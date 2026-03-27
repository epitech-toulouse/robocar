/**
 * @file navigation_controller.cpp
 * @brief GPS+LIDAR fusion navigation controller.
 *
 * Algorithm — priority-based fusion:
 *   1. Arrival check (distance < threshold → stop).
 *   2. Emergency obstacle (LIDAR front < STOP_DISTANCE → reverse).
 *   3. Obstacle avoidance (LIDAR front < SLOW_DISTANCE → blend avoidance).
 *   4. GPS heading correction (trajectory score → steering).
 *   5. Speed scaling by alignment quality + distance + obstacles.
 *
 * Steering output: 0.0 (left) → 0.5 (center) → 1.0 (right).
 * Score convention: positive = turn right, negative = turn left.
 */

#include "navigation_controller.hpp"
#include "heading_alignment.hpp"
#include "trajectory_score.hpp"
#include "obstacle_avoidance.hpp"
#include "target_point.hpp"

#include "esp_log.h"
#include <cmath>
#include <algorithm>
#include <iostream>

static const char *TAG = "nav_ctrl";

/// Clamp a float to [lo, hi].
static float clamp(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

NavCommands NavigationController::compute(const GpsFix &gps,
                                          const std::vector<LidarPoint> &scan,
                                          float target_lat, float target_lon) {
    NavCommands cmd = {NAV_STEER_CENTER, 0.0f};
    const TickType_t now = xTaskGetTickCount();

    /* -------------------------------------------------------------------- */
    /*  0. Check if heading is available                                     */
    /* -------------------------------------------------------------------- */
    float heading = gps.heading_deg;
    bool has_heading = !std::isnan(heading);

    /* -------------------------------------------------------------------- */
    /*  1. GPS alignment computation                                        */
    /* -------------------------------------------------------------------- */
    AlignmentResult align = {};
    float score = 0.0f;

    if (has_heading) {
        align = compute_alignment(gps.lat, gps.lon, heading,
                                  target_lat, target_lon);
        score = compute_trajectory_score(align.angular_error_deg);
    } else {
        // No heading — compute distance only, can't correct course
        align.distance_m  = compute_distance(gps.lat, gps.lon,
                                             target_lat, target_lon);
        align.bearing_deg = compute_bearing(gps.lat, gps.lon,
                                            target_lat, target_lon);
        align.angular_error_deg = 0.0f;
        align.is_aligned = false;
    }

    /* -------------------------------------------------------------------- */
    /*  2. Arrival check                                                     */
    /* -------------------------------------------------------------------- */
    if (align.distance_m < NAV_ARRIVAL_M) {
        nav_set_state(NavState::ARRIVED);
        cmd.duty  = 0.0f;
        cmd.steer = NAV_STEER_CENTER;
        ESP_LOGI(TAG, "ARRIVED at target (%.1fm)", align.distance_m);
        goto telemetry;
    }

    /* -------------------------------------------------------------------- */
    /*  3. LIDAR obstacle analysis                                          */
    /* -------------------------------------------------------------------- */
    {
        ObstacleResult obs = analyze_obstacles(scan);

        /* 3a. Check if we are in reverse mode */
        if (now < reverse_until) {
            cmd.steer = reverse_steer;
            cmd.duty  = NAV_REVERSE_SPEED;
            goto telemetry;
        }

        /* 3b. Emergency: obstacle critically close → reverse */
        if (obs.emergency) {
            // Choose reverse direction: steer away from closest side
            if (obs.left_min_m > obs.right_min_m) {
                reverse_steer = NAV_STEER_RIGHT;  // Back up while turning right
            } else {
                reverse_steer = NAV_STEER_LEFT;    // Back up while turning left
            }
            reverse_until = now + pdMS_TO_TICKS(NAV_REVERSE_DURATION_MS);
            cmd.steer = reverse_steer;
            cmd.duty  = NAV_REVERSE_SPEED;
            ESP_LOGW(TAG, "REVERSE: front=%.2fm L=%.2fm R=%.2fm",
                     obs.front_min_m, obs.left_min_m, obs.right_min_m);
            goto telemetry;
        }

        /* 3c. Obstacle within slow distance → blend LIDAR avoidance + GPS */
        if (obs.front_min_m < NAV_SLOW_DISTANCE_M) {
            // LIDAR avoidance has higher priority when obstacle is close
            float lidar_steer = obs.avoidance_steer;
            float gps_steer   = NAV_STEER_CENTER;

            if (has_heading) {
                // GPS correction: score → steer offset
                // score ∈ [-1, +1], map to steer offset around center
                float gps_offset = score * (NAV_STEER_RIGHT - NAV_STEER_CENTER);
                gps_steer = NAV_STEER_CENTER + gps_offset;
            }

            // Proximity factor: closer obstacle → more LIDAR influence
            float proximity = 1.0f - (obs.front_min_m / NAV_SLOW_DISTANCE_M);
            proximity = clamp(proximity, 0.0f, 1.0f);

            // Blend: as obstacle gets closer, LIDAR weight increases
            float lidar_w = NAV_LIDAR_WEIGHT + proximity * (1.0f - NAV_LIDAR_WEIGHT);
            float gps_w   = 1.0f - lidar_w;

            cmd.steer = lidar_w * lidar_steer + gps_w * gps_steer;
            cmd.steer = clamp(cmd.steer, NAV_STEER_LEFT, NAV_STEER_RIGHT);

            // Speed reduction near obstacles
            float dist_factor = obs.front_min_m / NAV_SLOW_DISTANCE_M;
            cmd.duty = NAV_SLOW_SPEED * dist_factor;
            goto telemetry;
        }

        /* 3d. Clear path → GPS-only steering */
        if (has_heading) {
            // Map trajectory score to steering:
            // score +1 → full right (0.8), score -1 → full left (0.2)
            float steer_offset = score * (NAV_STEER_RIGHT - NAV_STEER_CENTER);
            cmd.steer = NAV_STEER_CENTER + steer_offset;
            cmd.steer = clamp(cmd.steer, NAV_STEER_LEFT, NAV_STEER_RIGHT);
        } else {
            // No heading: go straight and hope for GPS update
            cmd.steer = NAV_STEER_CENTER;
        }

        /* Speed: scale by alignment quality and distance */
        {
            // Alignment factor: slow down when misaligned
            float align_factor = 1.0f;
            if (has_heading) {
                float abs_score = fabsf(score);
                // Full speed when score < 0.3, progressively slow when misaligned
                align_factor = 1.0f - 0.7f * abs_score;
                align_factor = clamp(align_factor, 0.3f, 1.0f);
            }

            // Distance factor: slow down when close to target
            float dist_factor = 1.0f;
            if (align.distance_m < 5.0f) {
                dist_factor = align.distance_m / 5.0f;
                dist_factor = clamp(dist_factor, 0.3f, 1.0f);
            }

            // Obstacle proximity factor (front in safe..inf range)
            float obs_factor = 1.0f;
            if (obs.front_min_m < NAV_SAFE_DISTANCE_M) {
                obs_factor = (obs.front_min_m - NAV_STOP_DISTANCE_M)
                           / (NAV_SAFE_DISTANCE_M - NAV_STOP_DISTANCE_M);
                obs_factor = clamp(obs_factor, 0.0f, 1.0f);
            }

            cmd.duty = NAV_MAX_SPEED * align_factor * dist_factor * obs_factor;
            cmd.duty = clamp(cmd.duty, 0.0f, NAV_MAX_SPEED);
        }
    }

telemetry:
    /* -------------------------------------------------------------------- */
    /*  Populate telemetry                                                   */
    /* -------------------------------------------------------------------- */
    last_telemetry.lat               = gps.lat;
    last_telemetry.lon               = gps.lon;
    last_telemetry.heading_deg       = heading;
    last_telemetry.target_bearing_deg = align.bearing_deg;
    last_telemetry.angular_error_deg = align.angular_error_deg;
    last_telemetry.score             = score;
    last_telemetry.distance_m        = align.distance_m;
    last_telemetry.speed             = cmd.duty;
    last_telemetry.steer             = cmd.steer;
    last_telemetry.obstacle_detected = (cmd.duty < 0.0f); // reverse = obstacle
    last_telemetry.has_gps           = gps.has_fix;
    last_telemetry.arrived           = (nav_get_state() == NavState::ARRIVED);

    // Throttled log output (every 500ms)
    if ((now - last_log_tick) > pdMS_TO_TICKS(500)) {
        last_log_tick = now;
        std::cout << "NAV "
                  << "dist=" << align.distance_m << "m "
                  << "bearing=" << align.bearing_deg << "° "
                  << "err=" << align.angular_error_deg << "° "
                  << "score=" << score << " "
                  << "steer=" << cmd.steer << " "
                  << "duty=" << cmd.duty
                  << (has_heading ? "" : " [NO_HDG]")
                  << std::endl;
    }

    return cmd;
}
