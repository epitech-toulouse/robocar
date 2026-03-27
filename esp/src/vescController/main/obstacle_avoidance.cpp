/**
 * @file obstacle_avoidance.cpp
 * @brief LIDAR sector analysis for obstacle detection.
 *
 * Scans front, left, and right sectors to determine nearest obstacles
 * and suggests an avoidance steering value.
 * Reuses the sector geometry from drive.hpp constants.
 */

#include "obstacle_avoidance.hpp"
#include "drive.hpp"  // For FRONT_WINDOW_DEG, SIDE_WINDOW_* constants
#include <cmath>
#include <algorithm>

/// Find nearest valid distance in a sector [minDeg, maxDeg].
/// Angles are in 0..360° space (as provided by LidarReader).
static float nearest_in_nav_sector(const std::vector<LidarPoint> &scan,
                                   float minDeg, float maxDeg) {
    float nearest = -1.0f;
    for (const auto &p : scan) {
        // Handle wrap-around: if minDeg > maxDeg, it wraps through 0°
        bool in_range;
        if (minDeg <= maxDeg) {
            in_range = (p.angleDeg >= minDeg && p.angleDeg <= maxDeg);
        } else {
            in_range = (p.angleDeg >= minDeg || p.angleDeg <= maxDeg);
        }
        if (!in_range) continue;

        float d = p.distanceMeters;
        if (d <= 0.0f) continue;
        if (nearest < 0.0f || d < nearest) {
            nearest = d;
        }
    }
    return nearest;
}

ObstacleResult analyze_obstacles(const std::vector<LidarPoint> &scan) {
    ObstacleResult result;

    if (scan.empty()) {
        // No data — assume clear (conservative: could also assume blocked)
        return result;
    }

    // Clean scan: handle -1 (touching sensor) and 0 (no reflection)
    std::vector<LidarPoint> clean;
    clean.reserve(scan.size());
    for (const auto &pt : scan) {
        LidarPoint cp = pt;
        if (cp.distanceMeters < 0.0f) {
            cp.distanceMeters = 0.01f;  // Obstacle touching sensor
        } else if (cp.distanceMeters == 0.0f) {
            cp.distanceMeters = NAV_SAFE_DISTANCE_M + 1.0f;  // No signal → treat as far
        }
        clean.push_back(cp);
    }

    // Front sector: 0..FRONT_WINDOW_DEG and (360-FRONT_WINDOW_DEG)..360
    float front_left  = nearest_in_nav_sector(clean, 0.0f, FRONT_WINDOW_DEG);
    float front_right = nearest_in_nav_sector(clean, 360.0f - FRONT_WINDOW_DEG, 360.0f);

    // Combine front: take the nearer of left-front and right-front
    if (front_left < 0.0f)       result.front_min_m = front_right;
    else if (front_right < 0.0f) result.front_min_m = front_left;
    else                         result.front_min_m = std::min(front_left, front_right);
    if (result.front_min_m < 0.0f) result.front_min_m = 999.0f;

    // Left sector (LIDAR angles: positive = left in 0..360 space)
    float left_near = nearest_in_nav_sector(clean, SIDE_WINDOW_MIN_DEG, SIDE_WINDOW_MAX_DEG);
    result.left_min_m = (left_near > 0.0f) ? left_near : 999.0f;

    // Right sector (symmetrical on the other side)
    float right_near = nearest_in_nav_sector(clean,
                                             360.0f - SIDE_WINDOW_MAX_DEG,
                                             360.0f - SIDE_WINDOW_MIN_DEG);
    result.right_min_m = (right_near > 0.0f) ? right_near : 999.0f;

    // Emergency: obstacle critically close in front
    result.emergency = (result.front_min_m < NAV_STOP_DISTANCE_M);

    // Suggested avoidance steering when obstacle is within slow distance
    if (result.front_min_m < NAV_SLOW_DISTANCE_M && !result.emergency) {
        // Steer toward the more open side
        if (result.left_min_m > result.right_min_m) {
            // More space on left → steer left
            float urgency = 1.0f - (result.front_min_m / NAV_SLOW_DISTANCE_M);
            result.avoidance_steer = NAV_STEER_CENTER
                - urgency * (NAV_STEER_CENTER - NAV_STEER_LEFT);
        } else {
            // More space on right → steer right
            float urgency = 1.0f - (result.front_min_m / NAV_SLOW_DISTANCE_M);
            result.avoidance_steer = NAV_STEER_CENTER
                + urgency * (NAV_STEER_RIGHT - NAV_STEER_CENTER);
        }
    }
    // If emergency, avoidance_steer is set by the controller (reverse logic)

    return result;
}
