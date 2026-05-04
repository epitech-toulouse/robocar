#include "drive.hpp"
#include <cmath>
#include <algorithm>
#if !__has_include("freertos/task.h")
#include <chrono>
#endif

static TickType_t get_tick_count()
{
#if __has_include("freertos/task.h")
    return xTaskGetTickCount();
#else
    using namespace std::chrono;
    return static_cast<TickType_t>(duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count());
#endif
}

static float clampf(float value, float lo, float hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static bool in_sector(float angleDeg, float minDeg, float maxDeg) {
    return angleDeg >= minDeg && angleDeg <= maxDeg;
}

static float nearest_in_sector(const std::vector<LidarPoint>& scan, float minDeg, float maxDeg) {
    float nearest = -1.0f;
    for (const auto& p : scan) {
        if (!in_sector(p.angleDeg, minDeg, maxDeg)) {
            continue;
        }
        if (nearest < 0.0f || p.distanceMeters < nearest) {
            nearest = p.distanceMeters;
        }
    }
    return nearest;
}

static float min_valid_distance(float a, float b) {
    if (a < 0.0f) return b;
    if (b < 0.0f) return a;
    return std::min(a, b);
}

static float compute_front_speed_cap(float frontNear)
{
    if (frontNear < 0.0f) {
        return SPEED_FORWARD;
    }
    if (frontNear <= STOP_DISTANCE_M) {
        return 0.0f;
    }
    if (frontNear <= SLOW_DISTANCE_M) {
        const float t = clampf((frontNear - STOP_DISTANCE_M) / (SLOW_DISTANCE_M - STOP_DISTANCE_M), 0.0f, 1.0f);
        return SPEED_CRAWL + (SPEED_SLOW - SPEED_CRAWL) * t;
    }
    if (frontNear >= SAFE_DISTANCE_M) {
        return SPEED_FORWARD;
    }

    const float t = clampf((frontNear - SLOW_DISTANCE_M) / (SAFE_DISTANCE_M - SLOW_DISTANCE_M), 0.0f, 1.0f);
    return SPEED_SLOW + (SPEED_FORWARD - SPEED_SLOW) * t;
}

static float sanitize_clearance(float distance)
{
    return (distance < 0.0f) ? FTG_MAX_RANGE_M : distance;
}

static float choose_escape_steer(float ftgSteer, float leftNear, float rightNear)
{
    const float leftClearance = sanitize_clearance(leftNear);
    const float rightClearance = sanitize_clearance(rightNear);

    if (leftClearance > rightClearance + SIDE_OPENNESS_EPSILON_M) {
        return STEER_LEFT;
    }
    if (rightClearance > leftClearance + SIDE_OPENNESS_EPSILON_M) {
        return STEER_RIGHT;
    }

    // If both sides look equally bad, keep the current FTG intent if it is already
    // meaningfully off-center, otherwise pick a deterministic left turn.
    if (ftgSteer < STEER_CENTER - 0.05f) {
        return STEER_LEFT;
    }
    if (ftgSteer > STEER_CENTER + 0.05f) {
        return STEER_RIGHT;
    }
    return STEER_LEFT;
}

static float map_auto_steer(float steer) {
    steer = clampf(steer, STEER_LEFT, STEER_RIGHT);
    if (AUTO_STEER_REVERSED) {
        steer = (STEER_LEFT + STEER_RIGHT) - steer;
    }
    return clampf(steer, STEER_LEFT, STEER_RIGHT);
}

static constexpr float FTG_FOV_RAD = 90.0f * (float)M_PI / 180.0f; // -90 to +90 degrees

struct PolarPoint {
    float angle_rad;
    float dist_m;
};

static float normalize_angle_rad(float angleDeg) {
    if (angleDeg > 180.0f) {
        return (angleDeg - 360.0f) * (float)M_PI / 180.0f; // negative for right
    }
    return angleDeg * (float)M_PI / 180.0f; // positive for left
}

static void compute_ftg_steer_speed(const std::vector<LidarPoint>& scan, float& out_steer, float& out_speed) {
    std::vector<PolarPoint> fov_scan;
    fov_scan.reserve(scan.size());

    // Filter to FOV and clamp range
    for (const auto& pt : scan) {
        float rad = normalize_angle_rad(pt.angleDeg);
        if (rad >= -FTG_FOV_RAD && rad <= FTG_FOV_RAD) {
            float d = std::min(pt.distanceMeters, FTG_MAX_RANGE_M);
            fov_scan.push_back({rad, d});
        }
    }

    if (fov_scan.empty()) {
        out_steer = STEER_CENTER;
        out_speed = SPEED_FORWARD;
        return;
    }

    // Sort by angle from right (negative) to left (positive)
    std::sort(fov_scan.begin(), fov_scan.end(), [](const PolarPoint& a, const PolarPoint& b) {
        return a.angle_rad < b.angle_rad;
    });

    // Disparity Extender
    std::vector<float> processed_dists(fov_scan.size());
    for (size_t i = 0; i < fov_scan.size(); ++i) {
        processed_dists[i] = fov_scan[i].dist_m;
    }

    for (size_t i = 0; i < fov_scan.size() - 1; ++i) {
        float orig_d1 = fov_scan[i].dist_m;   // Right side
        float orig_d2 = fov_scan[i+1].dist_m; // Left side
        float pdiff = std::abs(orig_d1 - orig_d2);

        if (pdiff > FTG_DISPARITY_THRESHOLD_M) {
            float closer_d = std::min(orig_d1, orig_d2);
            // angle_width = atan( (width / 2) / closer_d )
            float angle_width = std::atan2(FTG_CAR_WIDTH_M / 2.0f, closer_d);

            if (orig_d1 < orig_d2) { // point i is closer, extend forward to left
                float start_angle = fov_scan[i].angle_rad;
                for (size_t j = i + 1; j < fov_scan.size(); ++j) {
                    if (fov_scan[j].angle_rad - start_angle <= angle_width) {
                        processed_dists[j] = std::min(processed_dists[j], closer_d);
                    } else {
                        break;
                    }
                }
            } else { // point i+1 is closer, extend backward to right
                float start_angle = fov_scan[i+1].angle_rad;
                for (int j = (int)i; j >= 0; --j) {
                    if (start_angle - fov_scan[j].angle_rad <= angle_width) {
                        processed_dists[j] = std::min(processed_dists[j], closer_d);
                    } else {
                        break;
                    }
                }
            }
        }
    }

    // Find Best Point (Max distance array, find the maximum safely)
    float max_d = -1.0f;
    for (size_t i = 0; i < fov_scan.size(); ++i) {
        if (processed_dists[i] > max_d) {
            max_d = processed_dists[i];
        }
    }

    // Find the LARGEST continuous gap of points that are close to max_d
    int largest_gap_start = -1;
    int largest_gap_end = -1;
    int current_gap_start = -1;

    for (int i = 0; i < (int)fov_scan.size(); ++i) {
        if (processed_dists[i] >= max_d - 0.05f) {
            if (current_gap_start == -1) {
                current_gap_start = i;
            }
        } else {
            if (current_gap_start != -1) {
                int current_gap_len = i - current_gap_start;
                int largest_gap_len = (largest_gap_start != -1) ? (largest_gap_end - largest_gap_start + 1) : 0;

                if (current_gap_len > largest_gap_len) {
                    largest_gap_start = current_gap_start;
                    largest_gap_end = i - 1;
                }
                current_gap_start = -1;
            }
        }
    }
    // Handle gap at the end of the array
    if (current_gap_start != -1) {
        int current_gap_len = (int)fov_scan.size() - current_gap_start;
        int largest_gap_len = (largest_gap_start != -1) ? (largest_gap_end - largest_gap_start + 1) : 0;
        if (current_gap_len > largest_gap_len) {
            largest_gap_start = current_gap_start;
            largest_gap_end = (int)fov_scan.size() - 1;
        }
    }

    if (largest_gap_start == -1) {
        // Fallback (shouldn't happen since max_d exists)
        largest_gap_start = 0;
        largest_gap_end = 0;
    }

    size_t center_idx = (largest_gap_start + largest_gap_end) / 2;
    float best_angle = fov_scan[center_idx].angle_rad;

    // Compute Steer & Speed
    float steer_val = (best_angle * FTG_STEER_GAIN) / FTG_FOV_RAD;  // -1 to 1 (left to right)

    // Map it to [STEER_LEFT, STEER_RIGHT]
    out_steer = STEER_CENTER + steer_val * (STEER_LEFT - STEER_CENTER);
    out_steer = clampf(out_steer, STEER_LEFT, STEER_RIGHT);

    float angle_penalty = std::max(0.0f, std::cos(best_angle));
    float dist_factor = clampf((max_d - 0.5f) / (FTG_MAX_RANGE_M - 0.5f), 0.0f, 1.0f);

    out_speed = SPEED_SLOW + (SPEED_FORWARD - SPEED_SLOW) * dist_factor * angle_penalty;
}

DriveCommands AutonomousDriver::compute_commands(const std::vector<LidarPoint>& scan) {
    if (scan.empty()) {
        return {STEER_CENTER, 0.0f};
    }

    // Clean scan directly to avoid bugs with `-1` representing objects touching the LiDAR,
    // and `0` usually representing invalid points or "no reflection" (treated as Max Range).
    std::vector<LidarPoint> clean_scan;
    clean_scan.reserve(scan.size());
    for (const auto& pt : scan) {
        LidarPoint cp = pt;
        if (cp.distanceMeters < 0.0f) {
            cp.distanceMeters = 0.01f; // Critical obstacle touching the sensor
        } else if (cp.distanceMeters == 0.0f) {
            cp.distanceMeters = FTG_MAX_RANGE_M; // No signal / infinite distance
        }
        clean_scan.push_back(cp);
    }

    const float frontLeft = nearest_in_sector(clean_scan, 0.0f, FRONT_WINDOW_DEG);
    const float frontRight = nearest_in_sector(clean_scan, 360.0f - FRONT_WINDOW_DEG, 360.0f);
    const float frontNear = min_valid_distance(frontLeft, frontRight);

    const float leftNear = nearest_in_sector(clean_scan, SIDE_WINDOW_MIN_DEG, SIDE_WINDOW_MAX_DEG);
    const float rightNear = nearest_in_sector(clean_scan, 360.0f - SIDE_WINDOW_MAX_DEG, 360.0f - SIDE_WINDOW_MIN_DEG);

    const TickType_t now = get_tick_count();
    if (now < reverseUntil) {
        return {map_auto_steer(reverseSteer), SPEED_REVERSE};
    }

    // If there is a critical obstacle in front, back up and turn toward the more open side.
    if (frontNear > 0.0f && frontNear <= STOP_DISTANCE_M + 0.05f) {
        const bool leftMoreOpen = (leftNear < 0.0f) || (rightNear > 0.0f && rightNear > leftNear);
        reverseSteer = leftMoreOpen ? STEER_RIGHT : STEER_LEFT;
        reverseUntil = now + pdMS_TO_TICKS(REVERSE_DURATION_MS);
        return {map_auto_steer(reverseSteer), SPEED_REVERSE};
    }

    float steer = STEER_CENTER;
    float speed = SPEED_FORWARD;

    // Compute steer and speed using Disparity Extender FTG
    compute_ftg_steer_speed(clean_scan, steer, speed);

    if (frontNear > 0.0f && frontNear < ESCAPE_TURN_START_M) {
        const float escapeSteer = choose_escape_steer(steer, leftNear, rightNear);
        const float avoidanceWeight = clampf(
            (ESCAPE_TURN_START_M - frontNear) / (ESCAPE_TURN_START_M - STOP_DISTANCE_M),
            0.0f,
            1.0f);
        steer = steer * (1.0f - avoidanceWeight) + escapeSteer * avoidanceWeight;
        if (frontNear < SAFE_DISTANCE_M) {
            steer = escapeSteer;
        }
        speed = std::min(speed, SPEED_CRAWL + (SPEED_SLOW - SPEED_CRAWL) * (1.0f - avoidanceWeight));
    }

    steer = map_auto_steer(steer);

    // Clamp forward speed based on what is directly ahead so we don't keep charging
    // into a wall while FTG still sees lateral free space.
    speed = std::min(speed, compute_front_speed_cap(frontNear));

    // Emergency stop override if something is completely blocked right in front
    if (frontNear > 0.0f && frontNear <= STOP_DISTANCE_M) {
        speed = 0.0f;
    }

    return {steer, speed};
}
