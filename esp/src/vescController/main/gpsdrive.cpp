#include "gpsdrive.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

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

static float map_auto_steer(float steer) {
    steer = clampf(steer, STEER_LEFT, STEER_RIGHT);
    if (AUTO_STEER_REVERSED) {
        steer = (STEER_LEFT + STEER_RIGHT) - steer;
    }
    return clampf(steer, STEER_LEFT, STEER_RIGHT);
}

static constexpr float FTG_FOV_RAD = 90.0f * (float)M_PI / 180.0f;  // -90..+90
static constexpr float GPS_HEADING_FULL_SCALE_DEG = 90.0f;

struct PolarPoint {
    float angle_rad;
    float dist_m;
};

static float normalize_angle_rad(float angleDeg) {
    if (angleDeg > 180.0f) {
        return (angleDeg - 360.0f) * (float)M_PI / 180.0f;
    }
    return angleDeg * (float)M_PI / 180.0f;
}

static void preprocess_ftg_scan(const std::vector<LidarPoint>& scan,
                                std::vector<PolarPoint>& out_scan,
                                std::vector<float>& out_processed_dists) {
    out_scan.clear();
    out_processed_dists.clear();
    out_scan.reserve(scan.size());

    for (const auto& pt : scan) {
        float rad = normalize_angle_rad(pt.angleDeg);
        if (rad >= -FTG_FOV_RAD && rad <= FTG_FOV_RAD) {
            float d = std::min(pt.distanceMeters, FTG_MAX_RANGE_M);
            out_scan.push_back({rad, d});
        }
    }

    if (out_scan.empty()) {
        return;
    }

    std::sort(out_scan.begin(), out_scan.end(), [](const PolarPoint& a, const PolarPoint& b) {
        return a.angle_rad < b.angle_rad;
    });

    out_processed_dists.resize(out_scan.size());
    for (size_t i = 0; i < out_scan.size(); ++i) {
        out_processed_dists[i] = out_scan[i].dist_m;
    }

    for (size_t i = 0; i + 1 < out_scan.size(); ++i) {
        float orig_d1 = out_scan[i].dist_m;
        float orig_d2 = out_scan[i + 1].dist_m;
        float pdiff = std::abs(orig_d1 - orig_d2);

        if (pdiff <= FTG_DISPARITY_THRESHOLD_M) {
            continue;
        }

        float closer_d = std::min(orig_d1, orig_d2);
        float angle_width = std::atan2(FTG_CAR_WIDTH_M / 2.0f, closer_d);

        if (orig_d1 < orig_d2) {
            float start_angle = out_scan[i].angle_rad;
            for (size_t j = i + 1; j < out_scan.size(); ++j) {
                if (out_scan[j].angle_rad - start_angle <= angle_width) {
                    out_processed_dists[j] = std::min(out_processed_dists[j], closer_d);
                } else {
                    break;
                }
            }
        } else {
            float start_angle = out_scan[i + 1].angle_rad;
            for (int j = (int)i; j >= 0; --j) {
                if (start_angle - out_scan[(size_t)j].angle_rad <= angle_width) {
                    out_processed_dists[(size_t)j] = std::min(out_processed_dists[(size_t)j], closer_d);
                } else {
                    break;
                }
            }
        }
    }
}

static void find_largest_max_gap(const std::vector<float>& processed_dists,
                                 float max_d,
                                 int& largest_gap_start,
                                 int& largest_gap_end) {
    largest_gap_start = -1;
    largest_gap_end = -1;
    int current_gap_start = -1;

    for (int i = 0; i < (int)processed_dists.size(); ++i) {
        if (processed_dists[(size_t)i] >= max_d - 0.05f) {
            if (current_gap_start == -1) {
                current_gap_start = i;
            }
        } else if (current_gap_start != -1) {
            int current_gap_len = i - current_gap_start;
            int largest_gap_len = (largest_gap_start != -1) ? (largest_gap_end - largest_gap_start + 1) : 0;
            if (current_gap_len > largest_gap_len) {
                largest_gap_start = current_gap_start;
                largest_gap_end = i - 1;
            }
            current_gap_start = -1;
        }
    }

    if (current_gap_start != -1) {
        int current_gap_len = (int)processed_dists.size() - current_gap_start;
        int largest_gap_len = (largest_gap_start != -1) ? (largest_gap_end - largest_gap_start + 1) : 0;
        if (current_gap_len > largest_gap_len) {
            largest_gap_start = current_gap_start;
            largest_gap_end = (int)processed_dists.size() - 1;
        }
    }

    if (largest_gap_start == -1) {
        largest_gap_start = 0;
        largest_gap_end = 0;
    }
}

static int select_goal_biased_index(const std::vector<PolarPoint>& fov_scan,
                                    int gap_start,
                                    int gap_end,
                                    bool headingValid,
                                    float headingErrorDeg) {
    if (!headingValid) {
        return (gap_start + gap_end) / 2;
    }

    float desired_angle = clampf(
        (headingErrorDeg / GPS_HEADING_FULL_SCALE_DEG) * FTG_FOV_RAD,
        -FTG_FOV_RAD,
        FTG_FOV_RAD);

    int best_index = gap_start;
    float best_error = 1e9f;

    for (int i = gap_start; i <= gap_end; ++i) {
        float e = std::fabs(fov_scan[(size_t)i].angle_rad - desired_angle);
        if (e < best_error) {
            best_error = e;
            best_index = i;
        }
    }

    return best_index;
}

DriveCommands GpsAutonomousDriver::compute_commands(const std::vector<LidarPoint>& scan,
                                                    const GpsDriveInput& gpsInput) {
    if (scan.empty()) {
        return {STEER_CENTER, 0.0f};
    }

    if (gpsInput.goalReached) {
        return {STEER_CENTER, 0.0f};
    }

    std::vector<LidarPoint> clean_scan;
    clean_scan.reserve(scan.size());
    for (const auto& pt : scan) {
        LidarPoint cp = pt;
        if (cp.distanceMeters < 0.0f) {
            cp.distanceMeters = 0.01f;
        } else if (cp.distanceMeters == 0.0f) {
            cp.distanceMeters = FTG_MAX_RANGE_M;
        }
        clean_scan.push_back(cp);
    }

    const float frontLeft = nearest_in_sector(clean_scan, 0.0f, FRONT_WINDOW_DEG);
    const float frontRight = nearest_in_sector(clean_scan, 360.0f - FRONT_WINDOW_DEG, 360.0f);
    const float frontNear = min_valid_distance(frontLeft, frontRight);

    const float leftNear = nearest_in_sector(clean_scan, SIDE_WINDOW_MIN_DEG, SIDE_WINDOW_MAX_DEG);
    const float rightNear = nearest_in_sector(clean_scan, 360.0f - SIDE_WINDOW_MAX_DEG, 360.0f - SIDE_WINDOW_MIN_DEG);

    const TickType_t now = xTaskGetTickCount();
    if (now < reverseUntil) {
        return {map_auto_steer(reverseSteer), SPEED_REVERSE};
    }

    if (frontNear > 0.0f && frontNear < STOP_DISTANCE_M) {
        const bool leftMoreOpen = (leftNear < 0.0f) || (rightNear > 0.0f && rightNear > leftNear);
        reverseSteer = leftMoreOpen ? STEER_RIGHT : STEER_LEFT;
        reverseUntil = now + pdMS_TO_TICKS(REVERSE_DURATION_MS);
        return {map_auto_steer(reverseSteer), SPEED_REVERSE};
    }

    std::vector<PolarPoint> fov_scan;
    std::vector<float> processed_dists;
    preprocess_ftg_scan(clean_scan, fov_scan, processed_dists);

    if (fov_scan.empty()) {
        return {STEER_CENTER, SPEED_SLOW};
    }

    float max_d = -1.0f;
    for (float d : processed_dists) {
        if (d > max_d) {
            max_d = d;
        }
    }

    int gap_start = 0;
    int gap_end = 0;
    find_largest_max_gap(processed_dists, max_d, gap_start, gap_end);

    int chosen_idx = select_goal_biased_index(
        fov_scan,
        gap_start,
        gap_end,
        gpsInput.headingValid,
        gpsInput.headingErrorDeg);

    const float best_angle = fov_scan[(size_t)chosen_idx].angle_rad;

    float steer_val = (best_angle * FTG_STEER_GAIN) / FTG_FOV_RAD;
    float steer = STEER_CENTER + steer_val * (STEER_LEFT - STEER_CENTER);
    steer = clampf(steer, STEER_LEFT, STEER_RIGHT);

    float angle_penalty = std::max(0.0f, std::cos(best_angle));
    float dist_factor = clampf((max_d - 0.5f) / (FTG_MAX_RANGE_M - 0.5f), 0.0f, 1.0f);
    float speed = SPEED_SLOW + (SPEED_FORWARD - SPEED_SLOW) * dist_factor * angle_penalty;

    if (gpsInput.headingValid) {
        float heading_norm = std::fabs(clampf(gpsInput.headingErrorDeg / GPS_HEADING_FULL_SCALE_DEG, -1.0f, 1.0f));
        speed *= (1.0f - 0.35f * heading_norm);
    }

    steer = map_auto_steer(steer);

    if (frontNear > 0.0f && frontNear <= STOP_DISTANCE_M) {
        speed = 0.0f;
    }

    std::cout << "GPS-FTG steer=" << steer
              << " speed=" << speed
              << " distGoal=" << gpsInput.distanceToGoalM
              << "m headingErr=" << gpsInput.headingErrorDeg
              << "deg headingValid=" << (gpsInput.headingValid ? "yes" : "no")
              << std::endl;

    return {steer, speed};
}
