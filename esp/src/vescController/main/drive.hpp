#pragma once

#include <vector>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lidarReader.hpp"

// LiDAR-only driving parameters.
constexpr float FRONT_WINDOW_DEG = 45.0f;
constexpr float SIDE_WINDOW_MIN_DEG = 45.0f;
constexpr float SIDE_WINDOW_MAX_DEG = 80.0f;

constexpr float STOP_DISTANCE_M = 0.60f;
constexpr float SLOW_DISTANCE_M = 1.00f;
constexpr float SAFE_DISTANCE_M = 2.20f;

constexpr float SPEED_FORWARD = 0.050f;
constexpr float SPEED_SLOW = 0.025f;
constexpr float SPEED_REVERSE = -0.030f;

constexpr float STEER_CENTER = 0.50f;
constexpr float STEER_LEFT = 0.20f;
constexpr float STEER_RIGHT = 0.80f;
constexpr bool AUTO_STEER_REVERSED = true;

// FTG Parameters
constexpr float FTG_MAX_RANGE_M = 3.0f; 
constexpr float FTG_CAR_WIDTH_M = 0.50f; 
constexpr float FTG_DISPARITY_THRESHOLD_M = 0.15f; 
constexpr float FTG_STEER_GAIN = 2.2f;
constexpr float FTG_MIN_COMMIT_STEER_DELTA = 0.12f;

constexpr uint32_t REVERSE_DURATION_MS = 800;
constexpr uint32_t ESCAPE_DURATION_MS = 700;
constexpr uint32_t RECOVERY_COOLDOWN_MS = 500;
constexpr float ESCAPE_SPEED = 0.040f;

struct DriveCommands {
    float steer;
    float duty;
};

class AutonomousDriver {
public:
    AutonomousDriver() = default;

    // Computes the next driving commands based on lidar scan.
    DriveCommands compute_commands(const std::vector<LidarPoint>& scan);

private:
    TickType_t reverseUntil = 0;
    TickType_t escapeUntil = 0;
    TickType_t recoveryCooldownUntil = 0;
    float reverseSteer = STEER_CENTER;
    float escapeSteer = STEER_CENTER;
};
