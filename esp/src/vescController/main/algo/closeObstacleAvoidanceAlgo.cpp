#include "closeObstacleAvoidanceAlgo.hpp"
#include "algo/corridorLidar/drive.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

#include "../config.h"

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
        if (p.intensity == 0) continue;
        if (!in_sector(p.angleDeg, minDeg, maxDeg)) continue;
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

static float opposite_steer(float steer) {
    return clampf((STEER_LEFT + STEER_RIGHT) - steer, STEER_LEFT, STEER_RIGHT);
}

CloseObstacleAvoidanceAlgo::CloseObstacleAvoidanceAlgo(LidarSensorApi &lidar)
    : lidar(lidar), reverseUntil(0), escapeUntil(0), recoveryCooldownUntil(0), reverseSteer(STEER_CENTER), escapeSteer(STEER_CENTER) {}

bool CloseObstacleAvoidanceAlgo::available(void) {
    if (!lidar.isActive()) return false;

    const TickType_t now = xTaskGetTickCount();
    // Always available if we are in the middle of an escape/reverse maneuver
    if (now < escapeUntil || now < reverseUntil) {
        return true;
    }

    lidar_array_t rawData;
    if (!lidar.getData(rawData)) {
        return false;
    }

    // Convert rawData to scan to check front distance
    std::vector<LidarPoint> scan;
    scan.reserve(LIDAR_POINT_NUMBER);
    for (std::size_t i = 0; i < LIDAR_POINT_NUMBER; ++i) {
        const centimeter_t cm = rawData[i];
        if (cm == UNDEFINED_LIDAR_VALUE) continue;
        scan.push_back({
            static_cast<float>(i),
            static_cast<float>(cm) / 100.0f,
            1
        });
    }

    const float EMERGENCY_FRONT_WINDOW_DEG = 15.0f;
    const float frontLeft = nearest_in_sector(scan, 0.0f, EMERGENCY_FRONT_WINDOW_DEG);
    const float frontRight = nearest_in_sector(scan, 360.0f - EMERGENCY_FRONT_WINDOW_DEG, 360.0f);
    const float frontNear = min_valid_distance(frontLeft, frontRight);

    // If an obstacle is closer than AVOID_DISTANCE_M, we take over.
    if (frontNear > 0.0f && frontNear <= AVOID_DISTANCE_M) {
        return true;
    }

    return false;
}

bool CloseObstacleAvoidanceAlgo::compute(DrivingAlgorithmOutput &output) {
    lidar_array_t rawData;
    if (!lidar.getData(rawData)) {
        output = DEFAULT_DRIVING_ALGORITHM_OUTPUT;
        return false;
    }

    std::vector<LidarPoint> scan;
    scan.reserve(LIDAR_POINT_NUMBER);
    for (std::size_t i = 0; i < LIDAR_POINT_NUMBER; ++i) {
        const centimeter_t cm = rawData[i];
        if (cm == UNDEFINED_LIDAR_VALUE) continue;
        scan.push_back({
            static_cast<float>(i),
            static_cast<float>(cm) / 100.0f,
            1
        });
    }

    const float EMERGENCY_FRONT_WINDOW_DEG = 15.0f;
    const float frontLeft = nearest_in_sector(scan, 0.0f, EMERGENCY_FRONT_WINDOW_DEG);
    const float frontRight = nearest_in_sector(scan, 360.0f - EMERGENCY_FRONT_WINDOW_DEG, 360.0f);
    const float frontNear = min_valid_distance(frontLeft, frontRight);
    
    const float leftNear = nearest_in_sector(scan, SIDE_WINDOW_MIN_DEG, SIDE_WINDOW_MAX_DEG);
    const float rightNear = nearest_in_sector(scan, 360.0f - SIDE_WINDOW_MAX_DEG, 360.0f - SIDE_WINDOW_MIN_DEG);

    const TickType_t now = xTaskGetTickCount();

    // 1. Check if we are currently executing a reverse or escape maneuver
    if (now < reverseUntil) {
        // If we haven't reached a safe distance yet, keep extending the reverse duration
        if (frontNear > 0.0f && frontNear < SLOW_DISTANCE_M) {
            reverseUntil = now + pdMS_TO_TICKS(200); // Extend reverse by 200ms
            escapeUntil = reverseUntil + pdMS_TO_TICKS(ESCAPE_DURATION_MS);
            recoveryCooldownUntil = escapeUntil + pdMS_TO_TICKS(RECOVERY_COOLDOWN_MS);
        }
        
        output.target_speed = SPEED_REVERSE;
        output.target_steering = map_auto_steer(reverseSteer);
        output.computed_weight = 10.0f;
        return true;
    }
    if (escapeUntil != 0 && now >= reverseUntil && now < escapeUntil) {
        output.target_speed = ESCAPE_SPEED;
        output.target_steering = map_auto_steer(escapeSteer);
        output.computed_weight = 1.0f;
        return true;
    }
    if (escapeUntil != 0 && now >= escapeUntil) {
        escapeUntil = 0;
    }

    // 2. Too Close (Reverse Mode): Trigger if closer than STOP_DISTANCE_M (0.60m)
    if (now >= recoveryCooldownUntil && frontNear > 0.0f && frontNear <= STOP_DISTANCE_M) {
        const bool leftMoreOpen = (leftNear < 0.0f) || (rightNear > 0.0f && rightNear > leftNear);
        // Inverse direction: if left is more open, we want the front to go left.
        // Steering right in reverse makes the rear go right and the front swing left.
        reverseSteer = leftMoreOpen ? STEER_RIGHT : STEER_LEFT;
        escapeSteer = opposite_steer(reverseSteer);
        
        reverseUntil = now + pdMS_TO_TICKS(REVERSE_DURATION_MS);
        escapeUntil = reverseUntil + pdMS_TO_TICKS(ESCAPE_DURATION_MS);
        recoveryCooldownUntil = escapeUntil + pdMS_TO_TICKS(RECOVERY_COOLDOWN_MS);
        
        std::cout << "CloseObstacle: Reverse Mode, front=" << frontNear << "m" << std::endl;
        
        output.target_speed = SPEED_REVERSE;
        output.target_steering = map_auto_steer(reverseSteer);
        output.computed_weight = 10.0f;
        return true;
    }

    // 3. Really Close (Avoidance Mode): Trigger if between STOP_DISTANCE_M and AVOID_DISTANCE_M
    if (frontNear > 0.0f && frontNear <= AVOID_DISTANCE_M) {
        const bool leftMoreOpen = (leftNear < 0.0f) || (rightNear > 0.0f && rightNear > leftNear);
        float avoidSteer = leftMoreOpen ? STEER_LEFT : STEER_RIGHT;
        
        std::cout << "CloseObstacle: Avoid Mode, front=" << frontNear << "m steer=" << avoidSteer << std::endl;
        
        output.target_speed = SPEED_SLOW;
        output.target_steering = map_auto_steer(avoidSteer);
        output.computed_weight = 1.0f;
        return true;
    }

    // If we reach here, we shouldn't be active anymore (available() would have returned false)
    // But just in case, return false.
    output = DEFAULT_DRIVING_ALGORITHM_OUTPUT;
    return false;
}

float CloseObstacleAvoidanceAlgo::getPriority() {
    return LIDAR_AVOIDANCE_WEIGHT;
}
