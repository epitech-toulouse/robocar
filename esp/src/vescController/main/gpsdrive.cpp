#include "gpsdrive.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

static float clampf(float value, float lo, float hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static float map_auto_steer(float steer) {
    steer = clampf(steer, STEER_LEFT, STEER_RIGHT);
    if (AUTO_STEER_REVERSED) {
        steer = (STEER_LEFT + STEER_RIGHT) - steer;
    }
    return clampf(steer, STEER_LEFT, STEER_RIGHT);
}

static constexpr float GPS_HEADING_FULL_SCALE_DEG = 90.0f;
static constexpr float GPS_DISTANCE_FULL_SPEED_M = 3.0f;

DriveCommands GpsAutonomousDriver::compute_commands(const std::vector<LidarPoint>& scan,
                                                    const GpsDriveInput& gpsInput) {
    (void)scan;

    if (gpsInput.goalReached) {
        return {STEER_CENTER, 0.0f};
    }

    float steer = STEER_CENTER;
    float speed = SPEED_SLOW;

    if (gpsInput.headingValid) {
        const float headingNorm = clampf(gpsInput.headingErrorDeg / GPS_HEADING_FULL_SCALE_DEG, -1.0f, 1.0f);
        steer = STEER_CENTER + headingNorm * (STEER_LEFT - STEER_CENTER);

        const float turnPenalty = 1.0f - 0.60f * std::fabs(headingNorm);
        const float distanceFactor =
            (gpsInput.distanceToGoalM > 0.0f)
                ? clampf(gpsInput.distanceToGoalM / GPS_DISTANCE_FULL_SPEED_M, 0.15f, 1.0f)
                : 0.40f;

        speed = SPEED_SLOW + (SPEED_FORWARD - SPEED_SLOW) * distanceFactor;
        speed *= clampf(turnPenalty, 0.25f, 1.0f);

        if (std::fabs(headingNorm) > 0.85f) {
            speed = std::min(speed, SPEED_SLOW);
        }
    } else {
        steer = STEER_CENTER;
        speed = SPEED_SLOW * 0.7f;
    }

    steer = map_auto_steer(steer);

    std::cout << "GPS-only steer=" << steer
              << " speed=" << speed
              << " distGoal=" << gpsInput.distanceToGoalM
              << "m headingErr=" << gpsInput.headingErrorDeg
              << "deg headingValid=" << (gpsInput.headingValid ? "yes" : "no")
              << std::endl;

    return {steer, speed};
}
