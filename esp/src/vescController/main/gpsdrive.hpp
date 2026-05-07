#pragma once

#include <vector>

#include "algo/corridorLidar/drive.hpp"
#include "lidarReader.hpp"

struct GpsDriveInput {
    bool goalReached = false;
    bool headingValid = false;
    float headingErrorDeg = 0.0f;
    float distanceToGoalM = -1.0f;
};

class GpsAutonomousDriver {
public:
    GpsAutonomousDriver() = default;

    // Temporary GPS-only autonomous commands: drive toward goal heading and distance.
    DriveCommands compute_commands(const std::vector<LidarPoint>& scan, const GpsDriveInput& gpsInput);

private:
    TickType_t reverseUntil = 0;
    float reverseSteer = STEER_CENTER;
};
