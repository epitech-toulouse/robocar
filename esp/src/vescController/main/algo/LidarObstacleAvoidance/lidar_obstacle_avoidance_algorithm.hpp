#pragma once

#include "api/driving_algorithm_interface.hpp"
#include "api/lidar_sensor_api.hpp"
#include "drive.hpp"

class LidarObstacleAvoidanceAlgorithm : public IDrivingAlgorithm {
public:
    explicit LidarObstacleAvoidanceAlgorithm(ILidarSensor& lidarSensor);
    ~LidarObstacleAvoidanceAlgorithm() override = default;

    bool available(void) override;
    bool compute(DrivingAlgorithmOutput& output) override;
    float getPriority() override;

private:
    ILidarSensor& lidarSensor;
    AutonomousDriver driver;
};
