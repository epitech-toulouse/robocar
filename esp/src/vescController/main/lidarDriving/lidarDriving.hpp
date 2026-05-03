#pragma once

#include "../api/driving_algorithm_interface.hpp"
#include "../api/lidar_sensor_api.hpp"
#include "../drive.hpp"

#include <vector>

#ifndef LIDAR_DRIVING_HPP
#define LIDAR_DRIVING_HPP

class LidarDriving : public DrivingAlgorithmApi {
public:
    explicit LidarDriving(LidarSensorApi &lidar);
    ~LidarDriving() override = default;

    bool available(void) override;
    bool compute(DrivingAlgorithmOutput &output) override;
    float getPriority() override;

private:
    LidarSensorApi &lidar;
    AutonomousDriver driver;
};

#endif /* LIDAR_DRIVING_HPP */