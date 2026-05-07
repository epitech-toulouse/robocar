#ifndef CLOSE_OBSTACLE_AVOIDANCE_ALGO_HPP
#define CLOSE_OBSTACLE_AVOIDANCE_ALGO_HPP

#include "api/driving_algorithm_interface.hpp"
#include "api/lidar_sensor_api.hpp"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

class CloseObstacleAvoidanceAlgo : public DrivingAlgorithmApi {
public:
    explicit CloseObstacleAvoidanceAlgo(LidarSensorApi &lidar);
    ~CloseObstacleAvoidanceAlgo() override = default;

    bool available(void) override;
    bool compute(DrivingAlgorithmOutput &output) override;
    float getPriority() override;

private:
    LidarSensorApi &lidar;
    
    TickType_t reverseUntil;
    TickType_t escapeUntil;
    TickType_t recoveryCooldownUntil;
    float reverseSteer;
    float escapeSteer;
};

#endif /* CLOSE_OBSTACLE_AVOIDANCE_ALGO_HPP */
