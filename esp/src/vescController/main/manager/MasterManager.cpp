/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** master manager
*/

#include <memory>
#include "manager/MasterManager.hpp"

#include "CoupeCircuitManager.hpp"

#include "sensors/wifiControlServerSensor.hpp"
#include "sensors/gpsSensor.hpp"
#include "sensors/lidarSensor.hpp"

#include "algo/lidarDrivingAlgo.hpp"
#include "algo/closeObstacleAvoidanceAlgo.hpp"
#include "algo/gpsGoalAlgo.hpp"
#include "algo/userControllerAlgo.hpp"


#include "vesc/PhysicalVescController.hpp"
#include "esp_log.h"

MasterManager::MasterManager()
{
    this->vesc_controller_api = std::make_unique<PhysicalVescController>();
    this->gps_sensor_api = std::make_unique<GpsSensor>();
    this->lidar_sensor_api = std::make_unique<LidarSensor>();
    this->user_controller_api = std::make_unique<WifiControlServerSensor>(
        *this->vesc_controller_api,
        this->driving_mode_selector);
    this->coupe_circuit_manager = std::make_unique<CoupeCircuitManager>();

    this->vesc_controller_api->activate();

    //this->fusionEngine.addDrivingAlgorithm(std::make_unique<GpsGoalAlgo>(*this->gps_sensor_api));
    this->fusionEngine.addDrivingAlgorithm(std::make_unique<CloseObstacleAvoidanceAlgo>(*this->lidar_sensor_api));
    this->fusionEngine.addDrivingAlgorithm(std::make_unique<UserControllerAlgo>(*this->user_controller_api));
    this->fusionEngine.addDrivingAlgorithm(std::make_unique<LidarDrivingAlgo>(*this->lidar_sensor_api));
    this->corridor_lidar_algorithm = std::make_unique<LidarDrivingAlgo>(*this->lidar_sensor_api);
    this->close_obstacle_avoidance_algorithm = std::make_unique<CloseObstacleAvoidanceAlgo>(*this->lidar_sensor_api);
}

void MasterManager::iterate(void)
{
    static int iteration = 0;
    iteration++;
    DrivingAlgorithmOutput output = DEFAULT_DRIVING_ALGORITHM_OUTPUT;

    if (this->driving_mode_selector.isFusionMode()) {
        output = this->fusionEngine.computeOutput();
    } else if (this->close_obstacle_avoidance_algorithm != nullptr &&
               this->close_obstacle_avoidance_algorithm->available() &&
               this->close_obstacle_avoidance_algorithm->compute(output)) {
    } else if (this->corridor_lidar_algorithm != nullptr &&
               this->corridor_lidar_algorithm->available() &&
               this->corridor_lidar_algorithm->compute(output)) {
    } else {
        output = DEFAULT_DRIVING_ALGORITHM_OUTPUT;
    }

    if (!output.computed_weight)
        this->vesc_controller_api->stop();
    if (iteration % 100 == 0) { // Log every 100 iterations to avoid spamming logs
        ESP_LOGD("MasterManager",
                 "Iteration %d: mode=%s computed weight=%.3f",
                 iteration,
                 this->driving_mode_selector.modeString(),
                 output.computed_weight);
    }
    this->vesc_controller_api->set_speed(output.target_speed);
    this->vesc_controller_api->set_steering(output.target_steering);
}
