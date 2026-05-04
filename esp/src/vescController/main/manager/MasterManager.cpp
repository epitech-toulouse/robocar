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
#include "algo/gpsGoalAlgo.hpp"
#include "algo/userControllerAlgo.hpp"


#include "vesc/PhysicalVescController.hpp"
#include "esp_log.h"

MasterManager::MasterManager()
{

    this->vesc_controller_api = std::make_unique<PhysicalVescController>();
    this->user_controller_api = std::make_unique<WifiControlServerSensor>(*this->vesc_controller_api);


    this->coupe_circuit_manager = std::make_unique<CoupeCircuitManager>();

    // this->gps_sensor_api = std::make_unique<GpsSensor>();
    this->lidar_sensor_api = std::make_unique<LidarSensor>();

    this->vesc_controller_api->activate();

    this->fusionEngine.addDrivingAlgorithm(std::make_unique<LidarDrivingAlgo>(*this->lidar_sensor_api));
    this->fusionEngine.addDrivingAlgorithm(std::make_unique<UserControllerAlgo>(*this->user_controller_api));
}

void MasterManager::iterate(void)
{
    DrivingAlgorithmOutput output = this->fusionEngine.computeOutput();
    if (!output.computed_weight)
        this->vesc_controller_api->stop();
        
    ESP_LOGI("MasterManager", "Computed output: speed=%.3f steer=%.3f weight=%.3f",
             output.target_speed, output.target_steering, output.computed_weight);
    this->vesc_controller_api->set_speed(output.target_speed);
    this->vesc_controller_api->set_steering(output.target_steering);
}
