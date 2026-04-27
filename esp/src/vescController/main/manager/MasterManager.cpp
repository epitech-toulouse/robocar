/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** master manager
*/

#include <memory>
#include "manager/MasterManager.hpp"

#include "CoupeCircuitManager.hpp"
#include "wifi_control_server.hpp"

#include "sensors/gpsSensor.hpp"
#include "sensors/lidarSensor.hpp"

#include "algo/lidarDrivingAlgo.hpp"
#include "algo/gpsGoalAlgo.hpp"


#include "vesc/PhysicalVescController.hpp"
#include "esp_log.h"

MasterManager::MasterManager()
{
    this->user_controller_api = std::make_unique<WifiControlServer>();
    // Start WIFI (please put this inside the constructor :D)
    static_cast<WifiControlServer *>(user_controller_api.get())->start();

    this->vesc_controller_api = std::make_unique<PhysicalVescController>();
    // to do
    // this->coupe_circuit_manager = std::make_unique<CoupeCircuitManager>(*this->vesc_controller_api);

    this->gps_sensor_api = std::make_unique<GpsSensor>();
    this->lidar_sensor_api = std::make_unique<LidarSensor>();

    // this->vesc_controller_api->activate();

    this->fusionEngine.addDrivingAlgorithm(std::make_unique<GpsGoalAlgo>(*this->gps_sensor_api));
    this->fusionEngine.addDrivingAlgorithm(std::make_unique<LidarDrivingAlgo>(*this->lidar_sensor_api));
}

void MasterManager::iterate(void)
{
    DrivingAlgorithmOutput output = this->fusionEngine.computeOutput();
    if (!output.computed_weight)
        this->vesc_controller_api->stop();
    this->vesc_controller_api->set_speed(output.target_speed);
    this->vesc_controller_api->set_steering(output.target_steering);
}
