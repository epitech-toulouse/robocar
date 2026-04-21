/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** master manager
*/

#include "manager/MasterManager.hpp"

#include "wifi_control_server.hpp"
#include "tmp/vescController.hpp"
#include "tmp/gpsSensor.hpp"
#include "tmp/lidarSensor.hpp"
#include "tmp/demoAlgo.hpp"

MasterManager::MasterManager()
{
    this->user_controller_api = std::make_unique<WifiControlServer>();
    // Start WIFI (please put this inside the constructor :D)
    static_cast<WifiControlServer *>(user_controller_api.get())->start();

    this->vesc_controller_api = std::make_unique<VescController>();
    this->gps_sensor_api = std::make_unique<GpsSensor>();
    this->lidar_sensor_api = std::make_unique<LidarSensor>();

    this->fusionEngine.addDrivingAlgorithm(std::make_unique<DemoAlgo>(*this->gps_sensor_api));
}

void MasterManager::iterate(void)
{
    DrivingAlgorithmOutput output = this->fusionEngine.computeOutput();
    if (!output.computed_weight)
        this->vesc_controller_api->stop();
    this->vesc_controller_api->set_speed(output.target_speed);
    this->vesc_controller_api->set_steering(output.target_steering);
}
