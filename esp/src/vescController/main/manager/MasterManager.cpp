/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** master manager
*/

#include <memory>
#include "manager/MasterManager.hpp"

#include "CoupeCircuitManager.hpp"

#include "sensors/bluetoothControlServer.hpp"
#include "sensors/cameraBleSensor.hpp"
#include "sensors/gpsSensor.hpp"
#include "sensors/lidarSensor.hpp"

#include "algo/lidarDrivingAlgo.hpp"
#include "algo/cameraDrivingAlgo.hpp"
#include "algo/closeObstacleAvoidanceAlgo.hpp"
#include "algo/gpsGoalAlgo.hpp"
#include "algo/userControllerAlgo.hpp"


#include "vesc/PhysicalVescController.hpp"
#include "esp_log.h"

MasterManager::MasterManager()
{
    this->vesc_controller_api = std::make_unique<PhysicalVescController>();
    this->camera_sensor_api = std::make_unique<CameraBleSensor>();
    this->gps_sensor_api = std::make_unique<GpsSensor>();
    this->lidar_sensor_api = std::make_unique<LidarSensor>();
    this->user_controller_api = std::make_unique<BluetoothControlServer>(
        *this->vesc_controller_api,
        this->algorithm_selector,
        this->gps_goal_state_,
        *this->camera_sensor_api,
        *this->gps_sensor_api,
        *this->lidar_sensor_api);
    this->coupe_circuit_manager = std::make_unique<CoupeCircuitManager>();
    this->vesc_controller_api->activate();

    this->fusionEngine.addDrivingAlgorithm(SelectableAlgorithm::Gps,
                                           std::make_unique<GpsGoalAlgo>(*this->gps_sensor_api, this->gps_goal_state_));
    this->fusionEngine.addDrivingAlgorithm(SelectableAlgorithm::Camera,
                                           std::make_unique<CameraDrivingAlgo>(*this->camera_sensor_api));
    this->fusionEngine.addDrivingAlgorithm(SelectableAlgorithm::CloseObstacle,
                                           std::make_unique<CloseObstacleAvoidanceAlgo>(*this->lidar_sensor_api));
    this->fusionEngine.addDrivingAlgorithm(SelectableAlgorithm::Manual,
                                           std::make_unique<UserControllerAlgo>(*this->user_controller_api));
    this->fusionEngine.addDrivingAlgorithm(SelectableAlgorithm::LidarCorridor,
                                           std::make_unique<LidarDrivingAlgo>(*this->lidar_sensor_api));
}

void MasterManager::iterate(void)
{
    static int iteration = 0;
    iteration++;
    DrivingAlgorithmOutput output = DEFAULT_DRIVING_ALGORITHM_OUTPUT;

    this->user_controller_api->pollControlMessages();
    output = this->fusionEngine.computeOutput(this->algorithm_selector);

    if (!output.computed_weight)
        this->vesc_controller_api->stop();
    if (iteration % 100 == 0) { // Log every 100 iterations to avoid spamming logs
        ESP_LOGD("MasterManager",
                 "Iteration %d: selected_mask=0x%02lx computed weight=%.3f",
                 iteration,
                 static_cast<unsigned long>(this->algorithm_selector.getSelectedMask()),
                 output.computed_weight);
    }
    this->vesc_controller_api->set_speed(output.target_speed);
    this->vesc_controller_api->set_steering(output.target_steering);
}
