/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** master manager
*/

#ifndef MASTER_MANAGER_HPP
#define MASTER_MANAGER_HPP

#include <memory>
#include "api/driving_algorithm_interface.hpp"
#include "manager/AdvancedFusionEngine.hpp"
#include "manager/DrivingModeSelector.hpp"
#include "api/gps_sensor_api.hpp"
#include "api/lidar_sensor_api.hpp"
#include "api/user_controller_api.hpp"
#include "api/vesc_controller_api.hpp"
#include "api/camera_sensor_api.hpp"
#include "CoupeCircuitManager.hpp"

class MasterManager {
public:
    MasterManager();
    ~MasterManager() = default;

    // 
    void iterate(void);
private:
    AdvancedFusionEngine fusionEngine;
    DrivingModeSelector driving_mode_selector;
    std::unique_ptr<GpsSensorApi> gps_sensor_api = nullptr;
    std::unique_ptr<LidarSensorApi> lidar_sensor_api = nullptr;
    std::unique_ptr<VescControllerApi> vesc_controller_api = nullptr;
    std::unique_ptr<UserControllerApi> user_controller_api = nullptr;
    std::unique_ptr<DrivingAlgorithmApi> close_obstacle_avoidance_algorithm = nullptr;
    std::unique_ptr<CameraSensorApi> camera_sensor_api = nullptr;
    std::unique_ptr<DrivingAlgorithmApi> corridor_lidar_algorithm = nullptr;
    std::unique_ptr<CoupeCircuitManager> coupe_circuit_manager = nullptr;
};

#endif /* MASTER_MANAGER_HPP */
