/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** master manager
*/

#ifndef MASTER_MANAGER_HPP
#define MASTER_MANAGER_HPP

#include <memory>
#include "manager/AdvancedFusionEngine.hpp"
#include "api/gps_sensor_api.hpp"
#include "api/lidar_sensor_api.hpp"
#include "api/user_controller_api.hpp"
#include "api/vesc_controller_api.hpp"

class MasterManager {
public:
    MasterManager();
    ~MasterManager() = default;

    // 
    void iterate(void);
private:
    AdvancedFusionEngine fusionEngine;
    std::unique_ptr<GpsSensorApi> gps_sensor_api = nullptr;
    std::unique_ptr<LidarSensorApi> lidar_sensor_api = nullptr;
    std::unique_ptr<VescControllerApi> vesc_controller_api = nullptr;
    std::unique_ptr<UserControllerApi> user_controller_api = nullptr;
};

#endif /* MASTER_MANAGER_HPP */
