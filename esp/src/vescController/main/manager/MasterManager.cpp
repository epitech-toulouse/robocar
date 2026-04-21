/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** master manager
*/

#include "manager/MasterManager.hpp"

#include "wifi_control_server.hpp"

MasterManager::MasterManager()
{
    this->user_controller_api = std::make_unique<WifiControlServer>();
    static_cast<WifiControlServer *>(user_controller_api.get())->start();
    // Waiting here for
    // - VESC CONTROLLER
    // - GPS SENSOR
    // - LIDAR SENSOR
}

void MasterManager::iterate(void)
{
    
}
