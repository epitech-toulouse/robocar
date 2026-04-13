/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** master manager
*/

#ifndef MASTER_MANAGER_HPP
#define MASTER_MANAGER_HPP

#include "manager/AdvancedFusionEngine.hpp"
class MasterManager {
public:
    MasterManager() = default;
    ~MasterManager() = default;
private:
    AdvancedFusionEngine fusionEngine;
};

#endif /* MASTER_MANAGER_HPP */
