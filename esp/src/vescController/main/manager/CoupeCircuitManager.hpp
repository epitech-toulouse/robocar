/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** coupe circuit manager
*/

#ifndef COUPE_CIRCUIT_MANAGER_HPP
#define COUPE_CIRCUIT_MANAGER_HPP

#include "api/vesc_controller_api.hpp"

class CoupeCircuitManager
{
public:
    CoupeCircuitManager(VescControllerApi &vesc);
    ~CoupeCircuitManager();

    static void task(void *args);
private:
    VescControllerApi &vesc;
};

#endif /* COUPE_CIRCUIT_MANAGER_HPP */
