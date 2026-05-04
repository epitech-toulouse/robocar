/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** coupe circuit manager
*/

#ifndef COUPE_CIRCUIT_MANAGER_HPP
#define COUPE_CIRCUIT_MANAGER_HPP

#include <atomic>
#include <freertos/FreeRTOS.h>

extern std::atomic_bool coupe_circuit_connected;

class CoupeCircuitManager
{
public:
    CoupeCircuitManager();
private:
};

#endif /* COUPE_CIRCUIT_MANAGER_HPP */
