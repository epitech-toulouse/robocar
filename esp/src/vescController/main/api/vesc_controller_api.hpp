/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** vesc controller interface
*/

#ifndef VESC_CONTROLLER_API_HPP
#define VESC_CONTROLLER_API_HPP

#include "freertos/FreeRTOS.h"
#include <array>
#include <cstdint>

static TickType_t const VESC_MUTEX_TIMEOUT_TICK = pdMS_TO_TICKS(5);

class IVescController {
public:
    IVescController() = default;
    virtual ~IVescController() = default;

    // Return false if the VESC is not implemented or too old
    virtual bool isActive(void) = 0;

    // Set the speed to 0
    virtual void stop(void) = 0;

    virtual void deactivate(void) = 0;
    virtual void activate(void) = 0;

    // [-1.0;1.0]
    virtual void set_speed(float speed) = 0;
    // [0.0;1.0]
    virtual void set_steering(float steering) = 0;
};

#endif /* VESC_CONTROLLER_API_HPP */
