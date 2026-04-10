/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** User Controller interface
*/

#ifndef USER_USER_CONTROLLER_HPP
#define USER_USER_CONTROLLER_HPP

#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "portmacro.h"
#include <array>
#include <cstdint>

enum driving_mode_e : uint8_t {
    DRIVING_MODE_DISABLED = 0x00,
    DRIVING_MODE_USER = 0x01,
    DRIVING_MODE_GPS = 0x02,
    DRIVING_MODE_LIDAR = 0x03,
};
typedef enum driving_mode_e driving_mode_t;

static TickType_t const USER_MUTEX_TIMEOUT_TICK = pdMS_TO_TICKS(5);

static TickType_t const DATA_LIFESPAN = pdMS_TO_TICKS(20);

inline char const *driving_mode_str(driving_mode_t mode)
{
    switch (mode) {
        case DRIVING_MODE_DISABLED: return "DISABLED";
        case DRIVING_MODE_USER: return "USER";
        case DRIVING_MODE_GPS: return "GPS";
        case DRIVING_MODE_LIDAR: return "LIDAR";
    };
    return "UNKNOWN";
}

class UserControllerApi {
public:
    virtual ~UserControllerApi() = default;

    virtual bool isConnected(void) = 0;

    // Return current driving mode
    virtual driving_mode_t getDrivingMode(void) = 0;

    // [-1.0;1.0]
    virtual float getSpeed(void) = 0;

    // [0.0;1.0]
    virtual float getSteering(void) = 0;
};

#endif /* USER_USER_CONTROLLER_HPP */
