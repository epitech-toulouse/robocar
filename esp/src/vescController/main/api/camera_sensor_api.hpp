/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** camera sensor interface
*/

#ifndef CAMERA_SENSOR_API_HPP
#define CAMERA_SENSOR_API_HPP

#include "freertos/FreeRTOS.h"

struct CameraSteeringCommand {
    float steering_percent = 0.0f;
    float weight = 0.0f;
};

struct CameraStopCommand {
    float weight = 0.0f;
};

static TickType_t const CAMERA_COMMAND_TIMEOUT_TICK = pdMS_TO_TICKS(750);

class CameraSensorApi {
public:
    virtual ~CameraSensorApi() = default;

    virtual bool isActive(void) = 0;
    virtual void update(void) = 0;
    virtual bool getSteeringCommand(CameraSteeringCommand &output) = 0;
    virtual bool getStopCommand(CameraStopCommand &output) = 0;
};

#endif /* CAMERA_SENSOR_API_HPP */
