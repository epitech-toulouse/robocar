/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** camera sensor interface
*/

#ifndef CAMERA_SENSOR_API_HPP
#define CAMERA_SENSOR_API_HPP

#include "freertos/FreeRTOS.h"



CameraStatus {
    bool has_data = false;
    bool stop_detected = false;
    float steering_percent = 0.0f;
    float speed_percent = 0.0f;
};


class CameraSensorApi {
public:
    virtual ~CameraSensorApi() = default;

    // Return false if the camera is not implemented or too old
    virtual bool isActive(void) = 0;
    // Return false if no stop deected
    virtual bool getStop( bool &output) = 0;

    // Return the heading 
    virtual float getHeading( float &output) = 0;

    // Return the speed 
    virtual float getSpeed( float &output) = 0;

    // Return  true if camera status can be gathered
    virtual bool getStatus( CameraStatus &output) = 0;
};

#endif /* CAMERA_SENSOR_API_HPP */
