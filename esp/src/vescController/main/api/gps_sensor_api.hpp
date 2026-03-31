/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** gps sensor interface
*/

#ifndef GPS_SENSOR_API_HPP
#define GPS_SENSOR_API_HPP

#include "freertos/FreeRTOS.h"

struct GpsPosition {
    double latitude;
    double longitude;
};

struct GpsHeading {
    double degrees_to_north;
};

static TickType_t const GPS_MUTEX_TIMEOUT_TICK = pdMS_TO_TICKS(5);

// The GPS should use a task to update it's data
// The data should be accessed through a mutex using the timeout

class GpsSensorApi {
public:
    virtual ~GpsSensorApi() = default;

    // Return false if the GPS is not implemented or too old
    virtual bool isActive(void) = 0;
    // Return false if the data was not gathered or too old
    virtual bool getPosition(GpsPosition &output) = 0;
    // Return false if the data was not gathered or too old
    virtual bool getHeading(GpsHeading &output) = 0;
};

#endif /* GPS_SENSOR_API_HPP */
