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
    double latitude = 0.0;
    double longitude = 0.0;
};

struct GpsHeading {
    double degrees_to_north = 0.0;
};

enum class GpsFixMode {
    Invalid,
    Autonomous,
    Differential,
    RtkFloat,
    RtkFixed,
    Other,
};

struct GpsStatus {
    bool has_fix = false;
    int satellites = 0;
    GpsFixMode fix_mode = GpsFixMode::Invalid;
    bool is_rtk = false;
    bool is_rtk_fixed = false;
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
    // Return false if GPS status cannot be gathered
    virtual bool getStatus(GpsStatus &output) = 0;
};

#endif /* GPS_SENSOR_API_HPP */
