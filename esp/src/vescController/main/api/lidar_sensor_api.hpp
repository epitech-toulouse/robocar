/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** lidar sensor interface
*/

#ifndef LIDAR_SENSOR_API_HPP
#define LIDAR_SENSOR_API_HPP

#if __has_include("freertos/FreeRTOS.h")
#include "freertos/FreeRTOS.h"
#else
#include <cstdint>
using TickType_t = uint32_t;
#ifndef pdMS_TO_TICKS
#define pdMS_TO_TICKS(ms) (static_cast<TickType_t>(ms))
#endif
#endif
#include <array>
#include <cstdint>

// A number of meter
typedef uint16_t centimeter_t;
static centimeter_t const UNDEFINED_LIDAR_VALUE = 0;
// One per degree, wow !
#define LIDAR_POINT_NUMBER 360

typedef std::array<centimeter_t, LIDAR_POINT_NUMBER> lidar_array_t;

static uint32_t const LIDAR_HZ = 10;
static uint32_t const LIDAR_MAX_SKIP_RENEWAL = 3;

// If a single point was not renewed in this lifespan
// set it as UNDEFINED_LIDAR_VALUE
static TickType_t const LIDAR_POINT_LIFESPAN =
    pdMS_TO_TICKS(1000 * LIDAR_MAX_SKIP_RENEWAL / LIDAR_HZ);

static TickType_t const LIDAR_MUTEX_TIMEOUT_TICK = pdMS_TO_TICKS(5);

// The LIDAR should use a task to update it's data
// The data should be accessed through a mutex using the timeout

class ILidarSensor {
public:
    ILidarSensor() = default;
    virtual ~ILidarSensor() = default;

    // Return false if the LIDAR is not implemented or too old
    virtual bool isActive(void) = 0;
    // Return false if the data was not gathered or too old
    // This function returns a (nowingly) simple output
    // The purpose of this simple output is to simplify later treatment
    // 360° of distance is a great way to represent lidar output
    // with 0m representing either 0m or undefined value
    virtual bool getData(lidar_array_t &output) = 0;
};

#endif /* LIDAR_SENSOR_API_HPP */
