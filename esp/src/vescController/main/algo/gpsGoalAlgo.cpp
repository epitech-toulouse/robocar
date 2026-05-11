#include "gpsGoalAlgo.hpp"

#include <cmath>
#include <unistd.h>

#include "api/gps_sensor_api.hpp"
#include "config.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

bool GpsGoalAlgo::available(void)
{
    GpsStatus status{};
    return this->gps.isActive() && this->gps.getStatus(status) && status.has_fix;
}

bool GpsGoalAlgo::compute(DrivingAlgorithmOutput &output)
{
    output = DEFAULT_DRIVING_ALGORITHM_OUTPUT;
    const GpsGoalSnapshot goal = this->goalState.get();

    GpsPosition position{};
    GpsHeading heading{};

    // No goal, wait for one
    if (!goal.enabled) {
        ESP_LOGD(this->tag,
                 "return=false reason=goal_disabled weight=%.2f",
                 static_cast<double>(output.computed_weight));
        return false;
    }

    // No position, abort
    if (!this->gps.getPosition(position)) {
        ESP_LOGW(this->tag, "Unable to get position");
        return false;
    }

    // Compute distances to goal & desired bearing to goal
    const double distanceMeters = haversineDistanceMeters(
        position.latitude,
        position.longitude,
        this->goalLatitude,
        this->goalLongitude);

    const double desiredBearingDeg = initialBearingDegrees(
        position.latitude,
        position.longitude,
        this->goalLatitude,
        this->goalLongitude);

    // Goal reached, you won !
    if (distanceMeters <= this->goalReachedDistanceM) {
        output.target_speed = 0.0f;
        output.target_steering = 0.5f;
        output.computed_weight = 1.0f;
        return true;
    }

    const bool headingValid = this->gps.getHeading(heading);

    // No heading found, go straight and wait for one
    if (!headingValid) {
        output.target_steering = 0.5f;
        output.target_speed = WAIT_FOR_HEADING_SPEED;
        output.computed_weight = 1.0;
        return true; 
    }

    // Heading found, let's go towards it
    const double errorDeg = wrap180(desiredBearingDeg - heading.degrees_to_north);
    const float normalizedError = clampf(static_cast<float>(errorDeg / 90.0), -1.0f, 1.0f);
    output.target_steering = clampf(0.5f + normalizedError * maxSteeringDelta, 0.0f, 1.0f);
    output.target_speed = HEADING_FOUND_SPEED;
    output.computed_weight = 1.0f;

    return true;
}

float GpsGoalAlgo::getPriority()
{
    return GPS_WEIGHT;
}

float GpsGoalAlgo::clampf(float value, float lo, float hi)
{
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

double GpsGoalAlgo::toRadians(double degrees)
{
    return degrees * kPi / 180.0;
}

double GpsGoalAlgo::toDegrees(double radians)
{
    return radians * 180.0 / kPi;
}

double GpsGoalAlgo::wrap180(double degrees)
{
    while (degrees > 180.0) {
        degrees -= 360.0;
    }
    while (degrees < -180.0) {
        degrees += 360.0;
    }
    return degrees;
}

double GpsGoalAlgo::haversineDistanceMeters(double lat1, double lon1, double lat2, double lon2)
{
    const double dLat = toRadians(lat2 - lat1);
    const double dLon = toRadians(lon2 - lon1);
    const double a = std::sin(dLat / 2.0) * std::sin(dLat / 2.0)
        + std::cos(toRadians(lat1)) * std::cos(toRadians(lat2))
        * std::sin(dLon / 2.0) * std::sin(dLon / 2.0);
    const double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));
    return kEarthRadiusM * c;
}

double GpsGoalAlgo::initialBearingDegrees(double lat1, double lon1, double lat2, double lon2)
{
    const double lat1r = toRadians(lat1);
    const double lat2r = toRadians(lat2);
    const double dLon = toRadians(lon2 - lon1);

    const double y = std::sin(dLon) * std::cos(lat2r);
    const double x = std::cos(lat1r) * std::sin(lat2r)
        - std::sin(lat1r) * std::cos(lat2r) * std::cos(dLon);
    return std::fmod(toDegrees(std::atan2(y, x)) + 360.0, 360.0);
}
