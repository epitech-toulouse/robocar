#include "gpsGoalAlgo.hpp"

#include <cmath>
#include <cstdint>
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

#include <array>

#define GPS_POS_BUFFER_SIZE 2
std::array<GpsPosition, GPS_POS_BUFFER_SIZE> gps_positions = {
    {}
};
uint8_t gps_index = 0;
uint8_t old_index = 0;
double old_mani_heading = 0;
TickType_t last_update = 0;
TickType_t last_loop = 0;

bool GpsGoalAlgo::compute(DrivingAlgorithmOutput &output)
{
    output = DEFAULT_DRIVING_ALGORITHM_OUTPUT;
    const TickType_t now = xTaskGetTickCount();
    const bool shouldLog = (now - this->lastLogTick) >= this->logPeriodTicks;
    const GpsGoalSnapshot goal = this->goalState.get();

    GpsPosition position{};
    GpsHeading heading{};
    GpsStatus status{};

    if (!goal.enabled) {
        if (shouldLog) {
            ESP_LOGI(this->tag,
                     "return=false reason=goal_disabled weight=%.2f",
                     static_cast<double>(output.computed_weight));
            this->lastLogTick = now;
        }
        return false;
    }

    if (!this->gps.getPosition(position)) {
        if (shouldLog) {
            ESP_LOGI(this->tag,
                     "return=false reason mani =position_unavailable weight=%.2f",
                     static_cast<double>(output.computed_weight));
            this->lastLogTick = now;
        }
        return false;
    }

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


    ////// MANI HEADING COMPUTATION //////////////////////////////////
    bool should_update = gps_positions[old_index].latitude != position.latitude
        || gps_positions[old_index].longitude != position.longitude;

    TickType_t now_ = xTaskGetTickCount();

    double mani_heading = initialBearingDegrees(gps_positions[gps_index].latitude, gps_positions[gps_index].longitude, position.latitude, position.longitude);
    if (should_update) {
        old_mani_heading = mani_heading;
        gps_positions[gps_index].latitude = position.latitude;
        gps_positions[gps_index].longitude = position.longitude;
        old_index = gps_index;
        gps_index++;    
        gps_index %= GPS_POS_BUFFER_SIZE;
        // ESP_LOGI(tag, "DEBUG_01 Update of position (%d ms since last update)", pdTICKS_TO_MS(now - last_update));
        last_update = now_;
    } else {
        // ESP_LOGI(tag, "DEBUG_02 No update (%d ms since last loop)", pdTICKS_TO_MS(now - last_loop));
    }
    last_loop = now_;

    //////////////////////////////////////////////////////////////////////////////////////////////


    const bool statusValid = this->gps.getStatus(status);
    if (!statusValid || !status.has_fix) {
        if (shouldLog) {
            ESP_LOGI(this->tag,
                     "return=false reason=status_invalid_or_no_fix rtk_fixed=%d sats=%d weight=%.2f",
                     status.is_rtk_fixed,
                     status.satellites,
                     static_cast<double>(output.computed_weight));
            this->lastLogTick = now;
        }
        return false;
    }

    const bool rtkFixed = statusValid && status.is_rtk_fixed;



    if (distanceMeters <= this->goalReachedDistanceM) {
        output.target_speed = 0.0f;
        output.target_steering = 0.5f;
        output.computed_weight = 1.0f;
        if (shouldLog) {
            ESP_LOGI(this->tag,
                     "return=true mani reason=goal_reached dist=%.2fm goal_deg=%.1f speed=%.2f steer=%.2f rtk_fixed=%d sats=%d weight=%.2f",
                     distanceMeters,
                     desiredBearingDeg,
                     static_cast<double>(output.target_speed),
                     static_cast<double>(output.target_steering),
                     rtkFixed,
                     status.satellites,
                     static_cast<double>(output.computed_weight));
            this->lastLogTick = now;
        }
        return true;
    }

    const bool headingValid = this->gps.getHeading(heading);

    
    ///////////////////// HEADING UNAVAILABLE CASE - FALLBACK TO MANI HEADING /////////////////////
    if (!headingValid) {
            const float maxSteeringDelta2 = this->maxSteeringDelta;
            output.target_steering = 0.5;
            double DegreFromGoalMani = wrap180(desiredBearingDeg - mani_heading);
            if (shouldLog) {
                ESP_LOGI(tag,
                         "heading to point %f, fixed %d, distance %.2f",
                         DegreFromGoalMani,
                         status.is_rtk_fixed,
                         distanceMeters);
            }
    
            if (DegreFromGoalMani > 35) {
                output.target_steering = 1.0f;
            }
            else if (DegreFromGoalMani < -35) {
                output.target_steering = 0.0f;
            }
            else {
                output.target_steering = clampf(0.5f + (DegreFromGoalMani / 180.0f) * maxSteeringDelta2, 0.0f, 1.0f);
            }
            output.computed_weight = 1;
            output.target_speed = 0.03;


            ESP_LOGI(tag, "DEBUG_03 mani_heading[%.2f] heading_to_point[%.2f] desired[%.2f]", mani_heading, DegreFromGoalMani, desiredBearingDeg);
            // car_heading;goal_heading;degree_from_goal;lat;long;old_lat;old_long;distance_to_goal
            ESP_LOGE("CSV", "%f;%f;%f;%f;%f;%f;%f;%f", mani_heading, desiredBearingDeg, DegreFromGoalMani, position.latitude, position.longitude,
                   gps_positions[old_index].latitude, gps_positions[old_index].longitude, distanceMeters);
            if (shouldLog) {
            ESP_LOGI(this->tag,
                     "return=false reason=heading_unavailable mani dist=%.2fm goal_deg=%.1f speed=%.2f steer=%.2f rtk_fixed=%d sats=%d weight=%.2f",
                     distanceMeters,
                     DegreFromGoalMani,
                     static_cast<double>(output.target_speed),
                     static_cast<double>(output.target_steering),
                     rtkFixed,
                     status.satellites,
                     static_cast<double>(output.computed_weight));
            this->lastLogTick = now;
            }
            return true; 
    }


    /////////////////////// NORMAL CASE - HEADING AVAILABLE /////////////////////

    
    const float distanceScale = clampf(
        static_cast<float>(distanceMeters / this->fullSpeedDistanceM),
        this->minSpeedScale,
        1.0f);
    

    const double errorDeg = wrap180(desiredBearingDeg - heading.degrees_to_north);
    const float normalizedError = clampf(static_cast<float>(errorDeg / 90.0), -1.0f, 1.0f);

    const float maxSteeringDelta = rtkFixed ? this->maxSteeringDeltaRtkFixed : this->maxSteeringDelta;
    const float targetMaxSpeed = rtkFixed ? this->maxSpeedRtkFixed : this->maxSpeed;
    output.target_steering = clampf(0.5f + normalizedError * maxSteeringDelta, 0.0f, 1.0f);
    output.target_speed = this->baseSpeed + (targetMaxSpeed - this->baseSpeed) * distanceScale;
    output.computed_weight = rtkFixed ? this->computedWeightRtkFixed : 1.0f;

    // car_heading;bearing_to_goal_north;bearing_to_goal_front_of_car;latitude;longitude;distance
    ESP_LOGE("CSV", "%f;%f;%f;%f;%f;%f", heading.degrees_to_north, desiredBearingDeg, errorDeg, position.latitude, position.longitude, distanceMeters);
    if (shouldLog) {
        ESP_LOGI(this->tag,
                 "return=true reason=normal dist=%.2fm goal_deg=%.1f heading_err=%.1fdeg speed=%.2f steer=%.2f rtk_fixed=%d sats=%d weight=%.2f\n",
                 distanceMeters,
                 desiredBearingDeg,
                 errorDeg,
                 static_cast<double>(output.target_speed),
                 static_cast<double>(output.target_steering),
                 rtkFixed,
                 status.satellites,
                 static_cast<double>(output.computed_weight));
        this->lastLogTick = now;
    }
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
