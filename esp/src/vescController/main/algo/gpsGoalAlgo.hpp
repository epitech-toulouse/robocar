#ifndef GPS_GOAL_ALGO_HPP
#define GPS_GOAL_ALGO_HPP

#include <cmath>

#include "api/driving_algorithm_interface.hpp"
#include "api/gps_sensor_api.hpp"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "gps/gpsGoalState.hpp"

class GpsGoalAlgo : public DrivingAlgorithmApi
{
public:
    GpsGoalAlgo(GpsSensorApi &gps, GpsGoalState &goalState)
        : gps(gps),
          goalState(goalState)
    {
    }
    ~GpsGoalAlgo() = default;

    bool available(void) override;
    bool compute(DrivingAlgorithmOutput &output) override;
    float getPriority() override;

private:
    static constexpr double kPi = 3.14159265358979323846;
    static constexpr double kEarthRadiusM = 6371000.0;

    static float clampf(float value, float lo, float hi);
    static double toRadians(double degrees);
    static double toDegrees(double radians);
    static double wrap180(double degrees);
    static double haversineDistanceMeters(double lat1, double lon1, double lat2, double lon2);
    static double initialBearingDegrees(double lat1, double lon1, double lat2, double lon2);



    GpsSensorApi &gps;
    GpsGoalState &goalState;

    const float goalReachedDistanceM = 2.0f;
    const float fullSpeedDistanceM = 12.0f;
    const float baseSpeed = 0.08f;
    const float maxSpeed = 0.28f;
    const float maxSpeedRtkFixed = 0.34f;
    const float minSpeedScale = 0.25f;
    const float maxSteeringDelta = 0.45f;
    const float maxSteeringDeltaRtkFixed = 0.35f;
    const float fallbackWeight = 0.35f;
    const float fallbackWeightRtkFixed = 0.50f;
    const float computedWeightRtkFixed = 1.20f;

    static constexpr TickType_t logPeriodTicks = pdMS_TO_TICKS(500);
    const char *const tag = "GpsGoalAlgo";
    TickType_t lastLogTick = 0;
};

#endif /* GPS_GOAL_ALGO_HPP */
