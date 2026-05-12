#ifndef GPS_GOAL_ALGO_HPP
#define GPS_GOAL_ALGO_HPP

#include <cmath>
#include <cstdint>
#include <vector>

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
    // 48.62952718125309, 2.2619653410598435
    const double goalLatitude =  48.62952718125309;
    const double goalLongitude = 2.2619653410598435;
    std::vector<GpsPosition> positions = {
        { 48.62972588518544, 2.2622846371523013 },
        { 48.6296880943224, 2.2621471508285595 },
        { 48.62950635917635, 2.2619563263226214 },
        { 48.62942776961659, 2.2620747896650717 },
        { 48.62953677329812, 2.2619728419322995 },
        { 48.629682097628766, 2.262134603821977 },
        { 48.62967054536873, 2.2621736832135704 },
        { 48.62956328122276, 2.2622448567290885 },
        { 48.62969068720375, 2.262163500319441 },
    }; 
    uint8_t goal_index = 0;

    static constexpr TickType_t logPeriodTicks = pdMS_TO_TICKS(500);
    const char *const tag = "GpsGoalAlgo";
    TickType_t lastLogTick = 0;
};

#endif /* GPS_GOAL_ALGO_HPP */
