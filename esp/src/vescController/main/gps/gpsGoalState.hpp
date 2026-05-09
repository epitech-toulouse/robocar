#pragma once

#include <cmath>

#include "freertos/FreeRTOS.h"

struct GpsGoalSnapshot {
    double lat;
    double lon;
    bool enabled;
};

class GpsGoalState {
public:
    GpsGoalState(double lat, double lon, bool enabled = true)
        : goal_{lat, lon, enabled}
    {
    }

    GpsGoalSnapshot get() const
    {
        GpsGoalSnapshot copy{};
        portENTER_CRITICAL(&mux_);
        copy = goal_;
        portEXIT_CRITICAL(&mux_);
        return copy;
    }

    bool set(double lat, double lon, bool enabled = true)
    {
        if (!isValidLatitude(lat) || !isValidLongitude(lon) || !std::isfinite(lat) || !std::isfinite(lon)) {
            return false;
        }

        portENTER_CRITICAL(&mux_);
        goal_.lat = lat;
        goal_.lon = lon;
        goal_.enabled = enabled;
        portEXIT_CRITICAL(&mux_);
        return true;
    }

    static bool isValidLatitude(double lat)
    {
        return std::isfinite(lat) && lat >= -90.0 && lat <= 90.0;
    }

    static bool isValidLongitude(double lon)
    {
        return std::isfinite(lon) && lon >= -180.0 && lon <= 180.0;
    }

private:
    mutable portMUX_TYPE mux_ = portMUX_INITIALIZER_UNLOCKED;
    GpsGoalSnapshot goal_;
};
