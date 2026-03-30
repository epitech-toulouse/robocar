#pragma once

#include <stdbool.h>

struct GPSPoint {
    float lat;
    float lon;
    float alt;
    float heading; // in degrees True North
    float speed_knots;
    int sats;
    bool has_fix;
};

#ifdef __cplusplus
extern "C" {
#endif

void init_usb_gps(void);
struct GPSPoint get_latest_gps(void);

// Utility math functions for Navigation
float distance_haversine_m(float lat1, float lon1, float lat2, float lon2);
float initial_bearing_deg(float lat1, float lon1, float lat2, float lon2);
float wrap_180(float angle_deg);

#ifdef __cplusplus
}
#endif
