/**
 * @file heading_alignment.cpp
 * @brief GPS navigation math — bearing, distance, angular error.
 *
 * Bearing uses the forward-azimuth formula on a spherical Earth.
 * Distance uses the Haversine formula.
 * Angular error is the smallest signed difference between heading and bearing.
 */

#include "heading_alignment.hpp"
#include <cmath>

/* -------------------------------------------------------------------------- */
/*  Bearing: from (lat1,lon1) to (lat2,lon2) → 0..360°                        */
/* -------------------------------------------------------------------------- */
float compute_bearing(float lat1, float lon1, float lat2, float lon2) {
    float lat1_r = lat1 * DEG_TO_RAD;
    float lat2_r = lat2 * DEG_TO_RAD;
    float dlon_r = (lon2 - lon1) * DEG_TO_RAD;

    float x = sinf(dlon_r) * cosf(lat2_r);
    float y = cosf(lat1_r) * sinf(lat2_r)
            - sinf(lat1_r) * cosf(lat2_r) * cosf(dlon_r);

    float bearing_rad = atan2f(x, y);
    float bearing_deg = bearing_rad * RAD_TO_DEG;

    // Normalize to 0..360
    if (bearing_deg < 0.0f) bearing_deg += 360.0f;
    return bearing_deg;
}

/* -------------------------------------------------------------------------- */
/*  Distance: Haversine formula → meters                                       */
/* -------------------------------------------------------------------------- */
float compute_distance(float lat1, float lon1, float lat2, float lon2) {
    float dlat = (lat2 - lat1) * DEG_TO_RAD;
    float dlon = (lon2 - lon1) * DEG_TO_RAD;
    float lat1_r = lat1 * DEG_TO_RAD;
    float lat2_r = lat2 * DEG_TO_RAD;

    float a = sinf(dlat / 2.0f) * sinf(dlat / 2.0f)
            + cosf(lat1_r) * cosf(lat2_r)
            * sinf(dlon / 2.0f) * sinf(dlon / 2.0f);
    float c = 2.0f * atan2f(sqrtf(a), sqrtf(1.0f - a));

    return EARTH_RADIUS_M * c;
}

/* -------------------------------------------------------------------------- */
/*  Angular error: signed smallest difference → -180..+180°                    */
/*  Positive = need to turn RIGHT, Negative = need to turn LEFT.               */
/* -------------------------------------------------------------------------- */
float compute_angular_error(float heading, float bearing) {
    float diff = bearing - heading;
    // Normalize to -180..+180
    while (diff > 180.0f)  diff -= 360.0f;
    while (diff < -180.0f) diff += 360.0f;
    return diff;
}

/* -------------------------------------------------------------------------- */
/*  Full alignment result                                                      */
/* -------------------------------------------------------------------------- */
AlignmentResult compute_alignment(float cur_lat, float cur_lon,
                                  float cur_heading,
                                  float tgt_lat, float tgt_lon) {
    AlignmentResult r;
    r.bearing_deg       = compute_bearing(cur_lat, cur_lon, tgt_lat, tgt_lon);
    r.distance_m        = compute_distance(cur_lat, cur_lon, tgt_lat, tgt_lon);
    r.angular_error_deg = compute_angular_error(cur_heading, r.bearing_deg);
    r.is_aligned        = (fabsf(r.angular_error_deg) < NAV_DEADZONE_DEG);
    return r;
}
