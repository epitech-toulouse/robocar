#pragma once

/**
 * @file nav_types.h
 * @brief Shared types and configuration constants for GPS+LIDAR waypoint navigation.
 *
 * Conventions:
 *   - All angles in DEGREES.
 *   - Headings/bearings: 0..360° (0=North, 90=East, 180=South, 270=West).
 *   - Angular errors: -180..+180° (positive = turn right, negative = turn left).
 *   - Steering: 0.0 (full left) → 0.5 (center) → 1.0 (full right).
 *   - Score: -1.0 (max left correction) → 0.0 (aligned) → +1.0 (max right correction).
 *   - Distances in meters.
 *   - Coordinates in WGS84 decimal degrees.
 */

#include <cmath>
#include <cstdint>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/* -------------------------------------------------------------------------- */
/*  Navigation tuning constants                                                */
/* -------------------------------------------------------------------------- */

/// Distance (m) below which the robot considers it has arrived at the target.
static constexpr float NAV_ARRIVAL_M = 2.0f;

/// Angular error (deg) within which the robot is considered "on course".
static constexpr float NAV_DEADZONE_DEG = 5.0f;

/// Maximum forward duty cycle during navigation.
static constexpr float NAV_MAX_SPEED = 0.050f;

/// Reduced speed when misaligned or close to target.
static constexpr float NAV_SLOW_SPEED = 0.025f;

/// Reverse duty cycle when backing away from an obstacle.
static constexpr float NAV_REVERSE_SPEED = -0.030f;

/// GPS steering blend weight (0..1).  Higher = more GPS influence.
static constexpr float NAV_GPS_WEIGHT = 0.6f;

/// LIDAR avoidance blend weight (0..1). Higher = more obstacle-avoidance influence.
static constexpr float NAV_LIDAR_WEIGHT = 0.4f;

/// Maximum age (ms) of a GPS fix before it is considered stale → safety stop.
static constexpr uint32_t NAV_GPS_STALE_MS = 3000;

/// LIDAR obstacle distances (reuse existing values from drive.hpp).
static constexpr float NAV_STOP_DISTANCE_M  = 0.40f;
static constexpr float NAV_SLOW_DISTANCE_M  = 1.00f;
static constexpr float NAV_SAFE_DISTANCE_M  = 2.20f;

/// Reverse duration when obstacle is critically close.
static constexpr uint32_t NAV_REVERSE_DURATION_MS = 800;

/// Steering limits (matching drive.hpp).
static constexpr float NAV_STEER_CENTER = 0.50f;
static constexpr float NAV_STEER_LEFT   = 0.20f;
static constexpr float NAV_STEER_RIGHT  = 0.80f;

/// Earth's mean radius in meters (for Haversine calculations).
static constexpr float EARTH_RADIUS_M = 6371000.0f;

/// Degrees-to-radians conversion factor.
static constexpr float DEG_TO_RAD = (float)M_PI / 180.0f;

/// Radians-to-degrees conversion factor.
static constexpr float RAD_TO_DEG = 180.0f / (float)M_PI;

/* -------------------------------------------------------------------------- */
/*  Shared data structures                                                     */
/* -------------------------------------------------------------------------- */

/// GPS fix data populated by the USB GPS reader.
struct GpsFix {
    float lat        = 0.0f;  ///< Latitude  (decimal degrees, WGS84)
    float lon        = 0.0f;  ///< Longitude (decimal degrees, WGS84)
    float alt        = 0.0f;  ///< Altitude  (meters above MSL)
    float heading_deg = NAN;  ///< Course over ground 0-360°, NAN if unavailable
    float speed_mps  = 0.0f;  ///< Ground speed in m/s
    int   sats       = 0;     ///< Number of satellites in view
    bool  has_fix    = false;  ///< true if fix quality > 0
    TickType_t last_update_tick = 0;  ///< Tick count of last valid update
};

/// Result of obstacle analysis from LIDAR data.
struct ObstacleResult {
    float front_min_m  = 999.0f;  ///< Nearest obstacle in front sector (m)
    float left_min_m   = 999.0f;  ///< Nearest obstacle in left sector  (m)
    float right_min_m  = 999.0f;  ///< Nearest obstacle in right sector (m)
    bool  emergency    = false;    ///< true if obstacle < NAV_STOP_DISTANCE_M
    float avoidance_steer = NAV_STEER_CENTER; ///< Suggested steer (0..1), CENTER if clear
};

/// Result of heading alignment computation.
struct AlignmentResult {
    float bearing_deg      = 0.0f;  ///< Bearing to target 0..360°
    float angular_error_deg = 0.0f; ///< Signed error -180..+180° (+ = turn right)
    float distance_m       = 0.0f;  ///< Distance to target (m)
    bool  is_aligned       = false;  ///< true if |error| < NAV_DEADZONE_DEG
};

/// Navigation telemetry (for BLE reporting / logging).
struct NavTelemetry {
    float lat, lon;
    float heading_deg;
    float target_bearing_deg;
    float angular_error_deg;
    float score;
    float distance_m;
    float speed;
    float steer;
    bool  obstacle_detected;
    bool  has_gps;
    bool  arrived;
};
