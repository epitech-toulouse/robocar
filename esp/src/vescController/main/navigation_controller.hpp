#pragma once

/**
 * @file navigation_controller.hpp
 * @brief Fusion controller: GPS heading + LIDAR avoidance → motor commands.
 */

#include "nav_types.h"
#include "lidarReader.hpp"
#include <vector>

/// Motor commands (compatible with existing DriveCommands).
struct NavCommands {
    float steer;  ///< 0.0 (left) → 0.5 (center) → 1.0 (right)
    float duty;   ///< Duty cycle (positive = forward, negative = reverse)
};

class NavigationController {
public:
    NavigationController() = default;

    /**
     * @brief Compute navigation commands from GPS + LIDAR fusion.
     * @param gps        Latest GPS fix.
     * @param scan       Latest LIDAR 360° scan.
     * @param target_lat Target latitude (decimal degrees).
     * @param target_lon Target longitude (decimal degrees).
     * @return NavCommands with steer and duty values.
     */
    NavCommands compute(const GpsFix &gps,
                        const std::vector<LidarPoint> &scan,
                        float target_lat, float target_lon);

    /// Get the latest telemetry snapshot (for BLE reporting).
    NavTelemetry get_telemetry() const { return last_telemetry; }

private:
    /// Reverse state
    TickType_t reverse_until = 0;
    float reverse_steer = NAV_STEER_CENTER;

    /// Last telemetry snapshot
    NavTelemetry last_telemetry = {};

    /// Log throttle
    TickType_t last_log_tick = 0;
};
