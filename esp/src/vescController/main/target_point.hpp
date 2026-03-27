#pragma once

/**
 * @file target_point.hpp
 * @brief Manages the current navigation target point received via BLE.
 *
 * Thread-safe: called from BLE GATT callback (ISR-safe via portENTER_CRITICAL)
 * and from the main navigation task.
 */

#include "nav_types.h"
#include <atomic>

enum class NavState : uint8_t {
    IDLE,        ///< No target set
    NAVIGATING,  ///< Actively navigating to target
    ARRIVED,     ///< Reached the target
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Set a new navigation target. Called from BLE handler.
 * @return true if coordinates are valid.
 */
bool nav_set_target(float lat, float lon);

/// Clear the current navigation target (stop navigating).
void nav_clear_target(void);

/// Check if a navigation target is active.
bool nav_has_target(void);

/// Get the current target coordinates. Returns false if no target.
bool nav_get_target(float &lat, float &lon);

/// Get the current navigation state.
NavState nav_get_state(void);

/// Set the navigation state (called by NavigationController).
void nav_set_state(NavState state);

#ifdef __cplusplus
}
#endif
