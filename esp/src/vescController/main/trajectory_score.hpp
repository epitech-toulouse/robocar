#pragma once

/**
 * @file trajectory_score.hpp
 * @brief Computes a signed trajectory correction score from angular error.
 *
 * Score convention:
 *   +1.0 = maximum right correction needed
 *   -1.0 = maximum left correction needed
 *    0.0 = aligned (within deadzone)
 */

#include "nav_types.h"

/**
 * @brief Compute signed trajectory score from angular error.
 * @param angular_error_deg  Signed error in degrees (-180..+180).
 * @param deadzone_deg       Tolerance band (default = NAV_DEADZONE_DEG).
 * @return Score in [-1.0, +1.0].
 */
float compute_trajectory_score(float angular_error_deg,
                               float deadzone_deg = NAV_DEADZONE_DEG);
