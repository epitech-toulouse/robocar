/**
 * @file trajectory_score.cpp
 * @brief Signed trajectory correction score.
 *
 * Maps angular error to a [-1, +1] score:
 *   - Within deadzone → 0 (on course, no correction).
 *   - Outside deadzone → linearly scaled to ±1 at ±180°.
 *
 * The score serves two purposes:
 *   1. Steering correction signal (directly drives angular velocity).
 *   2. Telemetry metric (exposed to mobile app).
 */

#include "trajectory_score.hpp"
#include <cmath>

float compute_trajectory_score(float angular_error_deg, float deadzone_deg) {
    float abs_error = fabsf(angular_error_deg);

    // Within deadzone → perfectly aligned
    if (abs_error < deadzone_deg) {
        return 0.0f;
    }

    // Linear scaling: deadzone..180° → 0..1
    // The sign follows the error sign (+ = right, - = left).
    float effective_error = abs_error - deadzone_deg;
    float max_effective   = 180.0f - deadzone_deg;
    float magnitude = effective_error / max_effective;

    // Clamp to [0, 1]
    if (magnitude > 1.0f) magnitude = 1.0f;

    // Apply sign from original error
    return (angular_error_deg > 0.0f) ? magnitude : -magnitude;
}
