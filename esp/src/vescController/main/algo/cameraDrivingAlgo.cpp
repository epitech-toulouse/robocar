#include "cameraDrivingAlgo.hpp"

#include "config.h"

float CameraDrivingAlgo::clampf(float value, float lo, float hi)
{
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

bool CameraDrivingAlgo::available(void)
{
    CameraStatus status{};
    return this->cameraSensor.isActive() &&
        this->cameraSensor.getStatus(status) &&
        status.connected &&
        status.has_data;
}

bool CameraDrivingAlgo::compute(DrivingAlgorithmOutput &output)
{
    output = DEFAULT_DRIVING_ALGORITHM_OUTPUT;

    CameraStatus status{};
    if (!this->cameraSensor.getStatus(status) || !status.connected || !status.has_data) {
        return false;
    }

    output.target_steering = clampf(0.5f + (status.steering_percent / 200.0f), 0.0f, 1.0f);
    output.target_speed = laneCruiseSpeed;

    float weight = status.steering_weight > 0.0f ? status.steering_weight : 1.0f;
    if (status.stop_detected) {
        output.target_speed = 0.0f;
        weight = status.stop_weight > 0.0f
            ? status.stop_weight * stopPriorityBoost
            : stopPriorityBoost;
    }

    output.computed_weight = clampf(weight, 0.0f, 1000.0f);
    return output.computed_weight > 0.0f;
}

float CameraDrivingAlgo::getPriority()
{
    return CAMEDAR_WEIGHT;
}
