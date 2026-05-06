#include "cameraDrivingAlgo.hpp"

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

bool CameraDrivingAlgo::available(void)
{
    if (!cameraSensor.isActive()) {
        return false;
    }

    CameraSteeringCommand steeringCommand{};
    CameraStopCommand stopCommand{};
    return cameraSensor.getSteeringCommand(steeringCommand) || cameraSensor.getStopCommand(stopCommand);
}

bool CameraDrivingAlgo::compute(DrivingAlgorithmOutput &output)
{
    static TickType_t lastLogTick = 0;
    static constexpr TickType_t logPeriodTicks = pdMS_TO_TICKS(250);
    output = DEFAULT_DRIVING_ALGORITHM_OUTPUT;
    const TickType_t now = xTaskGetTickCount();

    CameraStopCommand stopCommand{};
    if (cameraSensor.getStopCommand(stopCommand)) {
        output.target_speed = 0.0f;
        output.target_steering = 0.5f;
        output.computed_weight = stopCommand.weight * stopPriorityBoost;
        if ((now - lastLogTick) >= logPeriodTicks) {
            ESP_LOGI("CameraDrivingAlgo", "output stop speed=%.2f steer=%.2f weight=%.2f",
                     static_cast<double>(output.target_speed),
                     static_cast<double>(output.target_steering),
                     static_cast<double>(output.computed_weight));
            lastLogTick = now;
        }
        return true;
    }

    CameraSteeringCommand steeringCommand{};
    if (!cameraSensor.getSteeringCommand(steeringCommand)) {
        return false;
    }

    output.target_speed = laneCruiseSpeed;
    output.target_steering = clampf(0.5f + steeringCommand.steering_percent / 200.0f, 0.0f, 1.0f);
    output.computed_weight = steeringCommand.weight;
    if ((now - lastLogTick) >= logPeriodTicks) {
        ESP_LOGI("CameraDrivingAlgo", "output lane input=%.1f%% speed=%.2f steer=%.2f weight=%.2f",
                 static_cast<double>(steeringCommand.steering_percent),
                 static_cast<double>(output.target_speed),
                 static_cast<double>(output.target_steering),
                 static_cast<double>(output.computed_weight));
        lastLogTick = now;
    }
    return steeringCommand.weight > 0.0f;
}

float CameraDrivingAlgo::getPriority()
{
    return 1.0f;
}

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
