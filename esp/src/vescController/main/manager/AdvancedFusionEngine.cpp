/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** advanced fusion engine
*/

#include "AdvancedFusionEngine.hpp"
#include "api/driving_algorithm_interface.hpp"
#include <vector>
#include <esp_log.h>

static const char *algorithmLogKey(SelectableAlgorithm id)
{
    const AlgorithmDescriptor *descriptor = algorithmDescriptor(id);

    return descriptor != nullptr ? descriptor->key : "unknown";
}

void AdvancedFusionEngine::addDrivingAlgorithm
(SelectableAlgorithm id, std::unique_ptr<DrivingAlgorithmApi> algorithm)
{
    this->driving_algorithms.push_back({id, std::move(algorithm)});
}

DrivingAlgorithmOutput AdvancedFusionEngine::computeOutput(const AlgorithmSelector &selector)
{
    std::vector<DrivingAlgorithmOutput> outputs;
    const TickType_t now = xTaskGetTickCount();
    const bool shouldLog = (now - this->lastLogTick) >= this->logPeriodTicks;
    const uint32_t selectedMask = selector.getSelectedMask();

    outputs.reserve(this->driving_algorithms.size());
    if (shouldLog) {
        ESP_LOGI("AdvancedFusionEngine",
                 "Fusion pass: selected_mask=0x%02lx",
                 static_cast<unsigned long>(selectedMask));
    }
    for (RegisteredAlgorithm &registered_algorithm : this->driving_algorithms) {
        if (!selector.isEnabled(registered_algorithm.id)) {
            if (shouldLog) {
                ESP_LOGI("AdvancedFusionEngine",
                         "Algo %s skipped: disabled by mask",
                         algorithmLogKey(registered_algorithm.id));
            }
            continue;
        }
        std::unique_ptr<DrivingAlgorithmApi> &driving_algo = registered_algorithm.algorithm;
        if (!driving_algo->available()) {
            if (shouldLog) {
                ESP_LOGI("AdvancedFusionEngine",
                         "Algo %s selected but unavailable",
                         algorithmLogKey(registered_algorithm.id));
            }
            continue;
        }
        float priority = driving_algo->getPriority();
        DrivingAlgorithmOutput output;
        if (!driving_algo->compute(output)) {
            if (shouldLog) {
                ESP_LOGI("AdvancedFusionEngine",
                         "Algo %s selected and available but compute returned false",
                         algorithmLogKey(registered_algorithm.id));
            }
            continue;
        }
        if (shouldLog) {
            ESP_LOGI("AdvancedFusionEngine",
                     "Algo %s computed: speed=%.2f steer=%.2f weight=%.3f priority=%.3f",
                     algorithmLogKey(registered_algorithm.id),
                     output.target_speed,
                     output.target_steering,
                     output.computed_weight,
                     priority);
        }

        float coef = priority * output.computed_weight;
        // Offset to allow computations
        output.target_steering -= 0.5;
        output.target_steering *= coef;
        output.target_speed *= coef;
        output.computed_weight = coef;
        outputs.push_back(output);
    }
    DrivingAlgorithmOutput final_output = {
        .target_speed = 0.0,
        .target_steering = 0.0,
        .computed_weight = 0.0
    };
    for (DrivingAlgorithmOutput &output : outputs) {
        final_output.target_speed += output.target_speed;
        final_output.target_steering += output.target_steering;
        final_output.computed_weight += output.computed_weight;
    }
    if (final_output.computed_weight <= 0.0) {
        if (shouldLog) {
            ESP_LOGI("AdvancedFusionEngine", "Fusion output: no computed algo, stop output");
            this->lastLogTick = now;
        }
        return {0.0, 0.5, 0.0}; // Return null output
    }
    final_output.target_speed /= final_output.computed_weight;
    final_output.target_steering /= final_output.computed_weight;
    // Offset back
    final_output.target_steering += 0.5;
    if (shouldLog) {
        ESP_LOGI("AdvancedFusionEngine",
                 "Fusion output: speed=%.2f steer=%.2f weight=%.3f",
                 final_output.target_speed,
                 final_output.target_steering,
                 final_output.computed_weight);
        this->lastLogTick = now;
    }
    return final_output;
}
