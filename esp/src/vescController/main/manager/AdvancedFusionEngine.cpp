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

void AdvancedFusionEngine::addDrivingAlgorithm
(std::unique_ptr<DrivingAlgorithmApi> algorithm)
{
    this->driving_algorithms.push_back(std::move(algorithm));
}

DrivingAlgorithmOutput AdvancedFusionEngine::computeOutput(void)
{
    std::vector<DrivingAlgorithmOutput> outputs;

    outputs.reserve(this->driving_algorithms.size());
    for (std::unique_ptr<DrivingAlgorithmApi> &driving_algo : this->driving_algorithms) {
        if (!driving_algo->available())
            continue;
        float priority = driving_algo->getPriority();
        DrivingAlgorithmOutput output;
        if (!driving_algo->compute(output))
            continue;
        auto gps_algo = dynamic_cast<GpsGoalAlgo*>(driving_algo.get());
        if (gps_algo) {
            ESP_LOGI("AdvancedFusionEngine", "Algorithm %p computed output: speed=%.2f steer=%.2f weight=%.3f",
                 driving_algo.get(), output.target_speed, output.target_steering, output.computed_weight);
        }
        // ESP_LOGI("AdvancedFusionEngine", "Algorithm %p computed output: speed=%.2f steer=%.2f weight=%.3f",
        //          driving_algo.get(), output.target_speed, output.target_steering, output.computed_weight);

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
        .target_steering = 0.5,
        .computed_weight = 0.0
    };
    for (DrivingAlgorithmOutput &output : outputs) {
        final_output.target_speed += output.target_speed;
        final_output.target_steering += output.target_steering;
        final_output.computed_weight += output.computed_weight;
    }
    if (final_output.computed_weight <= 0.0) {
        return {0.0, 0.5, 0.0}; // Return null output
    }
    final_output.target_speed /= final_output.computed_weight;
    final_output.target_steering /= final_output.computed_weight;
    // Offset back
    final_output.target_steering += 0.5;
    return final_output;
}
