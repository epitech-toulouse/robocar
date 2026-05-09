/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** advanced fusion engine
*/

#ifndef ADVANCED_FUSION_ENGINE_HPP
#define ADVANCED_FUSION_ENGINE_HPP

#include <memory>
#include <vector>

#include "api/driving_algorithm_interface.hpp"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "manager/AlgorithmSelector.hpp"

class AdvancedFusionEngine
{
public:
    AdvancedFusionEngine() = default;
    ~AdvancedFusionEngine() = default;

    void addDrivingAlgorithm(SelectableAlgorithm id,
                             std::unique_ptr<DrivingAlgorithmApi> algorithm);
    DrivingAlgorithmOutput computeOutput(const AlgorithmSelector &selector);
private:
    static constexpr TickType_t logPeriodTicks = pdMS_TO_TICKS(500);

    struct RegisteredAlgorithm {
        SelectableAlgorithm id;
        std::unique_ptr<DrivingAlgorithmApi> algorithm;
    };

    std::vector<RegisteredAlgorithm> driving_algorithms;
    TickType_t lastLogTick = 0;
};

#endif /* ADVANCED_FUSION_ENGINE_HPP */
