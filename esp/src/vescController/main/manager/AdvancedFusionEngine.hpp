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
    struct RegisteredAlgorithm {
        SelectableAlgorithm id;
        std::unique_ptr<DrivingAlgorithmApi> algorithm;
    };

    std::vector<RegisteredAlgorithm> driving_algorithms;
};

#endif /* ADVANCED_FUSION_ENGINE_HPP */
