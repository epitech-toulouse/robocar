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

class AdvancedFusionEngine
{
public:
    AdvancedFusionEngine() = default;
    ~AdvancedFusionEngine() = default;

    void addDrivingAlgorithm(std::unique_ptr<IDrivingAlgorithm> algorithm);
    DrivingAlgorithmOutput computeOutput(void);
private:
    std::vector<std::unique_ptr<IDrivingAlgorithm>> driving_algorithms;
};

#endif /* ADVANCED_FUSION_ENGINE_HPP */
