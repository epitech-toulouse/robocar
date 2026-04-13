/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** advanced fusion engine
*/

#ifndef ADVANCED_FUSION_ENGINE_HPP
#define ADVANCED_FUSION_ENGINE_HPP

#include "api/driving_algorithm_interface.hpp"
#include <vector>

class AdvancedFusionEngine
{
public:
    AdvancedFusionEngine() = default;
    ~AdvancedFusionEngine() = default;

    void addDrivingAlgorithm(DrivingAlgorithmApi *algorithm);
    DrivingAlgorithmOutput computeOutput(void);
private:
    std::vector<DrivingAlgorithmApi *> driving_algorithms;
};

#endif /* ADVANCED_FUSION_ENGINE_HPP */
