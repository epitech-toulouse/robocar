/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** driving algo api
*/

#ifndef DRIVING_ALGO_API_HPP
#define DRIVING_ALGO_API_HPP

struct DrivingAlgorithmOutput {
    // [-1.0;+1.0]
    float target_speed;
    // [0.0;1.0]
    float target_steering;
    // [0.0;+inf]
    float computed_weight;
};

static DrivingAlgorithmOutput const DEFAULT_DRIVING_ALGORITHM_OUTPUT = {
    // Stop car
    0.0, // target_speed
    // Center direction
    0.5, // target_steering
    // Null weight
    0.0  // computed_weight
};

class IDrivingAlgorithm {
public:
    IDrivingAlgorithm() = default;
    virtual ~IDrivingAlgorithm() = default;

    // Test if necessary inputs are present (but not if correct)
    virtual bool available(void) = 0;
    
    // Computation for a single tick
    // Returns false if no output can be computed (invalid inputs)
    // => output set to DEFAULT_DRIVING_ALGORITHM_OUTPUT
    // Returns true if output can be computer
    virtual bool compute(DrivingAlgorithmOutput &output) = 0;

    virtual float getPriority() = 0;
};

#endif /* DRIVING_ALGO_API_HPP */
