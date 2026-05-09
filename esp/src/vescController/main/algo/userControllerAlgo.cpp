#include "userControllerAlgo.hpp"
#include "config.h"
#include <limits>

float UserControllerAlgo::getPriority()
{
    return MANUAL_WEIGHT;
}

bool UserControllerAlgo::compute(DrivingAlgorithmOutput &output)
{
    if (!_userController.isConnected()) {
        output = DEFAULT_DRIVING_ALGORITHM_OUTPUT;
        return false;
    }

    output.target_speed = _userController.getSpeed();
    output.target_steering = _userController.getSteering();
    output.computed_weight = 1.0f;


    return true;
}
