#ifndef CAMERA_DRIVING_ALGO_HPP
#define CAMERA_DRIVING_ALGO_HPP

#include "api/camera_api.hpp"
#include "api/driving_algorithm_interface.hpp"

class CameraDrivingAlgo : public DrivingAlgorithmApi
{
public:
    explicit CameraDrivingAlgo(CameraSensorApi &cameraSensor)
        : cameraSensor(cameraSensor)
    {
    }

    bool available(void) override;
    bool compute(DrivingAlgorithmOutput &output) override;
    float getPriority() override;

private:
    static float clampf(float value, float lo, float hi);

    CameraSensorApi &cameraSensor;

    static constexpr float laneCruiseSpeed = 0.03f;
    static constexpr float stopPriorityBoost = 2.0f;
};

#endif /* CAMERA_DRIVING_ALGO_HPP */
