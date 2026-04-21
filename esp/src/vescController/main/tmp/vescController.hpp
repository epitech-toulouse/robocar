#ifndef VESC_CONTROLLER
#define VESC_CONTROLLER

#include "api/vesc_controller_api.hpp"

class VescController : public VescControllerApi
{
public:
    bool isActive(void) { return true; }

    void stop(void) {}
    void deactivate(void) {}
    void activate(void) {}

    void set_speed(float) {}
    void set_steering(float) {}
};

#endif