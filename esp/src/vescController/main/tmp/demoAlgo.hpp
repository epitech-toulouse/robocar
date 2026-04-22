#ifndef WTF_IS_THIS_WINK_WINK
#define WTF_IS_THIS_WINK_WINK

#include "api/driving_algorithm_interface.hpp"

class DemoAlgo : public DrivingAlgorithmApi
{
public:
    DemoAlgo(GpsSensorApi &gps)
        : gps(gps) {}
    // Should init an update loop and a mutex but I don't really care here

    bool available(void) { return true; }

    bool compute(DrivingAlgorithmOutput &output)
    {
        GpsHeading heading;
        GpsPosition position;

        if (!this->gps.getHeading(heading))
            return false;
        if (!this->gps.getPosition(position))
            return false;
        output.computed_weight = heading.degrees_to_north * 0.0 + 1.0;
        output.target_speed = position.latitude / 1'000'000.0 + 0.1;
        output.target_steering = position.longitude / 1'000'000.0 + 1.0;
        return true;
    }

    float getPriority() { return 1.0; }
private:
    GpsSensorApi &gps;
};

#endif