#ifndef LIDAR_SENSOR
#define LIDAR_SENSOR

#include "api/lidar_sensor_api.hpp"

class LidarSensor : public LidarSensorApi
{
public:
    bool isActive(void) { return true; }

    bool getData(lidar_array_t &output)
    {
        for (unsigned i = 0; i < 360; i++)
            output[i] = i;
        return true;
    }
};

#endif