#ifndef GPS_SENSOR
#define GPS_SENSOR

#include "api/gps_sensor_api.hpp"

class GpsSensor : public GpsSensorApi
{
public:
    bool isActive(void) { return true; }

    bool getPosition(GpsPosition &output)
    {
        output.latitude = 10.0;
        output.longitude = 15.0;
        return true;
    }

    bool getHeading(GpsHeading &output)
    {
        output.degrees_to_north = 0.8;
        return true;
    }
};

#endif