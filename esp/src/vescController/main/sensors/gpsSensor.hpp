#ifndef GPS_SENSOR
#define GPS_SENSOR

#include <cstdint>
#include <deque>

#include "api/gps_sensor_api.hpp"
#include "gpsUsbHost.hpp"

// Adapter that exposes the USB GPS host through the shared GPS sensor API.
// Position is taken from the latest fix; heading is derived from recent movement.
class GpsSensor : public IGpsSensor
{
public:
    GpsSensor() = default;
    ~GpsSensor() override;

    bool isActive(void) override;
    bool getPosition(GpsPosition &output) override;
    bool getStatus(GpsStatus &output) override;
    bool getHeading(GpsHeading &output) override;

private:
    static constexpr double kPi = 3.14159265358979323846;
    static constexpr double kEarthRadiusMeters = 6371000.0;

    static double toRadians(double degrees);
    static double toDegrees(double radians);
    static double normalizeDegrees(double degrees);
    static double planarDistanceMeters(double lat1, double lon1, double lat2, double lon2);
    static double initialBearingDegrees(double lat1, double lon1, double lat2, double lon2);
    static GpsFixMode mapFixMode(int fixQuality);

    void rememberFix(const GpsFix &fix);
    bool ensureStarted();

    UsbGpsHost gps;
    std::deque<GpsFix> headingHistory;
    uint32_t lastSeenFixCounter = 0;
    bool started = false;
};

#endif