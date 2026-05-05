#include "gpsSensor.hpp"

#include <cmath>

#include "esp_log.h"
#include "freertos/task.h"

namespace {
const char *TAG = "GpsSensor";
constexpr TickType_t HEADING_LOG_PERIOD_TICKS = pdMS_TO_TICKS(1000);
}

GpsSensor::~GpsSensor()
{
    if (this->started) {
        this->gps.stop();
    }
}

bool GpsSensor::isActive(void)
{
    return this->ensureStarted();
}

bool GpsSensor::getPosition(GpsPosition &output)
{
    if (!this->ensureStarted()) {
        return false;
    }

    const GpsFix fix = this->gps.getLatestFix();
    if (!fix.hasFix) {
        return false;
    }

    output.latitude = fix.latitude;
    output.longitude = fix.longitude;
    return true;
}

bool GpsSensor::getStatus(GpsStatus &output)
{
    if (!this->ensureStarted()) {
        return false;
    }

    const GpsFix fix = this->gps.getLatestFix();
    output.has_fix = fix.hasFix;
    output.satellites = fix.satellites;
    output.fix_mode = mapFixMode(fix.fixQuality);
    output.is_rtk = fix.isRtkFixed || fix.isRtkFloat;
    output.is_rtk_fixed = fix.isRtkFixed;
    return true;
}

bool GpsSensor::getHeading(GpsHeading &output)
{
    static TickType_t lastHeadingLogTick = 0;
    const TickType_t now = xTaskGetTickCount();
    const bool shouldLog = (now - lastHeadingLogTick) > HEADING_LOG_PERIOD_TICKS;

    if (!this->ensureStarted()) {
        if (shouldLog) {
            ESP_LOGI(TAG, "heading invalid: GPS host not started");
            lastHeadingLogTick = now;
        }
        return false;
    }

    const GpsFix fix = this->gps.getLatestFix();
    if (!fix.hasFix) {
        if (shouldLog) {
            ESP_LOGI(TAG, "heading invalid: no fix (sats=%d quality=%d)", fix.satellites, fix.fixQuality);
            lastHeadingLogTick = now;
        }
        return false;
    }

    this->rememberFix(fix);
    if (this->headingHistory.size() < 2) {
        if (shouldLog) {
            ESP_LOGI(TAG, "heading invalid: not enough history (%u points)", static_cast<unsigned>(this->headingHistory.size()));
            lastHeadingLogTick = now;
        }
        return false;
    }

    const GpsFix &previous = this->headingHistory[this->headingHistory.size() - 2];
    const GpsFix &current = this->headingHistory.back();
    if (!std::isfinite(previous.latitude) || !std::isfinite(previous.longitude) ||
        !std::isfinite(current.latitude) || !std::isfinite(current.longitude)) {
        if (shouldLog) {
            ESP_LOGI(TAG, "heading invalid: non-finite coordinates");
            lastHeadingLogTick = now;
        }
        return false;
    }

    const double movedMeters = planarDistanceMeters(previous.latitude,
                                                    previous.longitude,
                                                    current.latitude,
                                                    current.longitude);
    if (movedMeters < 0.5) {
        if (shouldLog) {
            ESP_LOGD(TAG,
                     "heading invalid: low movement moved=%.3fm (<0.5m) prev=(%.6f,%.6f) cur=(%.6f,%.6f)",
                     movedMeters,
                     previous.latitude,
                     previous.longitude,
                     current.latitude,
                     current.longitude);
            lastHeadingLogTick = now;
        }
        return false;
    }

    output.degrees_to_north = initialBearingDegrees(previous.latitude,
                                                    previous.longitude,
                                                    current.latitude,
                                                    current.longitude);
    if (shouldLog) {
        ESP_LOGI(TAG, "heading valid: %.1f deg (moved=%.2fm)", output.degrees_to_north, movedMeters);
        lastHeadingLogTick = now;
    }
    return true;
}

double GpsSensor::toRadians(double degrees)
{
    return degrees * kPi / 180.0;
}

double GpsSensor::toDegrees(double radians)
{
    return radians * 180.0 / kPi;
}

double GpsSensor::normalizeDegrees(double degrees)
{
    double normalized = std::fmod(degrees, 360.0);
    if (normalized < 0.0) {
        normalized += 360.0;
    }
    return normalized;
}

double GpsSensor::planarDistanceMeters(double lat1, double lon1, double lat2, double lon2)
{
    const double meanLatRad = toRadians((lat1 + lat2) * 0.5);
    const double metersPerDegLat = 111132.0;
    const double metersPerDegLon = 111320.0 * std::cos(meanLatRad);
    const double dLatM = (lat2 - lat1) * metersPerDegLat;
    const double dLonM = (lon2 - lon1) * metersPerDegLon;
    return std::sqrt(dLatM * dLatM + dLonM * dLonM);
}

double GpsSensor::initialBearingDegrees(double lat1, double lon1, double lat2, double lon2)
{
    const double startLat = toRadians(lat1);
    const double endLat = toRadians(lat2);
    const double dLon = toRadians(lon2 - lon1);

    const double y = std::sin(dLon) * std::cos(endLat);
    const double x = std::cos(startLat) * std::sin(endLat) -
                     std::sin(startLat) * std::cos(endLat) * std::cos(dLon);
    return normalizeDegrees(toDegrees(std::atan2(y, x)));
}

GpsFixMode GpsSensor::mapFixMode(int fixQuality)
{
    switch (fixQuality) {
        case 1:
            return GpsFixMode::Autonomous;
        case 2:
            return GpsFixMode::Differential;
        case 4:
            return GpsFixMode::RtkFixed;
        case 5:
            return GpsFixMode::RtkFloat;
        case 0:
            return GpsFixMode::Invalid;
        default:
            return GpsFixMode::Other;
    }
}

void GpsSensor::rememberFix(const GpsFix &fix)
{
    if (fix.updateCounter == 0 || fix.updateCounter == this->lastSeenFixCounter) {
        return;
    }

    this->lastSeenFixCounter = fix.updateCounter;
    this->headingHistory.push_back(fix);
    while (this->headingHistory.size() > 8) {
        this->headingHistory.pop_front();
    }
}

bool GpsSensor::ensureStarted()
{
    if (this->started) {
        return this->gps.isRunning();
    }

    if (this->gps.start() != ESP_OK) {
        return false;
    }

    this->started = true;
    return true;
}
