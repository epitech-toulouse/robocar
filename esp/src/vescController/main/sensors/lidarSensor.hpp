#ifndef LIDAR_SENSOR
#define LIDAR_SENSOR

#include "api/lidar_sensor_api.hpp"
#include "lidarReader.hpp"
#include "vescLidarUart.h"

#include <algorithm>
#include <cmath>

class LidarSensor : public ILidarSensor {
public:
    LidarSensor()
        : lidarReader(), lastSuccessfulUpdateTick(0), uartInitialized(false) {
        initIfNeeded();
        lidarReader.start();
    }

    ~LidarSensor() override {
        lidarReader.stop();
    }

    bool isActive(void) override {
        initIfNeeded();
        return lidarReader.isRunning();
    }

    bool getData(lidar_array_t &output) override {
        initIfNeeded();
        output.fill(UNDEFINED_LIDAR_VALUE);

        // Consume available UART bytes and refresh complete scans when available.
        if (lidarReader.update()) {
            lastSuccessfulUpdateTick = xTaskGetTickCount();
        }

        const TickType_t now = xTaskGetTickCount();
        if (lastSuccessfulUpdateTick == 0 || (now - lastSuccessfulUpdateTick) > LIDAR_POINT_LIFESPAN) {
            return false;
        }

        const std::vector<LidarPoint> points = lidarReader.getLatestScanPoints();
        if (points.empty()) {
            return false;
        }

        for (const LidarPoint &point : points) {
            const int degree = static_cast<int>(std::lround(point.angleDeg));
            int normalizedDegree = degree % static_cast<int>(LIDAR_POINT_NUMBER);
            if (normalizedDegree < 0) {
                normalizedDegree += LIDAR_POINT_NUMBER;
            }

            const float distanceCmF = point.distanceMeters * 100.0f;
            if (distanceCmF <= 0.0f) {
                continue;
            }

            const float clamped = std::min(distanceCmF, static_cast<float>(UINT16_MAX));
            const centimeter_t distanceCm = static_cast<centimeter_t>(clamped);

            const centimeter_t current = output[normalizedDegree];
            if (current == UNDEFINED_LIDAR_VALUE || distanceCm < current) {
                output[normalizedDegree] = distanceCm;
            }
        }
        return true;
    }

private:
    void initIfNeeded() {
        if (uartInitialized) {
            return;
        }
        init_lidar_uart();
        uartInitialized = true;
    }

    LidarReader lidarReader;
    TickType_t lastSuccessfulUpdateTick;
    bool uartInitialized;
};

#endif