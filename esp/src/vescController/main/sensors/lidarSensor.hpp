#ifndef LIDAR_SENSOR
#define LIDAR_SENSOR

#include "api/lidar_sensor_api.hpp"
#include "lidarReader.hpp"
#include "vescLidarUart.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include "freertos/semphr.h"

class LidarSensor : public LidarSensorApi {
public:
    LidarSensor()
        : lidarReader(), lastSuccessfulUpdateTick(0), uartInitialized(false) {
        initIfNeeded();
        lidarReader.start();
    }

    ~LidarSensor() override {
        // Stop task and reader, free mutex
        if (taskHandle != nullptr) {
            taskRunning = false;
            vTaskDelete(taskHandle);
            taskHandle = nullptr;
        }

        lidarReader.stop();

        if (dataMutex != nullptr) {
            vSemaphoreDelete(dataMutex);
            dataMutex = nullptr;
        }
    }

    bool isActive(void) override {
        initIfNeeded();
        return lidarReader.isRunning();
    }

    bool getData(lidar_array_t &output) override {
        initIfNeeded();

        // Return the last stored scan (task-updated). If data is too old, return false.
        output.fill(UNDEFINED_LIDAR_VALUE);

        const TickType_t now = xTaskGetTickCount();
        if (lastSuccessfulUpdateTick == 0 || (now - lastSuccessfulUpdateTick) > LIDAR_POINT_LIFESPAN) {
            return false;
        }

        if (dataMutex == nullptr) {
            return false;
        }

        if (xSemaphoreTake(dataMutex, LIDAR_MUTEX_TIMEOUT_TICK) != pdTRUE) {
            return false;
        }

        output = lastData;

        xSemaphoreGive(dataMutex);
        return true;
    }

private:
    void initIfNeeded() {
        if (uartInitialized) {
            return;
        }
        init_lidar_uart();
        uartInitialized = true;

        // Create mutex for data access
        dataMutex = xSemaphoreCreateMutex();
        if (dataMutex != nullptr) {
            // initialize stored data to undefined
            lastData.fill(UNDEFINED_LIDAR_VALUE);
        }

        // Start update task
        taskRunning = true;
        xTaskCreate(
            &LidarSensor::lidarTaskEntry,
            "lidar_update",
            4096,
            this,
            tskIDLE_PRIORITY + 1,
            &taskHandle);
    }

    LidarReader lidarReader;
    TickType_t lastSuccessfulUpdateTick;
    bool uartInitialized;
    // Task and synchronization
    TaskHandle_t taskHandle{nullptr};
    SemaphoreHandle_t dataMutex{nullptr};
    lidar_array_t lastData;
    bool taskRunning{false};
    uint32_t lastPublishedScanCount{0};

    static void lidarTaskEntry(void *pv) {
        LidarSensor *self = static_cast<LidarSensor *>(pv);
        if (!self) {
            vTaskDelete(nullptr);
            return;
        }

        const TickType_t delayTicks = pdMS_TO_TICKS(5);

        while (self->taskRunning) {
            self->lidarReader.update();

            const uint32_t completedScanCount = self->lidarReader.getCompletedScanCount();
            if (completedScanCount != self->lastPublishedScanCount) {
                const std::vector<LidarPoint> points = self->lidarReader.getCurrentWorld();
                if (!points.empty()) {
                    lidar_array_t newData;
                    newData.fill(UNDEFINED_LIDAR_VALUE);

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

                        const centimeter_t current = newData[normalizedDegree];
                        if (current == UNDEFINED_LIDAR_VALUE || distanceCm < current) {
                            newData[normalizedDegree] = distanceCm;
                        }
                    }

                    if (self->dataMutex != nullptr) {
                        if (xSemaphoreTake(self->dataMutex, LIDAR_MUTEX_TIMEOUT_TICK) == pdTRUE) {
                            self->lastData = newData;
                            self->lastSuccessfulUpdateTick = xTaskGetTickCount();
                            self->lastPublishedScanCount = completedScanCount;
                            xSemaphoreGive(self->dataMutex);
                        }
                    } else {
                        self->lastData = newData;
                        self->lastSuccessfulUpdateTick = xTaskGetTickCount();
                        self->lastPublishedScanCount = completedScanCount;
                    }
                }
            }

            vTaskDelay(delayTicks);
        }

        vTaskDelete(nullptr);
    }
};

#endif
