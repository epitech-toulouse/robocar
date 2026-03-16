#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "vescController.hpp"
#include "lidarReader.hpp"
#include <iostream>

#include "esp_err.h"

#define VESC_TX_PIN 17
#define VESC_RX_PIN 18
#define LIDAR_UART_NUM 2

#define LIDAR_RX_PIN 16

static constexpr float FRONT_WINDOW_DEG = 8.0f;

void vesc_control_task(void *pvParameters) {
    VescController vesc(VESC_TX_PIN, VESC_RX_PIN);
    // LD19 sends data from its TX line into ESP RX. We do not need ESP TX for LD19.
    LidarReader lidar(LIDAR_RX_PIN, -1, LIDAR_UART_NUM);
    std::cout << "LiDAR UART config: uart=" << LIDAR_UART_NUM << " rx=" << LIDAR_RX_PIN << std::endl;
    if (lidar.start() != ESP_OK) {
        std::cout << "Failed to start LidarReader" << std::endl;
    }

    // Sweep steering servo
    printf("Steering sweep\n");
    float pos = 0.5f;
    float step = 0.01f;

    while (1) {
        vesc.setDuty(0.01f);
        vesc.setSteering(pos);
        pos += step;
        if (pos >= 0.9f || pos <= 0.1f)
            step = -step;

        const bool gotUartBytes = lidar.poll();
        std::vector<LidarPoint> lastScan = lidar.getLatestScanPoints();

        float nearestPoint = -1.0f;
        for (const auto& point : lastScan) {
            if (point.angleDeg >= (360.0f - FRONT_WINDOW_DEG) || point.angleDeg <= FRONT_WINDOW_DEG) {
                if (nearestPoint < 0.0f || point.distanceMeters < nearestPoint) {
                    nearestPoint = point.distanceMeters;
                }
            }
        }

        if (lastScan.empty()) {
            std::cout << "LiDAR scan not ready yet. UART bytes=" << (gotUartBytes ? "yes" : "no") << std::endl;
        } else if (nearestPoint < 0.0f) {
            std::cout << "No valid point in +/-" << FRONT_WINDOW_DEG << " deg front window." << std::endl;
        } else {
            std::cout << "Nearest front point: " << nearestPoint << " meters (scan points=" << lastScan.size() << ")" << std::endl;
        }

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

extern "C" void app_main(void) {
    printf("Starting VESC Controller on ESP32-S3...\n");
    xTaskCreate(vesc_control_task, "vesc_task", 4096, NULL, 5, NULL);
}
