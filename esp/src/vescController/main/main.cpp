#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "vescController.hpp"
#include "lidarReader.hpp"
#include <iostream>

#include "esp_err.h"
#include "bluetooth_receiver.hpp"
#include "vescLidarUart.h"

// LiDAR-only driving parameters.
static constexpr float FRONT_WINDOW_DEG = 25.0f;
static constexpr float SIDE_WINDOW_MIN_DEG = 25.0f;
static constexpr float SIDE_WINDOW_MAX_DEG = 80.0f;

static constexpr float STOP_DISTANCE_M = 0.40f;
static constexpr float SLOW_DISTANCE_M = 1.00f;
static constexpr float SAFE_DISTANCE_M = 2.20f;

static constexpr float SPEED_FORWARD = 0.050f;
static constexpr float SPEED_SLOW = 0.025f;
static constexpr float SPEED_REVERSE = -0.030f;

static constexpr float STEER_CENTER = 0.50f;
static constexpr float STEER_LEFT = 0.20f;
static constexpr float STEER_RIGHT = 0.80f;

static constexpr TickType_t REVERSE_DURATION_TICKS = pdMS_TO_TICKS(800);
static constexpr TickType_t LIDAR_NO_DATA_TIMEOUT_TICKS = pdMS_TO_TICKS(3000);
static constexpr TickType_t LIDAR_LOG_PERIOD_TICKS = pdMS_TO_TICKS(1000);

static float clampf(float value, float lo, float hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static bool in_sector(float angleDeg, float minDeg, float maxDeg) {
    return angleDeg >= minDeg && angleDeg <= maxDeg;
}

static float nearest_in_sector(const std::vector<LidarPoint>& scan, float minDeg, float maxDeg) {
    float nearest = -1.0f;
    for (const auto& p : scan) {
        if (!in_sector(p.angleDeg, minDeg, maxDeg)) {
            continue;
        }
        if (nearest < 0.0f || p.distanceMeters < nearest) {
            nearest = p.distanceMeters;
        }
    }
    return nearest;
}

void vesc_control_task(void *pvParameters) {
    VescController vesc;
    // LD19 sends data from its TX line into ESP RX. We do not need ESP TX for LD19.
    LidarReader lidar;
    bool lidarEnabled = (lidar.start() == ESP_OK);
    TickType_t lidarNoDataSince = 0;
    TickType_t lastLidarLog = 0;

    vesc.setDuty(0.0f);
    vesc.setSteering(STEER_CENTER);
    vTaskDelay(pdMS_TO_TICKS(20));

    TickType_t reverseUntil = 0;
    float reverseSteer = STEER_CENTER;

    while (1) {
        float manualDuty, manualSteer;
        if (get_manual_control(manualDuty, manualSteer)) {
            vesc.setSteering(manualSteer);
            vesc.setDuty(manualDuty);
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        if (!lidarEnabled) {
            vesc.setDuty(0.0f);
            vesc.setSteering(STEER_CENTER);
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        const bool gotUartBytes = lidar.poll();
        std::vector<LidarPoint> lastScan = lidar.getLatestScanPoints();

        if (gotUartBytes) {
            lidarNoDataSince = 0;
        } else if (lidarNoDataSince == 0) {
            lidarNoDataSince = xTaskGetTickCount();
        }

        if (lastScan.empty()) {
            const TickType_t now = xTaskGetTickCount();
            if (lidarNoDataSince != 0 && (now - lidarNoDataSince) > LIDAR_NO_DATA_TIMEOUT_TICKS) {
                lidarEnabled = false;
                std::cout << "LiDAR timeout (no UART data) -> manual BLE mode only" << std::endl;
                vesc.setDuty(0.0f);
                vesc.setSteering(STEER_CENTER);
                vTaskDelay(pdMS_TO_TICKS(20));
                continue;
            }

            vesc.setDuty(0.0f);
            vesc.setSteering(STEER_CENTER);
            if ((now - lastLidarLog) > LIDAR_LOG_PERIOD_TICKS) {
                std::cout << "LiDAR scan not ready yet. UART bytes=" << (gotUartBytes ? "yes" : "no") << std::endl;
                lastLidarLog = now;
            }
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        const float frontNear = nearest_in_sector(lastScan, 0.0f, FRONT_WINDOW_DEG) < 0.0f
                                    ? nearest_in_sector(lastScan, 360.0f - FRONT_WINDOW_DEG, 360.0f)
                                    : std::min(nearest_in_sector(lastScan, 0.0f, FRONT_WINDOW_DEG),
                                               nearest_in_sector(lastScan, 360.0f - FRONT_WINDOW_DEG, 360.0f));

        const float leftNear = nearest_in_sector(lastScan, SIDE_WINDOW_MIN_DEG, SIDE_WINDOW_MAX_DEG);
        const float rightNear = nearest_in_sector(lastScan, 360.0f - SIDE_WINDOW_MAX_DEG, 360.0f - SIDE_WINDOW_MIN_DEG);

        const TickType_t now = xTaskGetTickCount();
        if (now < reverseUntil) {
            vesc.setSteering(reverseSteer);
            vesc.setDuty(SPEED_REVERSE);
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        // If there is a critical obstacle in front, back up and turn toward the more open side.
        if (frontNear > 0.0f && frontNear < STOP_DISTANCE_M) {
            const bool leftMoreOpen = (leftNear < 0.0f) || (rightNear > 0.0f && rightNear > leftNear);
            reverseSteer = leftMoreOpen ? STEER_RIGHT : STEER_LEFT;
            reverseUntil = now + REVERSE_DURATION_TICKS;
            vesc.setSteering(reverseSteer);
            vesc.setDuty(SPEED_REVERSE);
            std::cout << "Reverse: front=" << frontNear << "m left=" << leftNear << "m right=" << rightNear << "m" << std::endl;
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        float steer = STEER_CENTER;
        if (leftNear > 0.0f && rightNear > 0.0f) {
            if (leftNear > rightNear) {
                steer = STEER_LEFT;
            } else if (rightNear > leftNear) {
                steer = STEER_RIGHT;
            }
        } else if (leftNear > 0.0f && rightNear < 0.0f) {
            steer = STEER_LEFT;
        } else if (rightNear > 0.0f && leftNear < 0.0f) {
            steer = STEER_RIGHT;
        }

        float speed = SPEED_FORWARD;
        if (frontNear > 0.0f) {
            if (frontNear <= STOP_DISTANCE_M) {
                speed = 0.0f;
            } else if (frontNear < SAFE_DISTANCE_M) {
                const float ratio = (frontNear - STOP_DISTANCE_M) / (SAFE_DISTANCE_M - STOP_DISTANCE_M);
                speed = SPEED_SLOW + (SPEED_FORWARD - SPEED_SLOW) * clampf(ratio, 0.0f, 1.0f);
            }
        }

        vesc.setSteering(steer);
        vesc.setDuty(speed);

        std::cout << "AUTO front=" << frontNear
                  << "m left=" << leftNear
                  << "m right=" << rightNear
                  << "m steer=" << steer
                  << " speed=" << speed
                  << " pts=" << lastScan.size() << std::endl;

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

extern "C" void app_main(void) {
    printf("Starting VESC Controller on ESP32-S3...\n");
    init_bluetooth_receiver();
    init_lidar_uart();
    init_vesc_rmt_uart();
    xTaskCreate(vesc_control_task, "vesc_task", 4096, NULL, 5, NULL);
}
