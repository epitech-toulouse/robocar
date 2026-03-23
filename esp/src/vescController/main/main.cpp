#include <cstdint>
#include <stdio.h>
#include "config.h"
#include "driver/gpio.h"
#include "esp_attr.h"
#include "esp_intr_alloc.h"
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "freertos/task.h"
#include "hal/gpio_types.h"
#include "portmacro.h"
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

#include "esp_log.h"

static TaskHandle_t vesc_control_task_handle = nullptr;

void IRAM_ATTR coupe_circuit_handler(void *args)
{
    (void) args;
    BaseType_t priorityTaken = pdFALSE;

    if (vesc_control_task_handle)
        vTaskNotifyGiveFromISR(vesc_control_task_handle, &priorityTaken);
    if (priorityTaken != pdFALSE) {
        portYIELD_FROM_ISR(priorityTaken);
    }
/*
    VescController *vesc = (VescController *) vesc_ptr;

    int level = gpio_get_level(COUPE_CIRCUIT_PIN);
    // ESP_LOGE("COUPE_CIRCUIT", "Level = %d\n", level);
    if (level) // HIGH = disconnected
        vesc->deactivate();
    else
        vesc->activate();
*/
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
    gpio_set_direction(COUPE_CIRCUIT_PIN, GPIO_MODE_INPUT);
    gpio_set_pull_mode(COUPE_CIRCUIT_PIN, GPIO_PULLUP_ONLY);
    gpio_set_intr_type(COUPE_CIRCUIT_PIN, GPIO_INTR_ANYEDGE);
    ESP_ERROR_CHECK(gpio_install_isr_service(ESP_INTR_FLAG_LEVEL3 | ESP_INTR_FLAG_EDGE | ESP_INTR_FLAG_IRAM));
    ESP_ERROR_CHECK(gpio_isr_handler_add(COUPE_CIRCUIT_PIN, &coupe_circuit_handler, nullptr));
    gpio_intr_enable(COUPE_CIRCUIT_PIN);
    vTaskDelay(pdMS_TO_TICKS(20));

    TickType_t reverseUntil = 0;
    float reverseSteer = STEER_CENTER;
    uint32_t notification_value = 0;

    while (1) {
        if (xTaskNotifyWait(0, 0, &notification_value, pdMS_TO_TICKS(20)) == pdPASS) { // On interrupt on coupe circuit pin
            if (gpio_get_level(COUPE_CIRCUIT_PIN)) { // HIGH = disconnected
                vesc.deactivate();
            } else {
                vesc.activate();
            }
            continue;
        }
        float manualDuty, manualSteer;
        if (get_manual_control(manualDuty, manualSteer)) {
            vesc.setSteering(manualSteer);
            vesc.setDuty(manualDuty);
            continue;
        }

        if (!lidarEnabled) {
            vesc.setDuty(0.0f);
            vesc.setSteering(STEER_CENTER);
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
                continue;
            }

            vesc.setDuty(0.0f);
            vesc.setSteering(STEER_CENTER);
            if ((now - lastLidarLog) > LIDAR_LOG_PERIOD_TICKS) {
                std::cout << "LiDAR scan not ready yet. UART bytes=" << (gotUartBytes ? "yes" : "no") << std::endl;
                lastLidarLog = now;
            }
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
    }
}

extern "C" void app_main(void) {
    printf("Starting VESC Controller on ESP32-S3...\n");
    init_bluetooth_receiver();
    init_lidar_uart();
    init_vesc_rmt_uart();
    xTaskCreate(vesc_control_task, "vesc_task", 4096, NULL, 5, &vesc_control_task_handle);
}
