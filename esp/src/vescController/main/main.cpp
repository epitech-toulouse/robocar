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

#include "drive.hpp"

#include "esp_log.h"

static constexpr TickType_t LIDAR_NO_DATA_TIMEOUT_TICKS = pdMS_TO_TICKS(3000);
static constexpr TickType_t LIDAR_LOG_PERIOD_TICKS = pdMS_TO_TICKS(1000);

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
}

void vesc_control_task(void *pvParameters) {
    VescController vesc;
    // LD19 sends data from its TX line into ESP RX. We do not need ESP TX for LD19.
    LidarReader lidar;
    AutonomousDriver driver;
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
        bool s_emergency;
        if (get_manual_control(manualDuty, manualSteer, s_emergency)) {
            vesc.setSteering(manualSteer);
            vesc.setDuty(manualDuty);
            if (s_emergency) {
                vesc.deactivate();
            }
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

        DriveCommands cmds = driver.compute_commands(lastScan);
        vesc.setSteering(cmds.steer);
        vesc.setDuty(cmds.duty);
    }
}

extern "C" void app_main(void) {
    printf("Starting VESC Controller on ESP32-S3...\n");
    init_bluetooth_receiver();
    init_lidar_uart();
    init_vesc_rmt_uart();
    xTaskCreate(vesc_control_task, "vesc_task", 4096, NULL, 5, &vesc_control_task_handle);
}
