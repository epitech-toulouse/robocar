#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "vescController.hpp"

#define VESC_TX_PIN 17
#define VESC_RX_PIN 18

void vesc_control_task(void *pvParameters) {
    VescController vesc(VESC_TX_PIN, VESC_RX_PIN);

    // Spin motor at 5% duty for 2 seconds
    printf("Motor test: 5%% duty\n");
    for (int i = 0; i < 100; i++) {
        vesc.setDuty(0.05f);
        vTaskDelay(pdMS_TO_TICKS(20));
    }

    // Stop motor
    printf("Motor stop\n");
    vesc.setDuty(0.0f);
    vTaskDelay(pdMS_TO_TICKS(1000));

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
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

extern "C" void app_main(void) {
    printf("Starting VESC Controller on ESP32-S3...\n");
    xTaskCreate(vesc_control_task, "vesc_task", 4096, NULL, 5, NULL);
}