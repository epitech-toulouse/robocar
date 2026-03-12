#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "vescController.hpp"

#define VESC_TX_PIN 17
#define VESC_RX_PIN 18

void vesc_control_task(void *pvParameters) {
    vescController myVesc(VESC_TX_PIN, VESC_RX_PIN);
    
    float steeringPos = 0.5f; // Start at center
    float increment = 0.01f;   // For a simple "sweep" demo

    while (1) {
        printf("Setting steering position: %.2f\n", steeringPos);
        myVesc.setSteering(steeringPos);
        
        steeringPos += increment;
        if (steeringPos >= 0.9f || steeringPos <= 0.1f) {
            increment = -increment; // Reverse direction
        }

        // VESC expects a command at least every 1000ms. 
        // We will send one every 20ms (50Hz) for smooth steering.
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

extern "C" void app_main(void) {
    printf("Starting VESC Controller on ESP32-S3...\n");

    xTaskCreate(vesc_control_task, "vesc_task", 4096, NULL, 5, NULL);
}