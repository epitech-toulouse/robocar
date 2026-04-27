/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** coupe circuit manager
*/

#include "CoupeCircuitManager.hpp"
#include "api/vesc_controller_api.hpp"
#include "config.h"
#include "freertos/idf_additions.h"

#include <cstdint>
#include <freertos/FreeRTOS.h>
#include <driver/gpio.h>

static TaskHandle_t coupe_circuit_task_handle = nullptr;

void IRAM_ATTR coupe_circuit_handler(void *args)
{
    (void) args;
    BaseType_t priorityTaken = pdFALSE;

    if (coupe_circuit_task_handle)
        vTaskNotifyGiveFromISR(coupe_circuit_task_handle, &priorityTaken);
    if (priorityTaken != pdFALSE) {
        portYIELD_FROM_ISR(priorityTaken);
    }
}

CoupeCircuitManager::CoupeCircuitManager(VescControllerApi &vesc)
    : vesc(vesc)
{
    // Setup coupe circuit interruption
    gpio_set_direction(COUPE_CIRCUIT_GND_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(COUPE_CIRCUIT_GND_PIN, 0);
    gpio_set_direction(COUPE_CIRCUIT_PIN, GPIO_MODE_INPUT);
    gpio_set_pull_mode(COUPE_CIRCUIT_PIN, GPIO_PULLUP_ONLY);
    gpio_set_intr_type(COUPE_CIRCUIT_PIN, GPIO_INTR_ANYEDGE);
    ESP_ERROR_CHECK(gpio_install_isr_service(ESP_INTR_FLAG_LEVEL3 | ESP_INTR_FLAG_EDGE | ESP_INTR_FLAG_IRAM));
    ESP_ERROR_CHECK(gpio_isr_handler_add(COUPE_CIRCUIT_PIN, &coupe_circuit_handler, nullptr));
    gpio_intr_enable(COUPE_CIRCUIT_PIN);

    // Setup coupe circuit task
    xTaskCreate(this->task, "coupe_circuit_task", 4096, this, 1, &coupe_circuit_task_handle);
}

CoupeCircuitManager::~CoupeCircuitManager()
{
    this->vesc.deactivate();
}

void CoupeCircuitManager::task(void *args)
{
    uint32_t notification_value = 0;
    VescControllerApi *vesc = (VescControllerApi *) args;

    // On interrupt on "coupe circuit" pin
    if (xTaskNotifyWait(0, 0, &notification_value, pdMS_TO_TICKS(20)) == pdPASS) {
        if (gpio_get_level(COUPE_CIRCUIT_PIN)) { // HIGH = disconnected
            vesc->deactivate();
        } else {
            vesc->activate();
        }
    }
}
