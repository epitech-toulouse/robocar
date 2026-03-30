#include "vescLidarUart.h"
#include "config.h"
#include "driver/uart.h"
#include "hal/uart_types.h"
#include "esp_err.h"
#include "esp_log.h"
#include "driver/uart.h"

void init_lidar_uart(void)
{
    uart_driver_delete(1);
    uart_driver_delete(2);
    uart_config_t config = {
        .baud_rate = LIDAR_UART_BAUDRATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };
    ESP_ERROR_CHECK(uart_param_config(LIDAR_UART_PORT, &config));
    ESP_LOGI("lidar uart", "Config done.");

    ESP_ERROR_CHECK(uart_set_pin(LIDAR_UART_PORT, GPIO_NUM_4, LIDAR_UART_RX, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_LOGI("lidar uart", "Pins set.");

    ESP_ERROR_CHECK(uart_driver_install(LIDAR_UART_PORT, 1024, 0, 0, NULL, 0));
    ESP_LOGI("lidar uart", "Driver installed.");
}

void delete_vesc_lidar_uart(void)
{
    ESP_ERROR_CHECK(uart_driver_delete(LIDAR_UART_PORT));
    ESP_LOGI("lidar uart", "Driver deleted.");
}

void init_vesc_rmt_uart(void)
{
    // Replaced RMT with standard hardware UART1 for ESP-IDF v5 compatibility
    uart_config_t config = {
        .baud_rate = VESC_RMT_UART_BAUDRATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };
    ESP_ERROR_CHECK(uart_param_config(UART_NUM_1, &config));
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM_1, VESC_RMT_UART_TX, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_1, 1024, 0, 0, NULL, 0));
    ESP_LOGI("vesc rmt uart", "Hardware UART1 configured for VESC (replacing RMT).");
}

void delete_vesc_rmt_uart(void)
{
    ESP_ERROR_CHECK(uart_driver_delete(UART_NUM_1));
    ESP_LOGI("vesc rmt uart", "Driver deleted.");
}
