#include "vescLidarUart.h"
#include "config.h"
#include "driver/uart.h"
#include "hal/uart_types.h"
#include "esp_err.h"
#include "esp_log.h"

void init_vesc_lidar_uart(void)
{
    uart_config_t config = {
        .baud_rate = VESC_LIDAR_UART_BAUDRATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };
    ESP_ERROR_CHECK(uart_param_config(LIDAR_VESC_UART_PORT, &config));
    ESP_LOGI("vesc lidar uart", "Config done.");

    ESP_ERROR_CHECK(uart_set_pin(LIDAR_VESC_UART_PORT, VESC_UART_TX, LIDAR_UART_RX, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_LOGI("vesc lidar uart", "Pins set.");

    ESP_ERROR_CHECK(uart_driver_install(LIDAR_VESC_UART_PORT, 1024, 1024, 0, NULL, 0));
    ESP_LOGI("vesc lidar uart", "Driver installed.");
}

void delete_vesc_lidar_uart(void)
{
    ESP_ERROR_CHECK(uart_driver_delete(LIDAR_VESC_UART_PORT));
    ESP_LOGI("vesc lidar uart", "Driver deleted.");
}
