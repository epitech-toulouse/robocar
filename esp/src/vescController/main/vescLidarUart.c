#include "vescLidarUart.h"
#include "config.h"
#include "driver/uart.h"
#include "hal/uart_types.h"
#include "esp_err.h"
#include "esp_log.h"
#include "rmt_uart.h"

void init_lidar_uart(void)
{
    // uart_driver_delete(0);
    // uart_driver_delete(1);
    // uart_driver_delete(2);
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

    ESP_ERROR_CHECK(uart_driver_install(LIDAR_UART_PORT, 8192, 0, 0, NULL, 0));
    ESP_LOGI("lidar uart", "Driver installed.");
}

void delete_vesc_lidar_uart(void)
{
    ESP_ERROR_CHECK(uart_driver_delete(LIDAR_UART_PORT));
    ESP_LOGI("lidar uart", "Driver deleted.");
}

void init_vesc_rmt_uart(void)
{
    rmt_uart_config_t config = {
        .baud_rate = VESC_RMT_UART_BAUDRATE,
        .mode = RMT_UART_MODE_TX_ONLY,
        .data_bits = RMT_UART_DATA_8_BITS,
        .parity = RMT_UART_PARITY_DISABLE,
        .stop_bits = RMT_UART_STOP_BITS_1,
        .tx_io_num = VESC_RMT_UART_TX,
        .rx_io_num = 0,
        .buffer_size = 10000
    };
    ESP_ERROR_CHECK(rmt_uart_init(VESC_RMT_UART_PORT, &config));
    ESP_LOGI("vesc rmt uart", "Config done.");
}

void delete_vesc_rmt_uart(void)
{
    ESP_ERROR_CHECK(rmt_uart_deinit(VESC_RMT_UART_PORT));
    ESP_LOGI("vesc rmt uart", "Driver deleted.");
}
