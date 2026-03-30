#ifndef CONFIG_H
#define CONFIG_H

#include "hal/uart_types.h"
#include "soc/gpio_num.h"
#include <stdint.h>

/* ALL TX AND RX PINS are ESP-SIDE */

/* LIDAR */
static gpio_num_t const LIDAR_UART_RX = GPIO_NUM_16;
static uart_port_t const LIDAR_UART_PORT = UART_NUM_2;
static uint32_t const LIDAR_UART_BAUDRATE = 230400;

/* VESC CONTROLER */
static uint32_t const VESC_RMT_UART_BAUDRATE = 115200;
static gpio_num_t const VESC_RMT_UART_TX = GPIO_NUM_15;

/* CAMERA */
static uart_port_t const CAMERA_UART_PORT = UART_NUM_1;
static gpio_num_t const CAMERA_UART_TX = GPIO_NUM_17;
static gpio_num_t const CAMERA_UART_RX = GPIO_NUM_18;

/* GPS */
static uart_port_t const GPS_UART_PORT = UART_NUM_0;
static gpio_num_t const GPS_UART_TX = GPIO_NUM_43;
static gpio_num_t const GPS_UART_RX = GPIO_NUM_44;

/* COUPE CIRCUIT */
static gpio_num_t const COUPE_CIRCUIT_PIN = GPIO_NUM_19;

#endif /* CONFIG_H */
