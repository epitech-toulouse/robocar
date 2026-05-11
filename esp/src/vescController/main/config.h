#ifndef CONFIG_H
#define CONFIG_H

#include "hal/uart_types.h"
#include "rmt_uart.h"
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
static uint8_t const VESC_RMT_UART_PORT = RMT_UART_NUM_0;

/* CAMERA */
static uart_port_t const CAMERA_UART_PORT = UART_NUM_1;
static gpio_num_t const CAMERA_UART_TX = GPIO_NUM_17;
static gpio_num_t const CAMERA_UART_RX = GPIO_NUM_18;

/* GPS */
static uart_port_t const GPS_UART_PORT = UART_NUM_0;
static gpio_num_t const GPS_UART_TX = GPIO_NUM_43;
static gpio_num_t const GPS_UART_RX = GPIO_NUM_44;
static uint32_t const GPS_UART_BAUDRATE = 460800;

/* COUPE CIRCUIT */
// GPIO21/GPIO22 are adjacent and free in this project.
static gpio_num_t const COUPE_CIRCUIT_PIN = GPIO_NUM_36;
static gpio_num_t const COUPE_CIRCUIT_GND_PIN = GPIO_NUM_35;

static float const AVOID_DISTANCE_M = 1.0f;

/* WEIGHTS */
static float const MANUAL_WEIGHT = 100.0;
static float const LIDAR_AVOIDANCE_WEIGHT = 10.0;
static float const LIDAR_CORRIDOR_WEIGHT = 2.0;
static float const CAMEDAR_WEIGHT = 5.0;
static float const GPS_WEIGHT = 1.0;

/* SPEEDS */
static float const HEADING_FOUND_SPEED = 0.05;
static float const WAIT_FOR_HEADING_SPEED = 0.03;

#endif /* CONFIG_H */
