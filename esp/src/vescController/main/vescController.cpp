#include "vescController.hpp"
#include "driver/uart.h"
#include <cstring>

vescController::vescController(int txdPin, int rxdPin) {
    this->txdPin = txdPin;
    this->rxdPin = rxdPin;

    initVescUART();
}

vescController::~vescController() {
    //TODO
}

void vescController::initVescUART() {
    uart_config_t uart_config = {};
    uart_config.baud_rate = 115200;
    uart_config.data_bits = UART_DATA_8_BITS;
    uart_config.parity = UART_PARITY_DISABLE;
    uart_config.stop_bits = UART_STOP_BITS_1;
    uart_config.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
    uart_param_config(UART_NUM_1, &uart_config);
    uart_set_pin(UART_NUM_1, this->txdPin, this->rxdPin, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    uart_driver_install(UART_NUM_1, 1024, 0, 0, NULL, 0);
}

void vescController::sendPacket(uint8_t* payload, int len) {
    uint8_t frame[256];
    int count = 0;

    // 1. Start Byte (0x02 for small packets, 0x03 for large)
    frame[count++] = VESC_START_BYTE;
    
    // 2. Payload Length
    frame[count++] = (uint8_t)len;

    // 3. The Actual Data
    memcpy(&frame[count], payload, len);
    count += len;

    // 4. CRC16 Checksum
    uint16_t crc = crc16(payload, len);
    frame[count++] = (uint8_t)(crc >> 8);
    frame[count++] = (uint8_t)(crc & 0xFF);

    // 5. End Byte
    frame[count++] = VESC_END_BYTE;

    // Send it to the VESC
    uart_write_bytes(UART_NUM_1, (const char*)frame, count);
}

void vescController::set_vesc_duty(float duty) {
    uint8_t payload[5];
    // VESC expects duty cycle multiplied by 100,000
    int32_t d = (int32_t)(duty * 100000.0f);
    
    payload[0] = 5; // COMM_SET_DUTY
    payload[1] = (d >> 24) & 0xFF;
    payload[2] = (d >> 16) & 0xFF;
    payload[3] = (d >> 8) & 0xFF;
    payload[4] = d & 0xFF;
    
    sendPacket(payload, 5);
}

/// @brief set steering position of the servo, 0.0 (left) to 1.0 (right), 0.5 is center
/// @param position the sterring position
void vescController::setSteering(float position) {
    // VESC expects 0 to 1000 for the servo pulse
    uint16_t pos = (uint16_t)(position * 1000.0f);
    
    uint8_t payload[3];
    payload[0] = 7; // COMM_SET_SERVO
    payload[1] = (pos >> 8) & 0xFF;
    payload[2] = pos & 0xFF;

    sendPacket(payload, 3);
}

uint16_t vescController::crc16(const uint8_t* buf, int len) {
    uint16_t crc = 0;
    for (int i = 0; i < len; i++) {
        crc ^= (uint16_t)buf[i] << 8;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x8000)
                crc = (crc << 1) ^ 0x1021;
            else
                crc <<= 1;
        }
    }
    return crc;
}