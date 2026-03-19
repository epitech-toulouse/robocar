#include "vescController.hpp"
#include "config.h"
#include "driver/uart.h"
#include <cstring>

VescController::VescController() {}

VescController::~VescController() {}

void VescController::sendPacket(const uint8_t* payload, int len) {
    uint8_t frame[256];
    int idx = 0;

    frame[idx++] = START_BYTE;
    frame[idx++] = (uint8_t)len;
    memcpy(&frame[idx], payload, len);
    idx += len;

    uint16_t crc = crc16(payload, len);
    frame[idx++] = (uint8_t)(crc >> 8);
    frame[idx++] = (uint8_t)(crc & 0xFF);
    frame[idx++] = END_BYTE;

    uart_write_bytes(LIDAR_VESC_UART_PORT, (const char*)frame, idx);
}

void VescController::sendInt32Cmd(CommPacketId cmd, int32_t value) {
    uint8_t payload[5];
    payload[0] = cmd;
    payload[1] = (value >> 24) & 0xFF;
    payload[2] = (value >> 16) & 0xFF;
    payload[3] = (value >> 8)  & 0xFF;
    payload[4] = value & 0xFF;
    sendPacket(payload, 5);
}

/// @brief set the speed of the motor in duty cycle from -1.0 to 1.0, with a maximum speed for safety
/// @param duty the duty cycle
void VescController::setDuty(float duty) {
    duty = duty > VESC_MAX_MOTOR_SPEED ? VESC_MAX_MOTOR_SPEED : duty;
    if (duty < -VESC_MAX_MOTOR_SPEED) duty = -VESC_MAX_MOTOR_SPEED;
    int32_t d = (int32_t)(duty * 100000.0f);
    sendInt32Cmd(COMM_SET_DUTY, d);
}

/// @brief set the speed of the motor in amps, no maximum speed insafe
/// @param amps 
void VescController::setCurrent(float amps) {
    int32_t a = (int32_t)(amps * 1000.0f);
    sendInt32Cmd(COMM_SET_CURRENT, a);
}

/// @brief set the steering of the motor in amps
/// @param amps 
void VescController::setBrake(float amps) {
    int32_t a = (int32_t)(amps * 1000.0f);
    sendInt32Cmd(COMM_SET_BRAKE, a);
}

/// @brief set the steering of the motor in position from 0.0 to 1.0 (0.5 is the center)
/// @param position the steering position
void VescController::setSteering(float position) {
    uint16_t pos = (uint16_t)(position * 1000.0f);
    uint8_t payload[3];
    payload[0] = COMM_SET_SERVO_POS;
    payload[1] = (pos >> 8) & 0xFF;
    payload[2] = pos & 0xFF;
    sendPacket(payload, 3);
}

uint16_t VescController::crc16(const uint8_t* buf, int len) {
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
