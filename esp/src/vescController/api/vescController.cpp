#include "vescController.hpp"

VescController::VescController() {}

VescController::~VescController() {}

void VescController::activate() {
    this->active = true;
}

void VescController::deactivate() {
    this->active = false;
    this->set_speed(0.0f);
}

bool VescController::isActive() {
    return this->active;
}

void VescController::set_speed(float speed) {
    if (this->active) {
      speed = speed > VESC_MAX_MOTOR_SPEED ? VESC_MAX_MOTOR_SPEED : speed;
      if (speed < -VESC_MAX_MOTOR_SPEED)
        speed = -VESC_MAX_MOTOR_SPEED;
      int32_t s = (int32_t)(speed * 100000.0f);
      sendInt32Cmd(COMM_SET_DUTY, s);
    } else {
        sendInt32Cmd(COMM_SET_DUTY, 0);
    }
}

void VescController::stop() {
    set_speed(0.0f);
}

void VescController::set_steering(float steering) {
    if (this->active) {
        if (steering < 0.0f)
            steering = 0.0f;
        if (steering > 1.0f)
            steering = 1.0f;
    }
    else {
        steering = 0.5f; // center
    }

    int32_t pos = (int32_t)(steering * 1000.0f);
    uint8_t payload[3];
    payload[0] = COMM_SET_SERVO_POS;
    payload[1] = (pos >> 8) & 0xFF;
    payload[2] = pos & 0xFF;
    sendPacket(payload, 3);
}

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

    rmt_uart_write_bytes(VESC_RMT_UART_PORT, (const uint8_t*)frame, (size_t) idx);
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

// CRC-16-CCITT implementation
// This convert the payload into a 16-bit checksum used to verify the integrity of data being sent
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
