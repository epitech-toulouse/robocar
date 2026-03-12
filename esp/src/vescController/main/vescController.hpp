#pragma once

#include <cstdint>

#define VESC_MAX_MOTOR_SPEED 0.5f

class VescController {
public:
    VescController(int txPin, int rxPin);
    ~VescController();

    /// Set motor duty cycle, range: -1.0 (full reverse) to 1.0 (full forward)
    void setDuty(float duty);

    /// Set motor current in amps
    void setCurrent(float amps);

    /// Set brake current in amps
    void setBrake(float amps);

    /// Set steering servo position, range: 0.0 (left) to 1.0 (right), 0.5 = center
    void setSteering(float position);

private:
    static constexpr uint8_t START_BYTE = 0x02;
    static constexpr uint8_t END_BYTE   = 0x03;

    enum CommPacketId : uint8_t {
        COMM_SET_DUTY       = 5,
        COMM_SET_CURRENT    = 6,
        COMM_SET_BRAKE      = 7,
        COMM_SET_SERVO_POS  = 12,
    };

    int txPin;
    int rxPin;

    void initUart();
    void sendPacket(const uint8_t* payload, int len);
    void sendInt32Cmd(CommPacketId cmd, int32_t value);
    static uint16_t crc16(const uint8_t* buf, int len);
};