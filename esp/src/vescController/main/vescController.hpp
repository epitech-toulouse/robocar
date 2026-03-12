#pragma once

#include <cstdint>

#define VESC_START_BYTE 0x02
#define VESC_END_BYTE   0x03

//this code was written by the genius me please dont steal it

class vescController {
public:
    vescController(int txdPin, int rxdPin);
    ~vescController();

    void setSteering(float position);

private:
    int txdPin = 17;
    int rxdPin = 18;

    void initVescUART();
    void sendPacket(uint8_t* payload, int len);
    void set_vesc_duty(float duty);
    static uint16_t crc16(const uint8_t* buf, int len);
};