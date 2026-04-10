#ifndef VESC_CONTROLLER_HPP
#define VESC_CONTROLLER_HPP

#include "../main/api/vesc_controller_api.hpp"
#include <cstdint>

#define VESC_MAX_MOTOR_SPEED 0.5f

// this is the implementation of IVescController for physical vesc
class VescController : public IVescController {
public:
        VescController();
        ~VescController() override;

        void set_speed(float speed) override;
        void set_steering(float steering) override;
        void stop() override;
        void activate() override;
        void deactivate() override;
        bool isActive() override;

private:
        enum CommPacketId : uint8_t {
            COMM_SET_DUTY = 5,
            COMM_SET_SERVO_POS = 12,
        };

        static constexpr uint8_t START_BYTE = 0x02;
        static constexpr uint8_t END_BYTE = 0x03;

        void sendPacket(const uint8_t* payload, int len);
        void sendInt32Cmd(CommPacketId cmd, int32_t value);
        static uint16_t crc16(const uint8_t* buf, int len);
        
        bool active = false;

    };

#endif