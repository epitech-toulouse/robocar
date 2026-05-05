#include "../api/vesc_controller_api.hpp"
#include <cstdint>

#include "manager/CoupeCircuitManager.hpp"

#ifndef VESC_CONTROLLER_HPP
#define VESC_CONTROLLER_HPP

#define VESC_MAX_MOTOR_SPEED 0.5f

// this is the implementation of vescControllerAPI for physical vesc
class PhysicalVescController : public IVescController {
    public:
        PhysicalVescController();
        ~PhysicalVescController();

        void set_speed(float speed) override;
        void set_steering(float steering) override;
        void stop() override;
        void activate() override;
        void deactivate() override;
        bool isActive() override;

    private:
        static constexpr uint8_t START_BYTE = 0x02;
        static constexpr uint8_t END_BYTE   = 0x03;

        enum CommPacketId : uint8_t {
        COMM_SET_DUTY       = 5,
        COMM_SET_CURRENT    = 6,
        COMM_SET_BRAKE      = 7,
        COMM_SET_SERVO_POS  = 12,
        };
    
        void sendPacket(const uint8_t* payload, int len);
        void sendInt32Cmd(CommPacketId cmd, int32_t value);
        uint16_t crc16(const uint8_t* buf, int len);
        
        bool active = false;

};


#endif
