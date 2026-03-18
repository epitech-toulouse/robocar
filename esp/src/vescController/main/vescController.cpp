#include "vescController.hpp"
#include "driver/i2c_master.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_system.h"
#include "hal/i2c_types.h"
#include "soc/gpio_num.h"
#include <cstdint>
#include <cstring>

VescController::VescController(gpio_num_t scl_pin, gpio_num_t sda_pin)
    : scl_pin(scl_pin), sda_pin(sda_pin) {
    initIIC();
}

VescController::~VescController() {
  // deinit IIC
  ESP_ERROR_CHECK(i2c_del_master_bus(master_bus_handle));
}

void VescController::initIIC() {
  i2c_master_bus_config_t config = {.i2c_port = -1,
                                    .sda_io_num = sda_pin,
                                    .scl_io_num = scl_pin,
                                    .clk_source = I2C_CLK_SRC_DEFAULT,
                                    .glitch_ignore_cnt = 7,
                                    .intr_priority = 0,
                                    .trans_queue_depth = 0,
                                    .flags = {
                                        .enable_internal_pullup = true,
                                        .allow_pd = false,
                                    }};
  ESP_ERROR_CHECK(i2c_new_master_bus(&config, &master_bus_handle));
  ESP_LOGI("i²c", "Master bus init done");
  /*i2c_device_config_t dev_config = {.dev_addr_length = I2C_ADDR_BIT_LEN_7,
                                    .device_address = 0x00, // broadcast
                                    .scl_speed_hz = 100'000,
                                    .scl_wait_us = 0,
                                    .flags = {.disable_ack_check = true}};

  ESP_ERROR_CHECK(i2c_master_bus_add_device(master_bus_handle, &dev_config,
                                            &vesc_device_handle));
                                            ESP_LOGI("i²c", "Device added");*/
  uint16_t good_one = 0;
  for (uint16_t i = 1; i != 0; i++) {
    ESP_LOGI("i²c", "Probing : %hu", i);
    if (i2c_master_probe(master_bus_handle, 0x0000, 1) == ESP_OK) {
      ESP_LOGI("i²c", "Bus probed : %hu", i);
      good_one = i;
    }
  }
  ESP_LOGI("i²c", "GOOD_ONE : %hu", good_one);
  esp_restart();
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

    ESP_LOGI("i²c", "Sending %u bits", idx);
    ESP_ERROR_CHECK_WITHOUT_ABORT(
        i2c_master_transmit(vesc_device_handle, frame, idx, 5));
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
