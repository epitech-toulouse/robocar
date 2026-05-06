#ifndef CAMERA_SENSOR_HPP
#define CAMERA_SENSOR_HPP

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>

#include "api/camera_sensor_api.hpp"
#include "config.h"
#include "driver/uart.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "soc/gpio_num.h"

class CameraSensor : public CameraSensorApi
{
public:
    CameraSensor();
    ~CameraSensor() override = default;

    bool isActive(void) override;
    void update(void) override;
    bool getSteeringCommand(CameraSteeringCommand &output) override;
    bool getStopCommand(CameraStopCommand &output) override;

private:
    static constexpr uart_port_t kCameraUartPort = CAMERA_UART_PORT;
    static constexpr gpio_num_t kCameraRxPin = CAMERA_UART_RX;
    static constexpr int kCameraBaudRate = 115200;
    static constexpr int kRxBufferSize = 256;
    static constexpr size_t kMaxLineSize = 128;
    static constexpr const char *tag = "CameraSensor";

    static bool startsWith(const std::string &line, const char *prefix);
    static bool parseFloatValue(const std::string &value, float &parsed);
    static float clampf(float value, float lo, float hi);
    static bool isFresh(TickType_t tick, TickType_t now);

    void init();
    void handleBufferedLine();
    void handleLine(const std::string &line);

    bool initialized = false;
    std::string lineBuffer{};

    float steerPercent = 0.0f;
    float steerWeight = 0.0f;
    TickType_t lastSteerTick = 0;
    TickType_t lastSteerLogTick = 0;

    bool stopRequested = false;
    float stopWeight = 0.0f;
    TickType_t lastStopTick = 0;
    TickType_t lastStopLogTick = 0;
};

#endif /* CAMERA_SENSOR_HPP */
