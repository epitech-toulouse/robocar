#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

#include "api/camera_api.hpp"
#include "freertos/FreeRTOS.h"
#include "manager/BleManager.hpp"

class CameraBleSensor : public CameraSensorApi {
public:
    CameraBleSensor();
    ~CameraBleSensor() override = default;

    bool isActive(void) override;
    bool getStop(bool &output) override;
    bool getHeading(float &output) override;
    bool getSpeed(float &output) override;
    bool getStatus(CameraStatus &output) override;

private:
    static constexpr TickType_t kFreshDataTimeoutTicks = pdMS_TO_TICKS(1500);
    static constexpr size_t kMaxLineLength = 128;

    static CameraBleSensor *instance_;

    void pollIncomingMessages();
    void appendIncomingData(const uint8_t *data, size_t size);
    void handleLine(const std::string &line);
    bool ensureStarted();

    static bool startsWith(const std::string &value, const char *prefix);
    static bool parseFloatValue(const std::string &value, float &output);
    static float clampf(float value, float minValue, float maxValue);

    std::mutex mutex_;
    BleManager &bleManager_ = BleManager::instance();
    std::string rxBuffer_;
    uint32_t bleMessageCursor_ = 0;
    bool connected_ = false;
    bool started_ = false;
    bool stopRequested_ = false;
    float stopWeight_ = 0.0f;
    float steerPercent_ = 0.0f;
    float steerWeight_ = 0.0f;
    TickType_t lastSteerTick_ = 0;
    TickType_t lastStopTick_ = 0;
    TickType_t lastSteerLogTick_ = 0;
    TickType_t lastStopLogTick_ = 0;
};
