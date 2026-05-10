#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

#include "api/camera_api.hpp"
#include "freertos/FreeRTOS.h"

struct ble_gap_event;
struct ble_gatt_access_ctxt;

class CameraBleSensor : public CameraSensorApi {
public:
    CameraBleSensor();
    ~CameraBleSensor() override = default;

    bool isActive(void) override;
    bool getStop(bool &output) override;
    bool getHeading(float &output) override;
    bool getSpeed(float &output) override;
    bool getStatus(CameraStatus &output) override;

    static int gapEventHandler(struct ble_gap_event *event, void *arg);
    static int gattAccessHandler(uint16_t connHandle,
                                 uint16_t attrHandle,
                                 struct ble_gatt_access_ctxt *ctxt,
                                 void *arg);

private:
    static constexpr const char *kDeviceName = "robocar-camera";
    static constexpr TickType_t kFreshDataTimeoutTicks = pdMS_TO_TICKS(1500);
    static constexpr size_t kMaxLineLength = 128;

    static CameraBleSensor *instance_;

    static void onReset(int reason);
    static void onSync(void);
    static void hostTask(void *arg);

    static void startAdvertising();

    void appendIncomingData(const uint8_t *data, size_t size);
    void handleLine(const std::string &line);
    bool ensureStarted();

    static bool startsWith(const std::string &value, const char *prefix);
    static bool parseFloatValue(const std::string &value, float &output);
    static float clampf(float value, float minValue, float maxValue);

    std::mutex mutex_;
    std::string rxBuffer_;
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
