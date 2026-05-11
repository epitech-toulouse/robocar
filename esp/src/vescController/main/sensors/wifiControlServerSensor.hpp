#pragma once

#include <atomic>
#include <cstdint>

#include <cstddef>
#include <string>
#include <vector>

#include "api/gps_sensor_api.hpp"
#include "api/camera_api.hpp"
#include "api/lidar_sensor_api.hpp"
#include "api/user_controller_api.hpp"
#include "api/vesc_controller_api.hpp"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "gps/gpsGoalState.hpp"
#include "manager/AlgorithmSelector.hpp"

class WifiControlServerSensor : public UserControllerApi {
public:
    WifiControlServerSensor(VescControllerApi &vescController,
                            AlgorithmSelector &algorithmSelector,
                            GpsGoalState &gpsGoalState,
                            CameraSensorApi &cameraSensorApi,
                            GpsSensorApi &gpsSensorApi,
                            LidarSensorApi &lidarSensorApi)
        : _vescControllerApi(vescController),
          _algorithmSelector(algorithmSelector),
          _gpsGoalState(gpsGoalState),
          _cameraSensorApi(cameraSensorApi),
          _gpsSensorApi(gpsSensorApi),
          _lidarSensorApi(lidarSensorApi)
    {
        this->start();
    };
    ~WifiControlServerSensor() {};
    void start(void);
    void stop(void);
    bool isActivated(void);

    bool isConnected(void) override;
    driving_mode_t getDrivingMode(void) override;
    float getSpeed(void) override;
    float getSteering(void) override;

private:
    static constexpr float STEER_CENTER = 0.5f;
    static constexpr float STEER_LEFT = 0.0f;
    static constexpr float STEER_RIGHT = 1.0f;
    static constexpr float DUTY_FORWARD = 0.05f;
    static constexpr float DUTY_BACKWARD = -0.05f;
    static constexpr TickType_t MANUAL_TIMEOUT_MS = 2000;
    static constexpr size_t BLE_VALUE_MAX_SIZE = 512;
    static constexpr uint16_t BLE_APPEARANCE_GENERIC_REMOTE = 384;

    VescControllerApi &_vescControllerApi;

    void startBleServer();
    void stopBleServer();
    void restartAdvertising();
    void recomputeOutputFromState();
    void emergencyStop();
    bool applyProtocolChar(char c);
    void parseAndStore(const uint8_t *buf, int len);
    void handleBleCommand(const std::string &command);
    void handleLineCommand(const std::string &command);
    void handleAlgorithmCommand(const char *value);
    void handleGpsGoalCommand(const char *value);
    void handleSteeringCommand(const char *value);
    void setBleResponse(const std::string &payload);
    void notifySubscribers(const std::string &payload);
    void addConnection(uint16_t connHandle);
    void removeConnection(uint16_t connHandle);

    static void bleHostTask(void *arg);
    static int bleGapEvent(struct ble_gap_event *event, void *arg);
    static int bleGattAccess(uint16_t connHandle,
                             uint16_t attrHandle,
                             struct ble_gatt_access_ctxt *ctxt,
                             void *arg);
    static void bleOnSync();
    bool isManualDriveEnabled() const;
    bool isAlgorithmAvailable(SelectableAlgorithm id) const;
    std::string buildAlgorithmsJson(bool includeStatusEnvelope) const;
    std::string buildGpsGoalJson(bool includeStatusEnvelope) const;
    std::string buildStatusJson() const;
    void clearManualDriveState();
    void setSelectedAlgorithmsMask(uint32_t mask);

    std::atomic<float> duty_{0.0f};
    std::atomic<float> steer_{STEER_CENTER};
    std::atomic<int> lastTick_{0};
    std::atomic<bool> forward_{false};
    std::atomic<bool> backward_{false};
    std::atomic<bool> left_{false};
    std::atomic<bool> right_{false};
    std::atomic<bool> emergency_{false};
    std::atomic<bool> active_{false};
    std::atomic<int> connectedClients_{0};
    AlgorithmSelector &_algorithmSelector;
    GpsGoalState &_gpsGoalState;
    CameraSensorApi &_cameraSensorApi;
    GpsSensorApi &_gpsSensorApi;
    LidarSensorApi &_lidarSensorApi;
    uint16_t bleCharacteristicHandle_ = 0;
    std::string bleValue_ = "STATUS:{\"ok\":true,\"service\":\"robocar_ble\"}";
    std::vector<uint16_t> connHandles_;
};
