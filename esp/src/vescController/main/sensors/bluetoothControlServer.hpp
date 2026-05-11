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
#include "manager/BleManager.hpp"

class BluetoothControlServer : public UserControllerApi {
public:
    BluetoothControlServer(VescControllerApi &vescController,
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
    ~BluetoothControlServer() {};
    void start(void);
    void stop(void);
    bool isActivated(void);

    void pollControlMessages(void) override;
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

    VescControllerApi &_vescControllerApi;

    void pollIncomingMessages();
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

    bool isManualDriveEnabled() const;
    void notifyCameraStreamState(bool enabled);
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
    BleManager &bleManager_ = BleManager::instance();
    uint32_t bleMessageCursor_ = 0;
};
