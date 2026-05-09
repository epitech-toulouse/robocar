#pragma once

#include <atomic>
#include <cstdint>

#include <cstddef>
#include <string>

#include "api/gps_sensor_api.hpp"
#include "api/lidar_sensor_api.hpp"
#include "api/user_controller_api.hpp"
#include "api/vesc_controller_api.hpp"
#include "esp_err.h"
#include "esp_http_server.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "manager/AlgorithmSelector.hpp"




class WifiControlServerSensor : public UserControllerApi {
public:
    WifiControlServerSensor(VescControllerApi &vescController,
                            AlgorithmSelector &algorithmSelector,
                            GpsSensorApi &gpsSensorApi,
                            LidarSensorApi &lidarSensorApi)
        : _vescControllerApi(vescController),
          _algorithmSelector(algorithmSelector),
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
    static constexpr const char *WIFI_AP_SSID = "ROBOCAR_CTRL";
    static constexpr const char *WIFI_AP_PASSWORD = "YohannBoniface";
    static constexpr uint8_t WIFI_AP_CHANNEL = 1;
    static constexpr uint8_t WIFI_AP_MAX_CONN = 1;
    static constexpr int CONTROL_HTTP_PORT = 3333;
    static constexpr int CONTROL_TCP_PORT = 3334;
    static constexpr int RX_BUFFER_SIZE = 64;

    VescControllerApi &_vescControllerApi;

    void initWifiSoftAp();
    void startHttpServer();
    void runTcpServerTask();
    void recomputeOutputFromState();
    void emergencyStop();
    bool applyProtocolChar(char c);
    void parseAndStore(const uint8_t *buf, int len);

    static void tcpServerTask(void *arg);
    static void setHttpCommonHeaders(httpd_req_t *req);
    static WifiControlServerSensor *fromRequest(httpd_req_t *req);
    static esp_err_t httpCmdHandler(httpd_req_t *req);
    static esp_err_t httpCmdOptionsHandler(httpd_req_t *req);
    static esp_err_t httpLogsHandler(httpd_req_t *req);
    static esp_err_t httpLogsOptionsHandler(httpd_req_t *req);
    static esp_err_t httpAlgorithmsHandler(httpd_req_t *req);
    static esp_err_t httpAlgorithmsOptionsHandler(httpd_req_t *req);
    static esp_err_t httpStatusHandler(httpd_req_t *req);
    static esp_err_t httpStatusOptionsHandler(httpd_req_t *req);
    static esp_err_t httpRootHandler(httpd_req_t *req);
    bool isManualDriveEnabled() const;
    bool isAlgorithmAvailable(SelectableAlgorithm id) const;
    std::string buildAlgorithmsJson(bool includeStatusEnvelope) const;
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
    AlgorithmSelector &_algorithmSelector;
    GpsSensorApi &_gpsSensorApi;
    LidarSensorApi &_lidarSensorApi;
    httpd_handle_t httpServer_ = nullptr;
    TaskHandle_t tcpTaskHandle_ = nullptr;

  

};
