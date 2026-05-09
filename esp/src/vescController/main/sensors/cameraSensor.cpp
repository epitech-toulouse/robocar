#include "cameraSensor.hpp"

#include "freertos/task.h"

CameraSensor::CameraSensor()
{
    init();
}

bool CameraSensor::isActive(void)
{
    return initialized;
}

void CameraSensor::update(void)
{
    std::array<uint8_t, kRxBufferSize> rawBuffer{};
    const int bytesRead = uart_read_bytes(
        kCameraUartPort,
        rawBuffer.data(),
        rawBuffer.size(),
        0);

    if (bytesRead < 0) {
        ESP_LOGE(tag, "camera uart read failed");
        return;
    }

    for (int i = 0; i < bytesRead; ++i) {
        const char ch = static_cast<char>(rawBuffer[i]);
        if (ch == '\r') {
            continue;
        }

        if (ch == '\n') {
            handleBufferedLine();
            continue;
        }

        if (lineBuffer.size() < kMaxLineSize - 1) {
            lineBuffer.push_back(ch);
        } else {
            lineBuffer.clear();
            ESP_LOGW(tag, "camera uart line overflow, dropping partial frame");
        }
    }
}

bool CameraSensor::getSteeringCommand(CameraSteeringCommand &output)
{
    const TickType_t now = xTaskGetTickCount();
    if (!isFresh(lastSteerTick, now) || steerWeight <= 0.0f) {
        return false;
    }

    output.steering_percent = steerPercent;
    output.weight = steerWeight;
    return true;
}

bool CameraSensor::getStopCommand(CameraStopCommand &output)
{
    if (!stopRequested || stopWeight <= 0.0f) {
        return false;
    }

    output.weight = stopWeight;
    return true;
}

bool CameraSensor::startsWith(const std::string &line, const char *prefix)
{
    return line.rfind(prefix, 0) == 0;
}

bool CameraSensor::parseFloatValue(const std::string &value, float &parsed)
{
    errno = 0;
    char *end = nullptr;
    const float output = std::strtof(value.c_str(), &end);
    if (end == value.c_str() || errno != 0) {
        return false;
    }
    while (*end == ' ' || *end == '\t') {
        ++end;
    }
    if (*end != '\0') {
        return false;
    }
    parsed = output;
    return true;
}

float CameraSensor::clampf(float value, float lo, float hi)
{
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

bool CameraSensor::isFresh(TickType_t tick, TickType_t now)
{
    return tick != 0 && (now - tick) <= CAMERA_COMMAND_TIMEOUT_TICK;
}

void CameraSensor::init()
{
    uart_driver_delete(kCameraUartPort);

    const uart_config_t config = {
        .baud_rate = kCameraBaudRate,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .rx_flow_ctrl_thresh = 0,
        .source_clk = UART_SCLK_DEFAULT,
        .flags = {}
    };

    ESP_ERROR_CHECK(uart_param_config(kCameraUartPort, &config));
    ESP_ERROR_CHECK(uart_set_pin(
        kCameraUartPort,
        UART_PIN_NO_CHANGE,
        kCameraRxPin,
        UART_PIN_NO_CHANGE,
        UART_PIN_NO_CHANGE));
    ESP_ERROR_CHECK(uart_driver_install(kCameraUartPort, kRxBufferSize * 2, 0, 0, nullptr, 0));
    ESP_ERROR_CHECK(uart_flush_input(kCameraUartPort));

    initialized = true;
    ESP_LOGI(tag, "camera init successful on UART%d RX GPIO%d @ %d baud",
             static_cast<int>(kCameraUartPort),
             static_cast<int>(kCameraRxPin),
             kCameraBaudRate);
}

void CameraSensor::handleBufferedLine()
{
    if (lineBuffer.empty()) {
        return;
    }

    handleLine(lineBuffer);
    lineBuffer.clear();
}

void CameraSensor::handleLine(const std::string &line)
{
    const TickType_t now = xTaskGetTickCount();
    const bool canLogSteer = (now - lastSteerLogTick) >= pdMS_TO_TICKS(250);
    const bool canLogStop = (now - lastStopLogTick) >= pdMS_TO_TICKS(250);

    if (startsWith(line, "STEER:")) {
        float parsed = 0.0f;
        if (!parseFloatValue(line.substr(std::strlen("STEER:")), parsed)) {
            ESP_LOGW(tag, "invalid steering frame: %s", line.c_str());
            return;
        }
        steerPercent = clampf(parsed, -100.0f, 100.0f);
        lastSteerTick = now;
        if (canLogSteer) {
            ESP_LOGI(tag, "camera steer=%.1f%% weight=%.2f", static_cast<double>(steerPercent), static_cast<double>(steerWeight));
            lastSteerLogTick = now;
        }
        return;
    }

    if (startsWith(line, "STEER_WEIGHT:")) {
        float parsed = 0.0f;
        if (!parseFloatValue(line.substr(std::strlen("STEER_WEIGHT:")), parsed)) {
            ESP_LOGW(tag, "invalid steering weight frame: %s", line.c_str());
            return;
        }
        steerWeight = parsed < 0.0f ? 0.0f : parsed;
        lastSteerTick = now;
        if (canLogSteer) {
            ESP_LOGI(tag, "camera steer=%.1f%% weight=%.2f", static_cast<double>(steerPercent), static_cast<double>(steerWeight));
            lastSteerLogTick = now;
        }
        return;
    }

    if (line == "STOP") {
        stopRequested = true;
        if (stopWeight <= 0.0f) {
            stopWeight = 1.0f;
        }
        lastStopTick = now;
        if (canLogStop) {
            ESP_LOGI(tag, "camera stop requested weight=%.2f", static_cast<double>(stopWeight));
            lastStopLogTick = now;
        }
        return;
    }

    if (line == "GO") {
        stopRequested = false;
        stopWeight = 0.0f;
        lastStopTick = now;
        if (canLogStop) {
            ESP_LOGI(tag, "camera go received, releasing stop hold");
            lastStopLogTick = now;
        }
        return;
    }

    if (startsWith(line, "STOP_WEIGHT:")) {
        float parsed = 0.0f;
        if (!parseFloatValue(line.substr(std::strlen("STOP_WEIGHT:")), parsed)) {
            ESP_LOGW(tag, "invalid stop weight frame: %s", line.c_str());
            return;
        }
        stopRequested = parsed > 0.0f;
        stopWeight = parsed < 0.0f ? 0.0f : parsed;
        lastStopTick = now;
        if (canLogStop) {
            ESP_LOGI(tag, "camera stop active=%d weight=%.2f",
                     stopRequested ? 1 : 0,
                     static_cast<double>(stopWeight));
            lastStopLogTick = now;
        }
        return;
    }

    ESP_LOGW(tag, "unknown camera frame: %s", line.c_str());
}
