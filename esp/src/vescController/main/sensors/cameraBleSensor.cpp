#include "cameraBleSensor.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "esp_log.h"
#include "freertos/task.h"

namespace {
const char *TAG = "CameraBleSensor";
} // namespace

CameraBleSensor *CameraBleSensor::instance_ = nullptr;

CameraBleSensor::CameraBleSensor()
{
    this->ensureStarted();
}

bool CameraBleSensor::isActive(void)
{
    return this->ensureStarted();
}

bool CameraBleSensor::getStop(bool &output)
{
    if (!this->ensureStarted()) {
        return false;
    }
    this->pollIncomingMessages();

    std::lock_guard<std::mutex> lock(this->mutex_);
    output = this->stopRequested_;
    return true;
}

bool CameraBleSensor::getHeading(float &output)
{
    if (!this->ensureStarted()) {
        return false;
    }
    this->pollIncomingMessages();

    std::lock_guard<std::mutex> lock(this->mutex_);
    output = this->steerPercent_;
    return true;
}

bool CameraBleSensor::getSpeed(float &output)
{
    (void) output;
    return false;
}

bool CameraBleSensor::getStatus(CameraStatus &output)
{
    if (!this->ensureStarted()) {
        return false;
    }
    this->pollIncomingMessages();

    const TickType_t now = xTaskGetTickCount();

    std::lock_guard<std::mutex> lock(this->mutex_);
    const bool steerFresh = this->lastSteerTick_ != 0 &&
        (now - this->lastSteerTick_) <= kFreshDataTimeoutTicks;
    const bool stopFresh = this->lastStopTick_ != 0 &&
        (now - this->lastStopTick_) <= kFreshDataTimeoutTicks;

    output.has_data = steerFresh || stopFresh || this->stopRequested_;
    output.connected = this->connected_;
    output.stop_detected = this->stopRequested_;
    output.steering_percent = this->steerPercent_;
    output.steering_weight = this->steerWeight_;
    output.stop_weight = this->stopWeight_;
    output.speed_percent = 0.0f;
    return true;
}

void CameraBleSensor::pollIncomingMessages()
{
    const std::vector<BleMessage> messages =
        bleManager_.messagesSince(bleMessageCursor_, BleEndpoint::Camera);

    for (const BleMessage &message : messages) {
        bleMessageCursor_ = std::max(bleMessageCursor_, message.sequence);
        appendIncomingData(reinterpret_cast<const uint8_t *>(message.payload.data()),
                           message.payload.size());
    }

    std::lock_guard<std::mutex> lock(this->mutex_);
    this->connected_ = bleManager_.isConnected();
}

void CameraBleSensor::appendIncomingData(const uint8_t *data, size_t size)
{
    bool sawLineBreak = false;

    for (size_t i = 0; i < size; ++i) {
        const char c = static_cast<char>(data[i]);
        if (c == '\n' || c == '\r') {
            sawLineBreak = true;
            std::string line;
            {
                std::lock_guard<std::mutex> lock(this->mutex_);
                if (this->rxBuffer_.empty()) {
                    continue;
                }
                line.swap(this->rxBuffer_);
            }
            this->handleLine(line);
            continue;
        }

        std::lock_guard<std::mutex> lock(this->mutex_);
        if (this->rxBuffer_.size() >= kMaxLineLength) {
            ESP_LOGW(TAG, "dropping oversized BLE camera frame");
            this->rxBuffer_.clear();
            continue;
        }
        this->rxBuffer_.push_back(c);
    }

    if (!sawLineBreak && size > 0) {
        std::string line;
        {
            std::lock_guard<std::mutex> lock(this->mutex_);
            if (this->rxBuffer_.empty()) {
                return;
            }
            line.swap(this->rxBuffer_);
        }
        this->handleLine(line);
    }
}

void CameraBleSensor::handleLine(const std::string &line)
{
    const TickType_t now = xTaskGetTickCount();

    std::lock_guard<std::mutex> lock(this->mutex_);
    ESP_LOGI(TAG, "raw camera BLE line: %s", line.c_str());
    const bool canLogSteer = (now - this->lastSteerLogTick_) >= pdMS_TO_TICKS(250);
    const bool canLogStop = (now - this->lastStopLogTick_) >= pdMS_TO_TICKS(250);

    if (startsWith(line, "STEER:")) {
        float parsed = 0.0f;
        if (!parseFloatValue(line.substr(std::strlen("STEER:")), parsed)) {
            ESP_LOGW(TAG, "invalid steering frame: %s", line.c_str());
            return;
        }
        this->steerPercent_ = clampf(parsed, -100.0f, 100.0f);
        this->lastSteerTick_ = now;
        if (canLogSteer) {
            ESP_LOGI(TAG, "camera steer=%.1f%% weight=%.2f",
                     static_cast<double>(this->steerPercent_),
                     static_cast<double>(this->steerWeight_));
            this->lastSteerLogTick_ = now;
        }
        return;
    }

    if (startsWith(line, "STEER_WEIGHT:")) {
        float parsed = 0.0f;
        if (!parseFloatValue(line.substr(std::strlen("STEER_WEIGHT:")), parsed)) {
            ESP_LOGW(TAG, "invalid steering weight frame: %s", line.c_str());
            return;
        }
        this->steerWeight_ = parsed < 0.0f ? 0.0f : parsed;
        this->lastSteerTick_ = now;
        if (canLogSteer) {
            ESP_LOGI(TAG, "camera steer=%.1f%% weight=%.2f",
                     static_cast<double>(this->steerPercent_),
                     static_cast<double>(this->steerWeight_));
            this->lastSteerLogTick_ = now;
        }
        return;
    }

    if (line == "STOP") {
        this->stopRequested_ = true;
        if (this->stopWeight_ <= 0.0f) {
            this->stopWeight_ = 1.0f;
        }
        this->lastStopTick_ = now;
        if (canLogStop) {
            ESP_LOGI(TAG, "camera stop requested weight=%.2f",
                     static_cast<double>(this->stopWeight_));
            this->lastStopLogTick_ = now;
        }
        return;
    }

    if (line == "GO") {
        this->stopRequested_ = false;
        this->stopWeight_ = 0.0f;
        this->lastStopTick_ = now;
        if (canLogStop) {
            ESP_LOGI(TAG, "camera go received, releasing stop hold");
            this->lastStopLogTick_ = now;
        }
        return;
    }

    if (startsWith(line, "STOP_WEIGHT:")) {
        float parsed = 0.0f;
        if (!parseFloatValue(line.substr(std::strlen("STOP_WEIGHT:")), parsed)) {
            ESP_LOGW(TAG, "invalid stop weight frame: %s", line.c_str());
            return;
        }
        this->stopRequested_ = parsed > 0.0f;
        this->stopWeight_ = parsed < 0.0f ? 0.0f : parsed;
        this->lastStopTick_ = now;
        if (canLogStop) {
            ESP_LOGI(TAG, "camera stop active=%d weight=%.2f",
                     this->stopRequested_ ? 1 : 0,
                     static_cast<double>(this->stopWeight_));
            this->lastStopLogTick_ = now;
        }
        return;
    }

    ESP_LOGW(TAG, "unknown camera frame: %s", line.c_str());
}

bool CameraBleSensor::ensureStarted()
{
    if (this->started_) {
        return true;
    }
    if (instance_ != nullptr && instance_ != this) {
        ESP_LOGE(TAG, "only one CameraBleSensor instance is supported");
        return false;
    }

    if (!bleManager_.start()) {
        ESP_LOGE(TAG, "BLE manager failed to start");
        return false;
    }

    instance_ = this;
    bleMessageCursor_ = bleManager_.latestSequence();
    this->started_ = true;
    return true;
}

bool CameraBleSensor::startsWith(const std::string &value, const char *prefix)
{
    const size_t prefixLen = std::strlen(prefix);
    return value.size() >= prefixLen && value.compare(0, prefixLen, prefix) == 0;
}

bool CameraBleSensor::parseFloatValue(const std::string &value, float &output)
{
    errno = 0;
    char *end = nullptr;
    const float parsed = std::strtof(value.c_str(), &end);
    if (value.c_str() == end || errno != 0) {
        return false;
    }

    while (*end == ' ' || *end == '\t') {
        ++end;
    }
    if (*end != '\0') {
        return false;
    }

    output = parsed;
    return true;
}

float CameraBleSensor::clampf(float value, float minValue, float maxValue)
{
    return std::max(minValue, std::min(value, maxValue));
}
