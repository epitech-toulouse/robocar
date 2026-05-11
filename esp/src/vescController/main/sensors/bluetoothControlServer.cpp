#include "bluetoothControlServer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "esp_log.h"

static constexpr size_t WEB_LOG_MAX_ENTRIES = 128;
static constexpr size_t WEB_LOG_LINE_MAX = 192;
struct WebLogEntry {
    uint32_t seq;
    char line[WEB_LOG_LINE_MAX];
};

static const char TAG[] = "BleControlSensor";

static WebLogEntry s_webLogEntries[WEB_LOG_MAX_ENTRIES];
static std::atomic<uint32_t> s_webLogSeq;
static std::atomic<uint32_t> s_webLogCount;
static portMUX_TYPE s_webLogMux = portMUX_INITIALIZER_UNLOCKED;
using LogVprintfFn = int (*)(const char *, va_list);
static LogVprintfFn s_prevLogVprintf;
static std::atomic<bool> s_logRedirectInstalled;

enum class AlgorithmSelectionParseError : uint8_t {
    None = 0,
    Unknown,
    NotImplemented,
};

static void store_web_log_line(const char *text)
{
    if (text == nullptr || text[0] == '\0') {
        return;
    }

    const uint32_t seq = s_webLogSeq.fetch_add(1) + 1;
    const size_t index = static_cast<size_t>((seq - 1) % WEB_LOG_MAX_ENTRIES);

    portENTER_CRITICAL(&s_webLogMux);
    s_webLogEntries[index].seq = seq;
    std::strncpy(s_webLogEntries[index].line, text, WEB_LOG_LINE_MAX - 1);
    s_webLogEntries[index].line[WEB_LOG_LINE_MAX - 1] = '\0';
    portEXIT_CRITICAL(&s_webLogMux);

    uint32_t count = s_webLogCount.load();
    while (count < WEB_LOG_MAX_ENTRIES &&
           !s_webLogCount.compare_exchange_weak(count, count + 1)) {
    }
}

static int web_log_vprintf(const char *fmt, va_list args)
{
    char line[WEB_LOG_LINE_MAX];
    va_list copy;
    va_copy(copy, args);
    const int rc = std::vsnprintf(line, sizeof(line), fmt, copy);
    va_end(copy);

    if (rc > 0) {
        size_t len = std::strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[len - 1] = '\0';
            --len;
        }
        store_web_log_line(line);
    }

    if (s_prevLogVprintf != nullptr) {
        return s_prevLogVprintf(fmt, args);
    }
    return std::vprintf(fmt, args);
}

static void ensure_log_redirect_installed()
{
    bool expected = false;
    if (s_logRedirectInstalled.compare_exchange_strong(expected, true)) {
        s_prevLogVprintf = esp_log_set_vprintf(web_log_vprintf);
    }
}

static std::string get_logs_since(uint32_t since)
{
    const uint32_t latest = s_webLogSeq.load();
    const uint32_t count = s_webLogCount.load();
    if (latest == 0 || count == 0) {
        return std::string();
    }

    const uint32_t oldest = (latest > count) ? (latest - count + 1) : 1;
    uint32_t start = since + 1;
    if (start < oldest) {
        start = oldest;
    }
    if (start > latest) {
        return std::string();
    }

    std::string out;
    out.reserve((latest - start + 1) * 48);

    for (uint32_t seq = start; seq <= latest; ++seq) {
        const size_t index = static_cast<size_t>((seq - 1) % WEB_LOG_MAX_ENTRIES);
        WebLogEntry entry = {};
        portENTER_CRITICAL(&s_webLogMux);
        entry = s_webLogEntries[index];
        portEXIT_CRITICAL(&s_webLogMux);

        if (entry.seq != seq) {
            continue;
        }

        out += std::to_string(seq);
        out += "|";
        out += entry.line;
        out += "\n";
    }

    return out;
}

static bool parse_strict_double(const char *value, double &out)
{
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    char *end = nullptr;
    out = std::strtod(value, &end);
    return end != value && end != nullptr && *end == '\0' && std::isfinite(out);
}

static bool parse_strict_float(const char *value, float &out)
{
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    char *end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value || end == nullptr || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    out = parsed;
    return true;
}

static void trim_in_place(std::string &value)
{
    while (!value.empty() && (value.back() == '\r' || value.back() == '\n' ||
                              value.back() == ' ' || value.back() == '\t')) {
        value.pop_back();
    }
    size_t start = 0;
    while (start < value.size() &&
           (value[start] == ' ' || value[start] == '\t' ||
            value[start] == '\r' || value[start] == '\n')) {
        ++start;
    }
    if (start > 0) {
        value.erase(0, start);
    }
}

static bool parse_algorithm_selection_value(const char *value,
                                            uint32_t &mask,
                                            AlgorithmSelectionParseError &error,
                                            char *invalidToken,
                                            size_t invalidTokenSize)
{
    if (value == nullptr) {
        return false;
    }
    if (invalidToken != nullptr && invalidTokenSize > 0) {
        invalidToken[0] = '\0';
    }
    mask = 0;
    error = AlgorithmSelectionParseError::None;

    if (value[0] == '\0') {
        return true;
    }

    const char *cursor = value;
    while (*cursor != '\0') {
        while (*cursor == ',' || *cursor == ' ' || *cursor == '\t') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }

        char token[32] = {0};
        size_t tokenLen = 0;
        while (*cursor != '\0' && *cursor != ',') {
            if (tokenLen + 1 < sizeof(token)) {
                token[tokenLen++] = *cursor;
            }
            ++cursor;
        }
        while (tokenLen > 0 &&
               (token[tokenLen - 1] == ' ' || token[tokenLen - 1] == '\t')) {
            token[--tokenLen] = '\0';
        }
        if (tokenLen == 0) {
            continue;
        }

        const AlgorithmDescriptor *descriptor = findAlgorithmDescriptorByKey(token);
        if (descriptor == nullptr && std::strcmp(token, "close_obs") == 0) {
            descriptor = algorithmDescriptor(SelectableAlgorithm::CloseObstacle);
        } else if (descriptor == nullptr && std::strcmp(token, "lidar_cor") == 0) {
            descriptor = algorithmDescriptor(SelectableAlgorithm::LidarCorridor);
        }
        if (descriptor == nullptr) {
            error = AlgorithmSelectionParseError::Unknown;
        } else if (!descriptor->implemented) {
            error = AlgorithmSelectionParseError::NotImplemented;
        } else {
            mask |= algorithmBit(descriptor->id);
            continue;
        }
        if (invalidToken != nullptr && invalidTokenSize > 0) {
            std::strncpy(invalidToken, token, invalidTokenSize - 1);
            invalidToken[invalidTokenSize - 1] = '\0';
        }
        return false;
    }
    return true;
}

static void append_json_bool(std::string &payload, bool value)
{
    payload += value ? "true" : "false";
}

static void append_selected_algorithms_json(std::string &payload, uint32_t selectedMask)
{
    payload += "[";
    bool first = true;
    for (const AlgorithmDescriptor &descriptor : kSelectableAlgorithms) {
        if (!descriptor.implemented) {
            continue;
        }
        if ((selectedMask & algorithmBit(descriptor.id)) == 0) {
            continue;
        }
        if (!first) {
            payload += ",";
        }
        first = false;
        payload += "\"";
        payload += descriptor.key;
        payload += "\"";
    }
    payload += "]";
}

static void append_algorithm_object_json(std::string &payload,
                                         const AlgorithmDescriptor &descriptor,
                                         bool enabled,
                                         bool available)
{
    payload += "{";
    payload += "\"id\":\"";
    payload += descriptor.key;
    payload += "\",\"label\":\"";
    payload += descriptor.label;
    payload += "\",\"enabled\":";
    append_json_bool(payload, enabled);
    payload += ",\"available\":";
    append_json_bool(payload, available);
    payload += ",\"implemented\":";
    append_json_bool(payload, descriptor.implemented);
    payload += ",\"weight\":";
    payload += std::to_string(static_cast<double>(descriptor.weight));
    payload += "}";
}

static void append_gps_goal_json(std::string &payload, const GpsGoalSnapshot &goal)
{
    payload += "{";
    payload += "\"lat\":";
    payload += std::to_string(goal.lat);
    payload += ",\"lon\":";
    payload += std::to_string(goal.lon);
    payload += ",\"enabled\":";
    append_json_bool(payload, goal.enabled);
    payload += "}";
}

void BluetoothControlServer::setBleResponse(const std::string &payload)
{
    bleManager_.setControlResponse(payload);
}

bool BluetoothControlServer::isManualDriveEnabled() const
{
    return this->_algorithmSelector.isManualDriveEnabled();
}

void BluetoothControlServer::clearManualDriveState()
{
    forward_.store(false);
    backward_.store(false);
    left_.store(false);
    right_.store(false);
    duty_.store(0.0f);
    steer_.store(STEER_CENTER);
    lastTick_.store(0);
}

void BluetoothControlServer::setSelectedAlgorithmsMask(uint32_t mask)
{
    const bool manualWasEnabled = this->isManualDriveEnabled();

    this->_algorithmSelector.setSelectedMask(mask);
    const uint32_t appliedMask = this->_algorithmSelector.getSelectedMask();
    ESP_LOGI(TAG,
             "Algorithm mask update: requested=0x%02lx applied=0x%02lx manual=%s",
             static_cast<unsigned long>(mask),
             static_cast<unsigned long>(appliedMask),
             this->isManualDriveEnabled() ? "enabled" : "disabled");
    if (manualWasEnabled && !this->isManualDriveEnabled()) {
        ESP_LOGI(TAG, "Manual disabled by algorithm mask; clearing manual drive state");
        this->clearManualDriveState();
    }
}

bool BluetoothControlServer::isAlgorithmAvailable(SelectableAlgorithm id) const
{
    switch (id) {
        case SelectableAlgorithm::Manual:
            return this->active_.load() && this->connectedClients_.load() > 0;
        case SelectableAlgorithm::CloseObstacle:
        case SelectableAlgorithm::LidarCorridor:
            return this->_lidarSensorApi.isActive();
        case SelectableAlgorithm::Gps: {
            GpsStatus status{};
            return this->_gpsSensorApi.isActive() &&
                this->_gpsSensorApi.getStatus(status) &&
                status.has_fix;
        }
        case SelectableAlgorithm::Camera: {
            CameraStatus status{};
            return this->_cameraSensorApi.isActive() &&
                this->_cameraSensorApi.getStatus(status) &&
                status.connected &&
                status.has_data;
        }
        case SelectableAlgorithm::Count:
            break;
    }
    return false;
}

std::string BluetoothControlServer::buildAlgorithmsJson(bool includeStatusEnvelope) const
{
    std::string payload;
    payload.reserve(768);
    const uint32_t selectedMask = this->_algorithmSelector.getSelectedMask();

    payload += "{\"ok\":true,";
    if (includeStatusEnvelope) {
        payload += "\"manualDriveEnabled\":";
        append_json_bool(payload, this->isManualDriveEnabled());
        payload += ",";
    }
    payload += "\"selectedAlgorithms\":";
    append_selected_algorithms_json(payload, selectedMask);
    payload += ",\"algorithms\":[";
    bool first = true;
    for (const AlgorithmDescriptor &descriptor : kSelectableAlgorithms) {
        if (!first) {
            payload += ",";
        }
        first = false;
        append_algorithm_object_json(payload,
                                     descriptor,
                                     (selectedMask & algorithmBit(descriptor.id)) != 0,
                                     this->isAlgorithmAvailable(descriptor.id));
    }
    payload += "]}";
    return payload;
}

std::string BluetoothControlServer::buildStatusJson() const
{
    std::string payload;
    payload.reserve(1024);

    payload += "{\"ok\":true,\"service\":\"robocar_ble\",\"serviceActive\":";
    append_json_bool(payload, this->active_.load());
    payload += ",\"connectedClients\":";
    payload += std::to_string(this->connectedClients_.load());
    payload += ",\"vescActive\":";
    append_json_bool(payload, this->_vescControllerApi.isActive());
    payload += ",\"emergency\":";
    append_json_bool(payload, this->emergency_.load());
    payload += ",\"manualDriveEnabled\":";
    append_json_bool(payload, this->isManualDriveEnabled());
    payload += ",\"selectedAlgorithms\":";
    append_selected_algorithms_json(payload, this->_algorithmSelector.getSelectedMask());
    payload += ",\"gpsGoal\":";
    append_gps_goal_json(payload, this->_gpsGoalState.get());
    payload += ",\"algorithms\":[";
    bool first = true;
    for (const AlgorithmDescriptor &descriptor : kSelectableAlgorithms) {
        if (!first) {
            payload += ",";
        }
        first = false;
        append_algorithm_object_json(payload,
                                     descriptor,
                                     this->_algorithmSelector.isEnabled(descriptor.id),
                                     this->isAlgorithmAvailable(descriptor.id));
    }
    payload += "]}";
    return payload;
}

std::string BluetoothControlServer::buildGpsGoalJson(bool includeStatusEnvelope) const
{
    std::string payload;
    payload.reserve(128);

    payload += "{\"ok\":true";
    if (includeStatusEnvelope) {
        payload += ",\"serviceActive\":";
        append_json_bool(payload, this->active_.load());
    }
    payload += ",\"gpsGoal\":";
    append_gps_goal_json(payload, this->_gpsGoalState.get());
    payload += "}";
    return payload;
}

void BluetoothControlServer::recomputeOutputFromState()
{
    float duty = 0.0f;
    if (forward_.load() && !backward_.load()) {
        duty = DUTY_FORWARD;
    } else if (backward_.load() && !forward_.load()) {
        duty = DUTY_BACKWARD;
    }

    float steer = STEER_CENTER;
    if (left_.load() && !right_.load()) {
        steer = STEER_LEFT;
    } else if (right_.load() && !left_.load()) {
        steer = STEER_RIGHT;
    }

    duty_.store(duty);
    steer_.store(steer);
    lastTick_.store(static_cast<int>(xTaskGetTickCount()));
}

void BluetoothControlServer::emergencyStop()
{
    clearManualDriveState();
    emergency_.store(true);
    lastTick_.store(static_cast<int>(xTaskGetTickCount()));
    _vescControllerApi.stop();
    _vescControllerApi.deactivate();
}

bool BluetoothControlServer::applyProtocolChar(char c)
{
    switch (c) {
        case 'F':
            if (!isManualDriveEnabled()) {
                ESP_LOGI(TAG, "Manual command '%c' ignored: manual algorithm disabled", c);
                return true;
            }
            ESP_LOGI(TAG, "Manual command '%c' applied: forward on", c);
            forward_.store(true);
            recomputeOutputFromState();
            return true;
        case 'f':
            if (!isManualDriveEnabled()) {
                ESP_LOGI(TAG, "Manual command '%c' ignored: manual algorithm disabled", c);
                return true;
            }
            ESP_LOGI(TAG, "Manual command '%c' applied: forward off", c);
            forward_.store(false);
            recomputeOutputFromState();
            return true;
        case 'B':
            if (!isManualDriveEnabled()) {
                ESP_LOGI(TAG, "Manual command '%c' ignored: manual algorithm disabled", c);
                return true;
            }
            ESP_LOGI(TAG, "Manual command '%c' applied: backward on", c);
            backward_.store(true);
            recomputeOutputFromState();
            return true;
        case 'b':
            if (!isManualDriveEnabled()) {
                ESP_LOGI(TAG, "Manual command '%c' ignored: manual algorithm disabled", c);
                return true;
            }
            ESP_LOGI(TAG, "Manual command '%c' applied: backward off", c);
            backward_.store(false);
            recomputeOutputFromState();
            return true;
        case 'L':
            if (!isManualDriveEnabled()) {
                ESP_LOGI(TAG, "Manual command '%c' ignored: manual algorithm disabled", c);
                return true;
            }
            ESP_LOGI(TAG, "Manual command '%c' applied: left on", c);
            left_.store(true);
            recomputeOutputFromState();
            return true;
        case 'l':
            if (!isManualDriveEnabled()) {
                ESP_LOGI(TAG, "Manual command '%c' ignored: manual algorithm disabled", c);
                return true;
            }
            ESP_LOGI(TAG, "Manual command '%c' applied: left off", c);
            left_.store(false);
            recomputeOutputFromState();
            return true;
        case 'R':
            if (!isManualDriveEnabled()) {
                ESP_LOGI(TAG, "Manual command '%c' ignored: manual algorithm disabled", c);
                return true;
            }
            ESP_LOGI(TAG, "Manual command '%c' applied: right on", c);
            right_.store(true);
            recomputeOutputFromState();
            return true;
        case 'r':
            if (!isManualDriveEnabled()) {
                ESP_LOGI(TAG, "Manual command '%c' ignored: manual algorithm disabled", c);
                return true;
            }
            ESP_LOGI(TAG, "Manual command '%c' applied: right off", c);
            right_.store(false);
            recomputeOutputFromState();
            return true;
        case 'S':
            emergencyStop();
            setBleResponse("STATUS:" + buildStatusJson());
            return true;
        case 'A':
            emergency_.store(false);
            _vescControllerApi.activate();
            lastTick_.store(static_cast<int>(xTaskGetTickCount()));
            setBleResponse("STATUS:" + buildStatusJson());            return true;
        default:
            return false;
    }
}

void BluetoothControlServer::parseAndStore(const uint8_t *buf, int len)
{
    if (len <= 0 || buf == nullptr) {
        return;
    }

    for (int i = 0; i < len; ++i) {
        const uint8_t b = buf[i];
        if (b == 0x00 || b == '\n' || b == '\r' || b == ' ' || b == '\t') {
            continue;
        }
        if (!applyProtocolChar(static_cast<char>(b))) {
            ESP_LOGW(TAG, "Unknown protocol char: '%c' (0x%02X)", static_cast<char>(b), b);
            setBleResponse("ERR:unknown_protocol_char");
        }
    }
}

void BluetoothControlServer::handleAlgorithmCommand(const char *value)
{
    uint32_t selectedMask = 0;
    AlgorithmSelectionParseError parseError = AlgorithmSelectionParseError::None;
    char invalidToken[32] = {0};

    if (!parse_algorithm_selection_value(value,
                                         selectedMask,
                                         parseError,
                                         invalidToken,
                                         sizeof(invalidToken))) {
        std::string payload = "ERR:";
        payload += (parseError == AlgorithmSelectionParseError::NotImplemented)
            ? "algorithm_not_implemented:"
            : "unknown_algorithm:";
        payload += invalidToken;
        setBleResponse(payload);
        return;
    }

    ESP_LOGI(TAG,
             "Algorithm command received: value=\"%s\" parsed_mask=0x%02lx",
             value != nullptr ? value : "",
             static_cast<unsigned long>(selectedMask));
    setSelectedAlgorithmsMask(selectedMask);
    setBleResponse("ALGORITHMS:" + buildAlgorithmsJson(true));
}

void BluetoothControlServer::handleGpsGoalCommand(const char *value)
{
    if (value == nullptr) {
        setBleResponse("ERR:missing_gps_goal");
        return;
    }

    const char *sep = std::strchr(value, ',');
    if (sep == nullptr) {
        setBleResponse("ERR:missing_lat_or_lon");
        return;
    }

    std::string latText(value, sep - value);
    std::string lonText(sep + 1);
    trim_in_place(latText);
    trim_in_place(lonText);

    double lat = 0.0;
    double lon = 0.0;
    if (!parse_strict_double(latText.c_str(), lat)) {
        setBleResponse("ERR:invalid_latitude_format");
        return;
    }
    if (!parse_strict_double(lonText.c_str(), lon)) {
        setBleResponse("ERR:invalid_longitude_format");
        return;
    }
    if (!GpsGoalState::isValidLatitude(lat)) {
        setBleResponse("ERR:latitude_out_of_range");
        return;
    }
    if (!GpsGoalState::isValidLongitude(lon)) {
        setBleResponse("ERR:longitude_out_of_range");
        return;
    }
    if (!_gpsGoalState.set(lat, lon, true)) {
        setBleResponse("ERR:invalid_gps_goal");
        return;
    }
    setBleResponse("GPS:" + buildGpsGoalJson(true));
}

void BluetoothControlServer::handleSteeringCommand(const char *value)
{
    float steeringPercent = 0.0f;
    if (!parse_strict_float(value, steeringPercent)) {
        setBleResponse("ERR:invalid_steering");
        return;
    }

    const float clamped = std::max(-100.0f, std::min(100.0f, steeringPercent));
    steer_.store((clamped + 100.0f) / 200.0f);
    forward_.store(true);
    backward_.store(false);
    left_.store(false);
    right_.store(false);
    duty_.store(DUTY_FORWARD);
    lastTick_.store(static_cast<int>(xTaskGetTickCount()));
}

void BluetoothControlServer::handleLineCommand(const std::string &command)
{
    if (command.empty()) {
        return;
    }

    if (command == "STATUS?") {
        setBleResponse("STATUS:" + buildStatusJson());
        return;
    }
    if (command.rfind("LOGS:", 0) == 0) {
        const uint32_t since = static_cast<uint32_t>(std::strtoul(command.c_str() + 5, nullptr, 10));
        setBleResponse("LOGS:" + get_logs_since(since));
        return;
    }
    if (command.rfind("ALG:", 0) == 0) {
        handleAlgorithmCommand(command.c_str() + 4);
        return;
    }
    if (command.rfind("GPS:", 0) == 0) {
        handleGpsGoalCommand(command.c_str() + 4);
        return;
    }
    if (command.rfind("STEER:", 0) == 0) {
        handleSteeringCommand(command.c_str() + 6);
        return;
    }
    if (command.rfind("CV:", 0) == 0) {
        lastTick_.store(static_cast<int>(xTaskGetTickCount()));
        return;
    }
    if (command == "STOP") {
        emergencyStop();
        setBleResponse("STATUS:" + buildStatusJson());
        return;
    }
    if (command == "GO") {
        emergency_.store(false);
        _vescControllerApi.activate();
        lastTick_.store(static_cast<int>(xTaskGetTickCount()));
        setBleResponse("STATUS:" + buildStatusJson());
        return;
    }
    if (command.size() == 1 && applyProtocolChar(command[0])) {
        return;
    }

    setBleResponse("ERR:unknown_command");
}

void BluetoothControlServer::handleBleCommand(const std::string &payload)
{
    if (payload.empty()) {
        return;
    }

    bool hasSeparator = false;
    for (char c : payload) {
        if (c == ':' || c == '?' || c == '\n' || c == '\r') {
            hasSeparator = true;
            break;
        }
    }

    if (!hasSeparator) {
        parseAndStore(reinterpret_cast<const uint8_t *>(payload.data()),
                      static_cast<int>(payload.size()));
        return;
    }

    size_t start = 0;
    while (start < payload.size()) {
        size_t end = payload.find_first_of("\r\n", start);
        std::string command = payload.substr(start, end == std::string::npos
            ? std::string::npos
            : end - start);
        trim_in_place(command);
        handleLineCommand(command);
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
}

void BluetoothControlServer::pollIncomingMessages()
{
    const int connectedClients = bleManager_.connectedClientCount();
    if (connectedClients == 0 && connectedClients_.load() > 0) {
        clearManualDriveState();
    }
    connectedClients_.store(connectedClients);

    const std::vector<BleMessage> messages =
        bleManager_.messagesSince(bleMessageCursor_, BleEndpoint::Control);

    for (const BleMessage &message : messages) {
        bleMessageCursor_ = std::max(bleMessageCursor_, message.sequence);
        handleBleCommand(message.payload);
    }
}

void BluetoothControlServer::start(void)
{
    if (active_.load()) {
        return;
    }

    ensure_log_redirect_installed();
    active_.store(true);
    setBleResponse("STATUS:" + buildStatusJson());
    if (!bleManager_.start()) {
        active_.store(false);
        ESP_LOGE(TAG, "BLE manager failed to start");
        return;
    }
    bleMessageCursor_ = bleManager_.latestSequence();
    ESP_LOGI(TAG, "BLE control service initialized; Wi-Fi/HTTP control disabled");
}

void BluetoothControlServer::stop(void)
{
    if (!active_.load()) {
        return;
    }
    clearManualDriveState();
    active_.store(false);
    ESP_LOGI(TAG, "BLE control service stopped");
}

bool BluetoothControlServer::isActivated(void)
{
    return active_.load();
}

void BluetoothControlServer::pollControlMessages(void)
{
    pollIncomingMessages();
}

bool BluetoothControlServer::isConnected(void)
{
    pollIncomingMessages();
    if (!active_.load() || connectedClients_.load() <= 0) {
        return false;
    }

    const bool hasActiveCommand =
        forward_.load() || backward_.load() || left_.load() || right_.load();
    if (hasActiveCommand) {
        return true;
    }

    const int last = lastTick_.load();
    if (last == 0) {
        return false;
    }

    TickType_t now = xTaskGetTickCount();
    return (now - static_cast<TickType_t>(last)) <= pdMS_TO_TICKS(MANUAL_TIMEOUT_MS);
}

driving_mode_t BluetoothControlServer::getDrivingMode(void)
{
    if (!isConnected() || !isManualDriveEnabled()) {
        return DRIVING_MODE_DISABLED;
    }
    return DRIVING_MODE_USER;
}

float BluetoothControlServer::getSpeed(void)
{
    if (!isConnected() || !isManualDriveEnabled()) {
        return 0.0f;
    }
    return duty_.load();
}

float BluetoothControlServer::getSteering(void)
{
    if (!isConnected() || !isManualDriveEnabled()) {
        return STEER_CENTER;
    }
    return steer_.load();
}
