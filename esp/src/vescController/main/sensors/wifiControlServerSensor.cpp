#include "wifiControlServerSensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "esp_log.h"
#include "nvs_flash.h"
#include "host/ble_hs.h"
#include "host/ble_uuid.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

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
static std::atomic<bool> s_bleStarted;
static WifiControlServerSensor *s_bleService;

static ble_uuid128_t s_serviceUuid = BLE_UUID128_INIT(
    0x31, 0x56, 0x26, 0xc0, 0xa8, 0x60, 0x4d, 0x2f,
    0x98, 0x7b, 0x66, 0x6d, 0xaf, 0xaa, 0x00, 0x01);
static ble_uuid128_t s_characteristicUuid = BLE_UUID128_INIT(
    0x31, 0x56, 0x26, 0xc0, 0xa8, 0x60, 0x4d, 0x2f,
    0x98, 0x7b, 0x66, 0x6d, 0xaf, 0xaa, 0x00, 0x02);

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

void WifiControlServerSensor::bleHostTask(void *arg)
{
    (void)arg;
    nimble_port_run();
    nimble_port_freertos_deinit();
}

int WifiControlServerSensor::bleGapEvent(struct ble_gap_event *event, void *arg)
{
    auto *self = static_cast<WifiControlServerSensor *>(arg);
    if (self == nullptr) {
        self = s_bleService;
    }

    switch (event->type) {
        case BLE_GAP_EVENT_CONNECT:
            if (event->connect.status == 0) {
                if (self != nullptr) {
                    self->addConnection(event->connect.conn_handle);
                }
                ESP_LOGI(TAG, "BLE client connected handle=%d", event->connect.conn_handle);
            } else {
                ESP_LOGW(TAG, "BLE connect failed status=%d", event->connect.status);
                if (self != nullptr) {
                    self->restartAdvertising();
                }
            }
            return 0;
        case BLE_GAP_EVENT_DISCONNECT:
            if (self != nullptr) {
                self->removeConnection(event->disconnect.conn.conn_handle);
                self->restartAdvertising();
            }
            ESP_LOGI(TAG, "BLE client disconnected handle=%d",
                     event->disconnect.conn.conn_handle);
            return 0;
        case BLE_GAP_EVENT_SUBSCRIBE:
            ESP_LOGI(TAG, "BLE subscribe conn=%d attr=%d notify=%d",
                     event->subscribe.conn_handle,
                     event->subscribe.attr_handle,
                     event->subscribe.cur_notify);
            return 0;
        case BLE_GAP_EVENT_MTU:
            ESP_LOGI(TAG, "BLE mtu conn=%d mtu=%d",
                     event->mtu.conn_handle,
                     event->mtu.value);
            return 0;
        default:
            return 0;
    }
}

int WifiControlServerSensor::bleGattAccess(uint16_t connHandle,
                                           uint16_t attrHandle,
                                           struct ble_gatt_access_ctxt *ctxt,
                                           void *arg)
{
    (void)connHandle;
    (void)attrHandle;
    (void)arg;
    WifiControlServerSensor *self = s_bleService;
    if (self == nullptr) {
        return BLE_ATT_ERR_UNLIKELY;
    }

    switch (ctxt->op) {
        case BLE_GATT_ACCESS_OP_READ_CHR: {
            const int rc = os_mbuf_append(ctxt->om,
                                          self->bleValue_.data(),
                                          self->bleValue_.size());
            return rc == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
        }
        case BLE_GATT_ACCESS_OP_WRITE_CHR: {
            std::string payload;
            payload.resize(OS_MBUF_PKTLEN(ctxt->om));
            if (!payload.empty()) {
                const int rc = ble_hs_mbuf_to_flat(ctxt->om,
                                                   payload.data(),
                                                   payload.size(),
                                                   nullptr);
                if (rc != 0) {
                    return BLE_ATT_ERR_UNLIKELY;
                }
            }
            self->handleBleCommand(payload);
            return 0;
        }
        default:
            return BLE_ATT_ERR_UNLIKELY;
    }
}

void WifiControlServerSensor::bleOnSync()
{
    if (s_bleService != nullptr) {
        s_bleService->restartAdvertising();
    }
}

void WifiControlServerSensor::restartAdvertising()
{
    ble_gap_adv_stop();

    ble_hs_adv_fields fields = {};
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.name = reinterpret_cast<const uint8_t *>("ROBOCAR_BLE");
    fields.name_len = std::strlen("ROBOCAR_BLE");
    fields.name_is_complete = 1;
    fields.appearance = BLE_APPEARANCE_GENERIC_REMOTE;
    fields.uuids128 = &s_serviceUuid;
    fields.num_uuids128 = 1;
    fields.uuids128_is_complete = 1;
    int rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "Failed to set BLE advertising fields rc=%d", rc);
        return;
    }

    ble_gap_adv_params advParams = {};
    advParams.conn_mode = BLE_GAP_CONN_MODE_UND;
    advParams.disc_mode = BLE_GAP_DISC_MODE_GEN;
    rc = ble_gap_adv_start(BLE_OWN_ADDR_PUBLIC,
                           nullptr,
                           BLE_HS_FOREVER,
                           &advParams,
                           bleGapEvent,
                           this);
    if (rc != 0) {
        ESP_LOGE(TAG, "Failed to start BLE advertising rc=%d", rc);
    } else {
        ESP_LOGI(TAG, "BLE advertising as ROBOCAR_BLE");
    }
}

void WifiControlServerSensor::startBleServer()
{
    bool expected = false;
    if (!s_bleStarted.compare_exchange_strong(expected, true)) {
        ESP_LOGW(TAG, "BLE host already started");
        return;
    }

    s_bleService = this;

    int rc = nimble_port_init();
    if (rc != 0) {
        ESP_LOGE(TAG, "nimble_port_init failed rc=%d", rc);
        active_.store(false);
        return;
    }

    ble_svc_gap_init();
    ble_svc_gatt_init();
    ble_svc_gap_device_name_set("ROBOCAR_BLE");
    ble_hs_cfg.sync_cb = bleOnSync;
    ble_hs_cfg.store_status_cb = nullptr;

    static ble_gatt_chr_def characteristicDefs[] = {
        {
            .uuid = &s_characteristicUuid.u,
            .access_cb = WifiControlServerSensor::bleGattAccess,
            .arg = nullptr,
            .descriptors = nullptr,
            .flags = BLE_GATT_CHR_F_READ |
                     BLE_GATT_CHR_F_WRITE |
                     BLE_GATT_CHR_F_WRITE_NO_RSP |
                     BLE_GATT_CHR_F_NOTIFY,
            .min_key_size = 0,
            .val_handle = &s_bleService->bleCharacteristicHandle_,
            .cpfd = nullptr,
        },
        {
            .uuid = nullptr,
            .access_cb = nullptr,
            .arg = nullptr,
            .descriptors = nullptr,
            .flags = 0,
            .min_key_size = 0,
            .val_handle = nullptr,
            .cpfd = nullptr,
        }
    };
    static ble_gatt_svc_def services[] = {
        {
            .type = BLE_GATT_SVC_TYPE_PRIMARY,
            .uuid = &s_serviceUuid.u,
            .includes = nullptr,
            .characteristics = characteristicDefs,
        },
        {
            .type = 0,
            .uuid = nullptr,
            .includes = nullptr,
            .characteristics = nullptr,
        }
    };

    rc = ble_gatts_count_cfg(services);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gatts_count_cfg failed rc=%d", rc);
        return;
    }
    rc = ble_gatts_add_svcs(services);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gatts_add_svcs failed rc=%d", rc);
        return;
    }

    nimble_port_freertos_init(bleHostTask);
}

void WifiControlServerSensor::stopBleServer()
{
    if (!s_bleStarted.load()) {
        return;
    }
    ble_gap_adv_stop();
    nimble_port_stop();
}

void WifiControlServerSensor::addConnection(uint16_t connHandle)
{
    if (std::find(connHandles_.begin(), connHandles_.end(), connHandle) == connHandles_.end()) {
        connHandles_.push_back(connHandle);
    }
    connectedClients_.store(static_cast<int>(connHandles_.size()));
}

void WifiControlServerSensor::removeConnection(uint16_t connHandle)
{
    connHandles_.erase(std::remove(connHandles_.begin(), connHandles_.end(), connHandle),
                       connHandles_.end());
    connectedClients_.store(static_cast<int>(connHandles_.size()));
    if (connHandles_.empty()) {
        clearManualDriveState();
    }
}

void WifiControlServerSensor::setBleResponse(const std::string &payload)
{
    if (payload.size() > BLE_VALUE_MAX_SIZE) {
        bleValue_ = payload.substr(0, BLE_VALUE_MAX_SIZE);
    } else {
        bleValue_ = payload;
    }
    notifySubscribers(bleValue_);
}

void WifiControlServerSensor::notifySubscribers(const std::string &payload)
{
    if (bleCharacteristicHandle_ == 0) {
        return;
    }

    const size_t notifyLen = std::min(payload.size(), BLE_VALUE_MAX_SIZE);
    for (uint16_t connHandle : connHandles_) {
        os_mbuf *om = ble_hs_mbuf_from_flat(payload.data(), notifyLen);
        if (om == nullptr) {
            continue;
        }
        const int rc = ble_gatts_notify_custom(connHandle, bleCharacteristicHandle_, om);
        if (rc != 0) {
            ESP_LOGW(TAG, "notify failed conn=%d rc=%d", connHandle, rc);
        }
    }
}

bool WifiControlServerSensor::isManualDriveEnabled() const
{
    return this->_algorithmSelector.isManualDriveEnabled();
}

void WifiControlServerSensor::clearManualDriveState()
{
    forward_.store(false);
    backward_.store(false);
    left_.store(false);
    right_.store(false);
    duty_.store(0.0f);
    steer_.store(STEER_CENTER);
    lastTick_.store(0);
}

void WifiControlServerSensor::setSelectedAlgorithmsMask(uint32_t mask)
{
    const bool manualWasEnabled = this->isManualDriveEnabled();

    this->_algorithmSelector.setSelectedMask(mask);
    if (manualWasEnabled && !this->isManualDriveEnabled()) {
        this->clearManualDriveState();
    }
}

bool WifiControlServerSensor::isAlgorithmAvailable(SelectableAlgorithm id) const
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

std::string WifiControlServerSensor::buildAlgorithmsJson(bool includeStatusEnvelope) const
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

std::string WifiControlServerSensor::buildStatusJson() const
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

std::string WifiControlServerSensor::buildGpsGoalJson(bool includeStatusEnvelope) const
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

void WifiControlServerSensor::recomputeOutputFromState()
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

void WifiControlServerSensor::emergencyStop()
{
    clearManualDriveState();
    emergency_.store(true);
    lastTick_.store(static_cast<int>(xTaskGetTickCount()));
    _vescControllerApi.stop();
    _vescControllerApi.deactivate();
}

bool WifiControlServerSensor::applyProtocolChar(char c)
{
    switch (c) {
        case 'F':
            if (!isManualDriveEnabled()) return true;
            forward_.store(true);
            recomputeOutputFromState();
            return true;
        case 'f':
            if (!isManualDriveEnabled()) return true;
            forward_.store(false);
            recomputeOutputFromState();
            return true;
        case 'B':
            if (!isManualDriveEnabled()) return true;
            backward_.store(true);
            recomputeOutputFromState();
            return true;
        case 'b':
            if (!isManualDriveEnabled()) return true;
            backward_.store(false);
            recomputeOutputFromState();
            return true;
        case 'L':
            if (!isManualDriveEnabled()) return true;
            left_.store(true);
            recomputeOutputFromState();
            return true;
        case 'l':
            if (!isManualDriveEnabled()) return true;
            left_.store(false);
            recomputeOutputFromState();
            return true;
        case 'R':
            if (!isManualDriveEnabled()) return true;
            right_.store(true);
            recomputeOutputFromState();
            return true;
        case 'r':
            if (!isManualDriveEnabled()) return true;
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
            setBleResponse("STATUS:" + buildStatusJson());
            return true;
        default:
            return false;
    }
}

void WifiControlServerSensor::parseAndStore(const uint8_t *buf, int len)
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

void WifiControlServerSensor::handleAlgorithmCommand(const char *value)
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

    setSelectedAlgorithmsMask(selectedMask);
    setBleResponse("ALGORITHMS:" + buildAlgorithmsJson(true));
}

void WifiControlServerSensor::handleGpsGoalCommand(const char *value)
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

void WifiControlServerSensor::handleSteeringCommand(const char *value)
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

void WifiControlServerSensor::handleLineCommand(const std::string &command)
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

void WifiControlServerSensor::handleBleCommand(const std::string &payload)
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

void WifiControlServerSensor::start(void)
{
    if (active_.load()) {
        return;
    }

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ensure_log_redirect_installed();
    active_.store(true);
    setBleResponse("STATUS:" + buildStatusJson());
    startBleServer();
    ESP_LOGI(TAG, "BLE control service initialized; Wi-Fi/HTTP control disabled");
}

void WifiControlServerSensor::stop(void)
{
    if (!active_.load()) {
        return;
    }
    clearManualDriveState();
    stopBleServer();
    active_.store(false);
    ESP_LOGI(TAG, "BLE control service stopped");
}

bool WifiControlServerSensor::isActivated(void)
{
    return active_.load();
}

bool WifiControlServerSensor::isConnected(void)
{
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

driving_mode_t WifiControlServerSensor::getDrivingMode(void)
{
    if (!isConnected() || !isManualDriveEnabled()) {
        return DRIVING_MODE_DISABLED;
    }
    return DRIVING_MODE_USER;
}

float WifiControlServerSensor::getSpeed(void)
{
    if (!isConnected() || !isManualDriveEnabled()) {
        return 0.0f;
    }
    return duty_.load();
}

float WifiControlServerSensor::getSteering(void)
{
    if (!isConnected() || !isManualDriveEnabled()) {
        return STEER_CENTER;
    }
    return steer_.load();
}
