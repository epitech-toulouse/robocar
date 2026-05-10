#include "cameraBleSensor.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>

#include "esp_bt.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/task.h"
#include "host/ble_gap.h"
#include "host/ble_gatt.h"
#include "host/ble_hs.h"
#include "os/os_mbuf.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "nvs_flash.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

namespace {
const char *TAG = "CameraBleSensor";

const ble_uuid128_t kCameraServiceUuid =
    BLE_UUID128_INIT(0x31, 0x56, 0x26, 0xc0, 0xa8, 0x60, 0x4d, 0x2f,
                     0x98, 0x7b, 0x66, 0x6d, 0xaf, 0xaa, 0x00, 0x01);
const ble_uuid128_t kCameraRxCharUuid =
    BLE_UUID128_INIT(0x31, 0x56, 0x26, 0xc0, 0xa8, 0x60, 0x4d, 0x2f,
                     0x98, 0x7b, 0x66, 0x6d, 0xaf, 0xaa, 0x00, 0x02);

uint16_t g_cameraRxHandle = 0;
ble_gatt_chr_def gattCharacteristics[] = {
    {
        .uuid = &kCameraRxCharUuid.u,
        .access_cb = CameraBleSensor::gattAccessHandler,
        .flags = BLE_GATT_CHR_F_WRITE | BLE_GATT_CHR_F_WRITE_NO_RSP,
        .val_handle = &g_cameraRxHandle,
    },
    {},
};

const ble_gatt_svc_def gattServices[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &kCameraServiceUuid.u,
        .characteristics = gattCharacteristics,
    },
    {},
};
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

    std::lock_guard<std::mutex> lock(this->mutex_);
    output = this->stopRequested_;
    return true;
}

bool CameraBleSensor::getHeading(float &output)
{
    if (!this->ensureStarted()) {
        return false;
    }

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

int CameraBleSensor::gapEventHandler(struct ble_gap_event *event, void *arg)
{
    (void) arg;

    if (instance_ == nullptr) {
        return 0;
    }

    switch (event->type) {
        case BLE_GAP_EVENT_CONNECT: {
            bool connected = false;
            {
                std::lock_guard<std::mutex> lock(instance_->mutex_);
                instance_->connected_ = (event->connect.status == 0);
                connected = instance_->connected_;
            }
            if (connected) {
                ESP_LOGI(TAG, "camera BLE connected");
            } else {
                ESP_LOGW(TAG, "camera BLE connect failed, restart advertising");
                startAdvertising();
            }
            break;
        }
        case BLE_GAP_EVENT_DISCONNECT: {
            {
                std::lock_guard<std::mutex> lock(instance_->mutex_);
                instance_->connected_ = false;
            }
            ESP_LOGW(TAG, "camera BLE disconnected, reason=%d", event->disconnect.reason);
            startAdvertising();
            break;
        }
        case BLE_GAP_EVENT_ADV_COMPLETE:
            startAdvertising();
            break;
        default:
            break;
    }

    return 0;
}

int CameraBleSensor::gattAccessHandler(uint16_t connHandle,
                                       uint16_t attrHandle,
                                       struct ble_gatt_access_ctxt *ctxt,
                                       void *arg)
{
    (void) connHandle;
    (void) attrHandle;
    (void) arg;

    if (instance_ == nullptr) {
        return BLE_ATT_ERR_UNLIKELY;
    }
    if (ctxt->op != BLE_GATT_ACCESS_OP_WRITE_CHR) {
        return BLE_ATT_ERR_UNLIKELY;
    }

    const uint16_t packetSize = OS_MBUF_PKTLEN(ctxt->om);
    if (packetSize == 0) {
        return 0;
    }

    std::string buffer;
    buffer.resize(packetSize);
    if (ble_hs_mbuf_to_flat(ctxt->om, buffer.data(), packetSize, nullptr) != 0) {
        return BLE_ATT_ERR_UNLIKELY;
    }

    instance_->appendIncomingData(reinterpret_cast<const uint8_t *>(buffer.data()), buffer.size());
    return 0;
}

void CameraBleSensor::onReset(int reason)
{
    ESP_LOGE(TAG, "BLE host reset, reason=%d", reason);
}

void CameraBleSensor::onSync(void)
{
    uint8_t addrType = 0;
    const int rc = ble_hs_id_infer_auto(0, &addrType);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_hs_id_infer_auto failed: rc=%d", rc);
        return;
    }

    if (ble_svc_gap_device_name_set(kDeviceName) != 0) {
        ESP_LOGW(TAG, "failed to set BLE device name");
    }

    ESP_LOGI(TAG, "BLE camera host ready, advertising as %s", kDeviceName);
    startAdvertising();
}

void CameraBleSensor::hostTask(void *arg)
{
    (void) arg;
    nimble_port_run();
    nimble_port_freertos_deinit();
}

void CameraBleSensor::startAdvertising()
{
    ble_hs_adv_fields fields{};
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.uuids128 = &kCameraServiceUuid;
    fields.num_uuids128 = 1;
    fields.uuids128_is_complete = 1;

    int rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gap_adv_set_fields failed: rc=%d", rc);
        return;
    }

    ble_hs_adv_fields scanResponse{};
    scanResponse.name = reinterpret_cast<const uint8_t *>(kDeviceName);
    scanResponse.name_len = static_cast<uint8_t>(std::strlen(kDeviceName));
    scanResponse.name_is_complete = 1;

    rc = ble_gap_adv_rsp_set_fields(&scanResponse);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gap_adv_rsp_set_fields failed: rc=%d", rc);
        return;
    }

    ble_gap_adv_params advParams{};
    advParams.conn_mode = BLE_GAP_CONN_MODE_UND;
    advParams.disc_mode = BLE_GAP_DISC_MODE_GEN;

    uint8_t addrType = 0;
    rc = ble_hs_id_infer_auto(0, &addrType);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_hs_id_infer_auto failed before advertise: rc=%d", rc);
        return;
    }

    rc = ble_gap_adv_start(addrType, nullptr, BLE_HS_FOREVER, &advParams, gapEventHandler, nullptr);
    if (rc != 0 && rc != BLE_HS_EALREADY) {
        ESP_LOGE(TAG, "ble_gap_adv_start failed: rc=%d", rc);
    }
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

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "nvs_flash_init failed: %s", esp_err_to_name(ret));
        return false;
    }

    const esp_bt_controller_status_t controllerStatus = esp_bt_controller_get_status();
    ESP_LOGI(TAG, "BT controller status before init: %d", static_cast<int>(controllerStatus));

    ret = esp_bt_controller_mem_release(ESP_BT_MODE_CLASSIC_BT);
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "esp_bt_controller_mem_release failed: %s", esp_err_to_name(ret));
        return false;
    }

    ret = nimble_port_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "nimble_port_init failed: %s", esp_err_to_name(ret));
        return false;
    }

    ble_hs_cfg.reset_cb = onReset;
    ble_hs_cfg.sync_cb = onSync;

    ble_svc_gap_init();
    ble_svc_gatt_init();

    int rc = ble_gatts_count_cfg(gattServices);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gatts_count_cfg failed: rc=%d", rc);
        return false;
    }

    rc = ble_gatts_add_svcs(gattServices);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gatts_add_svcs failed: rc=%d", rc);
        return false;
    }

    instance_ = this;
    nimble_port_freertos_init(hostTask);
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
