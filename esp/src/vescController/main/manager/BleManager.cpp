#include "manager/BleManager.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>

#include "esp_bt.h"
#include "esp_err.h"
#include "esp_log.h"
#include "host/ble_gap.h"
#include "host/ble_gatt.h"
#include "host/ble_hs.h"
#include "host/ble_uuid.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "nvs_flash.h"
#include "os/os_mbuf.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

namespace {
const char *TAG = "BleManager";
constexpr const char *kDeviceName = "ROBOCAR_BLE";

ble_uuid128_t g_serviceUuid = BLE_UUID128_INIT(
    0x31, 0x56, 0x26, 0xc0, 0xa8, 0x60, 0x4d, 0x2f,
    0x98, 0x7b, 0x66, 0x6d, 0xaf, 0xaa, 0x00, 0x01);
ble_uuid128_t g_controlCharacteristicUuid = BLE_UUID128_INIT(
    0x31, 0x56, 0x26, 0xc0, 0xa8, 0x60, 0x4d, 0x2f,
    0x98, 0x7b, 0x66, 0x6d, 0xaf, 0xaa, 0x00, 0x02);
ble_uuid128_t g_cameraCharacteristicUuid = BLE_UUID128_INIT(
    0x31, 0x56, 0x26, 0xc0, 0xa8, 0x60, 0x4d, 0x2f,
    0x98, 0x7b, 0x66, 0x6d, 0xaf, 0xaa, 0x00, 0x03);

const char *endpointName(BleEndpoint endpoint)
{
    switch (endpoint) {
        case BleEndpoint::Control:
            return "control";
        case BleEndpoint::Camera:
            return "camera";
        default:
            return "unknown";
    }
}

std::string payloadPreview(const std::string &payload)
{
    std::string preview;
    preview.reserve(payload.size());
    for (unsigned char c : payload) {
        preview.push_back(std::isprint(c) ? static_cast<char>(c) : '.');
    }
    return preview;
}

std::string payloadHex(const std::string &payload)
{
    static constexpr char kHex[] = "0123456789ABCDEF";

    std::string hex;
    if (payload.empty()) {
        return hex;
    }
    hex.reserve(payload.size() * 3 - 1);
    for (size_t i = 0; i < payload.size(); ++i) {
        if (i > 0) {
            hex.push_back(' ');
        }
        const uint8_t byte = static_cast<uint8_t>(payload[i]);
        hex.push_back(kHex[(byte >> 4) & 0x0F]);
        hex.push_back(kHex[byte & 0x0F]);
    }
    return hex;
}
} // namespace

BleManager &BleManager::instance()
{
    static BleManager manager;
    return manager;
}

bool BleManager::start()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (started_) {
            return true;
        }
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

    ret = esp_bt_controller_mem_release(ESP_BT_MODE_CLASSIC_BT);
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "esp_bt_controller_mem_release failed: %s", esp_err_to_name(ret));
        return false;
    }

    int rc = nimble_port_init();
    if (rc != 0) {
        ESP_LOGE(TAG, "nimble_port_init failed rc=%d", rc);
        return false;
    }

    ble_svc_gap_init();
    ble_svc_gatt_init();
    ble_svc_gap_device_name_set(kDeviceName);
    ble_hs_cfg.reset_cb = onReset;
    ble_hs_cfg.sync_cb = onSync;
    ble_hs_cfg.store_status_cb = nullptr;

    static ble_gatt_chr_def characteristicDefs[] = {
        {
            .uuid = &g_controlCharacteristicUuid.u,
            .access_cb = BleManager::gattAccessHandler,
            .arg = nullptr,
            .descriptors = nullptr,
            .flags = BLE_GATT_CHR_F_READ |
                     BLE_GATT_CHR_F_WRITE |
                     BLE_GATT_CHR_F_WRITE_NO_RSP |
                     BLE_GATT_CHR_F_NOTIFY,
            .min_key_size = 0,
            .val_handle = &BleManager::instance().controlCharacteristicHandle_,
            .cpfd = nullptr,
        },
        {
            .uuid = &g_cameraCharacteristicUuid.u,
            .access_cb = BleManager::gattAccessHandler,
            .arg = nullptr,
            .descriptors = nullptr,
            .flags = BLE_GATT_CHR_F_WRITE | BLE_GATT_CHR_F_WRITE_NO_RSP,
            .min_key_size = 0,
            .val_handle = &BleManager::instance().cameraCharacteristicHandle_,
            .cpfd = nullptr,
        },
        {}
    };
    static ble_gatt_svc_def services[] = {
        {
            .type = BLE_GATT_SVC_TYPE_PRIMARY,
            .uuid = &g_serviceUuid.u,
            .includes = nullptr,
            .characteristics = characteristicDefs,
        },
        {}
    };

    rc = ble_gatts_count_cfg(services);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gatts_count_cfg failed rc=%d", rc);
        return false;
    }
    rc = ble_gatts_add_svcs(services);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gatts_add_svcs failed rc=%d", rc);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        started_ = true;
    }
    nimble_port_freertos_init(hostTask);
    return true;
}

void BleManager::stop()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!started_) {
        return;
    }
    ble_gap_adv_stop();
    nimble_port_stop();
    started_ = false;
    connHandles_.clear();
}

bool BleManager::isStarted() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return started_;
}

bool BleManager::isConnected() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return !connHandles_.empty();
}

int BleManager::connectedClientCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(connHandles_.size());
}

uint32_t BleManager::latestSequence() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return sequence_;
}

std::vector<BleMessage> BleManager::messagesSince(uint32_t sequence, BleEndpoint endpoint) const
{
    std::vector<BleMessage> result;
    std::lock_guard<std::mutex> lock(mutex_);
    for (const BleMessage &message : messages_) {
        if (message.sequence > sequence && message.endpoint == endpoint) {
            result.push_back(message);
        }
    }
    return result;
}

void BleManager::setControlResponse(const std::string &payload)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        controlValue_ = payload.size() > kMaxValueSize
            ? payload.substr(0, kMaxValueSize)
            : payload;
    }
    notifyControlSubscribers(payload);
}

std::string BleManager::controlResponse() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return controlValue_;
}

void BleManager::notifyControlSubscribers(const std::string &payload)
{
    std::vector<uint16_t> handles;
    uint16_t charHandle = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        handles = connHandles_;
        charHandle = controlCharacteristicHandle_;
    }

    if (charHandle == 0) {
        return;
    }

    const size_t notifyLen = std::min(payload.size(), kMaxValueSize);
    for (uint16_t connHandle : handles) {
        os_mbuf *om = ble_hs_mbuf_from_flat(payload.data(), notifyLen);
        if (om == nullptr) {
            continue;
        }
        const int rc = ble_gatts_notify_custom(connHandle, charHandle, om);
        if (rc != 0) {
            ESP_LOGW(TAG, "notify failed conn=%d rc=%d", connHandle, rc);
        }
    }
}

int BleManager::gapEventHandler(struct ble_gap_event *event, void *arg)
{
    (void)arg;
    BleManager &manager = BleManager::instance();

    switch (event->type) {
        case BLE_GAP_EVENT_CONNECT:
            if (event->connect.status == 0) {
                manager.addConnection(event->connect.conn_handle);
                ESP_LOGI(TAG, "BLE client connected handle=%d", event->connect.conn_handle);
            } else {
                ESP_LOGW(TAG, "BLE connect failed status=%d", event->connect.status);
                manager.restartAdvertising();
            }
            return 0;
        case BLE_GAP_EVENT_DISCONNECT:
            manager.removeConnection(event->disconnect.conn.conn_handle);
            manager.restartAdvertising();
            ESP_LOGI(TAG, "BLE client disconnected handle=%d",
                     event->disconnect.conn.conn_handle);
            return 0;
        case BLE_GAP_EVENT_ADV_COMPLETE:
            manager.restartAdvertising();
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

int BleManager::gattAccessHandler(uint16_t connHandle,
                                  uint16_t attrHandle,
                                  struct ble_gatt_access_ctxt *ctxt,
                                  void *arg)
{
    (void)arg;
    return BleManager::instance().handleAccess(connHandle, attrHandle, ctxt);
}

void BleManager::onSync()
{
    BleManager::instance().restartAdvertising();
}

void BleManager::onReset(int reason)
{
    ESP_LOGE(TAG, "BLE host reset reason=%d", reason);
}

void BleManager::hostTask(void *arg)
{
    (void)arg;
    nimble_port_run();
    nimble_port_freertos_deinit();
}

void BleManager::restartAdvertising()
{
    if (ble_gap_adv_active()) {
        ble_gap_adv_stop();
    }

    ble_hs_adv_fields fields = {};
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.uuids128 = &g_serviceUuid;
    fields.num_uuids128 = 1;
    fields.uuids128_is_complete = 1;

    int rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gap_adv_set_fields failed rc=%d", rc);
        return;
    }

    ble_hs_adv_fields scanResponse = {};
    scanResponse.name = reinterpret_cast<const uint8_t *>(kDeviceName);
    scanResponse.name_len = std::strlen(kDeviceName);
    scanResponse.name_is_complete = 1;
    scanResponse.appearance = kAppearanceGenericRemote;
    scanResponse.appearance_is_present = 1;

    rc = ble_gap_adv_rsp_set_fields(&scanResponse);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gap_adv_rsp_set_fields failed rc=%d", rc);
        return;
    }

    ble_gap_adv_params advParams = {};
    advParams.conn_mode = BLE_GAP_CONN_MODE_UND;
    advParams.disc_mode = BLE_GAP_DISC_MODE_GEN;

    uint8_t addrType = 0;
    rc = ble_hs_id_infer_auto(0, &addrType);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_hs_id_infer_auto failed rc=%d", rc);
        return;
    }

    rc = ble_gap_adv_start(addrType,
                           nullptr,
                           BLE_HS_FOREVER,
                           &advParams,
                           gapEventHandler,
                           nullptr);
    if (rc != 0 && rc != BLE_HS_EALREADY) {
        ESP_LOGE(TAG, "ble_gap_adv_start failed rc=%d", rc);
    } else {
        ESP_LOGI(TAG, "BLE advertising as %s", kDeviceName);
    }
}

void BleManager::addConnection(uint16_t connHandle)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (std::find(connHandles_.begin(), connHandles_.end(), connHandle) == connHandles_.end()) {
        connHandles_.push_back(connHandle);
    }
}

void BleManager::removeConnection(uint16_t connHandle)
{
    std::lock_guard<std::mutex> lock(mutex_);
    connHandles_.erase(std::remove(connHandles_.begin(), connHandles_.end(), connHandle),
                       connHandles_.end());
}

void BleManager::appendMessage(BleEndpoint endpoint,
                               uint16_t connHandle,
                               const std::string &payload)
{
    std::lock_guard<std::mutex> lock(mutex_);
    BleMessage message;
    message.sequence = ++sequence_;
    message.tick = xTaskGetTickCount();
    message.connHandle = connHandle;
    message.endpoint = endpoint;
    message.payload = payload;
    messages_.push_back(message);
    if (messages_.size() > kMaxMessages) {
        messages_.erase(messages_.begin(),
                        messages_.begin() + (messages_.size() - kMaxMessages));
    }
}

int BleManager::handleAccess(uint16_t connHandle,
                             uint16_t attrHandle,
                             struct ble_gatt_access_ctxt *ctxt)
{
    switch (ctxt->op) {
        case BLE_GATT_ACCESS_OP_READ_CHR: {
            std::string value;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (attrHandle != controlCharacteristicHandle_) {
                    return BLE_ATT_ERR_UNLIKELY;
                }
                value = controlValue_;
            }
            const int rc = os_mbuf_append(ctxt->om, value.data(), value.size());
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

            BleEndpoint endpoint = BleEndpoint::Control;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (attrHandle == cameraCharacteristicHandle_) {
                    endpoint = BleEndpoint::Camera;
                } else if (attrHandle != controlCharacteristicHandle_) {
                    return BLE_ATT_ERR_UNLIKELY;
                }
            }
            ESP_LOGI(TAG,
                     "RX endpoint=%s conn=%u len=%u text=\"%s\" hex=%s",
                     endpointName(endpoint),
                     static_cast<unsigned>(connHandle),
                     static_cast<unsigned>(payload.size()),
                     payloadPreview(payload).c_str(),
                     payloadHex(payload).c_str());
            appendMessage(endpoint, connHandle, payload);
            return 0;
        }
        default:
            return BLE_ATT_ERR_UNLIKELY;
    }
}
