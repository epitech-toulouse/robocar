#include "bluetooth_receiver.hpp"

#include "sdkconfig.h"
#include "nvs_flash.h"
#if CONFIG_BT_BLUEDROID_ENABLED && CONFIG_BT_BLE_ENABLED
#include "esp_bt.h"
#include "esp_bt_main.h"
#include "esp_gap_ble_api.h"
#include "esp_gatts_api.h"
#include "esp_gatt_common_api.h"

#endif
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

static const char *TAG = "ble_recv";

static std::atomic<float> s_duty{0.0f};
static std::atomic<float> s_steer{0.5f};
static std::atomic<int> s_last_tick{0};
static std::atomic<bool> s_forward{false};
static std::atomic<bool> s_backward{false};
static std::atomic<bool> s_left{false};
static std::atomic<bool> s_right{false};
static constexpr float STEER_CENTER = 0.5f;
static constexpr float STEER_LEFT = 0.2f;
static constexpr float STEER_RIGHT = 0.8f;
static constexpr float DUTY_FORWARD = 0.05f;
static constexpr float DUTY_BACKWARD = -0.05f;

static constexpr TickType_t MANUAL_TIMEOUT_MS = 2000;

#if CONFIG_BT_BLUEDROID_ENABLED && CONFIG_BT_BLE_ENABLED
static constexpr uint16_t GATTS_APP_ID = 0x55;
static constexpr uint16_t SERVICE_UUID = 0xFFE0;
static constexpr uint16_t CHAR_UUID = 0xFFE1;

static uint16_t s_service_handle = 0;
static uint16_t s_char_handle = 0;
static esp_gatt_if_t s_gatts_if = ESP_GATT_IF_NONE;

static esp_ble_adv_data_t s_adv_data = {
    .set_scan_rsp = false,
    .include_name = true,
    .include_txpower = false,
    .min_interval = 0,
    .max_interval = 0,
    .appearance = 0,
    .manufacturer_len = 0,
    .p_manufacturer_data = nullptr,
    .service_data_len = 0,
    .p_service_data = nullptr,
    .service_uuid_len = 0,
    .p_service_uuid = nullptr,
    .flag = ESP_BLE_ADV_FLAG_GEN_DISC | ESP_BLE_ADV_FLAG_BREDR_NOT_SPT,
};

static esp_ble_adv_params_t make_adv_params() {
    esp_ble_adv_params_t p = {};
    p.adv_int_min = 0x20;
    p.adv_int_max = 0x40;
    p.adv_type = ADV_TYPE_IND;
    p.own_addr_type = BLE_ADDR_TYPE_PUBLIC;
    p.channel_map = ADV_CHNL_ALL;
    p.adv_filter_policy = ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY;
    return p;
}

static esp_ble_adv_params_t s_adv_params = make_adv_params();

static void recompute_output_from_state() {
    float duty = 0.0f;
    if (s_forward.load() && !s_backward.load()) {
        duty = DUTY_FORWARD;
    } else if (s_backward.load() && !s_forward.load()) {
        duty = DUTY_BACKWARD;
    }

    float steer = STEER_CENTER;
    if (s_left.load() && !s_right.load()) {
        steer = STEER_LEFT;
    } else if (s_right.load() && !s_left.load()) {
        steer = STEER_RIGHT;
    }

    s_duty.store(duty);
    s_steer.store(steer);
    s_last_tick.store(static_cast<int>(xTaskGetTickCount()));
}

static void emergency_stop() {
    s_forward.store(false);
    s_backward.store(false);
    s_left.store(false);
    s_right.store(false);
    s_duty.store(0.0f);
    s_steer.store(STEER_CENTER);
    s_last_tick.store(static_cast<int>(xTaskGetTickCount()));
}

static bool apply_protocol_char(char c) {
    switch (c) {
        case 'F': s_forward.store(true);  recompute_output_from_state(); return true;
        case 'f': s_forward.store(false); recompute_output_from_state(); return true;
        case 'B': s_backward.store(true);  recompute_output_from_state(); return true;
        case 'b': s_backward.store(false); recompute_output_from_state(); return true;
        case 'L': s_left.store(true);  recompute_output_from_state(); return true;
        case 'l': s_left.store(false); recompute_output_from_state(); return true;
        case 'R': s_right.store(true);  recompute_output_from_state(); return true;
        case 'r': s_right.store(false); recompute_output_from_state(); return true;
        case 'S': emergency_stop(); return true;
        default: return false;
    }
}

static void parse_and_store(const uint8_t *buf, int len) {
    // Protocol: F/f/B/b/L/l/R/r/S (1 ASCII char per action)
    if (len <= 0 || buf == nullptr) {
        return;
    }

    for (int i = 0; i < len; ++i) {
        const uint8_t b = buf[i];
        if (b == 0x00 || b == '\n' || b == '\r' || b == ' ' || b == '\t') {
            continue;
        }
        if (!apply_protocol_char(static_cast<char>(b))) {
            ESP_LOGW(TAG, "Unknown protocol char: '%c' (0x%02X)", static_cast<char>(b), b);
        }
    }

    ESP_LOGI(TAG, "State F=%d B=%d L=%d R=%d -> duty=%.3f steer=%.3f",
             s_forward.load(), s_backward.load(), s_left.load(), s_right.load(),
             s_duty.load(), s_steer.load());
}

static void gap_event_handler(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param) {
    switch (event) {
        case ESP_GAP_BLE_ADV_DATA_SET_COMPLETE_EVT:
            esp_ble_gap_start_advertising(&s_adv_params);
            break;
        case ESP_GAP_BLE_ADV_START_COMPLETE_EVT:
            if (param->adv_start_cmpl.status != ESP_BT_STATUS_SUCCESS) {
                ESP_LOGE(TAG, "Advertising start failed: %d", param->adv_start_cmpl.status);
            } else {
                ESP_LOGI(TAG, "BLE advertising started");
            }
            break;
        default:
            break;
    }
}

static void gatts_event_handler(esp_gatts_cb_event_t event,
                                esp_gatt_if_t gatts_if,
                                esp_ble_gatts_cb_param_t *param) {
    switch (event) {
        case ESP_GATTS_REG_EVT: {
            s_gatts_if = gatts_if;
            esp_err_t name_ret = esp_ble_gap_set_device_name("ESP32S3_BLE_CTRL");
            if (name_ret != ESP_OK) {
                ESP_LOGE(TAG, "set device name failed: %s", esp_err_to_name(name_ret));
            }

            esp_err_t cfg_ret = esp_ble_gap_config_adv_data(&s_adv_data);
            if (cfg_ret != ESP_OK) {
                ESP_LOGE(TAG, "config adv data failed: %s", esp_err_to_name(cfg_ret));
            }

            esp_gatt_srvc_id_t service_id = {};
            service_id.is_primary = true;
            service_id.id.inst_id = 0x00;
            service_id.id.uuid.len = ESP_UUID_LEN_16;
            service_id.id.uuid.uuid.uuid16 = SERVICE_UUID;

            esp_err_t create_ret = esp_ble_gatts_create_service(gatts_if, &service_id, 4);
            if (create_ret != ESP_OK) {
                ESP_LOGE(TAG, "create service failed: %s", esp_err_to_name(create_ret));
            }
            break;
        }

        case ESP_GATTS_CREATE_EVT: {
            s_service_handle = param->create.service_handle;
            esp_ble_gatts_start_service(s_service_handle);

            esp_bt_uuid_t char_uuid = {};
            char_uuid.len = ESP_UUID_LEN_16;
            char_uuid.uuid.uuid16 = CHAR_UUID;

            constexpr esp_gatt_perm_t perm = ESP_GATT_PERM_WRITE;
            constexpr esp_gatt_char_prop_t prop = ESP_GATT_CHAR_PROP_BIT_WRITE | ESP_GATT_CHAR_PROP_BIT_WRITE_NR;

            esp_err_t add_ret = esp_ble_gatts_add_char(
                s_service_handle,
                &char_uuid,
                perm,
                prop,
                nullptr,
                nullptr);
            if (add_ret != ESP_OK) {
                ESP_LOGE(TAG, "add char failed: %s", esp_err_to_name(add_ret));
            }
            break;
        }

        case ESP_GATTS_ADD_CHAR_EVT:
            s_char_handle = param->add_char.attr_handle;
            ESP_LOGI(TAG, "BLE write characteristic ready, handle=%u", s_char_handle);
            break;

        case ESP_GATTS_CONNECT_EVT:
            ESP_LOGI(TAG, "BLE client connected");
            break;

        case ESP_GATTS_DISCONNECT_EVT:
            ESP_LOGI(TAG, "BLE client disconnected, restart advertising");
            emergency_stop();
            esp_ble_gap_start_advertising(&s_adv_params);
            break;

        case ESP_GATTS_WRITE_EVT:
            if (param->write.handle == s_char_handle && param->write.len > 0 && param->write.value) {
                parse_and_store(param->write.value, param->write.len);
            }
            if (param->write.need_rsp && s_gatts_if != ESP_GATT_IF_NONE) {
                esp_ble_gatts_send_response(s_gatts_if,
                                            param->write.conn_id,
                                            param->write.trans_id,
                                            ESP_GATT_OK,
                                            nullptr);
            }
            break;

        default:
            break;
    }
}

void init_bluetooth_receiver() {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_bt_controller_mem_release(ESP_BT_MODE_CLASSIC_BT));

    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_bt_controller_init(&bt_cfg));
    ESP_ERROR_CHECK(esp_bt_controller_enable(ESP_BT_MODE_BLE));

    ESP_ERROR_CHECK(esp_bluedroid_init());
    ESP_ERROR_CHECK(esp_bluedroid_enable());

    ESP_ERROR_CHECK(esp_ble_gatts_register_callback(gatts_event_handler));
    ESP_ERROR_CHECK(esp_ble_gap_register_callback(gap_event_handler));
    ESP_ERROR_CHECK(esp_ble_gatts_app_register(GATTS_APP_ID));

    ESP_LOGI(TAG, "BLE receiver initialized");
}
#else
void init_bluetooth_receiver() {
    ESP_LOGW(TAG, "BLE disabled in sdkconfig (CONFIG_BT_BLUEDROID_ENABLED/CONFIG_BT_BLE_ENABLED)");
}
#endif

bool get_manual_control(float &duty, float &steer) {
    const bool hasActiveCommand =
        s_forward.load() || s_backward.load() || s_left.load() || s_right.load();

    if (hasActiveCommand) {
        duty = s_duty.load();
        steer = s_steer.load();
        return true;
    }

    int last = s_last_tick.load();
    if (last == 0) {
        return false;
    }

    TickType_t now = xTaskGetTickCount();
    if ((now - static_cast<TickType_t>(last)) > pdMS_TO_TICKS(MANUAL_TIMEOUT_MS)) {
        return false;
    }

    duty = s_duty.load();
    steer = s_steer.load();
    return true;
}
