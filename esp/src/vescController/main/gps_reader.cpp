/**
 * @file gps_reader.cpp
 * @brief USB Host CDC GPS reader — merged from esp32_s3_usb_gps_host.
 *
 * Runs a FreeRTOS task that manages the USB Host stack and a CDC-ACM
 * connection to the GPS module.  Incoming bytes are assembled into NMEA
 * lines and parsed for GGA (position) and RMC (heading/speed) sentences.
 *
 * Thread-safe access via atomic copy of GpsFix.
 */

#include "gps_reader.hpp"
#include "nav_types.h"

#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "usb/cdc_acm_host.h"
#include "usb/usb_host.h"
#include "usb/vcp_ch34x.h"
#include "usb/vcp_cp210x.h"
#include "usb/vcp_ftdi.h"

#include <atomic>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>

static const char *TAG = "gps_usb";

/* -------------------------------------------------------------------------- */
/*  GPS baud rate — Point One Navigation uses 460800                           */
/* -------------------------------------------------------------------------- */
static constexpr int GPS_BAUD_RATE = 460800;

/* -------------------------------------------------------------------------- */
/*  Atomic GPS fix shared between USB task callback and main task              */
/* -------------------------------------------------------------------------- */
static GpsFix s_gps_fix;
static portMUX_TYPE s_gps_lock = portMUX_INITIALIZER_UNLOCKED;

bool get_gps_fix(GpsFix &fix) {
    portENTER_CRITICAL(&s_gps_lock);
    fix = s_gps_fix;
    portEXIT_CRITICAL(&s_gps_lock);

    if (!fix.has_fix) return false;

    // Check staleness
    TickType_t age = xTaskGetTickCount() - fix.last_update_tick;
    if (age > pdMS_TO_TICKS(NAV_GPS_STALE_MS)) return false;

    return true;
}

/* -------------------------------------------------------------------------- */
/*  NMEA parsing                                                               */
/* -------------------------------------------------------------------------- */

/// Convert NMEA coordinate (DDDMM.MMMMM, N/S/E/W) to decimal degrees.
static float nmea_to_decimal(const char *nmea_coord, char dir) {
    if (!nmea_coord || strlen(nmea_coord) < 4) return 0.0f;

    const char *dot = strchr(nmea_coord, '.');
    if (!dot) return 0.0f;

    int min_digits = 2;
    int deg_len = (int)(dot - nmea_coord) - min_digits;
    if (deg_len <= 0 || deg_len > 3) return 0.0f;

    char deg_str[4] = {0};
    strncpy(deg_str, nmea_coord, deg_len);
    float degrees = (float)atof(deg_str);
    float minutes = (float)atof(nmea_coord + deg_len);

    float decimal = degrees + (minutes / 60.0f);
    if (dir == 'S' || dir == 'W') decimal = -decimal;
    return decimal;
}

/// Tokenize an NMEA sentence by commas and asterisk.
static int tokenize_nmea(char *line, char **tokens, int max_tokens) {
    int count = 0;
    tokens[count++] = line;
    for (int i = 0; line[i] != '\0' && count < max_tokens; i++) {
        if (line[i] == ',' || line[i] == '*') {
            line[i] = '\0';
            tokens[count++] = &line[i + 1];
        }
    }
    return count;
}

/**
 * Parse a single NMEA sentence and update the global GPS fix.
 *
 * Supported sentences:
 *   $GNGGA / $GPGGA — lat, lon, alt, fix quality, sats
 *   $GNRMC / $GPRMC — lat, lon, speed (knots), course over ground (heading)
 */
static void parse_nmea(char *line) {
    const bool is_gga = (strncmp(line, "$GNGGA", 6) == 0 || strncmp(line, "$GPGGA", 6) == 0);
    const bool is_rmc = (strncmp(line, "$GNRMC", 6) == 0 || strncmp(line, "$GPRMC", 6) == 0);

    if (!is_gga && !is_rmc) return;

    char *tokens[20];
    int count = tokenize_nmea(line, tokens, 20);

    if (is_gga && count >= 10) {
        /* GGA: $xxGGA,time,lat,N/S,lon,E/W,fix,sats,hdop,alt,M,...
         *        0     1    2   3   4   5   6   7    8    9  10     */
        if (strlen(tokens[2]) == 0 || strlen(tokens[4]) == 0) return;

        float lat = nmea_to_decimal(tokens[2], tokens[3][0]);
        float lon = nmea_to_decimal(tokens[4], tokens[5][0]);
        int   fix = atoi(tokens[6]);
        int   sats = atoi(tokens[7]);
        float alt = (float)atof(tokens[9]);

        portENTER_CRITICAL(&s_gps_lock);
        s_gps_fix.lat      = lat;
        s_gps_fix.lon      = lon;
        s_gps_fix.alt      = alt;
        s_gps_fix.sats     = sats;
        s_gps_fix.has_fix  = (fix > 0);
        s_gps_fix.last_update_tick = xTaskGetTickCount();
        portEXIT_CRITICAL(&s_gps_lock);

        if (fix > 0) {
            ESP_LOGI(TAG, "GGA Fix=%d Sats=%d Lat=%.6f Lon=%.6f Alt=%.1f",
                     fix, sats, lat, lon, alt);
        }
    }

    if (is_rmc && count >= 9) {
        /* RMC: $xxRMC,time,status,lat,N/S,lon,E/W,speed_kn,course,date,...
         *        0     1    2      3   4   5   6   7         8      9      */
        char status = tokens[2][0];
        if (status != 'A') return; // 'A' = active, 'V' = void

        float speed_knots = (float)atof(tokens[7]);
        float course_deg  = (strlen(tokens[8]) > 0) ? (float)atof(tokens[8]) : NAN;

        // Convert knots to m/s (1 knot = 0.514444 m/s)
        float speed_mps = speed_knots * 0.514444f;

        portENTER_CRITICAL(&s_gps_lock);
        s_gps_fix.speed_mps = speed_mps;
        // Only update heading if we have a valid course and some speed
        // (course is unreliable when stationary)
        if (!std::isnan(course_deg) && speed_mps > 0.3f) {
            s_gps_fix.heading_deg = course_deg;
        }
        s_gps_fix.last_update_tick = xTaskGetTickCount();
        portEXIT_CRITICAL(&s_gps_lock);
    }
}

/* -------------------------------------------------------------------------- */
/*  USB Host CDC logic (merged from esp32_s3_usb_gps_host)                     */
/* -------------------------------------------------------------------------- */

static constexpr int MAX_CDC_DEVICES = 3;
static cdc_acm_dev_hdl_t s_cdc_devices[MAX_CDC_DEVICES] = {};

static QueueHandle_t s_app_queue = nullptr;

enum AppEventId { APP_QUIT, APP_DEVICE_CONNECTED, APP_DEVICE_DISCONNECTED };

struct AppMessage {
    AppEventId id;
    union {
        struct { uint16_t vid; uint16_t pid; } new_dev;
        int device_slot;
    } data;
};

/// Line buffer for assembling NMEA sentences from USB RX fragments.
static char s_line_buf[256];
static int  s_line_pos = 0;

static int find_free_slot() {
    for (int i = 0; i < MAX_CDC_DEVICES; i++) {
        if (s_cdc_devices[i] == nullptr) return i;
    }
    return -1;
}

/// USB RX callback — accumulates bytes into NMEA lines and parses them.
static bool handle_rx(const uint8_t *data, size_t data_len, void *arg) {
    for (size_t i = 0; i < data_len; i++) {
        char c = (char)data[i];
        if (c == '$') {
            s_line_pos = 0;
            s_line_buf[s_line_pos++] = c;
        } else if (s_line_pos > 0) {
            if (c == '\r' || c == '\n' || s_line_pos >= 255) {
                s_line_buf[s_line_pos] = '\0';
                if (s_line_pos > 10) {
                    parse_nmea(s_line_buf);
                }
                s_line_pos = 0;
            } else {
                s_line_buf[s_line_pos++] = c;
            }
        }
    }
    return true;
}

static void handle_event(const cdc_acm_host_dev_event_data_t *event, void *user_ctx) {
    switch (event->type) {
        case CDC_ACM_HOST_ERROR:
            ESP_LOGE(TAG, "CDC-ACM error: %i", event->data.error);
            break;
        case CDC_ACM_HOST_DEVICE_DISCONNECTED:
            if (s_app_queue) {
                AppMessage msg = {};
                msg.id = APP_DEVICE_DISCONNECTED;
                msg.data.device_slot = (int)(intptr_t)user_ctx;
                xQueueSend(s_app_queue, &msg, 0);
            } else {
                cdc_acm_host_close(event->data.cdc_hdl);
            }
            break;
        case CDC_ACM_HOST_SERIAL_STATE:
            break;
        default:
            break;
    }
}

static void new_dev_cb(usb_device_handle_t usb_dev) {
    const usb_device_desc_t *desc;
    if (usb_host_get_device_descriptor(usb_dev, &desc) != ESP_OK) return;

    ESP_LOGI(TAG, "USB Device: VID=0x%04X PID=0x%04X", desc->idVendor, desc->idProduct);

    if (s_app_queue) {
        AppMessage msg = {};
        msg.id = APP_DEVICE_CONNECTED;
        msg.data.new_dev.vid = desc->idVendor;
        msg.data.new_dev.pid = desc->idProduct;
        xQueueSend(s_app_queue, &msg, 0);
    }
}

static cdc_acm_dev_hdl_t open_cdc_device(uint16_t vid, uint16_t pid,
                                          const cdc_acm_host_device_config_t *cfg) {
    cdc_acm_dev_hdl_t dev = nullptr;
    esp_err_t err;

    switch (vid) {
        case FTDI_VID:
            err = ftdi_vcp_open(pid, 0, cfg, &dev);
            break;
        case NANJING_QINHENG_MICROE_VID:
            err = ch34x_vcp_open(pid, 0, cfg, &dev);
            break;
        case SILICON_LABS_VID:
            err = cp210x_vcp_open(pid, 0, cfg, &dev);
            break;
        default:
            err = cdc_acm_host_open(vid, pid, 0, cfg, &dev);
            break;
    }

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open CDC VID=0x%04X PID=0x%04X", vid, pid);
        return nullptr;
    }
    return dev;
}

static void configure_gps_usb(int slot) {
    cdc_acm_dev_hdl_t dev = s_cdc_devices[slot];
    cdc_acm_host_set_control_line_state(dev, true, true);
    vTaskDelay(pdMS_TO_TICKS(100));

    cdc_acm_line_coding_t lc = {};
    esp_err_t err = cdc_acm_host_line_coding_get(dev, &lc);
    lc.dwDTERate   = GPS_BAUD_RATE;
    lc.bDataBits   = 8;
    lc.bParityType = 0;
    lc.bCharFormat = 0;
    err = cdc_acm_host_line_coding_set(dev, &lc);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "GPS USB baud set to %d", GPS_BAUD_RATE);
    } else {
        ESP_LOGW(TAG, "GPS USB baud set attempt (may be unsupported): %s", esp_err_to_name(err));
    }
    ESP_LOGI(TAG, "GPS USB ready, waiting for NMEA...");
}

static void free_cdc_device(int slot) {
    if (slot < 0 || slot >= MAX_CDC_DEVICES || s_cdc_devices[slot] == nullptr) return;
    cdc_acm_host_close(s_cdc_devices[slot]);
    s_cdc_devices[slot] = nullptr;
}

/* -------------------------------------------------------------------------- */
/*  USB Host library task + GPS event task                                     */
/* -------------------------------------------------------------------------- */

/// Low-level USB Host library pump task.
static void usb_lib_task(void *arg) {
    const usb_host_config_t host_config = {
        .skip_phy_setup = false,
        .intr_flags = ESP_INTR_FLAG_LOWMED,
    };
    ESP_ERROR_CHECK(usb_host_install(&host_config));

    const cdc_acm_host_driver_config_t driver_config = {
        .driver_task_stack_size = 4096,
        .driver_task_priority   = 20,
        .xCoreID = 0,
        .new_dev_cb = new_dev_cb,
    };
    ESP_ERROR_CHECK(cdc_acm_host_install(&driver_config));
    xTaskNotifyGive((TaskHandle_t)arg);

    while (true) {
        uint32_t event_flags;
        usb_host_lib_handle_events(portMAX_DELAY, &event_flags);
        // We never uninstall, so just keep pumping events.
    }
}

/// GPS connection management task.
static void gps_event_task(void *arg) {
    cdc_acm_host_device_config_t dev_config = {
        .connection_timeout_ms = 0,
        .out_buffer_size = 512,
        .in_buffer_size  = 2048,
        .user_arg  = nullptr,
        .event_cb  = handle_event,
        .data_cb   = handle_rx,
    };

    while (true) {
        AppMessage msg;
        if (xQueueReceive(s_app_queue, &msg, portMAX_DELAY) != pdPASS) continue;

        switch (msg.id) {
            case APP_DEVICE_CONNECTED: {
                int slot = find_free_slot();
                if (slot < 0) continue;
                dev_config.user_arg = (void *)(intptr_t)slot;
                cdc_acm_dev_hdl_t dev = open_cdc_device(
                    msg.data.new_dev.vid, msg.data.new_dev.pid, &dev_config);
                if (!dev) continue;
                s_cdc_devices[slot] = dev;
                configure_gps_usb(slot);
                break;
            }
            case APP_DEVICE_DISCONNECTED:
                ESP_LOGW(TAG, "GPS USB disconnected (slot %d)", msg.data.device_slot);
                free_cdc_device(msg.data.device_slot);
                // Mark fix as lost
                portENTER_CRITICAL(&s_gps_lock);
                s_gps_fix.has_fix = false;
                portEXIT_CRITICAL(&s_gps_lock);
                break;
            case APP_QUIT:
                break;
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  Public init                                                                */
/* -------------------------------------------------------------------------- */

void init_gps_usb(void) {
    s_app_queue = xQueueCreate(10, sizeof(AppMessage));

    // Start USB Host library pump task
    TaskHandle_t current = xTaskGetCurrentTaskHandle();
    xTaskCreate(usb_lib_task, "usb_lib", 4096, (void *)current, 20, nullptr);
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    // Start GPS event management task
    xTaskCreate(gps_event_task, "gps_evt", 4096, nullptr, 10, nullptr);

    ESP_LOGI(TAG, "GPS USB Host initialized, waiting for device...");
}
