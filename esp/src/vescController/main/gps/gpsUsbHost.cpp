#include "gpsUsbHost.hpp"

#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "esp_log.h"
#include "esp_intr_alloc.h"
#include "config.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "usb/cdc_acm_host.h"
#include "usb/cdc_host_types.h"
#include "usb/usb_host.h"
#include "usb/vcp_ch34x.h"
#include "usb/vcp_cp210x.h"
#include "usb/vcp_ftdi.h"

struct UsbGpsHost::AppMessage {
    enum Id {
        Quit,
        DeviceConnected,
        DeviceDisconnected,
    } id;

    union {
        struct {
            uint16_t vid;
            uint16_t pid;
        } newDevice;
        int deviceSlot;
    } data;
};

namespace {
const char *TAG = "UsbGpsHost";
}

UsbGpsHost *UsbGpsHost::s_instance = nullptr;

UsbGpsHost::UsbGpsHost()
    : appQueue(nullptr),
      usbTaskHandle(nullptr),
      appTaskHandle(nullptr),
      fixMutex(nullptr),
            startNotifyTarget(nullptr),
      cdcDevices{nullptr},
      lineBuffer{0},
      linePos(0),
      running(false),
    fixUpdateCounter(0),
      latestFix{} {
}

UsbGpsHost::~UsbGpsHost() {
    stop();
}

esp_err_t UsbGpsHost::start() {
    ESP_LOGD(TAG, "start() called running=%d s_instance=%p", running, static_cast<void *>(s_instance));
    if (running) {
        ESP_LOGW(TAG, "start() ignored because host is already running");
        return ESP_OK;
    }
    if (s_instance != nullptr) {
        ESP_LOGE(TAG, "start() rejected because another UsbGpsHost instance is active (%p)", static_cast<void *>(s_instance));
        return ESP_ERR_INVALID_STATE;
    }

    fixMutex = xSemaphoreCreateMutex();
    if (fixMutex == nullptr) {
        ESP_LOGE(TAG, "Failed to create fix mutex");
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGD(TAG, "Created fix mutex");

    appQueue = xQueueCreate(10, sizeof(AppMessage));
    if (appQueue == nullptr) {
        ESP_LOGE(TAG, "Failed to create app queue");
        vSemaphoreDelete(fixMutex);
        fixMutex = nullptr;
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGD(TAG, "Created app queue");

    s_instance = this;
    startNotifyTarget = xTaskGetCurrentTaskHandle();
    ESP_LOGD(TAG, "Registered singleton instance=%p startNotifyTarget=%p", static_cast<void *>(this), static_cast<void *>(startNotifyTarget));

    BaseType_t usbTaskCreated = xTaskCreate(
        &UsbGpsHost::usbLibTask,
        "gps_usb_lib",
        4096,
        this,
        EXAMPLE_USB_HOST_PRIORITY,
        &usbTaskHandle);
    if (usbTaskCreated != pdTRUE) {
        ESP_LOGE(TAG, "Failed to create usbLibTask");
        s_instance = nullptr;
        vQueueDelete(appQueue);
        appQueue = nullptr;
        vSemaphoreDelete(fixMutex);
        fixMutex = nullptr;
        return ESP_FAIL;
    }
    ESP_LOGD(TAG, "usbLibTask created handle=%p", static_cast<void *>(usbTaskHandle));

    // Wait until USB host stack is installed before creating the app task.
    ESP_LOGD(TAG, "Waiting for usbLibTask initialization notification");
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    ESP_LOGD(TAG, "Received usbLibTask initialization notification");

    BaseType_t appTaskCreated = xTaskCreate(
        &UsbGpsHost::appTask,
        "gps_app",
        4096,
        this,
        EXAMPLE_USB_HOST_PRIORITY,
        &appTaskHandle);
    if (appTaskCreated != pdTRUE) {
        ESP_LOGE(TAG, "Failed to create appTask");
        stop();
        return ESP_FAIL;
    }
    ESP_LOGD(TAG, "appTask created handle=%p", static_cast<void *>(appTaskHandle));

    running = true;
    ESP_LOGI(TAG, "USB GPS host started");
    return ESP_OK;
}

void UsbGpsHost::stop() {
    ESP_LOGD(TAG, "stop() called running=%d appQueue=%p fixMutex=%p", running, static_cast<void *>(appQueue), static_cast<void *>(fixMutex));
    if (!running && appQueue == nullptr && fixMutex == nullptr) {
        ESP_LOGD(TAG, "stop() no-op because host is already fully stopped");
        return;
    }

    if (appQueue != nullptr) {
        AppMessage quitMsg{};
        quitMsg.id = AppMessage::Quit;
        const BaseType_t sent = xQueueSend(appQueue, &quitMsg, 0);
        ESP_LOGD(TAG, "Queued Quit message result=%ld", static_cast<long>(sent));
    }

    if (appQueue != nullptr) {
        ESP_LOGD(TAG, "Deleting app queue");
        vQueueDelete(appQueue);
        appQueue = nullptr;
    }

    if (fixMutex != nullptr) {
        ESP_LOGD(TAG, "Deleting fix mutex");
        vSemaphoreDelete(fixMutex);
        fixMutex = nullptr;
    }

    s_instance = nullptr;
    running = false;
    usbTaskHandle = nullptr;
    appTaskHandle = nullptr;
    startNotifyTarget = nullptr;
    linePos = 0;
    std::memset(lineBuffer, 0, sizeof(lineBuffer));
    latestFix = GpsFix{};
    fixUpdateCounter = 0;

    ESP_LOGI(TAG, "USB GPS host stopped");
}

bool UsbGpsHost::isRunning() const {
    return running;
}

GpsFix UsbGpsHost::getLatestFix() const {
    GpsFix copy = latestFix;
    if (fixMutex == nullptr) {
        ESP_LOGD(TAG, "getLatestFix() returning without mutex protection");
        return copy;
    }

    if (xSemaphoreTake(fixMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        copy = latestFix;
        xSemaphoreGive(fixMutex);
    } else {
        ESP_LOGW(TAG, "getLatestFix() timed out waiting for mutex");
    }

    ESP_LOGD(TAG,
             "getLatestFix() hasFix=%d sats=%d lat=%.6f lon=%.6f alt=%.2f",
             copy.hasFix,
             copy.satellites,
             copy.latitude,
             copy.longitude,
             static_cast<double>(copy.altitudeMeters));
    return copy;
}

void UsbGpsHost::usbLibTask(void *arg) {
    auto *self = static_cast<UsbGpsHost *>(arg);
    ESP_LOGD(TAG, "usbLibTask started self=%p", static_cast<void *>(self));

    usb_host_config_t hostConfig{};
    hostConfig.skip_phy_setup = false;
    hostConfig.intr_flags = ESP_INTR_FLAG_LOWMED;
    ESP_LOGD(TAG, "Installing USB host library");
    ESP_ERROR_CHECK(usb_host_install(&hostConfig));
    ESP_LOGD(TAG, "USB host library installed");

    const cdc_acm_host_driver_config_t driverConfig = {
        .driver_task_stack_size = 4096,
        .driver_task_priority = EXAMPLE_USB_HOST_PRIORITY + 1,
        .xCoreID = 0,
        .new_dev_cb = &UsbGpsHost::newDeviceCallback,
    };
    ESP_LOGD(TAG, "Installing CDC ACM host driver");
    ESP_ERROR_CHECK(cdc_acm_host_install(&driverConfig));
    ESP_LOGD(TAG, "CDC ACM host driver installed");

    if (self->startNotifyTarget != nullptr) {
        ESP_LOGD(TAG, "Notifying starter task %p", static_cast<void *>(self->startNotifyTarget));
        xTaskNotifyGive(self->startNotifyTarget);
    }

    bool hasClients = true;
    while (true) {
        uint32_t eventFlags = 0;
        usb_host_lib_handle_events(portMAX_DELAY, &eventFlags);
        ESP_LOGD(TAG, "usb_host_lib_handle_events flags=0x%08" PRIX32, eventFlags);

        if (eventFlags & USB_HOST_LIB_EVENT_FLAGS_NO_CLIENTS) {
            hasClients = false;
            ESP_LOGW(TAG, "USB host reports no clients, attempting to free all devices");
            if (usb_host_device_free_all() == ESP_OK) {
                ESP_LOGD(TAG, "All USB devices freed");
                break;
            }
            ESP_LOGW(TAG, "usb_host_device_free_all() did not return ESP_OK yet");
        }

        if ((eventFlags & USB_HOST_LIB_EVENT_FLAGS_ALL_FREE) && !hasClients) {
            ESP_LOGD(TAG, "USB host reports all resources free");
            break;
        }
    }

    ESP_LOGD(TAG, "Uninstalling CDC ACM host driver");
    ESP_ERROR_CHECK(cdc_acm_host_uninstall());
    ESP_LOGD(TAG, "Uninstalling USB host library");
    ESP_ERROR_CHECK(usb_host_uninstall());
    ESP_LOGD(TAG, "usbLibTask exiting");

    vTaskDelete(nullptr);
}

void UsbGpsHost::appTask(void *arg) {
    auto *self = static_cast<UsbGpsHost *>(arg);
    ESP_LOGD(TAG, "appTask started self=%p", static_cast<void *>(self));

    cdc_acm_host_device_config_t devConfig = {};
    devConfig.connection_timeout_ms = 0;
    devConfig.out_buffer_size = 512;
    devConfig.in_buffer_size = 2048;
    devConfig.user_arg = nullptr;
    devConfig.event_cb = &UsbGpsHost::eventCallback;
    devConfig.data_cb = &UsbGpsHost::dataCallback;

    while (true) {
        AppMessage msg;
        if (xQueueReceive(self->appQueue, &msg, portMAX_DELAY) != pdTRUE) {
            ESP_LOGW(TAG, "xQueueReceive failed in appTask");
            continue;
        }
        ESP_LOGD(TAG, "appTask received msg id=%d", msg.id);

        switch (msg.id) {
            case AppMessage::DeviceConnected: {
                ESP_LOGI(TAG,
                         "Handling DeviceConnected VID=0x%04X PID=0x%04X",
                         msg.data.newDevice.vid,
                         msg.data.newDevice.pid);
                const int slot = self->findFreeSlot();
                if (slot < 0) {
                    ESP_LOGW(TAG, "No free CDC slots");
                    break;
                }
                ESP_LOGD(TAG, "Using CDC slot %d", slot);

                devConfig.user_arg = reinterpret_cast<void *>(static_cast<intptr_t>(slot));
                cdc_acm_dev_hdl_t cdcDev = self->openCdcDevice(msg.data.newDevice.vid, msg.data.newDevice.pid, &devConfig);
                if (cdcDev == nullptr) {
                    ESP_LOGW(TAG,
                             "openCdcDevice failed for VID=0x%04X PID=0x%04X",
                             msg.data.newDevice.vid,
                             msg.data.newDevice.pid);
                    break;
                }

                self->cdcDevices[slot] = cdcDev;
                ESP_LOGI(TAG, "CDC device opened in slot %d handle=%p", slot, cdcDev);
                self->configureGpsUsb(slot);
                break;
            }

            case AppMessage::DeviceDisconnected:
                ESP_LOGI(TAG, "Handling DeviceDisconnected slot=%d", msg.data.deviceSlot);
                self->freeCdcDevice(msg.data.deviceSlot);
                break;

            case AppMessage::Quit:
                ESP_LOGI(TAG, "Handling Quit message");
                self->freeAllCdcDevices();
                ESP_LOGD(TAG, "appTask exiting");
                vTaskDelete(nullptr);
                return;
        }
    }
}

void UsbGpsHost::newDeviceCallback(usb_device_handle_t usb_dev) {
    ESP_LOGD(TAG, "newDeviceCallback usb_dev=%p", usb_dev);
    if (s_instance == nullptr || s_instance->appQueue == nullptr) {
        ESP_LOGW(TAG, "Ignoring new device because instance/queue is not ready");
        return;
    }

    const usb_device_desc_t *deviceDesc = nullptr;
    esp_err_t err = usb_host_get_device_descriptor(usb_dev, &deviceDesc);
    if (err != ESP_OK || deviceDesc == nullptr) {
        ESP_LOGE(TAG, "usb_host_get_device_descriptor failed: %s", esp_err_to_name(err));
        return;
    }

    const uint16_t vid = deviceDesc->idVendor;
    const uint16_t pid = deviceDesc->idProduct;
    ESP_LOGI(TAG, "USB device connected VID=0x%04X PID=0x%04X", vid, pid);

    AppMessage msg = {
        .id = AppMessage::DeviceConnected,
        .data = {.newDevice = {.vid = vid, .pid = pid}},
    };
    const BaseType_t sent = xQueueSend(s_instance->appQueue, &msg, 0);
    if (sent != pdTRUE) {
        ESP_LOGW(TAG, "Failed to queue DeviceConnected message");
    }
}

void UsbGpsHost::eventCallback(const cdc_acm_host_dev_event_data_t *event, void *user_ctx) {
    if (event == nullptr) {
        ESP_LOGW(TAG, "eventCallback received null event");
        return;
    }

    if (s_instance == nullptr) {
        ESP_LOGW(TAG, "eventCallback ignored because instance is null");
        return;
    }

    ESP_LOGD(TAG, "eventCallback type=%d user_ctx=%p", event->type, user_ctx);

    switch (event->type) {
        case CDC_ACM_HOST_ERROR:
            ESP_LOGE(TAG, "CDC-ACM error err_no=%d", event->data.error);
            break;

        case CDC_ACM_HOST_DEVICE_DISCONNECTED: {
            if (s_instance->appQueue != nullptr) {
                AppMessage msg = {
                    .id = AppMessage::DeviceDisconnected,
                    .data = {.deviceSlot = static_cast<int>(reinterpret_cast<intptr_t>(user_ctx))},
                };
                const BaseType_t sent = xQueueSend(s_instance->appQueue, &msg, 0);
                if (sent != pdTRUE) {
                    ESP_LOGW(TAG, "Failed to queue DeviceDisconnected for slot %d", msg.data.deviceSlot);
                }
            } else {
                ESP_LOGW(TAG, "App queue unavailable, closing CDC handle directly");
                ESP_ERROR_CHECK(cdc_acm_host_close(event->data.cdc_hdl));
            }
            break;
        }

        case CDC_ACM_HOST_SERIAL_STATE:
            ESP_LOGI(TAG, "Serial state notif 0x%04X", event->data.serial_state.val);
            break;

        default:
            break;
    }
}

bool UsbGpsHost::dataCallback(const uint8_t *data, size_t data_len, void *arg) {
    const int slot = static_cast<int>(reinterpret_cast<intptr_t>(arg));
    ESP_LOGD(TAG, "dataCallback slot=%d len=%u", slot, static_cast<unsigned>(data_len));

    if (s_instance == nullptr) {
        ESP_LOGW(TAG, "dataCallback ignored because instance is null");
        return true;
    }

    if (data == nullptr || data_len == 0) {
        ESP_LOGD(TAG, "dataCallback received empty payload");
        return true;
    }

    s_instance->processIncomingBytes(data, data_len);
    return true;
}

int UsbGpsHost::findFreeSlot() const {
    for (int i = 0; i < MAX_CDC_DEVICES; ++i) {
        if (cdcDevices[i] == nullptr) {
            ESP_LOGD(TAG, "findFreeSlot found slot %d", i);
            return i;
        }
    }
    ESP_LOGW(TAG, "findFreeSlot found no available slots");
    return -1;
}

cdc_acm_dev_hdl_t UsbGpsHost::openCdcDevice(uint16_t vid, uint16_t pid, const cdc_acm_host_device_config_t *dev_config) {
    cdc_acm_dev_hdl_t cdcDev = nullptr;
    esp_err_t err = ESP_FAIL;

    ESP_LOGI(TAG, "Opening CDC device VID=0x%04X PID=0x%04X", vid, pid);

    switch (vid) {
        case FTDI_VID:
            ESP_LOGD(TAG, "Using FTDI VCP open path");
            err = ftdi_vcp_open(pid, 0, dev_config, &cdcDev);
            break;
        case NANJING_QINHENG_MICROE_VID:
            ESP_LOGD(TAG, "Using CH34x VCP open path");
            err = ch34x_vcp_open(pid, 0, dev_config, &cdcDev);
            break;
        case SILICON_LABS_VID:
            ESP_LOGD(TAG, "Using CP210x VCP open path");
            err = cp210x_vcp_open(pid, 0, dev_config, &cdcDev);
            break;
        default:
            ESP_LOGD(TAG, "Using generic CDC ACM open path");
            err = cdc_acm_host_open(vid, pid, 0, dev_config, &cdcDev);
            break;
    }

    if (err != ESP_OK) {
        ESP_LOGE(TAG,
                 "Failed to open CDC VID=0x%04X PID=0x%04X err=%s",
                 vid,
                 pid,
                 esp_err_to_name(err));
        return nullptr;
    }

    ESP_LOGI(TAG, "CDC open success handle=%p", cdcDev);

    return cdcDev;
}

void UsbGpsHost::configureGpsUsb(int slot) {
    if (slot < 0 || slot >= MAX_CDC_DEVICES || cdcDevices[slot] == nullptr) {
        ESP_LOGW(TAG, "configureGpsUsb invalid slot=%d", slot);
        return;
    }

    cdc_acm_dev_hdl_t cdcDev = cdcDevices[slot];
    ESP_LOGD(TAG, "configureGpsUsb slot=%d handle=%p", slot, cdcDev);

    ESP_LOGI(TAG, "Configuring GPS baudrate to %" PRIu32, GPS_UART_BAUDRATE);

    esp_err_t ctrlErr = cdc_acm_host_set_control_line_state(cdcDev, true, true);
    if (ctrlErr != ESP_OK) {
        ESP_LOGW(TAG, "Failed to set control line state: %s", esp_err_to_name(ctrlErr));
    }
    vTaskDelay(pdMS_TO_TICKS(100));

    cdc_acm_line_coding_t lineCoding = {
        .dwDTERate = GPS_UART_BAUDRATE,
        .bCharFormat = 0,
        .bParityType = 0,
        .bDataBits = 8,
    };

    esp_err_t err = cdc_acm_host_line_coding_get(cdcDev, &lineCoding);
    if (err == ESP_OK) {
        ESP_LOGD(TAG,
                 "Current line coding before update: rate=%" PRIu32 " bits=%u parity=%u stop=%u",
                 lineCoding.dwDTERate,
                 lineCoding.bDataBits,
                 lineCoding.bParityType,
                 lineCoding.bCharFormat);
        lineCoding.dwDTERate = GPS_UART_BAUDRATE;
        lineCoding.bDataBits = 8;
        lineCoding.bParityType = 0;
        lineCoding.bCharFormat = 0;
        err = cdc_acm_host_line_coding_set(cdcDev, &lineCoding);
    } else {
        err = cdc_acm_host_line_coding_set(cdcDev, &lineCoding);
    }

    if (err == ESP_OK) {
        ESP_LOGI(TAG, "GPS line coding configured");
    } else {
        ESP_LOGW(TAG, "Failed to set GPS line coding: %s", esp_err_to_name(err));
    }
}

void UsbGpsHost::freeCdcDevice(int slot) {
    if (slot < 0 || slot >= MAX_CDC_DEVICES || cdcDevices[slot] == nullptr) {
        ESP_LOGD(TAG, "freeCdcDevice ignored for slot=%d", slot);
        return;
    }

    ESP_LOGD(TAG, "Closing CDC device in slot %d handle=%p", slot, cdcDevices[slot]);
    cdc_acm_host_close(cdcDevices[slot]);
    cdcDevices[slot] = nullptr;
    ESP_LOGW(TAG, "GPS CDC device in slot %d disconnected", slot);
}

void UsbGpsHost::freeAllCdcDevices() {
    ESP_LOGD(TAG, "freeAllCdcDevices() scanning %d slots", MAX_CDC_DEVICES);
    for (int i = 0; i < MAX_CDC_DEVICES; ++i) {
        if (cdcDevices[i] != nullptr) {
            freeCdcDevice(i);
        }
    }
}

float UsbGpsHost::nmeaToDecimal(const char *nmea_coord, char dir) const {
    if (nmea_coord == nullptr || std::strlen(nmea_coord) < 4) {
        ESP_LOGD(TAG, "nmeaToDecimal invalid coord input");
        return 0.0f;
    }

    const char *dot = std::strchr(nmea_coord, '.');
    if (dot == nullptr) {
        ESP_LOGD(TAG, "nmeaToDecimal missing dot in '%s'", nmea_coord);
        return 0.0f;
    }

    const int minutesDigits = 2;
    const int degLen = static_cast<int>((dot - nmea_coord) - minutesDigits);
    if (degLen <= 0 || degLen > 3) {
        ESP_LOGD(TAG, "nmeaToDecimal invalid degree length=%d coord='%s'", degLen, nmea_coord);
        return 0.0f;
    }

    char degStr[4] = {0};
    std::strncpy(degStr, nmea_coord, static_cast<size_t>(degLen));

    const float degrees = std::atof(degStr);
    const float minutes = std::atof(nmea_coord + degLen);

    float decimal = degrees + (minutes / 60.0f);
    if (dir == 'S' || dir == 'W') {
        decimal = -decimal;
    }
    ESP_LOGD(TAG, "nmeaToDecimal coord='%s' dir=%c -> %.6f", nmea_coord, dir, static_cast<double>(decimal));
    return decimal;
}

void UsbGpsHost::parseNmeaLine(char *line) {
    if (line == nullptr) {
        ESP_LOGD(TAG, "parseNmeaLine called with null line");
        return;
    }

    ESP_LOGD(TAG, "parseNmeaLine input='%s'", line);

    if (std::strncmp(line, "$GNGGA", 6) != 0 && std::strncmp(line, "$GPGGA", 6) != 0) {
        ESP_LOGD(TAG, "Ignoring non-GGA sentence");
        return;
    }

    ESP_LOGD(TAG, "GGA sentence received: %s", line);

    char *tokens[20] = {nullptr};
    int count = 0;

    tokens[count++] = line;
    for (int i = 0; line[i] != '\0'; ++i) {
        if (line[i] == ',' || line[i] == '*') {
            line[i] = '\0';
            if (count < 20) {
                tokens[count++] = &line[i + 1];
            }
        }
    }

    if (count < 10 || std::strlen(tokens[2]) == 0 || std::strlen(tokens[4]) == 0) {
        ESP_LOGD(TAG, "GGA sentence missing required fields count=%d: %s", count, line);
        return;
    }

    GpsFix nextFix;
    const int fixQuality = std::atoi(tokens[6]);
    nextFix.latitude = static_cast<double>(nmeaToDecimal(tokens[2], tokens[3][0]));
    nextFix.longitude = static_cast<double>(nmeaToDecimal(tokens[4], tokens[5][0]));
    nextFix.fixQuality = fixQuality;
    nextFix.hasFix = (fixQuality > 0);
    nextFix.isRtkFixed = (fixQuality == 4);
    nextFix.isRtkFloat = (fixQuality == 5);
    nextFix.satellites = std::atoi(tokens[7]);
    nextFix.altitudeMeters = static_cast<float>(std::atof(tokens[9]));
    nextFix.updateCounter = fixUpdateCounter + 1;
    nextFix.updateTick = xTaskGetTickCount();

    if (!nextFix.hasFix) {
        ESP_LOGW(TAG,
                 "Parsed GGA but no fix q=%d sats=%d lat=%.6f lon=%.6f",
                 nextFix.fixQuality,
                 nextFix.satellites,
                 nextFix.latitude,
                 nextFix.longitude);
    }

    ESP_LOGI(TAG,
             "Parsed fix hasFix=%d q=%d rtkFixed=%d rtkFloat=%d sats=%d lat=%.6f lon=%.6f alt=%.2f",
             nextFix.hasFix,
             nextFix.fixQuality,
             nextFix.isRtkFixed,
             nextFix.isRtkFloat,
             nextFix.satellites,
             nextFix.latitude,
             nextFix.longitude,
             static_cast<double>(nextFix.altitudeMeters));

    if (fixMutex != nullptr && xSemaphoreTake(fixMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        latestFix = nextFix;
        fixUpdateCounter = nextFix.updateCounter;
        xSemaphoreGive(fixMutex);
    } else if (fixMutex == nullptr) {
        ESP_LOGW(TAG, "Cannot store fix because mutex is null");
    } else {
        ESP_LOGW(TAG, "Timed out while storing parsed GPS fix");
    }
}

void UsbGpsHost::processIncomingBytes(const uint8_t *data, size_t len) {
    if (data == nullptr || len == 0) {
        ESP_LOGD(TAG, "processIncomingBytes called with empty payload");
        return;
    }

    ESP_LOGD(TAG, "processIncomingBytes len=%u", static_cast<unsigned>(len));

    for (size_t i = 0; i < len; ++i) {
        const char c = static_cast<char>(data[i]);

        if (c == '$') {
            linePos = 0;
            lineBuffer[linePos++] = c;
            ESP_LOGD(TAG, "Detected start of NMEA sentence");
            continue;
        }

        if (linePos <= 0) {
            continue;
        }

        if (c == '\r' || c == '\n' || linePos >= static_cast<int>(sizeof(lineBuffer) - 1)) {
            lineBuffer[linePos] = '\0';
            if (linePos > 10) {
                ESP_LOGD(TAG, "Completed NMEA sentence (%d chars): %s", linePos, lineBuffer);
                parseNmeaLine(lineBuffer);
            } else {
                ESP_LOGD(TAG, "Discarded short line (%d chars)", linePos);
            }
            if (linePos >= static_cast<int>(sizeof(lineBuffer) - 1)) {
                ESP_LOGD(TAG, "Line buffer reached capacity (%d) and was flushed", linePos);
            }
            linePos = 0;
            continue;
        }

        lineBuffer[linePos++] = c;
    }
}
