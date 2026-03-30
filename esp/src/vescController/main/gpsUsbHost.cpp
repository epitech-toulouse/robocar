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
      latestFix{} {
}

UsbGpsHost::~UsbGpsHost() {
    stop();
}

esp_err_t UsbGpsHost::start() {
    if (running) {
        return ESP_OK;
    }
    if (s_instance != nullptr) {
        return ESP_ERR_INVALID_STATE;
    }

    fixMutex = xSemaphoreCreateMutex();
    if (fixMutex == nullptr) {
        return ESP_ERR_NO_MEM;
    }

    appQueue = xQueueCreate(10, sizeof(AppMessage));
    if (appQueue == nullptr) {
        vSemaphoreDelete(fixMutex);
        fixMutex = nullptr;
        return ESP_ERR_NO_MEM;
    }

    s_instance = this;
    startNotifyTarget = xTaskGetCurrentTaskHandle();

    BaseType_t usbTaskCreated = xTaskCreate(
        &UsbGpsHost::usbLibTask,
        "gps_usb_lib",
        4096,
        this,
        EXAMPLE_USB_HOST_PRIORITY,
        &usbTaskHandle);
    if (usbTaskCreated != pdTRUE) {
        s_instance = nullptr;
        vQueueDelete(appQueue);
        appQueue = nullptr;
        vSemaphoreDelete(fixMutex);
        fixMutex = nullptr;
        return ESP_FAIL;
    }

    // Wait until USB host stack is installed before creating the app task.
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    BaseType_t appTaskCreated = xTaskCreate(
        &UsbGpsHost::appTask,
        "gps_app",
        4096,
        this,
        EXAMPLE_USB_HOST_PRIORITY,
        &appTaskHandle);
    if (appTaskCreated != pdTRUE) {
        stop();
        return ESP_FAIL;
    }

    running = true;
    ESP_LOGI(TAG, "USB GPS host started");
    return ESP_OK;
}

void UsbGpsHost::stop() {
    if (!running && appQueue == nullptr && fixMutex == nullptr) {
        return;
    }

    if (appQueue != nullptr) {
        AppMessage quitMsg = {.id = AppMessage::Quit};
        xQueueSend(appQueue, &quitMsg, 0);
    }

    if (appQueue != nullptr) {
        vQueueDelete(appQueue);
        appQueue = nullptr;
    }

    if (fixMutex != nullptr) {
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

    ESP_LOGI(TAG, "USB GPS host stopped");
}

bool UsbGpsHost::isRunning() const {
    return running;
}

GpsFix UsbGpsHost::getLatestFix() const {
    GpsFix copy = latestFix;
    if (fixMutex == nullptr) {
        return copy;
    }

    if (xSemaphoreTake(fixMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        copy = latestFix;
        xSemaphoreGive(fixMutex);
    }
    return copy;
}

void UsbGpsHost::usbLibTask(void *arg) {
    auto *self = static_cast<UsbGpsHost *>(arg);

    const usb_host_config_t hostConfig = {
        .skip_phy_setup = false,
        .intr_flags = ESP_INTR_FLAG_LOWMED,
    };
    ESP_ERROR_CHECK(usb_host_install(&hostConfig));

    const cdc_acm_host_driver_config_t driverConfig = {
        .driver_task_stack_size = 4096,
        .driver_task_priority = EXAMPLE_USB_HOST_PRIORITY + 1,
        .xCoreID = 0,
        .new_dev_cb = &UsbGpsHost::newDeviceCallback,
    };
    ESP_ERROR_CHECK(cdc_acm_host_install(&driverConfig));

    if (self->startNotifyTarget != nullptr) {
        xTaskNotifyGive(self->startNotifyTarget);
    }

    bool hasClients = true;
    while (true) {
        uint32_t eventFlags = 0;
        usb_host_lib_handle_events(portMAX_DELAY, &eventFlags);

        if (eventFlags & USB_HOST_LIB_EVENT_FLAGS_NO_CLIENTS) {
            hasClients = false;
            if (usb_host_device_free_all() == ESP_OK) {
                break;
            }
        }

        if ((eventFlags & USB_HOST_LIB_EVENT_FLAGS_ALL_FREE) && !hasClients) {
            break;
        }
    }

    ESP_ERROR_CHECK(cdc_acm_host_uninstall());
    ESP_ERROR_CHECK(usb_host_uninstall());

    vTaskDelete(nullptr);
}

void UsbGpsHost::appTask(void *arg) {
    auto *self = static_cast<UsbGpsHost *>(arg);

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
            continue;
        }

        switch (msg.id) {
            case AppMessage::DeviceConnected: {
                const int slot = self->findFreeSlot();
                if (slot < 0) {
                    ESP_LOGW(TAG, "No free CDC slots");
                    break;
                }

                devConfig.user_arg = reinterpret_cast<void *>(static_cast<intptr_t>(slot));
                cdc_acm_dev_hdl_t cdcDev = self->openCdcDevice(msg.data.newDevice.vid, msg.data.newDevice.pid, &devConfig);
                if (cdcDev == nullptr) {
                    break;
                }

                self->cdcDevices[slot] = cdcDev;
                self->configureGpsUsb(slot);
                break;
            }

            case AppMessage::DeviceDisconnected:
                self->freeCdcDevice(msg.data.deviceSlot);
                break;

            case AppMessage::Quit:
                self->freeAllCdcDevices();
                vTaskDelete(nullptr);
                return;
        }
    }
}

void UsbGpsHost::newDeviceCallback(usb_device_handle_t usb_dev) {
    if (s_instance == nullptr || s_instance->appQueue == nullptr) {
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
    xQueueSend(s_instance->appQueue, &msg, 0);
}

void UsbGpsHost::eventCallback(const cdc_acm_host_dev_event_data_t *event, void *user_ctx) {
    if (s_instance == nullptr) {
        return;
    }

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
                xQueueSend(s_instance->appQueue, &msg, 0);
            } else {
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
    (void)arg;
    if (s_instance == nullptr) {
        return true;
    }
    s_instance->processIncomingBytes(data, data_len);
    return true;
}

int UsbGpsHost::findFreeSlot() const {
    for (int i = 0; i < MAX_CDC_DEVICES; ++i) {
        if (cdcDevices[i] == nullptr) {
            return i;
        }
    }
    return -1;
}

cdc_acm_dev_hdl_t UsbGpsHost::openCdcDevice(uint16_t vid, uint16_t pid, const cdc_acm_host_device_config_t *dev_config) {
    cdc_acm_dev_hdl_t cdcDev = nullptr;
    esp_err_t err = ESP_FAIL;

    switch (vid) {
        case FTDI_VID:
            err = ftdi_vcp_open(pid, 0, dev_config, &cdcDev);
            break;
        case NANJING_QINHENG_MICROE_VID:
            err = ch34x_vcp_open(pid, 0, dev_config, &cdcDev);
            break;
        case SILICON_LABS_VID:
            err = cp210x_vcp_open(pid, 0, dev_config, &cdcDev);
            break;
        default:
            err = cdc_acm_host_open(vid, pid, 0, dev_config, &cdcDev);
            break;
    }

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open CDC VID=0x%04X PID=0x%04X", vid, pid);
        return nullptr;
    }

    return cdcDev;
}

void UsbGpsHost::configureGpsUsb(int slot) {
    if (slot < 0 || slot >= MAX_CDC_DEVICES || cdcDevices[slot] == nullptr) {
        return;
    }

    cdc_acm_dev_hdl_t cdcDev = cdcDevices[slot];

    ESP_LOGI(TAG, "Configuring GPS baudrate to %" PRIu32, GPS_UART_BAUDRATE);

    cdc_acm_host_set_control_line_state(cdcDev, true, true);
    vTaskDelay(pdMS_TO_TICKS(100));

    cdc_acm_line_coding_t lineCoding = {
        .dwDTERate = GPS_UART_BAUDRATE,
        .bCharFormat = 0,
        .bParityType = 0,
        .bDataBits = 8,
    };

    esp_err_t err = cdc_acm_host_line_coding_get(cdcDev, &lineCoding);
    if (err == ESP_OK) {
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
        return;
    }

    cdc_acm_host_close(cdcDevices[slot]);
    cdcDevices[slot] = nullptr;
    ESP_LOGW(TAG, "GPS CDC device in slot %d disconnected", slot);
}

void UsbGpsHost::freeAllCdcDevices() {
    for (int i = 0; i < MAX_CDC_DEVICES; ++i) {
        if (cdcDevices[i] != nullptr) {
            freeCdcDevice(i);
        }
    }
}

float UsbGpsHost::nmeaToDecimal(const char *nmea_coord, char dir) const {
    if (nmea_coord == nullptr || std::strlen(nmea_coord) < 4) {
        return 0.0f;
    }

    const char *dot = std::strchr(nmea_coord, '.');
    if (dot == nullptr) {
        return 0.0f;
    }

    const int minutesDigits = 2;
    const int degLen = static_cast<int>((dot - nmea_coord) - minutesDigits);
    if (degLen <= 0 || degLen > 3) {
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
    return decimal;
}

void UsbGpsHost::parseNmeaLine(char *line) {
    if (line == nullptr) {
        return;
    }

    if (std::strncmp(line, "$GNGGA", 6) != 0 && std::strncmp(line, "$GPGGA", 6) != 0) {
        return;
    }

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
        return;
    }

    GpsFix nextFix;
    nextFix.latitude = static_cast<double>(nmeaToDecimal(tokens[2], tokens[3][0]));
    nextFix.longitude = static_cast<double>(nmeaToDecimal(tokens[4], tokens[5][0]));
    nextFix.hasFix = (std::atoi(tokens[6]) > 0);
    nextFix.satellites = std::atoi(tokens[7]);
    nextFix.altitudeMeters = static_cast<float>(std::atof(tokens[9]));

    if (fixMutex != nullptr && xSemaphoreTake(fixMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        latestFix = nextFix;
        xSemaphoreGive(fixMutex);
    }
}

void UsbGpsHost::processIncomingBytes(const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        const char c = static_cast<char>(data[i]);

        if (c == '$') {
            linePos = 0;
            lineBuffer[linePos++] = c;
            continue;
        }

        if (linePos <= 0) {
            continue;
        }

        if (c == '\r' || c == '\n' || linePos >= static_cast<int>(sizeof(lineBuffer) - 1)) {
            lineBuffer[linePos] = '\0';
            if (linePos > 10) {
                parseNmeaLine(lineBuffer);
            }
            linePos = 0;
            continue;
        }

        lineBuffer[linePos++] = c;
    }
}
