#pragma once

#include <array>
#include <cstdint>
#include <queue>

#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "usb/cdc_acm_host.h"
#include "usb/usb_host.h"

struct GpsFix {
    bool hasFix = false;
    int satellites = 0;
    int fixQuality = 0;
    bool isRtkFloat = false;
    bool isRtkFixed = false;
    double latitude = 0.0;
    double longitude = 0.0;
    float altitudeMeters = 0.0f;
    uint32_t updateCounter = 0;
    TickType_t updateTick = 0;
};

class UsbGpsHost {
public:
    UsbGpsHost();
    ~UsbGpsHost();

    esp_err_t start();
    void stop();

    bool isRunning() const;
    GpsFix getLatestFix() const;
    std::array<GpsFix, 20> getFixArray() {
        if (fixMutex == nullptr || xSemaphoreTake(this->fixMutex, pdMS_TO_TICKS(5)) != pdTRUE)
            return {};
        std::array<GpsFix, 20> array = this->fixArray;
        xSemaphoreGive(this->fixMutex);
        return array;
    }

private:
    struct AppMessage;

    static constexpr int EXAMPLE_USB_HOST_PRIORITY = 20;
    static constexpr int MAX_CDC_DEVICES = 5;

    static void usbLibTask(void *arg);
    static void appTask(void *arg);

    static void newDeviceCallback(usb_device_handle_t usb_dev);
    static void eventCallback(const cdc_acm_host_dev_event_data_t *event, void *user_ctx);
    static bool dataCallback(const uint8_t *data, size_t data_len, void *arg);

    int findFreeSlot() const;
    cdc_acm_dev_hdl_t openCdcDevice(uint16_t vid, uint16_t pid, const cdc_acm_host_device_config_t *dev_config);
    void configureGpsUsb(int slot);
    void freeCdcDevice(int slot);
    void freeAllCdcDevices();

    float nmeaToDecimal(const char *nmea_coord, char dir) const;
    void parseNmeaLine(char *line);
    void processIncomingBytes(const uint8_t *data, size_t len);

    static UsbGpsHost *s_instance;

    QueueHandle_t appQueue;
    TaskHandle_t usbTaskHandle;
    TaskHandle_t appTaskHandle;
    SemaphoreHandle_t fixMutex;
    TaskHandle_t startNotifyTarget;

    cdc_acm_dev_hdl_t cdcDevices[MAX_CDC_DEVICES];

    char lineBuffer[512];
    int linePos;

    bool running;
    uint32_t fixUpdateCounter;
    GpsFix latestFix;
    std::array<GpsFix, 20> fixArray;
};
