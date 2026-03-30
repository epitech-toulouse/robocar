#include <cstdint>
#include <stdio.h>
#include "config.h"
#include "driver/gpio.h"
#include "esp_attr.h"
#include "esp_intr_alloc.h"
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "freertos/task.h"
#include "hal/gpio_types.h"
#include "portmacro.h"
#include "vescController.hpp"
#include "lidarReader.hpp"
#include "gpsUsbHost.hpp"
#include <iostream>

#include "esp_err.h"
#include "bluetooth_receiver.hpp"
#include "vescLidarUart.h"

#include "drive.hpp"
#include "gpsdrive.hpp"

#include "esp_log.h"
#include <cmath>

static constexpr TickType_t LIDAR_NO_DATA_TIMEOUT_TICKS = pdMS_TO_TICKS(3000);
static constexpr TickType_t LIDAR_LOG_PERIOD_TICKS = pdMS_TO_TICKS(1000);
static constexpr TickType_t GPS_LOG_PERIOD_TICKS = pdMS_TO_TICKS(2000);
static constexpr TickType_t GPS_DRIVE_LOG_PERIOD_TICKS = pdMS_TO_TICKS(1000);
static constexpr float GPS_GOAL_ACCEPTANCE_RADIUS_M = 10.0f;
static constexpr float GPS_HEADING_MIN_MOVEMENT_M = 1.5f;

static TaskHandle_t vesc_control_task_handle = nullptr;

struct gps_goal_t {
    double lat;
    double lon;
    bool enabled;
};

static double deg_to_rad(double deg)
{
    return deg * M_PI / 180.0;
}

static double rad_to_deg(double rad)
{
    return rad * 180.0 / M_PI;
}

static double wrap180(double deg)
{
    while (deg > 180.0) {
        deg -= 360.0;
    }
    while (deg < -180.0) {
        deg += 360.0;
    }
    return deg;
}

static double haversine_distance_m(double lat1, double lon1, double lat2, double lon2)
{
    static constexpr double kEarthRadiusM = 6371000.0;

    const double lat1r = deg_to_rad(lat1);
    const double lat2r = deg_to_rad(lat2);
    const double dlat = deg_to_rad(lat2 - lat1);
    const double dlon = deg_to_rad(lon2 - lon1);

    const double a = std::sin(dlat / 2.0) * std::sin(dlat / 2.0)
        + std::cos(lat1r) * std::cos(lat2r) * std::sin(dlon / 2.0) * std::sin(dlon / 2.0);
    const double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));
    return kEarthRadiusM * c;
}

static double initial_bearing_deg(double lat1, double lon1, double lat2, double lon2)
{
    const double lat1r = deg_to_rad(lat1);
    const double lat2r = deg_to_rad(lat2);
    const double dlon = deg_to_rad(lon2 - lon1);

    const double y = std::sin(dlon) * std::cos(lat2r);
    const double x = std::cos(lat1r) * std::sin(lat2r)
        - std::sin(lat1r) * std::cos(lat2r) * std::cos(dlon);
    const double brng = rad_to_deg(std::atan2(y, x));
    return std::fmod(brng + 360.0, 360.0);
}

void IRAM_ATTR coupe_circuit_handler(void *args)
{
    (void) args;
    BaseType_t priorityTaken = pdFALSE;

    if (vesc_control_task_handle)
        vTaskNotifyGiveFromISR(vesc_control_task_handle, &priorityTaken);
    if (priorityTaken != pdFALSE) {
        portYIELD_FROM_ISR(priorityTaken);
    }
}

void vesc_control_task(void *pvParameters) {
    VescController vesc;
    // LD19 sends data from its TX line into ESP RX. We do not need ESP TX for LD19.
    LidarReader lidar;
    UsbGpsHost gps;
    AutonomousDriver driver;
    GpsAutonomousDriver gpsDriver;
    bool lidarEnabled = (lidar.start() == ESP_OK);
    bool gpsEnabled = (gps.start() == ESP_OK);
    TickType_t lidarNoDataSince = 0;
    TickType_t lastLidarLog = 0;
    TickType_t lastGpsLog = 0;

    gps_goal_t goal = {
        .lat = 0.0,
        .lon = 0.0,
        .enabled = false,
    };

    GpsFix previousFix = {};
    bool havePreviousFix = false;
    TickType_t lastGpsDriveLog = 0;

    if (!gpsEnabled) {
        ESP_LOGW("main", "USB GPS host failed to start");
    }

    vesc.setDuty(0.0f);
    vesc.setSteering(STEER_CENTER);
    gpio_set_direction(COUPE_CIRCUIT_GND_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(COUPE_CIRCUIT_GND_PIN, 0);
    gpio_set_direction(COUPE_CIRCUIT_PIN, GPIO_MODE_INPUT);
    gpio_set_pull_mode(COUPE_CIRCUIT_PIN, GPIO_PULLUP_ONLY);
    gpio_set_intr_type(COUPE_CIRCUIT_PIN, GPIO_INTR_ANYEDGE);
    ESP_ERROR_CHECK(gpio_install_isr_service(ESP_INTR_FLAG_LEVEL3 | ESP_INTR_FLAG_EDGE | ESP_INTR_FLAG_IRAM));
    ESP_ERROR_CHECK(gpio_isr_handler_add(COUPE_CIRCUIT_PIN, &coupe_circuit_handler, nullptr));
    gpio_intr_enable(COUPE_CIRCUIT_PIN);
    vTaskDelay(pdMS_TO_TICKS(20));

    uint32_t notification_value = 0;

    while (1) {
        const TickType_t now = xTaskGetTickCount();
        if (gpsEnabled && (now - lastGpsLog) > GPS_LOG_PERIOD_TICKS) {
            const GpsFix fix = gps.getLatestFix();
            if (fix.hasFix) {
                ESP_LOGI("gps", "fix sats=%d lat=%.6f lon=%.6f alt=%.1f",
                         fix.satellites, fix.latitude, fix.longitude, fix.altitudeMeters);
            } else {
                ESP_LOGI("gps", "waiting for fix sats=%d", fix.satellites);
            }
            lastGpsLog = now;
        }

        GpsDriveInput gpsInput = {};
        bool gpsModeActive = false;
        if (goal.enabled && gpsEnabled) {
            const GpsFix currentFix = gps.getLatestFix();
            if (currentFix.hasFix) {
                const double distToGoal = haversine_distance_m(
                    currentFix.latitude,
                    currentFix.longitude,
                    goal.lat,
                    goal.lon);

                gpsInput.distanceToGoalM = static_cast<float>(distToGoal);
                gpsInput.goalReached = (distToGoal <= GPS_GOAL_ACCEPTANCE_RADIUS_M);

                if (!gpsInput.goalReached && havePreviousFix && previousFix.hasFix) {
                    const double movedM = haversine_distance_m(
                        previousFix.latitude,
                        previousFix.longitude,
                        currentFix.latitude,
                        currentFix.longitude);
                    if (movedM >= GPS_HEADING_MIN_MOVEMENT_M) {
                        const double headingDeg = initial_bearing_deg(
                            previousFix.latitude,
                            previousFix.longitude,
                            currentFix.latitude,
                            currentFix.longitude);
                        const double bearingToGoalDeg = initial_bearing_deg(
                            currentFix.latitude,
                            currentFix.longitude,
                            goal.lat,
                            goal.lon);
                        gpsInput.headingErrorDeg = static_cast<float>(
                            wrap180(bearingToGoalDeg - headingDeg));
                        gpsInput.headingValid = true;
                    }
                }

                previousFix = currentFix;
                havePreviousFix = true;
                gpsModeActive = true;
            }
        }

        if (xTaskNotifyWait(0, 0, &notification_value, pdMS_TO_TICKS(20)) == pdPASS) { // On interrupt on coupe circuit pin
            if (gpio_get_level(COUPE_CIRCUIT_PIN)) { // HIGH = disconnected
                vesc.deactivate();
            } else {
                vesc.activate();
            }
            continue;
        }
        float manualDuty, manualSteer;
        bool s_emergency;
        if (get_manual_control(manualDuty, manualSteer, s_emergency)) {
            vesc.setSteering(manualSteer);
            vesc.setDuty(manualDuty);
            if (s_emergency) {
                vesc.deactivate();
            }
            continue;
        }

        if (!lidarEnabled) {
            vesc.setDuty(0.0f);
            vesc.setSteering(STEER_CENTER);
            continue;
        }

        const bool gotUartBytes = lidar.poll();
        std::vector<LidarPoint> lastScan = lidar.getLatestScanPoints();

        if (gotUartBytes) {
            lidarNoDataSince = 0;
        } else if (lidarNoDataSince == 0) {
            lidarNoDataSince = xTaskGetTickCount();
        }

        if (lastScan.empty()) {
            if (lidarNoDataSince != 0 && (now - lidarNoDataSince) > LIDAR_NO_DATA_TIMEOUT_TICKS) {
                lidarEnabled = false;
                std::cout << "LiDAR timeout (no UART data) -> manual BLE mode only" << std::endl;
                vesc.setDuty(0.0f);
                vesc.setSteering(STEER_CENTER);
                continue;
            }

            vesc.setDuty(0.0f);
            vesc.setSteering(STEER_CENTER);
            if ((now - lastLidarLog) > LIDAR_LOG_PERIOD_TICKS) {
                std::cout << "LiDAR scan not ready yet. UART bytes=" << (gotUartBytes ? "yes" : "no") << std::endl;
                lastLidarLog = now;
            }
            continue;
        }

        DriveCommands cmds;
        if (gpsModeActive) {
            if (gpsInput.goalReached) {
                vesc.setSteering(STEER_CENTER);
                vesc.setDuty(0.0f);
                if ((now - lastGpsDriveLog) > GPS_DRIVE_LOG_PERIOD_TICKS) {
                    ESP_LOGI("gps_drive", "Goal reached (<= %.1fm), holding position",
                             GPS_GOAL_ACCEPTANCE_RADIUS_M);
                    lastGpsDriveLog = now;
                }
                continue;
            }

            cmds = gpsDriver.compute_commands(lastScan, gpsInput);
            if ((now - lastGpsDriveLog) > GPS_DRIVE_LOG_PERIOD_TICKS) {
                ESP_LOGI("gps_drive", "active dist=%.1fm headingErr=%.1fdeg headingValid=%d",
                         gpsInput.distanceToGoalM,
                         gpsInput.headingErrorDeg,
                         gpsInput.headingValid);
                lastGpsDriveLog = now;
            }
        } else {
            cmds = driver.compute_commands(lastScan);
        }
        vesc.setSteering(cmds.steer);
        vesc.setDuty(cmds.duty);
    }
}

extern "C" void app_main(void) {
    printf("Starting VESC Controller on ESP32-S3...\n");
    init_bluetooth_receiver();
    init_lidar_uart();
    init_vesc_rmt_uart();
    xTaskCreate(vesc_control_task, "vesc_task", 4096, NULL, 5, &vesc_control_task_handle);
}
