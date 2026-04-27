/*
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
#include "api/user_controller_api.hpp"
#include "wifi_control_server.hpp"
#include "vescLidarUart.h"

#include "drive.hpp"
#include "gpsdrive.hpp"

#include "esp_log.h"
#include <cmath>
#include <deque>

static constexpr TickType_t LIDAR_NO_DATA_TIMEOUT_TICKS = pdMS_TO_TICKS(3000);
static constexpr TickType_t LIDAR_LOG_PERIOD_TICKS = pdMS_TO_TICKS(1000);
static constexpr TickType_t GPS_LOG_PERIOD_TICKS = pdMS_TO_TICKS(2000);
static constexpr TickType_t GPS_DRIVE_LOG_PERIOD_TICKS = pdMS_TO_TICKS(1000);
static constexpr TickType_t GPS_POLL_LOG_PERIOD_TICKS = pdMS_TO_TICKS(1000);
static constexpr TickType_t GPS_STALE_WARN_PERIOD_TICKS = pdMS_TO_TICKS(2000);
static constexpr float GPS_GOAL_ACCEPTANCE_RADIUS_M = 2.0f;
static constexpr float GPS_HEADING_MIN_MOVEMENT_M = 1.5f;
static constexpr float GPS_HEADING_SEGMENT_MIN_MOVEMENT_M = 0.35f;
// Keep a wider history so low-speed runs can still accumulate usable motion.
static constexpr size_t GPS_HEADING_HISTORY_POINTS = 40;
static constexpr size_t GPS_HEADING_GROUP_STRIDE = 3;
static constexpr int GPS_HEADING_MIN_GROUPS = 3;
static constexpr float GPS_HEADING_MIN_DIRECTION_CONFIDENCE = 0.55f;

static TaskHandle_t vesc_control_task_handle = nullptr;

struct gps_goal_t {
    double lat;
    double lon;
    bool enabled;
};

struct GpsRuntimeState {
    TickType_t lastGpsLog = 0;
    TickType_t lastGpsPollLog = 0;
    TickType_t lastGpsStaleWarnLog = 0;
    uint32_t lastGpsUpdateCounterSeen = 0;
    std::deque<GpsFix> headingHistory;
};

struct LidarRuntimeState {
    bool enabled = false;
    TickType_t noDataSince = 0;
    TickType_t lastLog = 0;
};

enum class LidarPollStatus {
    Ready,
    Waiting,
    TimedOut,
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

static void setup_gps_log_levels(bool gpsEnabled)
{
    if (!gpsEnabled) {
        ESP_LOGW("main", "USB GPS host failed to start");
        return;
    }

    // Force these tags to be visible even when global log level is stricter.
    esp_log_level_set("gps", ESP_LOG_INFO);
    esp_log_level_set("gps_drive", ESP_LOG_INFO);
    esp_log_level_set("UsbGpsHost", ESP_LOG_INFO);
}

static void log_periodic_gps_status(UsbGpsHost& gps, TickType_t now, GpsRuntimeState& state)
{
    if ((now - state.lastGpsLog) > GPS_LOG_PERIOD_TICKS) {
        const GpsFix fix = gps.getLatestFix();
        if (fix.hasFix) {
            ESP_LOGI("gps", "fix sats=%d lat=%.6f lon=%.6f alt=%.1f",
                     fix.satellites, fix.latitude, fix.longitude, fix.altitudeMeters);
        } else {
            ESP_LOGI("gps", "waiting for fix sats=%d", fix.satellites);
        }
        state.lastGpsLog = now;
    }
}

static bool build_gps_drive_input(UsbGpsHost& gps,
                                  const gps_goal_t& goal,
                                  TickType_t now,
                                  GpsRuntimeState& state,
                                  GpsDriveInput& gpsInput)
{
    gpsInput = GpsDriveInput{};

    if (!goal.enabled) {
        return false;
    }

    const GpsFix currentFix = gps.getLatestFix();

    if ((now - state.lastGpsPollLog) > GPS_POLL_LOG_PERIOD_TICKS) {
        ESP_LOGI("gps", "polled hasFix=%d sats=%d lat=%.6f lon=%.6f",
                 currentFix.hasFix,
                 currentFix.satellites,
                 currentFix.latitude,
                 currentFix.longitude);
        state.lastGpsPollLog = now;
    }

    if (currentFix.updateCounter != 0 && currentFix.updateCounter != state.lastGpsUpdateCounterSeen) {
        state.lastGpsUpdateCounterSeen = currentFix.updateCounter;
        const uint32_t ageMs = static_cast<uint32_t>(pdTICKS_TO_MS(now - currentFix.updateTick));
        ESP_LOGI("gps", "NEW fix #%lu age=%lums hasFix=%d sats=%d lat=%.6f lon=%.6f",
                 static_cast<unsigned long>(currentFix.updateCounter),
                 static_cast<unsigned long>(ageMs),
                 currentFix.hasFix,
                 currentFix.satellites,
                 currentFix.latitude,
                 currentFix.longitude);
    } else if ((now - state.lastGpsStaleWarnLog) > GPS_STALE_WARN_PERIOD_TICKS) {
        const uint32_t ageMs = static_cast<uint32_t>(pdTICKS_TO_MS(now - currentFix.updateTick));
        ESP_LOGW("gps", "no new fix yet (counter=%lu age=%lums)",
                 static_cast<unsigned long>(currentFix.updateCounter),
                 static_cast<unsigned long>(ageMs));
        state.lastGpsStaleWarnLog = now;
    }

    if (!currentFix.hasFix) {
        return false;
    }

    const double distToGoal = haversine_distance_m(
        currentFix.latitude,
        currentFix.longitude,
        goal.lat,
        goal.lon);

    gpsInput.distanceToGoalM = static_cast<float>(distToGoal);
    gpsInput.goalReached = (distToGoal <= GPS_GOAL_ACCEPTANCE_RADIUS_M);

    state.headingHistory.push_back(currentFix);
    while (state.headingHistory.size() > GPS_HEADING_HISTORY_POINTS) {
        state.headingHistory.pop_front();
    }

    if (!gpsInput.goalReached && state.headingHistory.size() > GPS_HEADING_GROUP_STRIDE) {
        double sumSin = 0.0;
        double sumCos = 0.0;
        double totalMovedM = 0.0;
        int usedGroups = 0;

        // Use grouped points (not immediate neighbors) to reduce high-frequency GPS jitter impact.
        for (size_t i = GPS_HEADING_GROUP_STRIDE; i < state.headingHistory.size(); ++i) {
            const GpsFix& p0 = state.headingHistory[i - GPS_HEADING_GROUP_STRIDE];
            const GpsFix& p1 = state.headingHistory[i];
            const double movedM = haversine_distance_m(
                p0.latitude,
                p0.longitude,
                p1.latitude,
                p1.longitude);

            if (movedM < GPS_HEADING_SEGMENT_MIN_MOVEMENT_M) {
                continue;
            }

            const double segHeadingDeg = initial_bearing_deg(
                p0.latitude,
                p0.longitude,
                p1.latitude,
                p1.longitude);
            const double segHeadingRad = deg_to_rad(segHeadingDeg);
            sumSin += std::sin(segHeadingRad) * movedM;
            sumCos += std::cos(segHeadingRad) * movedM;
            totalMovedM += movedM;
            ++usedGroups;
        }

        const double headingVectorNorm = std::hypot(sumSin, sumCos);
        const double headingConfidence =
            (totalMovedM > 1e-6) ? (headingVectorNorm / totalMovedM) : 0.0;

        if (totalMovedM >= GPS_HEADING_MIN_MOVEMENT_M &&
            usedGroups >= GPS_HEADING_MIN_GROUPS &&
            headingConfidence >= GPS_HEADING_MIN_DIRECTION_CONFIDENCE &&
            (std::fabs(sumSin) > 1e-6 || std::fabs(sumCos) > 1e-6)) {
            const double headingDeg = std::fmod(rad_to_deg(std::atan2(sumSin, sumCos)) + 360.0, 360.0);
            const double bearingToGoalDeg = initial_bearing_deg(
                currentFix.latitude,
                currentFix.longitude,
                goal.lat,
                goal.lon);
            gpsInput.headingErrorDeg = static_cast<float>(
                wrap180(bearingToGoalDeg - headingDeg));
            gpsInput.headingValid = true;
        } else {
            ESP_LOGI("gps",
                     "heading invalid moved=%.2fm groups=%d conf=%.2f (need moved>=%.2f groups>=%d conf>=%.2f)",
                     totalMovedM,
                     usedGroups,
                     headingConfidence,
                     static_cast<double>(GPS_HEADING_MIN_MOVEMENT_M),
                     GPS_HEADING_MIN_GROUPS,
                     static_cast<double>(GPS_HEADING_MIN_DIRECTION_CONFIDENCE));
        }
    }

    return true;
}

static LidarPollStatus poll_lidar_scan(LidarReader& lidar,
                                       TickType_t now,
                                       LidarRuntimeState& state,
                                       std::vector<LidarPoint>& lastScan)
{
    const bool gotUartBytes = lidar.poll();
    lastScan = lidar.getLatestScanPoints();

    if (gotUartBytes) {
        state.noDataSince = 0;
    } else if (state.noDataSince == 0) {
        state.noDataSince = now;
    }

    if (!lastScan.empty()) {
        return LidarPollStatus::Ready;
    }

    if (state.noDataSince != 0 && (now - state.noDataSince) > LIDAR_NO_DATA_TIMEOUT_TICKS) {
        state.enabled = false;
        std::cout << "LiDAR timeout (no UART data) -> manual BLE mode only" << std::endl;
        return LidarPollStatus::TimedOut;
    }

    if ((now - state.lastLog) > LIDAR_LOG_PERIOD_TICKS) {
        std::cout << "LiDAR scan not ready yet. UART bytes=" << (gotUartBytes ? "yes" : "no") << std::endl;
        state.lastLog = now;
    }

    return LidarPollStatus::Waiting;
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
    (void)pvParameters;
    VescController vesc;
    // LD19 sends data from its TX line into ESP RX. We do not need ESP TX for LD19.
    LidarReader lidar;
    UsbGpsHost gps;
    AutonomousDriver driver;
    GpsAutonomousDriver gpsDriver;
    LidarRuntimeState lidarState;
    UserControllerApi &manualControl = wifiControlServer();

    lidarState.enabled = (lidar.start() == ESP_OK);
    const bool gpsEnabled = (gps.start() == ESP_OK);
    GpsRuntimeState gpsState;

    gps_goal_t goal = { //43.612139, 1.430194

        .lat = 43.612139,
        .lon = 1.430194,
        .enabled = true,
    };

    TickType_t lastGpsDriveLog = 0;
    setup_gps_log_levels(gpsEnabled);

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
        if (gpsEnabled) {
            log_periodic_gps_status(gps, now, gpsState);
        }

        GpsDriveInput gpsInput = {};
        const bool gpsModeActive = gpsEnabled &&
                                   build_gps_drive_input(gps, goal, now, gpsState, gpsInput);

        if (xTaskNotifyWait(0, 0, &notification_value, pdMS_TO_TICKS(20)) == pdPASS) { // On interrupt on coupe circuit pin
            if (gpio_get_level(COUPE_CIRCUIT_PIN)) { // HIGH = disconnected
                vesc.deactivate();
            } else {
                vesc.activate();
            }
            continue;
        }
        if (manualControl.isConnected() && manualControl.getDrivingMode() == DRIVING_MODE_USER) {
            vesc.setSteering(manualControl.getSteering());
            vesc.setDuty(manualControl.getSpeed());
            continue;
        }

        if (!lidarState.enabled) {
            vesc.setDuty(0.0f);
            vesc.setSteering(STEER_CENTER);
            continue;
        }

        std::vector<LidarPoint> lastScan;
        const LidarPollStatus lidarStatus = poll_lidar_scan(lidar, now, lidarState, lastScan);
        if (lidarStatus != LidarPollStatus::Ready) {
            vesc.setDuty(0.0f);
            vesc.setSteering(STEER_CENTER);
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
    wifiControlService().start();
    init_lidar_uart();
    init_vesc_rmt_uart();
    xTaskCreate(vesc_control_task, "vesc_task", 4096, NULL, 5, &vesc_control_task_handle);
}
*/

#include <esp_log.h>
#include <freertos/FreeRTOS.h>

#include "manager/MasterManager.hpp"
#include <cstddef>

static char const *const TAG = "MAIN";

static TickType_t const MAIN_LOOP_DELAY = 0;

static TaskHandle_t main_loop_task_handle = nullptr;

void main_loop(void *) {
    ESP_LOGI(TAG, "Initiating master manager.");

    MasterManager master;

    while (true) {
        master.iterate();

        if (MAIN_LOOP_DELAY)
            vTaskDelay(MAIN_LOOP_DELAY);
    }
}

extern "C" {
void app_main(void) {
    ESP_LOGI(TAG, "Starting program.");

    xTaskCreatePinnedToCore(main_loop, "main_loop", 4096, NULL, 5, &main_loop_task_handle, 0);
}
}
