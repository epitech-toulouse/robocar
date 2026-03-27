/**
 * @file target_point.cpp
 * @brief Thread-safe navigation target point management.
 */

#include "target_point.hpp"
#include "esp_log.h"

static const char *TAG = "nav_target";

static portMUX_TYPE s_lock = portMUX_INITIALIZER_UNLOCKED;
static float  s_target_lat = 0.0f;
static float  s_target_lon = 0.0f;
static bool   s_has_target = false;
static NavState s_state = NavState::IDLE;

bool nav_set_target(float lat, float lon) {
    // Validate coordinates
    if (lat < -90.0f || lat > 90.0f || lon < -180.0f || lon > 180.0f) {
        ESP_LOGW(TAG, "Invalid target: lat=%.6f lon=%.6f", lat, lon);
        return false;
    }

    portENTER_CRITICAL(&s_lock);
    s_target_lat = lat;
    s_target_lon = lon;
    s_has_target = true;
    s_state = NavState::NAVIGATING;
    portEXIT_CRITICAL(&s_lock);

    ESP_LOGI(TAG, "Target set: lat=%.6f lon=%.6f", lat, lon);
    return true;
}

void nav_clear_target(void) {
    portENTER_CRITICAL(&s_lock);
    s_has_target = false;
    s_state = NavState::IDLE;
    portEXIT_CRITICAL(&s_lock);

    ESP_LOGI(TAG, "Target cleared");
}

bool nav_has_target(void) {
    portENTER_CRITICAL(&s_lock);
    bool has = s_has_target;
    portEXIT_CRITICAL(&s_lock);
    return has;
}

bool nav_get_target(float &lat, float &lon) {
    portENTER_CRITICAL(&s_lock);
    if (!s_has_target) {
        portEXIT_CRITICAL(&s_lock);
        return false;
    }
    lat = s_target_lat;
    lon = s_target_lon;
    portEXIT_CRITICAL(&s_lock);
    return true;
}

NavState nav_get_state(void) {
    portENTER_CRITICAL(&s_lock);
    NavState st = s_state;
    portEXIT_CRITICAL(&s_lock);
    return st;
}

void nav_set_state(NavState state) {
    portENTER_CRITICAL(&s_lock);
    s_state = state;
    if (state == NavState::ARRIVED || state == NavState::IDLE) {
        s_has_target = false;
    }
    portEXIT_CRITICAL(&s_lock);
}
