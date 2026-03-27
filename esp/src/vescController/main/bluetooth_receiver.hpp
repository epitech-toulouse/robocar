#pragma once

void init_bluetooth_receiver();
bool get_manual_control(float &duty, float &steer);

/// Navigation target API (set from JSON BLE commands)
bool get_nav_target_ble(float &lat, float &lon);
void clear_nav_target_ble();
