#pragma once

#include <vector>
#if __has_include("freertos/FreeRTOS.h")
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#else
#include <cstdint>
using TickType_t = uint32_t;
#ifndef pdMS_TO_TICKS
#define pdMS_TO_TICKS(ms) (static_cast<TickType_t>(ms))
#endif
#endif
#include "lidarReader.hpp"

// LiDAR-only driving parameters.
constexpr float FRONT_WINDOW_DEG = 60.0f;
constexpr float SIDE_WINDOW_MIN_DEG = 55.0f;
constexpr float SIDE_WINDOW_MAX_DEG = 125.0f;

constexpr float STOP_DISTANCE_M = 0.55f;
constexpr float SLOW_DISTANCE_M = 1.20f;
constexpr float SAFE_DISTANCE_M = 2.50f;
constexpr float ESCAPE_TURN_START_M = 2.80f;
constexpr float SIDE_OPENNESS_EPSILON_M = 0.08f;

constexpr float SPEED_FORWARD = 0.150f;  // Vitesse de pointe augmentée (15% vs 5%)
constexpr float SPEED_SLOW = 0.050f;     // Vitesse lente augmentée
constexpr float SPEED_CRAWL = 0.025f;    // Approche très lente quand l'avant se ferme
constexpr float SPEED_REVERSE = -0.015f; // Marche arrière beaucoup plus douce (env 5km/h)

constexpr float STEER_CENTER = 0.50f;
constexpr float STEER_LEFT = 0.00f;      // Débattement direction maximum absolu (0% PWM)
constexpr float STEER_RIGHT = 1.00f;     // Débattement direction maximum absolu (100% PWM)
constexpr bool AUTO_STEER_REVERSED = true;

// FTG Parameters
constexpr float FTG_MAX_RANGE_M = 3.0f;
constexpr float FTG_CAR_WIDTH_M = 0.15f; // Véritable largeur (15cm)
constexpr float FTG_DISPARITY_THRESHOLD_M = 0.15f;
constexpr float FTG_STEER_GAIN = 2.5f; // Sensibilité du volant augmentée (Tourne plus fort, plus tôt)

constexpr uint32_t REVERSE_DURATION_MS = 1000;

struct DriveCommands {
    float steer;
    float duty;
};

class AutonomousDriver {
public:
    AutonomousDriver() = default;

    // Computes the next driving commands based on lidar scan.
    DriveCommands compute_commands(const std::vector<LidarPoint>& scan);

private:
    TickType_t reverseUntil = 0;
    float reverseSteer = STEER_CENTER;
};
