#ifndef LIDAR_DRIVING_ALGO_HPP
#define LIDAR_DRIVING_ALGO_HPP

#include "api/driving_algorithm_interface.hpp"
#include "api/lidar_sensor_api.hpp"
#include <esp_log.h>
#include "algo/corridorLidar/drive.hpp"
#include "config.h"

#include <vector>

class LidarDrivingAlgo : public DrivingAlgorithmApi {
public:
    explicit LidarDrivingAlgo(LidarSensorApi &lidar)
        : lidar(lidar), driver() {}

    bool available(void) override {
        return lidar.isActive();
    }

    bool compute(DrivingAlgorithmOutput &output) override {
        lidar_array_t rawData;
        if (!lidar.getData(rawData)) {
            output = DEFAULT_DRIVING_ALGORITHM_OUTPUT;
            return false;
        }

        std::vector<LidarPoint> scan;
        scan.reserve(LIDAR_POINT_NUMBER);
        float min_distance = -1.0f;

        for (std::size_t i = 0; i < LIDAR_POINT_NUMBER; ++i) {
            const centimeter_t cm = rawData[i];
            const float meters = (cm == UNDEFINED_LIDAR_VALUE)
                ? 0.0f
                : static_cast<float>(cm) / 100.0f;

            if (cm != UNDEFINED_LIDAR_VALUE && meters > 0.0f) {
                if (min_distance < 0.0f || meters < min_distance) {
                    min_distance = meters;
                }
            }

            scan.push_back({
                static_cast<float>(i),
                meters,
                static_cast<uint8_t>((cm == UNDEFINED_LIDAR_VALUE) ? 0 : 1),
            });
        }

        const DriveCommands command = driver.compute_commands(scan);

        output.target_speed = command.duty;
        output.target_steering = command.steer;
        
        float computed_weight = 0.0f;
        if (min_distance > 0.0f) {
            const float max_influence_dist = 3.0f;
            if (min_distance < max_influence_dist) {
                computed_weight = 1.0f - (min_distance / max_influence_dist);
            }
        }
        
        output.computed_weight = computed_weight;
        return true;
    }

    float getPriority() override {
        return LIDAR_CORRIDOR_WEIGHT;
    }

private:
    LidarSensorApi &lidar;
    AutonomousDriver driver;
};

#endif /* LIDAR_DRIVING_ALGO_HPP */
