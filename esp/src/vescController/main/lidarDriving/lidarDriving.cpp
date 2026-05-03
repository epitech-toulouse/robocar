#include "lidarDriving.hpp"
#include "lidarDriving.hpp"

LidarDriving::LidarDriving(LidarSensorApi &lidar) : lidar(lidar), driver() {}

bool LidarDriving::available(void) {
    return lidar.isActive();
}

bool LidarDriving::compute(DrivingAlgorithmOutput &output) {
    lidar_array_t rawData;
    if (!lidar.getData(rawData)) {
        output = DEFAULT_DRIVING_ALGORITHM_OUTPUT;
        return false;
    }

    std::vector<LidarPoint> scan;
    scan.reserve(LIDAR_POINT_NUMBER);

    for (std::size_t i = 0; i < LIDAR_POINT_NUMBER; ++i) {
        const centimeter_t cm = rawData[i];
        const float meters = (cm == UNDEFINED_LIDAR_VALUE)
            ? 0.0f
            : static_cast<float>(cm) / 100.0f;

        scan.push_back({
            static_cast<float>(i),
            meters,
            static_cast<uint8_t>((cm == UNDEFINED_LIDAR_VALUE) ? 0 : 1),
        });
    }

    const DriveCommands command = driver.compute_commands(scan);

    output.target_speed = command.duty;
    output.target_steering = command.steer;
    output.computed_weight = 1.0f;
    return true;
}

float LidarDriving::getPriority() {
    return 1.0f;
}