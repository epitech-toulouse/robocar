#include "lidar_obstacle_avoidance_algorithm.hpp"

#include <vector>

LidarObstacleAvoidanceAlgorithm::LidarObstacleAvoidanceAlgorithm(ILidarSensor& lidarSensor)
    : lidarSensor(lidarSensor)
{
}

bool LidarObstacleAvoidanceAlgorithm::available(void)
{
    return lidarSensor.isActive();
}

bool LidarObstacleAvoidanceAlgorithm::compute(DrivingAlgorithmOutput& output)
{
    output = DEFAULT_DRIVING_ALGORITHM_OUTPUT;

    if (!available()) {
        return false;
    }

    lidar_array_t lidarRaw{};
    if (!lidarSensor.getData(lidarRaw)) {
        return false;
    }

    std::vector<LidarPoint> scan;
    scan.reserve(LIDAR_POINT_NUMBER);

    for (int angle = 0; angle < LIDAR_POINT_NUMBER; ++angle) {
        LidarPoint point;
        point.angleDeg = static_cast<float>(angle);
        point.distanceMeters = (lidarRaw[angle] == UNDEFINED_LIDAR_VALUE)
            ? 0.0f
            : static_cast<float>(lidarRaw[angle]) / 100.0f;
        point.intensity = 0;
        scan.push_back(point);
    }

    const DriveCommands commands = driver.compute_commands(scan);
    output.target_speed = commands.duty;
    output.target_steering = commands.steer;
    return true;
}
