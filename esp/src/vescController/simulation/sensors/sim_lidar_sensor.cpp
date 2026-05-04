#include "sim_lidar_sensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace {

constexpr float PI_F = 3.14159265358979323846f;
constexpr float MIN_RANGE_M = 0.05f;
constexpr float MAX_RANGE_M = 12.0f;
constexpr float ANGLE_OFFSET_DEG = 15.0f;
constexpr auto FRESH_DATA_TIMEOUT = std::chrono::milliseconds(300);

float normalize_angle_deg(float angleDeg)
{
	angleDeg = std::fmod(angleDeg, 360.0f);
	if (angleDeg < 0.0f) {
		angleDeg += 360.0f;
	}
	return angleDeg;
}

} // namespace

bool SimLidarSensor::isFreshLocked() const
{
	return hasFreshData && (std::chrono::steady_clock::now() - lastUpdateTime) <= FRESH_DATA_TIMEOUT;
}

bool SimLidarSensor::isActive(void)
{
	std::lock_guard<std::mutex> lock(dataMutex);
	return isFreshLocked();
}

bool SimLidarSensor::getData(lidar_array_t& output)
{
	std::lock_guard<std::mutex> lock(dataMutex);
	if (!isFreshLocked()) {
		output = {};
		return false;
	}
	output = lastData;
	return true;
}

void SimLidarSensor::updateFromScan(const sensor_msgs::msg::LaserScan& scanMsg)
{
	lidar_array_t data{};
	data.fill(UNDEFINED_LIDAR_VALUE);

	for (std::size_t i = 0; i < scanMsg.ranges.size(); ++i) {
		const float range = scanMsg.ranges[i];
		if (!std::isfinite(range) || range <= MIN_RANGE_M || range >= MAX_RANGE_M) {
			continue;
		}

		const float angleRad = scanMsg.angle_min + static_cast<float>(i) * scanMsg.angle_increment;
		const float angleDeg = normalize_angle_deg(angleRad * 180.0f / PI_F + ANGLE_OFFSET_DEG);
		const int angleIndex = static_cast<int>(std::lround(angleDeg)) % LIDAR_POINT_NUMBER;
		const centimeter_t distanceCm = static_cast<centimeter_t>(
			std::clamp(range * 100.0f, 0.0f, static_cast<float>(std::numeric_limits<centimeter_t>::max())));

		if (data[angleIndex] == UNDEFINED_LIDAR_VALUE || distanceCm < data[angleIndex]) {
			data[angleIndex] = distanceCm;
		}
	}

	{
		std::lock_guard<std::mutex> lock(dataMutex);
		lastData = data;
		hasFreshData = true;
		lastUpdateTime = std::chrono::steady_clock::now();
	}
}
