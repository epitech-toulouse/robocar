#pragma once

#include <chrono>
#include <mutex>

#include "sensor_msgs/msg/laser_scan.hpp"

#include "api/lidar_sensor_api.hpp"

class SimLidarSensor : public ILidarSensor {
public:
	SimLidarSensor() = default;
	~SimLidarSensor() override = default;

	bool isActive(void) override;
	bool getData(lidar_array_t& output) override;

	void updateFromScan(const sensor_msgs::msg::LaserScan& scanMsg);

private:
	bool isFreshLocked() const;

	mutable std::mutex dataMutex;
	lidar_array_t lastData{};
	bool hasFreshData = false;
	std::chrono::steady_clock::time_point lastUpdateTime{};
};
