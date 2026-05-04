#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
	#include "std_msgs/msg/bool.hpp"
	#include "std_msgs/msg/float32_multi_array.hpp"
	#include "std_msgs/msg/float64.hpp"
	#include "std_msgs/msg/header.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "algo/LidarObstacleAvoidance/lidar_obstacle_avoidance_algorithm.hpp"
#include "api/driving_algorithm_interface.hpp"
#include "output/sim_vesc_controller.hpp"
#include "sensors/sim_lidar_sensor.hpp"

class RobocarSimControllerNode : public rclcpp::Node {
public:
	RobocarSimControllerNode();

private:
	struct SimLidarPoint {
		float angleDeg;
		float distanceMeters;
		uint8_t intensity;
	};

	SimControlConfig build_vesc_config();

	void on_scan(const sensor_msgs::msg::LaserScan::SharedPtr& msg);
	void publish_command_tick();
	void on_mode_update(const std_msgs::msg::Bool::SharedPtr& msg);
	void on_manual_cmd_update(const geometry_msgs::msg::Twist::SharedPtr& msg);

	void publish_flat_points(const std::vector<SimLidarPoint>& points);
	void publish_cluster_markers(const std::vector<SimLidarPoint>& points, const std_msgs::msg::Header& header);
	float nearest_distance_in_sector(const std::vector<SimLidarPoint>& points, float minDeg, float maxDeg) const;
	void print_scan_summary(const std::vector<SimLidarPoint>& points);
	void print_menu() const;
	void menu_loop();

	std::string scanTopic;
	std::string cmdTopic;
	std::string pointsTopic;
	std::string clusterTopic;
	std::string frontLeftSteeringTopic;
	std::string frontRightSteeringTopic;

	bool menuEnabled = true;
	bool autoPrint = false;
	float clusterDistanceThresholdM = 0.18f;
	float clusterMaxAngleGapDeg = 2.0f;
	int clusterMinPoints = 3;

	std::mutex dataMutex;
	std::vector<SimLidarPoint> lastPoints;
	std::chrono::steady_clock::time_point lastScanReceiveWallTime{};
	bool hasScanReceiveWallTime = false;
	double scanRateHz = 0.0;
	float commandLinear = 0.0f;
	float commandAngular = 0.0f;
	float lastPublishedLinear = 0.0f;
	float lastPublishedAngular = 0.0f;
	float lastSteeringAngleDeg = 0.0f;
	float lastAutoTargetSpeed = 0.0f;
	float lastAutoTargetSteering = 0.5f;
	float lastScaledAutoSpeed = 0.0f;
	float lastFrontNearM = -1.0f;
	float lastLeftNearM = -1.0f;
	float lastRightNearM = -1.0f;
	float lastRearNearM = -1.0f;
	bool lastAutoComputeOk = false;
	bool autonomousEnabled = true;

	std::atomic<bool> menuThreadRunning{false};
	std::thread menuThread;

	SimLidarSensor lidarSensor;
	std::unique_ptr<IDrivingAlgorithm> drivingAlgorithm;
	SimVescController vescController;

	rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scanSubscriber;
	rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr modeSubscriber;
	rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr manualCmdSubscriber;
		rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmdPublisher;
		rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr frontLeftSteeringPublisher;
		rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr frontRightSteeringPublisher;
		rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pointsPublisher;
	rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr clusterPublisher;
	rclcpp::TimerBase::SharedPtr commandTimer;
	rclcpp::TimerBase::SharedPtr statusTimer;
};
