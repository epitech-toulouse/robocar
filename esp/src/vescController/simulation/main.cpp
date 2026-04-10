#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

using namespace std::chrono_literals;

namespace {

constexpr float MIN_RANGE_M = 0.05f;
constexpr float MAX_RANGE_M = 12.0f;
constexpr float PI_F = 3.14159265358979323846f;

struct SimLidarPoint {
	float angleDeg;
	float distanceMeters;
	uint8_t intensity;
};

float radians_to_degrees(float rad)
{
	return rad * 180.0f / PI_F;
}

float normalize_angle_deg(float angleDeg)
{
	angleDeg = std::fmod(angleDeg, 360.0f);
	if (angleDeg < 0.0f) {
		angleDeg += 360.0f;
	}
	return angleDeg;
}

bool is_front_sector(float angleDeg)
{
	return angleDeg <= 45.0f || angleDeg >= 315.0f;
}

} // namespace

class RobocarSimControllerNode : public rclcpp::Node {
public:
	RobocarSimControllerNode()
		: rclcpp::Node("robocar_sim_controller")
	{
		scanTopic = this->declare_parameter<std::string>("scan_topic", "/scan");
		cmdTopic = this->declare_parameter<std::string>("cmd_topic", "/cmd_vel");
		pointsTopic = this->declare_parameter<std::string>("points_topic", "/robocar/lidar_points_flat");
		menuEnabled = this->declare_parameter<bool>("menu_enabled", true);
		autoPrint = this->declare_parameter<bool>("auto_print", false);

		cmdPublisher = this->create_publisher<geometry_msgs::msg::Twist>(cmdTopic, 10);
		pointsPublisher = this->create_publisher<std_msgs::msg::Float32MultiArray>(pointsTopic, 10);

		scanSubscriber = this->create_subscription<sensor_msgs::msg::LaserScan>(
			scanTopic,
			10,
			[this](const sensor_msgs::msg::LaserScan::SharedPtr msg) {
				this->on_scan(msg);
			});

		commandTimer = this->create_wall_timer(100ms, [this]() {
			geometry_msgs::msg::Twist cmd;
			{
				std::lock_guard<std::mutex> lock(dataMutex);
				cmd.linear.x = commandLinear;
				cmd.angular.z = commandAngular;
			}
			cmdPublisher->publish(cmd);
		});

		statusTimer = this->create_wall_timer(1000ms, [this]() {
			std::size_t count = 0;
			float hz = 0.0f;
			{
				std::lock_guard<std::mutex> lock(dataMutex);
				count = lastPoints.size();
				hz = static_cast<float>(scanRateHz);
			}
			RCLCPP_INFO(this->get_logger(), "LiDAR points=%zu rate=%.2f Hz", count, hz);
		});

		if (menuEnabled) {
			menuThreadRunning.store(true);
			menuThread = std::thread([this]() {
				this->menu_loop();
			});
			menuThread.detach();
		}

		RCLCPP_INFO(this->get_logger(), "robocar_sim_controller ready");
		if (menuEnabled) {
			print_menu();
		}
	}

private:
	void on_scan(const sensor_msgs::msg::LaserScan::SharedPtr& msg)
	{
		std::vector<SimLidarPoint> parsedPoints;
		parsedPoints.reserve(msg->ranges.size());

		const bool hasIntensity = !msg->intensities.empty();

		for (std::size_t i = 0; i < msg->ranges.size(); ++i) {
			const float range = msg->ranges[i];
			if (!std::isfinite(range)) {
				continue;
			}

			if (range <= MIN_RANGE_M || range >= MAX_RANGE_M) {
				continue;
			}

			uint8_t intensity = 255;
			if (hasIntensity) {
				const float rawIntensity = (i < msg->intensities.size()) ? msg->intensities[i] : 0.0f;
				if (rawIntensity <= 0.0f) {
					continue;
				}
				intensity = static_cast<uint8_t>(std::clamp(rawIntensity, 0.0f, 255.0f));
			}

			const float angleRad = msg->angle_min + static_cast<float>(i) * msg->angle_increment;
			const float angleDeg = normalize_angle_deg(radians_to_degrees(angleRad));

			parsedPoints.push_back({angleDeg, range, intensity});
		}

		std::sort(parsedPoints.begin(), parsedPoints.end(), [](const SimLidarPoint& a, const SimLidarPoint& b) {
			return a.angleDeg < b.angleDeg;
		});

		publish_flat_points(parsedPoints);

		const rclcpp::Time now = this->now();
		{
			std::lock_guard<std::mutex> lock(dataMutex);
			if (lastScanTime.nanoseconds() != 0) {
				const double dt = (now - lastScanTime).seconds();
				if (dt > 1e-6) {
					const double instantHz = 1.0 / dt;
					scanRateHz = (scanRateHz <= 0.0) ? instantHz : (0.85 * scanRateHz + 0.15 * instantHz);
				}
			}
			lastScanTime = now;
			lastPoints = parsedPoints;
		}

		if (autoPrint) {
			print_scan_summary(parsedPoints);
		}
	}

	void publish_flat_points(const std::vector<SimLidarPoint>& points)
	{
		std_msgs::msg::Float32MultiArray msg;
		msg.data.reserve(points.size() * 3);

		for (const auto& p : points) {
			msg.data.push_back(p.angleDeg);
			msg.data.push_back(p.distanceMeters);
			msg.data.push_back(static_cast<float>(p.intensity));
		}

		pointsPublisher->publish(msg);
	}

	void print_scan_summary(const std::vector<SimLidarPoint>& points)
	{
		if (points.empty()) {
			RCLCPP_INFO(this->get_logger(), "LiDAR: no valid points");
			return;
		}

		float nearestFront = std::numeric_limits<float>::max();
		for (const auto& p : points) {
			if (is_front_sector(p.angleDeg)) {
				nearestFront = std::min(nearestFront, p.distanceMeters);
			}
		}

		if (nearestFront == std::numeric_limits<float>::max()) {
			RCLCPP_INFO(this->get_logger(), "LiDAR: points=%zu front=none", points.size());
		} else {
			RCLCPP_INFO(this->get_logger(), "LiDAR: points=%zu nearestFront=%.2f m", points.size(), nearestFront);
		}
	}

	void print_menu() const
	{
		std::cout << "\n=== Robocar Sim Menu ===\n"
				  << "h          : help menu\n"
				  << "s          : status\n"
				  << "a          : toggle auto print lidar summary\n"
				  << "cmd v w    : set cmd_vel (linear v, angular w)\n"
				  << "stop       : set cmd_vel to 0 0\n"
				  << "q          : close menu input thread\n"
				  << "========================\n";
	}

	void menu_loop()
	{
		while (rclcpp::ok() && menuThreadRunning.load()) {
			std::string line;
			if (!std::getline(std::cin, line)) {
				return;
			}

			if (line == "h") {
				print_menu();
				continue;
			}

			if (line == "a") {
				autoPrint = !autoPrint;
				std::cout << "auto_print=" << (autoPrint ? "ON" : "OFF") << "\n";
				continue;
			}

			if (line == "s") {
				std::size_t count = 0;
				double hz = 0.0;
				float v = 0.0f;
				float w = 0.0f;
				{
					std::lock_guard<std::mutex> lock(dataMutex);
					count = lastPoints.size();
					hz = scanRateHz;
					v = commandLinear;
					w = commandAngular;
				}
				std::cout << std::fixed << std::setprecision(2)
						  << "status: points=" << count
						  << " scan_hz=" << hz
						  << " cmd_vel=(" << v << ", " << w << ")\n";
				continue;
			}

			if (line == "stop") {
				std::lock_guard<std::mutex> lock(dataMutex);
				commandLinear = 0.0f;
				commandAngular = 0.0f;
				std::cout << "cmd_vel set to 0 0\n";
				continue;
			}

			if (line == "q") {
				menuThreadRunning.store(false);
				std::cout << "menu thread stopped\n";
				return;
			}

			if (line.rfind("cmd ", 0) == 0) {
				std::istringstream iss(line);
				std::string token;
				float v = 0.0f;
				float w = 0.0f;
				iss >> token >> v >> w;
				if (!iss.fail()) {
					std::lock_guard<std::mutex> lock(dataMutex);
					commandLinear = v;
					commandAngular = w;
					std::cout << "cmd_vel set to " << v << " " << w << "\n";
				} else {
					std::cout << "invalid command, expected: cmd <linear> <angular>\n";
				}
				continue;
			}

			std::cout << "unknown command. type h for help\n";
		}
	}

	std::string scanTopic;
	std::string cmdTopic;
	std::string pointsTopic;

	bool menuEnabled = true;
	bool autoPrint = false;

	std::mutex dataMutex;
	std::vector<SimLidarPoint> lastPoints;
	rclcpp::Time lastScanTime;
	double scanRateHz = 0.0;
	float commandLinear = 0.0f;
	float commandAngular = 0.0f;

	std::atomic<bool> menuThreadRunning{false};
	std::thread menuThread;

	rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scanSubscriber;
	rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmdPublisher;
	rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pointsPublisher;
	rclcpp::TimerBase::SharedPtr commandTimer;
	rclcpp::TimerBase::SharedPtr statusTimer;
};

int main(int argc, char** argv)
{
	rclcpp::init(argc, argv);
	auto node = std::make_shared<RobocarSimControllerNode>();
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
