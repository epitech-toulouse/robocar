#include "robocar_sim_controller_node.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>

using namespace std::chrono_literals;

namespace {

constexpr float MIN_RANGE_M = 0.05f;
constexpr float MAX_RANGE_M = 12.0f;
constexpr float PI_F = 3.14159265358979323846f;
constexpr float AUTO_FULL_FORWARD_DUTY = 0.150f;
constexpr char kAlgorithmName[] = "lidar_obstacle_avoidance";

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

float degrees_to_radians(float deg)
{
	return deg * PI_F / 180.0f;
}

bool is_front_sector(float angleDeg)
{
	return angleDeg <= 45.0f || angleDeg >= 315.0f;
}

bool angle_in_sector(float angleDeg, float minDeg, float maxDeg)
{
	if (minDeg <= maxDeg) {
		return angleDeg >= minDeg && angleDeg <= maxDeg;
	}
	return angleDeg >= minDeg || angleDeg <= maxDeg;
}

} // namespace

RobocarSimControllerNode::RobocarSimControllerNode()
	: rclcpp::Node("robocar_sim_controller")
	, vescController(build_vesc_config())
{
	scanTopic = this->declare_parameter<std::string>("scan_topic", "/scan");
	cmdTopic = this->declare_parameter<std::string>("cmd_topic", "/cmd_vel");
	pointsTopic = this->declare_parameter<std::string>("points_topic", "/robocar/lidar_points_flat");
	clusterTopic = this->declare_parameter<std::string>("cluster_topic", "/robocar/lidar_clusters");
	frontLeftSteeringTopic = this->declare_parameter<std::string>("front_left_steering_topic", "/robocar/front_left_steering_cmd");
	frontRightSteeringTopic = this->declare_parameter<std::string>("front_right_steering_topic", "/robocar/front_right_steering_cmd");
	menuEnabled = this->declare_parameter<bool>("menu_enabled", true);
	autoPrint = this->declare_parameter<bool>("auto_print", false);
	autonomousEnabled = this->declare_parameter<bool>("autonomous_enabled", true);
	clusterDistanceThresholdM = this->declare_parameter<float>("cluster_distance_threshold_m", 0.18f);
	clusterMaxAngleGapDeg = this->declare_parameter<float>("cluster_max_angle_gap_deg", 2.0f);
	clusterMinPoints = this->declare_parameter<int>("cluster_min_points", 3);

	drivingAlgorithm = std::make_unique<LidarObstacleAvoidanceAlgorithm>(lidarSensor);

	cmdPublisher = this->create_publisher<geometry_msgs::msg::Twist>(cmdTopic, 10);
	frontLeftSteeringPublisher = this->create_publisher<std_msgs::msg::Float64>(frontLeftSteeringTopic, 10);
	frontRightSteeringPublisher = this->create_publisher<std_msgs::msg::Float64>(frontRightSteeringTopic, 10);
	pointsPublisher = this->create_publisher<std_msgs::msg::Float32MultiArray>(pointsTopic, 10);
	clusterPublisher = this->create_publisher<visualization_msgs::msg::MarkerArray>(clusterTopic, 10);

	scanSubscriber = this->create_subscription<sensor_msgs::msg::LaserScan>(
		scanTopic,
		10,
		[this](const sensor_msgs::msg::LaserScan::SharedPtr msg) {
			this->on_scan(msg);
		});

	modeSubscriber = this->create_subscription<std_msgs::msg::Bool>(
		"/robocar/menu/autonomous_enabled",
		10,
		[this](const std_msgs::msg::Bool::SharedPtr msg) {
			this->on_mode_update(msg);
		});

	manualCmdSubscriber = this->create_subscription<geometry_msgs::msg::Twist>(
		"/robocar/menu/manual_cmd_vel",
		10,
		[this](const geometry_msgs::msg::Twist::SharedPtr msg) {
			this->on_manual_cmd_update(msg);
		});

	commandTimer = this->create_wall_timer(50ms, [this]() {
		this->publish_command_tick();
	});

	statusTimer = this->create_wall_timer(1000ms, [this]() {
		std::size_t count = 0;
		float hz = 0.0f;
		bool autoMode = true;
		float outV = 0.0f;
		float outW = 0.0f;
		bool algoOk = false;
		float steeringDeg = 0.0f;
		float autoTargetSpeed = 0.0f;
		float autoTargetSteering = 0.5f;
		float scaledAutoSpeed = 0.0f;
		float frontNear = -1.0f;
		float leftNear = -1.0f;
		float rightNear = -1.0f;
		float rearNear = -1.0f;
		double scanAgeSec = -1.0;
		{
			std::lock_guard<std::mutex> lock(dataMutex);
			count = lastPoints.size();
			hz = static_cast<float>(scanRateHz);
			autoMode = autonomousEnabled;
			outV = lastPublishedLinear;
			outW = lastPublishedAngular;
			algoOk = lastAutoComputeOk;
			steeringDeg = lastSteeringAngleDeg;
			autoTargetSpeed = lastAutoTargetSpeed;
			autoTargetSteering = lastAutoTargetSteering;
			scaledAutoSpeed = lastScaledAutoSpeed;
			frontNear = lastFrontNearM;
			leftNear = lastLeftNearM;
			rightNear = lastRightNearM;
			rearNear = lastRearNearM;
			if (hasScanReceiveWallTime) {
				scanAgeSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - lastScanReceiveWallTime).count();
			}
		}
		RCLCPP_INFO(this->get_logger(),
			"LiDAR points=%zu rate=%.2f Hz age=%.2f s mode=%s algo=%s cmd=(%.3f, %.3f) steer=%.1fdeg auto=(speed=%.3f scaled=%.3f steering=%.3f ok=%s) lidar(front=%.2f left=%.2f right=%.2f rear=%.2f)",
			count,
			hz,
			scanAgeSec,
			autoMode ? "AUTO" : "MANUAL",
			kAlgorithmName,
			outV,
			outW,
			steeringDeg,
			autoTargetSpeed,
			scaledAutoSpeed,
			autoTargetSteering,
			algoOk ? "ok" : "no",
			frontNear,
			leftNear,
			rightNear,
			rearNear);

		if (autoMode && algoOk && autoTargetSpeed < 0.0f) {
			RCLCPP_WARN(this->get_logger(),
				"Auto reverse requested by obstacle avoidance: target_speed=%.3f front=%.2f left=%.2f right=%.2f rear=%.2f",
				autoTargetSpeed,
				frontNear,
				leftNear,
				rightNear,
				rearNear);
		}

		if (scanAgeSec > 2.0) {
			RCLCPP_WARN(this->get_logger(), "LiDAR scan stream stale: last update %.2f s ago", scanAgeSec);
		}
	});

	if (menuEnabled) {
		std::cout.setf(std::ios::unitbuf);
		menuThreadRunning.store(true);
		menuThread = std::thread([this]() {
			this->menu_loop();
		});
		menuThread.detach();
	}

	RCLCPP_INFO(this->get_logger(), "robocar_sim_controller ready");
	if (menuEnabled) {
		RCLCPP_INFO(this->get_logger(), "interactive menu enabled (type 'h' for help)");
		print_menu();
	}
}

SimControlConfig RobocarSimControllerNode::build_vesc_config()
{
	SimControlConfig config;
	config.maxLinearSpeedMps = this->declare_parameter<float>("sim_max_linear_speed_mps", 5.0f);
	config.maxAccelerationMps2 = this->declare_parameter<float>("sim_max_acceleration_mps2", 1.5f);
	config.maxAngularSpeedRadps = this->declare_parameter<float>("sim_max_angular_speed_radps", 2.2f);
	const float maxSteeringAngleDeg = this->declare_parameter<float>("sim_max_steering_angle_deg", 35.0f);
	config.maxSteeringAngleRad = maxSteeringAngleDeg * PI_F / 180.0f;
	config.reverseSteering = this->declare_parameter<bool>("sim_reverse_steering", false);
	if (config.maxLinearSpeedMps > 10.0f) {
		RCLCPP_WARN(
			this->get_logger(),
			"sim_max_linear_speed_mps=%.2f is very high for the current LiDAR avoidance distances. "
			"The controller will command it, but the car can overshoot obstacles, reverse, slide, or look slow after impacts. "
			"Use about 2-5 m/s unless the avoidance stop/slow distances are retuned.",
			config.maxLinearSpeedMps);
	}
	if (config.maxAccelerationMps2 > 20.0f) {
		RCLCPP_WARN(
			this->get_logger(),
			"sim_max_acceleration_mps2=%.2f gives almost instant throttle changes and can destabilize the Gazebo car physics.",
			config.maxAccelerationMps2);
	}
	return config;
}

float RobocarSimControllerNode::nearest_distance_in_sector(
	const std::vector<SimLidarPoint>& points,
	float minDeg,
	float maxDeg) const
{
	float nearest = -1.0f;
	for (const auto& point : points) {
		if (!angle_in_sector(point.angleDeg, minDeg, maxDeg)) {
			continue;
		}
		if (nearest < 0.0f || point.distanceMeters < nearest) {
			nearest = point.distanceMeters;
		}
	}
	return nearest;
}

void RobocarSimControllerNode::on_scan(const sensor_msgs::msg::LaserScan::SharedPtr& msg)
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
			if (std::isfinite(rawIntensity) && rawIntensity > 0.0f) {
				intensity = static_cast<uint8_t>(std::clamp(rawIntensity, 0.0f, 255.0f));
			}
		}

		const float angleRad = msg->angle_min + static_cast<float>(i) * msg->angle_increment;
		const float angleDeg = normalize_angle_deg(radians_to_degrees(angleRad));

		parsedPoints.push_back({angleDeg, range, intensity});
	}

	std::sort(parsedPoints.begin(), parsedPoints.end(), [](const SimLidarPoint& a, const SimLidarPoint& b) {
		return a.angleDeg < b.angleDeg;
	});

	publish_flat_points(parsedPoints);
	publish_cluster_markers(parsedPoints, msg->header);
	lidarSensor.updateFromScan(*msg);

	const auto now = std::chrono::steady_clock::now();
	{
		std::lock_guard<std::mutex> lock(dataMutex);
		if (hasScanReceiveWallTime) {
			const double dt = std::chrono::duration<double>(now - lastScanReceiveWallTime).count();
			if (dt > 1e-6) {
				const double instantHz = 1.0 / dt;
				scanRateHz = (scanRateHz <= 0.0) ? instantHz : (0.85 * scanRateHz + 0.15 * instantHz);
			}
		}
		lastScanReceiveWallTime = now;
		hasScanReceiveWallTime = true;
		lastPoints = parsedPoints;
		lastFrontNearM = nearest_distance_in_sector(parsedPoints, 315.0f, 45.0f);
		lastLeftNearM = nearest_distance_in_sector(parsedPoints, 45.0f, 135.0f);
		lastRearNearM = nearest_distance_in_sector(parsedPoints, 135.0f, 225.0f);
		lastRightNearM = nearest_distance_in_sector(parsedPoints, 225.0f, 315.0f);
	}

	if (autoPrint) {
		print_scan_summary(parsedPoints);
	}
}

void RobocarSimControllerNode::publish_command_tick()
{
	geometry_msgs::msg::Twist cmd;
	float steeringAngle = 0.0f;
	bool autoMode = false;
	float manualV = 0.0f;
	float manualW = 0.0f;
	bool autoComputeOk = false;
	{
		std::lock_guard<std::mutex> lock(dataMutex);
		autoMode = autonomousEnabled;
		manualV = commandLinear;
		manualW = commandAngular;
	}

	if (autoMode && drivingAlgorithm) {
		DrivingAlgorithmOutput output = DEFAULT_DRIVING_ALGORITHM_OUTPUT;
		autoComputeOk = drivingAlgorithm->compute(output);
		float scaledAutoSpeed = 0.0f;
		if (autoComputeOk) {
			scaledAutoSpeed = std::clamp(output.target_speed / AUTO_FULL_FORWARD_DUTY, -1.0f, 1.0f);
			vescController.set_speed(scaledAutoSpeed);
			vescController.set_steering(output.target_steering);
		} else {
			vescController.stop();
			vescController.set_steering(0.5f);
		}
		cmd = vescController.toTwistCommand();
		steeringAngle = vescController.toSteeringAngleCommand();
		{
			std::lock_guard<std::mutex> lock(dataMutex);
			lastAutoTargetSpeed = output.target_speed;
			lastAutoTargetSteering = output.target_steering;
			lastScaledAutoSpeed = scaledAutoSpeed;
		}
	} else {
		cmd.linear.x = manualV;
		cmd.angular.z = 0.0;
		steeringAngle = vescController.steeringAngleFromAngularCommand(manualW);
	}

	{
		std::lock_guard<std::mutex> lock(dataMutex);
		lastPublishedLinear = static_cast<float>(cmd.linear.x);
		lastPublishedAngular = static_cast<float>(cmd.angular.z);
		lastSteeringAngleDeg = radians_to_degrees(steeringAngle);
		lastAutoComputeOk = autoComputeOk;
	}

	cmdPublisher->publish(cmd);
	std_msgs::msg::Float64 steeringMsg;
	steeringMsg.data = static_cast<double>(steeringAngle);
	frontLeftSteeringPublisher->publish(steeringMsg);
	frontRightSteeringPublisher->publish(steeringMsg);
}

void RobocarSimControllerNode::on_mode_update(const std_msgs::msg::Bool::SharedPtr& msg)
{
	std::lock_guard<std::mutex> lock(dataMutex);
	autonomousEnabled = msg->data;
}

void RobocarSimControllerNode::on_manual_cmd_update(const geometry_msgs::msg::Twist::SharedPtr& msg)
{
	std::lock_guard<std::mutex> lock(dataMutex);
	commandLinear = static_cast<float>(msg->linear.x);
	commandAngular = static_cast<float>(msg->angular.z);
}

void RobocarSimControllerNode::publish_flat_points(const std::vector<SimLidarPoint>& points)
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

void RobocarSimControllerNode::publish_cluster_markers(
	const std::vector<SimLidarPoint>& points,
	const std_msgs::msg::Header& header)
{
	visualization_msgs::msg::MarkerArray markerArray;

	visualization_msgs::msg::Marker clearMarker;
	clearMarker.header = header;
	clearMarker.ns = "lidar_clusters";
	clearMarker.id = 0;
	clearMarker.action = visualization_msgs::msg::Marker::DELETEALL;
	markerArray.markers.push_back(clearMarker);

	if (points.empty()) {
		clusterPublisher->publish(markerArray);
		return;
	}

	const float clusterDistanceSq = clusterDistanceThresholdM * clusterDistanceThresholdM;
	const int minClusterPoints = std::max(clusterMinPoints, 2);
	std::vector<geometry_msgs::msg::Point> currentCluster;
	currentCluster.reserve(points.size());
	auto to_cartesian_point = [](const SimLidarPoint& point) {
		geometry_msgs::msg::Point out;
		const float angleRad = degrees_to_radians(point.angleDeg);
		out.x = static_cast<double>(point.distanceMeters * std::cos(angleRad));
		out.y = static_cast<double>(point.distanceMeters * std::sin(angleRad));
		out.z = 0.0;
		return out;
	};

	auto flush_cluster = [&](int markerId) {
		if (static_cast<int>(currentCluster.size()) < minClusterPoints) {
			currentCluster.clear();
			return false;
		}

		visualization_msgs::msg::Marker marker;
		marker.header = header;
		marker.ns = "lidar_clusters";
		marker.id = markerId;
		marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
		marker.action = visualization_msgs::msg::Marker::ADD;
		marker.pose.orientation.w = 1.0;
		marker.scale.x = 0.025;
		marker.color.r = 0.15f;
		marker.color.g = 0.95f;
		marker.color.b = 0.35f;
		marker.color.a = 1.0f;
		marker.lifetime = rclcpp::Duration::from_seconds(0.25);
		marker.points = currentCluster;
		markerArray.markers.push_back(std::move(marker));
		currentCluster.clear();
		return true;
	};

	int markerId = 1;
	currentCluster.push_back(to_cartesian_point(points.front()));
	for (std::size_t i = 1; i < points.size(); ++i) {
		const auto& previous = points[i - 1];
		const auto& current = points[i];
		const float angleGapDeg = current.angleDeg - previous.angleDeg;
		const geometry_msgs::msg::Point currentPoint = to_cartesian_point(current);
		const geometry_msgs::msg::Point previousPoint = to_cartesian_point(previous);
		const double dx = currentPoint.x - previousPoint.x;
		const double dy = currentPoint.y - previousPoint.y;
		const double distanceSq = dx * dx + dy * dy;
		const bool sameCluster =
			angleGapDeg <= clusterMaxAngleGapDeg &&
			distanceSq <= static_cast<double>(clusterDistanceSq);

		if (!sameCluster) {
			if (flush_cluster(markerId)) {
				++markerId;
			}
		}
		currentCluster.push_back(currentPoint);
	}
	flush_cluster(markerId);

	clusterPublisher->publish(markerArray);
}

void RobocarSimControllerNode::print_scan_summary(const std::vector<SimLidarPoint>& points)
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

void RobocarSimControllerNode::print_menu() const
{
	std::cout << "\n=== Robocar Sim Menu ===\n"
			  << "h                : help menu\n"
			  << "s                : status\n"
			  << "a                : toggle auto print lidar summary\n"
			  << "mode auto        : enable autonomous mode\n"
			  << "mode manual      : enable manual cmd_vel mode\n"
			  << "cmd v w          : set manual cmd_vel (linear v, angular w)\n"
			  << "vesc s st        : set normalized VESC command (speed [-1..1], steering [0..1])\n"
			  << "stop             : set speed to 0\n"
			  << "q                : close menu input thread\n"
			  << "========================\n"
			  << std::flush;
}

void RobocarSimControllerNode::menu_loop()
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

		if (line == "mode auto") {
			std::lock_guard<std::mutex> lock(dataMutex);
			autonomousEnabled = true;
			std::cout << "mode=AUTO\n";
			continue;
		}

		if (line == "mode manual") {
			std::lock_guard<std::mutex> lock(dataMutex);
			autonomousEnabled = false;
			std::cout << "mode=MANUAL\n";
			continue;
		}

		if (line == "s") {
			std::size_t count = 0;
			double hz = 0.0;
			float v = 0.0f;
			float w = 0.0f;
			bool autoMode = true;
			{
				std::lock_guard<std::mutex> lock(dataMutex);
				count = lastPoints.size();
				hz = scanRateHz;
				v = commandLinear;
				w = commandAngular;
				autoMode = autonomousEnabled;
			}
			std::cout << std::fixed << std::setprecision(2)
					  << "status: points=" << count
					  << " scan_hz=" << hz
					  << " mode=" << (autoMode ? "AUTO" : "MANUAL")
					  << " manual_cmd_vel=(" << v << ", " << w << ")\n";
			continue;
		}

		if (line == "stop") {
			{
				std::lock_guard<std::mutex> lock(dataMutex);
				commandLinear = 0.0f;
				commandAngular = 0.0f;
			}
			vescController.stop();
			std::cout << "speed set to 0\n";
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
				std::cout << "manual cmd_vel set to " << v << " " << w << "\n";
			} else {
				std::cout << "invalid command, expected: cmd <linear> <angular>\n";
			}
			continue;
		}

		if (line.rfind("vesc ", 0) == 0) {
			std::istringstream iss(line);
			std::string token;
			float speed = 0.0f;
			float steering = 0.5f;
			iss >> token >> speed >> steering;
			if (!iss.fail()) {
				vescController.set_speed(speed);
				vescController.set_steering(steering);
				std::cout << "vesc set_speed=" << speed << " set_steering=" << steering << "\n";
			} else {
				std::cout << "invalid command, expected: vesc <speed> <steering>\n";
			}
			continue;
		}

		std::cout << "unknown command. type h for help\n";
	}
}
