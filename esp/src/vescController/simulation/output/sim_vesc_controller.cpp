#include "sim_vesc_controller.hpp"

#include <algorithm>
#include <cmath>

SimVescController::SimVescController(SimControlConfig config)
	: config(config)
{
}

bool SimVescController::isActive(void)
{
	std::lock_guard<std::mutex> lock(dataMutex);
	return active;
}

void SimVescController::stop(void)
{
	std::lock_guard<std::mutex> lock(dataMutex);
	speedCmd = 0.0f;
}

void SimVescController::deactivate(void)
{
	std::lock_guard<std::mutex> lock(dataMutex);
	active = false;
	speedCmd = 0.0f;
	steeringCmd = 0.5f;
}

void SimVescController::activate(void)
{
	std::lock_guard<std::mutex> lock(dataMutex);
	active = true;
}

void SimVescController::set_speed(float speed)
{
	std::lock_guard<std::mutex> lock(dataMutex);
	if (!active) {
		speedCmd = 0.0f;
		return;
	}
	speedCmd = std::clamp(speed, -1.0f, 1.0f);
}

void SimVescController::set_steering(float steering)
{
	std::lock_guard<std::mutex> lock(dataMutex);
	if (!active) {
		steeringCmd = 0.5f;
		return;
	}
	steeringCmd = std::clamp(steering, 0.0f, 1.0f);
}

geometry_msgs::msg::Twist SimVescController::toTwistCommand() const
{
	std::lock_guard<std::mutex> lock(dataMutex);

	geometry_msgs::msg::Twist cmd;
	if (!active) {
		return cmd;
	}

	const float signedSteering = (steeringCmd - 0.5f) * 2.0f;
	const float steer = config.reverseSteering ? -signedSteering : signedSteering;
	const float linear = speedCmd * config.maxLinearSpeedMps;
	const float curvatureScale = std::clamp(std::abs(speedCmd), 0.0f, 1.0f);
	const float angular = steer * config.maxAngularSpeedRadps * curvatureScale;

	cmd.linear.x = linear;
	cmd.angular.z = angular;
	return cmd;
}

