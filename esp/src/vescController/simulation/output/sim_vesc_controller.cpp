#include "sim_vesc_controller.hpp"

#include <algorithm>
#include <chrono>
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
	currentLinearMps = 0.0f;
	hasLastCommandTime = false;
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

	const auto now = std::chrono::steady_clock::now();
	float dt = 0.1f;
	if (hasLastCommandTime) {
		dt = std::chrono::duration<float>(now - lastCommandTime).count();
	}
	lastCommandTime = now;
	hasLastCommandTime = true;

	const float targetLinear = speedCmd * config.maxLinearSpeedMps;
	const float acceleration = std::max(0.0f, config.maxAccelerationMps2);
	const float maxDelta = acceleration * std::clamp(dt, 0.0f, 0.25f);
	if (acceleration <= 0.0f) {
		currentLinearMps = targetLinear;
	} else {
		currentLinearMps += std::clamp(targetLinear - currentLinearMps, -maxDelta, maxDelta);
	}

	cmd.linear.x = currentLinearMps;
	cmd.angular.z = 0.0;
	return cmd;
}

float SimVescController::toSteeringAngleCommand() const
{
	std::lock_guard<std::mutex> lock(dataMutex);
	const float signedSteering = (steeringCmd - 0.5f) * 2.0f;
	const float steer = config.reverseSteering ? -signedSteering : signedSteering;
	return std::clamp(
		steer * config.maxSteeringAngleRad,
		-config.maxSteeringAngleRad,
		config.maxSteeringAngleRad);
}

float SimVescController::steeringAngleFromAngularCommand(float angular) const
{
	const float steer = std::clamp(angular / std::max(config.maxAngularSpeedRadps, 0.01f), -1.0f, 1.0f);
	return std::clamp(
		steer * config.maxSteeringAngleRad,
		-config.maxSteeringAngleRad,
		config.maxSteeringAngleRad);
}
