#pragma once

#include <mutex>

#include "geometry_msgs/msg/twist.hpp"

#include "api/vesc_controller_api.hpp"
#include "simulation_types.hpp"

class SimVescController : public IVescController {
public:
	explicit SimVescController(SimControlConfig config = {});
	~SimVescController() override = default;

	bool isActive(void) override;
	void stop(void) override;
	void deactivate(void) override;
	void activate(void) override;
	void set_speed(float speed) override;
	void set_steering(float steering) override;

	geometry_msgs::msg::Twist toTwistCommand() const;

private:
	SimControlConfig config;

	mutable std::mutex dataMutex;
	bool active = true;
	float speedCmd = 0.0f;
	float steeringCmd = 0.5f;
};

