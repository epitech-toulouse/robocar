#pragma once

struct SimControlConfig {
	float maxLinearSpeedMps = 1.8f;
	float maxAccelerationMps2 = 1.5f;
	float maxAngularSpeedRadps = 2.2f;
	float maxSteeringAngleRad = 0.6108652382f;
	bool reverseSteering = false;
};
