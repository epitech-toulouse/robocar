/*
** EPITECH PROJECT, 2026
** robocar
** File description:
** camera sensor interface
*/

#ifndef CAMERA_SENSOR_API_HPP
#define CAMERA_SENSOR_API_HPP

struct CameraStatus {
    bool has_data = false;
    bool connected = false;
    bool stop_detected = false;
    float steering_percent = 0.0f;
    float steering_weight = 0.0f;
    float stop_weight = 0.0f;
    float speed_percent = 0.0f;
};

class CameraSensorApi {
public:
    virtual ~CameraSensorApi() = default;

    // Return false if the camera sensor is not available.
    virtual bool isActive(void) = 0;
    // Return false if no stop status can be gathered.
    virtual bool getStop(bool &output) = 0;
    // Legacy API name kept for compatibility; returns steering percent.
    virtual bool getHeading(float &output) = 0;
    // Return false when no speed information is available from the camera.
    virtual bool getSpeed(float &output) = 0;
    // Return true if camera status can be gathered.
    virtual bool getStatus(CameraStatus &output) = 0;
};

#endif /* CAMERA_SENSOR_API_HPP */
