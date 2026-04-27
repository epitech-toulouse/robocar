#include "../api/driving_algorithm_interface.hpp"
#include "../api/lidar_sensor_api.hpp"

#ifndef LIDAR_DRIVING_HPP
#define LIDAR_DRIVING_HPP

class LidarDriving : public DrivingAlgorithmApi {
public:
    LidarDriving(LidarSensorApi &lidar);
    ~LidarDriving() override;
    
    bool available(void) override;
    bool compute(DrivingAlgorithmOutput &output) override;
private:
    LidarSensorApi &lidar;
};

#endif /* LIDAR_DRIVING_HPP */