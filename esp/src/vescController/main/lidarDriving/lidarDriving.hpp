#include "../api/driving_algorithm_interface.hpp"
#include "../api/lidar_sensor_api.hpp"

#ifndef LIDAR_DRIVING_HPP
#define LIDAR_DRIVING_HPP

class LidarDriving : public IDrivingAlgorithm {
public:
    LidarDriving(ILidarSensor &lidar);
    ~LidarDriving() override;
    
    bool available(void) override;
    bool compute(DrivingAlgorithmOutput &output) override;
private:
    ILidarSensor &lidar;
};

#endif /* LIDAR_DRIVING_HPP */