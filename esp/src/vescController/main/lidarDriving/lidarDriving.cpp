#include "lidarDriving.hpp"

LidarDriving::LidarDriving(ILidarSensor &lidar) : lidar(lidar) {
    
}

LidarDriving::~LidarDriving() {
    
}

bool LidarDriving::available(void) {
    return false;
}

bool LidarDriving::compute(DrivingAlgorithmOutput &output) {
    output = DEFAULT_DRIVING_ALGORITHM_OUTPUT;
    return false;
}

bool LidarDriving::feedLidarData(const lidar_array_t &data) {
    return false;
}