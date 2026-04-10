idf_component_register(SRCS "gpsdrive.cpp" "gpsUsbHost.cpp" "main.cpp" "vescController.cpp" "lidarReader.cpp" "bluetooth_receiver.cpp" "vescLidarUart.c" "rmt-uart/rmt_uart.c" "drive.cpp" "algo/LidarObstacleAvoidance/lidar_obstacle_avoidance_algorithm.cpp"
                       PRIV_REQUIRES driver bt nvs_flash usb
                       INCLUDE_DIRS "." "rmt-uart")
