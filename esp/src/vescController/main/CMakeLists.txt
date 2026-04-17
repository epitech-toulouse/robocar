idf_component_register(SRCS "gpsdrive.cpp" "gpsUsbHost.cpp" "main.cpp" "vescController.cpp" "lidarReader.cpp" "bluetooth_receiver.cpp" "wifi_control_server.cpp" "vescLidarUart.c" "rmt-uart/rmt_uart.c" "drive.cpp" "algo/LidarObstacleAvoidance/lidar_obstacle_avoidance_algorithm.cpp"
                       PRIV_REQUIRES driver bt esp_wifi esp_netif esp_event nvs_flash lwip esp_http_server usb esp_driver_uart esp_driver_gpio
                       INCLUDE_DIRS "." "rmt-uart")
