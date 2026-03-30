idf_component_register(SRCS "gpsdrive.cpp" "gpsUsbHost.cpp" "main.cpp" "vescController.cpp" "lidarReader.cpp" "wifi_receiver.cpp" "vescLidarUart.c" "drive.cpp"
                       PRIV_REQUIRES driver esp_wifi esp_netif esp_event nvs_flash lwip esp_http_server usb esp_driver_uart esp_driver_gpio
                       INCLUDE_DIRS ".")
