idf_component_register(SRCS "main.cpp" "vescController.cpp" "lidarReader.cpp" "wifi_receiver.cpp" "vescLidarUart.c" "drive.cpp"
                       PRIV_REQUIRES driver esp_wifi esp_netif esp_event nvs_flash lwip esp_http_server esp_hal_uart esp_driver_uart esp_driver_gpio
                       INCLUDE_DIRS ".")
