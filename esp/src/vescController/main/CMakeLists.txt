idf_component_register(SRCS "main.cpp" "vescController.cpp" "lidarReader.cpp" "wifi_receiver.cpp" "vescLidarUart.c" "rmt-uart/rmt_uart.c" "drive.cpp"
                       PRIV_REQUIRES driver esp_wifi esp_netif esp_event nvs_flash lwip esp_http_server
                       INCLUDE_DIRS "." "rmt-uart")
