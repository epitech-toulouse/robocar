idf_component_register(SRCS "gpsUsbHost.cpp" "main.cpp" "vescController.cpp" "lidarReader.cpp" "bluetooth_receiver.cpp" "vescLidarUart.c" "rmt-uart/rmt_uart.c" "drive.cpp"
                       PRIV_REQUIRES driver bt nvs_flash usb
                       INCLUDE_DIRS "." "rmt-uart")
