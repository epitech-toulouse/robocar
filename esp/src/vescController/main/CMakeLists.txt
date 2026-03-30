idf_component_register(SRCS "main.cpp" "vescController.cpp" "lidarReader.cpp" "bluetooth_receiver.cpp" "vescLidarUart.c" "rmt-uart/rmt_uart.c" "drive.cpp" "usb_gps.cpp"
                       PRIV_REQUIRES driver bt nvs_flash
                       INCLUDE_DIRS "." "rmt-uart")
