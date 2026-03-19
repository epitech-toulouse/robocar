idf_component_register(SRCS "main.cpp" "vescController.cpp" "lidarReader.cpp" "bluetooth_receiver.cpp" "vescLidarUart.c"
                       PRIV_REQUIRES driver bt nvs_flash
                       INCLUDE_DIRS ".")

                       
