idf_component_register(SRCS "main.cpp" "vescController.cpp" "lidarReader.cpp" "bluetooth_receiver.cpp"
                       PRIV_REQUIRES driver bt nvs_flash
                       INCLUDE_DIRS ".")

                       
