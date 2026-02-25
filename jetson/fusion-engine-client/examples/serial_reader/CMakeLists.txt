# Add the executable.
add_executable(serial_reader serial_reader.cc)

# Link against the FusionEngine library and the common examples library.
target_link_libraries(serial_reader PUBLIC fusion_engine_client)
target_link_libraries(serial_reader PUBLIC print_message)

