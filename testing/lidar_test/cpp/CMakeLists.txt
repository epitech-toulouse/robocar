cmake_minimum_required(VERSION 3.10)
project(LD19_SFML)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SFML
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# Add executable
add_executable(ld19_lidar main.cpp)

# Link SFML libraries
target_link_libraries(ld19_lidar sfml-graphics sfml-window sfml-system)

# Compiler flags
target_compile_options(ld19_lidar PRIVATE -Wall -Wextra -O2)

# For Fedora Silverblue: Enable access to /dev/ttyUSB0
# Run with: sudo ./ld19_lidar
# Or add user to dialout group: sudo usermod -aG dialout $USER
