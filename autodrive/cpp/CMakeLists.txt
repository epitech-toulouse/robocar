cmake_minimum_required(VERSION 3.10)
project(lidar_parser)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build shared library
add_library(lidar_parser SHARED
    lidar_parser.cpp
)

target_compile_options(lidar_parser PRIVATE
    -Wall
    -Wextra
    -O2
    -fPIC
)

# Link pthread for threading
target_link_libraries(lidar_parser
    pthread
)

# Install to parent directory for easy Python access
install(TARGETS lidar_parser
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}
)
