cmake_minimum_required(VERSION 3.10)
project(LD19_SFML)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use pkg-config to find SFML (required on Ubuntu 18.04 ARM)
find_package(PkgConfig REQUIRED)

pkg_check_modules(SFML REQUIRED
    sfml-graphics
    sfml-window
    sfml-system
)

add_executable(ld19_lidar
    main.cpp
)

target_include_directories(ld19_lidar PRIVATE
    ${SFML_INCLUDE_DIRS}
)

target_link_libraries(ld19_lidar
    ${SFML_LIBRARIES}
)

target_compile_options(ld19_lidar PRIVATE
    -Wall
    -Wextra
    -O2
)

