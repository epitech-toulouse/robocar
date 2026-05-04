cmake_minimum_required(VERSION 3.10)
project(robocar_sim)

if(NOT CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

get_filename_component(SIM_DIR_REAL "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
get_filename_component(VESC_CONTROLLER_ROOT "${SIM_DIR_REAL}/.." REALPATH)

set(SHARED_LIDAR_ALGO_DIR "${VESC_CONTROLLER_ROOT}/main/algo/LidarObstacleAvoidance")
set(SHARED_MAIN_DIR "${VESC_CONTROLLER_ROOT}/main")

add_executable(robocar_sim_controller
	main.cpp
	robocar_sim_controller_node.cpp
	sensors/sim_lidar_sensor.cpp
	output/sim_vesc_controller.cpp
	${SHARED_LIDAR_ALGO_DIR}/drive.cpp
	${SHARED_LIDAR_ALGO_DIR}/lidar_obstacle_avoidance_algorithm.cpp
)

target_include_directories(robocar_sim_controller PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${SHARED_MAIN_DIR}
)

ament_target_dependencies(robocar_sim_controller
	rclcpp
	sensor_msgs
	geometry_msgs
	std_msgs
	visualization_msgs
)

install(TARGETS robocar_sim_controller
	DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
	tools/robocar_menu_qt.py
	DESTINATION lib/${PROJECT_NAME}
	RENAME robocar_menu_qt
)

install(DIRECTORY
	launch
	worlds
	config
	DESTINATION share/${PROJECT_NAME}
)

ament_package()
