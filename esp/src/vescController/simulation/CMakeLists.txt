cmake_minimum_required(VERSION 3.10)
project(robocar_sim)

if(NOT CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
add_executable(robocar_sim_controller
	main.cpp
)

install(TARGETS robocar_sim_controller
	DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
	launch
	worlds
	config
	DESTINATION share/${PROJECT_NAME}
)

ament_package()
