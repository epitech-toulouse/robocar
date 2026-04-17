#include "rclcpp/rclcpp.hpp"
#include "robocar_sim_controller_node.hpp"

int main(int argc, char** argv)
{
	rclcpp::init(argc, argv);
	auto node = std::make_shared<RobocarSimControllerNode>();
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
