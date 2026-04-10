from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("robocar_sim")

    world_arg = DeclareLaunchArgument(
        "world",
        default_value=PathJoinSubstitution([package_share, "worlds", "robocar_empty.sdf"]),
        description="Path to Gazebo world",
    )

    world = LaunchConfiguration("world")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])
        ),
        launch_arguments={"gz_args": ["-r ", world]}.items(),
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="robocar_ros_gz_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/gps/fix@sensor_msgs/msg/NavSatFix[gz.msgs.NavSat",
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
        ],
    )

    controller = Node(
        package="robocar_sim",
        executable="robocar_sim_controller",
        name="robocar_sim_controller",
        output="screen",
        parameters=[
            PathJoinSubstitution([package_share, "config", "controller.yaml"]),
        ],
    )

    return LaunchDescription([
        world_arg,
        gazebo,
        bridge,
        controller,
    ])
