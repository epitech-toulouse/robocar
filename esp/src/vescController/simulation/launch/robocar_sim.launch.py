from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
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

    start_controller_arg = DeclareLaunchArgument(
        "start_controller",
        default_value="true",
        description="Start robocar_sim_controller from launch",
    )

    start_controller = LaunchConfiguration("start_controller")

    controller_menu_enabled_arg = DeclareLaunchArgument(
        "controller_menu_enabled",
        default_value="true",
        description="Enable stdin menu in robocar_sim_controller",
    )

    controller_menu_enabled = LaunchConfiguration("controller_menu_enabled")

    start_rviz_arg = DeclareLaunchArgument(
        "start_rviz",
        default_value="false",
        description="Start RViz with LiDAR view",
    )

    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=PathJoinSubstitution([package_share, "config", "robocar_lidar.rviz"]),
        description="Path to RViz config",
    )

    start_rviz = LaunchConfiguration("start_rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    startup_delay_arg = DeclareLaunchArgument(
        "startup_delay",
        default_value="2.0",
        description="Delay non-Gazebo nodes startup to let the GUI/world settle",
    )

    startup_delay = LaunchConfiguration("startup_delay")

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
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/robocar/front_left_steering_cmd@std_msgs/msg/Float64]gz.msgs.Double",
            "/robocar/front_right_steering_cmd@std_msgs/msg/Float64]gz.msgs.Double",
        ],
    )

    controller = Node(
        package="robocar_sim",
        executable="robocar_sim_controller",
        name="robocar_sim_controller",
        output="screen",
        condition=IfCondition(start_controller),
        parameters=[
            PathJoinSubstitution([package_share, "config", "controller.yaml"]),
            {"menu_enabled": controller_menu_enabled},
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="robocar_rviz",
        output="screen",
        condition=IfCondition(start_rviz),
        parameters=[{"use_sim_time": True}],
        arguments=["-d", rviz_config],
    )

    static_tf_lidar = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="robocar_lidar_static_tf",
        output="screen",
        arguments=["0", "0", "0", "0", "0", "0", "map", "robocar/lidar_link/top_lidar"],
    )

    return LaunchDescription([
        world_arg,
        start_controller_arg,
        controller_menu_enabled_arg,
        start_rviz_arg,
        rviz_config_arg,
        startup_delay_arg,
        gazebo,
        TimerAction(
            period=startup_delay,
            actions=[
                bridge,
                controller,
                static_tf_lidar,
                rviz,
            ],
        ),
    ])
