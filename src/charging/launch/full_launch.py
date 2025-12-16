from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.events import Shutdown
from launch.actions import IncludeLaunchDescription  
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, EmitEvent
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # MoveIt 
    ur_type = LaunchConfiguration("ur_type", default="ur7e")

    # Path to the MoveIt launch file
    moveit_launch_file = os.path.join(
                get_package_share_directory("ur_moveit_config"),
                "launch",
                "ur_moveit.launch.py"
            )

    # Include the MoveIt launch description
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(moveit_launch_file),
        launch_arguments={
            "ur_type": ur_type,
        }.items(),
    )


    calibration_node = Node(
        package='charging',
        executable='calibration',
        name='calibration_node',
        output='screen',
    )

    cv_node = Node(
        package='charging',
        executable='cv',
        name='cv_node',
        output='screen',
    )

    spiral = Node(
        package='charging',
        executable='spiral',
        name='spiral_node',
        output='screen',
    )

    handeye_node = Node(
        package='charging',
        executable='handeye',
        name='handeye_node',
        output='screen',
    )

    # Static transform publisher for aruco
    # G matrix: ar_marker_10 -> base_link (aruco to base)
    # Translation: [0.0, 0.16, -0.13]
    # Rotation: [-1, 0, 0; 0, 0, 1; 0, 1, 0] = quaternion [0.5, -0.5, -0.5, 0.5]
    static_transform_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_aruco_static_transform_publisher',
        arguments=['0.0', '0.16', '-0.13', '0', '0.7071068', '0.7071068', '0', 'base_link', 'ar_marker_10'],
        output='screen',
    )

    # # Static transform publisher for camera
    # static_transform_publisher = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='camera_static_transform_publisher',
    #     arguments=['0.0508', '0', '0', '0', '0', '0', 'tool0', 'camera_link'],
    #     output='screen',
    # )
    # z downwards, x right of image, y down of image

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('charging'), 'launch', 'camera_launch.py')
        )
    )

    return LaunchDescription([
        moveit_launch,
        calibration_node,
        handeye_node,
        spiral,
        cv_node,
        static_transform_publisher,
        camera_launch
    ])
