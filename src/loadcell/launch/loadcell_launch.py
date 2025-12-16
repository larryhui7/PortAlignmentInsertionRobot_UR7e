from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """
    Launch file for the loadcell node.
    Allows configuration of serial port, baud rate, and publish rate.
    """
    
    # Declare launch arguments
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyUSB0',
        description='Serial port path (e.g., /dev/ttyUSB0, /dev/ttyACM0)'
    )
    
    baud_rate_arg = DeclareLaunchArgument(
        'baud_rate',
        default_value='115200',
        description='Serial baud rate'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='10.0',
        description='Publish rate in Hz'
    )
    
    # Loadcell node
    loadcell_node = Node(
        package='loadcell',
        executable='loadcell_node',
        name='loadcell_node',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baud_rate': LaunchConfiguration('baud_rate'),
            'publish_rate': LaunchConfiguration('publish_rate'),
        }]
    )
    
    return LaunchDescription([
        serial_port_arg,
        baud_rate_arg,
        publish_rate_arg,
        loadcell_node,
    ])

