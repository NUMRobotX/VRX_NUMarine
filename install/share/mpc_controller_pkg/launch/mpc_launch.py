from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mpc_controller_pkg',
            executable='mpc_node',
            name='mpc_node',
            output='screen',
        ),
    ])