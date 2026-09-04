"""Launches vla_policy_client against either backend.

For Isaac Sim: start ../../isaac/pick_place_scene_bridge.py separately
first (with Isaac Sim's own python.sh -- it needs the Kit runtime, so it
isn't launched from here), and start openpi's policy server
(scripts/serve_policy.py in your openpi checkout) with the fine-tuned
pi0_ur5e_pick_place checkpoint, then run this with robot_backend:=isaac_sim.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('vla_bridge')
    params_file = os.path.join(pkg_share, 'config', 'params.yaml')

    robot_backend = LaunchConfiguration('robot_backend')

    return LaunchDescription([
        DeclareLaunchArgument('robot_backend', default_value='isaac_sim',
                               description="'rtde' (real UR5e) or 'isaac_sim' "
                                           "(needs ../../isaac/pick_place_scene_bridge.py running separately)"),
        Node(
            package='vla_bridge',
            executable='vla_policy_client',
            name='vla_policy_client',
            parameters=[params_file, {'robot_backend': robot_backend}],
            output='screen',
        ),
    ])
