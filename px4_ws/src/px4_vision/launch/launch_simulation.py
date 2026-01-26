#!/usr/bin/env python3
"""
ROS2 Launch file for PX4 Gazebo Simulation with MPC Controller
Save as: ~/px4_ws/src/px4_vision/launch/launch_simulation.py

Launch with: ros2 launch px4_vision launch_simulation.py
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    """Generate launch description for complete simulation"""
    
    # Launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='obstacle_course',
        description='Gazebo world name'
    )
    
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo headless'
    )
    
    goal_x = DeclareLaunchArgument('goal_x', default_value='15.0')
    goal_y = DeclareLaunchArgument('goal_y', default_value='10.0')
    goal_z = DeclareLaunchArgument('goal_z', default_value='-2.0')
    
    # PX4 SITL
    px4_sitl = ExecuteProcess(
        cmd=[
            'make', 'px4_sitl', 
            'gz_iris_depth_camera'
        ],
        cwd=os.path.expanduser('~/PX4-Autopilot'),
        output='screen',
        shell=True
    )
    
    # MicroXRCE Agent (PX4-ROS2 bridge)
    micro_xrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen'
    )
    
    # Wait for PX4 to start
    delayed_agent = TimerAction(
        period=5.0,
        actions=[micro_xrce_agent]
    )
    
    # MPC Controller
    mpc_controller = Node(
        package='px4_vision',
        executable='mpc_vision_controller',
        name='mpc_vision_controller',
        output='screen',
        parameters=[{
            'goal_x': LaunchConfiguration('goal_x'),
            'goal_y': LaunchConfiguration('goal_y'),
            'goal_z': LaunchConfiguration('goal_z'),
            'prediction_horizon': 10,
            'control_horizon': 5,
            'dt': 0.2,
            'max_velocity': 2.0,
            'obstacle_threshold': 2.5,
            'safety_distance': 1.5
        }]
    )
    
    # Wait for everything to initialize
    delayed_controller = TimerAction(
        period=10.0,
        actions=[mpc_controller]
    )
    
    # RViz for visualization
    rviz_config = os.path.join(
        os.path.expanduser('~/px4_ws/src/px4_vision/rviz'),
        'px4_vision.rviz'
    )
    
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )
    
    delayed_rviz = TimerAction(
        period=8.0,
        actions=[rviz]
    )
    
    return LaunchDescription([
        world_arg,
        headless_arg,
        goal_x,
        goal_y,
        goal_z,
        px4_sitl,
        delayed_agent,
        delayed_controller,
        delayed_rviz
    ])