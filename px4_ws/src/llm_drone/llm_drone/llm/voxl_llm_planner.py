#!/usr/bin/env python3
"""
VOXL2 HITL LLM Trajectory Planner
Tailored for voxl-mpa-to-ros2 HITL setup

Subscribes to VOXL MPA topics:
  - /tracking (tracking camera - sensor_msgs/Image)
  - /tof_depth (ToF depth - sensor_msgs/Image) 
  - /qvio (VIO odometry - nav_msgs/Odometry)
  - /imu_apps (IMU - sensor_msgs/Imu)

Publishes to PX4 via micro-XRCE-DDS:
  - /fmu/in/trajectory_setpoint (px4_msgs/TrajectorySetpoint)
  - /llm/trajectory (geometry_msgs/PoseStamped)

Usage:
  ros2 run px4_vision voxl_llm_planner
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode, VehicleCommand
from cv_bridge import CvBridge
import cv2
import numpy as np
import base64
import json
from openai import OpenAI
import os


class VoxlLLMPlanner(Node):
    
    def __init__(self):
        super().__init__('voxl_llm_planner')
        
        # OpenAI setup
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.get_logger().error('OPENAI_API_KEY not set!')
            raise ValueError('Set OPENAI_API_KEY')
        
        self.client = OpenAI(api_key=api_key)
        
        # Load prompt
        prompt_file = os.path.join(
            os.path.dirname(__file__), 
            '../config/voxl_llm_prompt.txt'
        )
        try:
            with open(prompt_file, 'r') as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            self.get_logger().warn(f'Prompt file not found: {prompt_file}')
            self.system_prompt = self._get_default_prompt()
        
        # Parameters
        self.declare_parameter('goal_x', 15.0)
        self.declare_parameter('goal_y', 10.0)
        self.declare_parameter('goal_z', 2.0)  # Positive Z is up in NED
        self.declare_parameter('update_rate', 1.0)
        self.declare_parameter('gpt_model', 'gpt-4o')
        
        self.goal = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value,
            self.get_parameter('goal_z').value
        ])
        
        # State
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.tracking_image = None
        self.depth_image = None
        self.bridge = CvBridge()
        self.offboard_enabled = False
        
        # Trajectory storage
        self.llm_trajectory = []
        
        # VOXL MPA Subscribers
        self.tracking_sub = self.create_subscription(
            Image, '/tracking', self.tracking_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/tof_depth', self.depth_callback, 10)
        self.qvio_sub = self.create_subscription(
            Odometry, '/qvio', self.qvio_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu_apps', self.imu_callback, 10)
        
        # PX4 Publishers (micro-XRCE-DDS)
        self.traj_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.llm_traj_pub = self.create_publisher(
            PoseStamped, '/llm/trajectory', 10)
        
        # Timers
        update_period = 1.0 / self.get_parameter('update_rate').value
        self.planner_timer = self.create_timer(update_period, self.plan_trajectory)
        self.offboard_timer = self.create_timer(0.1, self.publish_offboard_heartbeat)
        
        self.get_logger().info('VOXL2 HITL LLM Planner initialized')
        self.get_logger().info(f'Goal: {self.goal}')
    
    def _get_default_prompt(self):
        return """You are a drone motion planning system using VOXL2 sensors."""
    
    def tracking_callback(self, msg):
        """VOXL tracking camera (640x480 grayscale)"""
        try:
            self.tracking_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
        except Exception as e:
            self.get_logger().error(f'Tracking image error: {e}')
    
    def depth_callback(self, msg):
        """VOXL ToF depth sensor"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            self.get_logger().error(f'Depth image error: {e}')
    
    def qvio_callback(self, msg):
        """VOXL qVIO odometry (NED frame)"""
        self.current_position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.current_velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
    
    def imu_callback(self, msg):
        """VOXL IMU data (for reference if needed)"""
        pass
    
    def publish_offboard_heartbeat(self):
        """Required heartbeat for PX4 offboard mode"""
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        self.offboard_mode_pub.publish(msg)
    
    def arm_vehicle(self):
        """Arm the vehicle"""
        cmd = VehicleCommand()
        cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        cmd.param1 = 1.0  # Arm
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        self.vehicle_command_pub.publish(cmd)
    
    def encode_image(self, image):
        """Encode image to base64"""
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        image_small = cv2.resize(image, (512, 384))
        _, buffer = cv2.imencode('.jpg', image_small)
        return base64.b64encode(buffer).decode('utf-8')
    
    def create_depth_visualization(self):
        """Create colored depth map"""
        if self.depth_image is None:
            return None
        
        depth_norm = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        
        return depth_colored
    
    def plan_trajectory(self):
        """Query LLM for next trajectory point"""
        
        if self.tracking_image is None or self.depth_image is None:
            self.get_logger().warn('Waiting for camera data...', throttle_duration_sec=2.0)
            return
        
        # Check goal
        dist_to_goal = np.linalg.norm(self.goal - self.current_position)
        if dist_to_goal < 0.5:
            self.get_logger().info('Goal reached!', throttle_duration_sec=1.0)
            return
        
        # Query LLM
        try:
            next_position = self.query_llm()
            
            if next_position is not None:
                self.llm_trajectory.append(next_position)
                
                # Publish trajectory
                traj_msg = PoseStamped()
                traj_msg.header.stamp = self.get_clock().now().to_msg()
                traj_msg.header.frame_id = 'map'
                traj_msg.pose.position.x = next_position[0]
                traj_msg.pose.position.y = next_position[1]
                traj_msg.pose.position.z = next_position[2]
                self.llm_traj_pub.publish(traj_msg)
                
                # Publish to PX4
                self.publish_trajectory_setpoint(next_position)
                
        except Exception as e:
            self.get_logger().error(f'LLM query failed: {e}')
    
    def query_llm(self):
        """Query GPT-4 Vision"""
        
        # Encode images
        tracking_base64 = self.encode_image(self.tracking_image)
        depth_colored = self.create_depth_visualization()
        depth_base64 = self.encode_image(depth_colored)
        
        user_message = f"""
Current State (NED Frame):
- Position: [{self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f}] m
- Velocity: [{self.current_velocity[0]:.2f}, {self.current_velocity[1]:.2f}, {self.current_velocity[2]:.2f}] m/s
- Goal: [{self.goal[0]:.2f}, {self.goal[1]:.2f}, {self.goal[2]:.2f}] m
- Distance to goal: {np.linalg.norm(self.goal - self.current_position):.2f} m

Sensors:
- VOXL Tracking Camera (640x480 grayscale)
- VOXL ToF Depth (colored: red=close, blue=far)

Generate next waypoint (x, y, z) in NED coordinates.
Respond with JSON: {{"x": float, "y": float, "z": float, "reasoning": "..."}}
"""
        
        response = self.client.chat.completions.create(
            model=self.get_parameter('gpt_model').value,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{tracking_base64}"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{depth_base64}"}
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        content = response.choices[0].message.content
        self.get_logger().info(f'LLM: {content}')
        
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]
            data = json.loads(json_str)
            
            next_position = np.array([
                float(data['x']),
                float(data['y']),
                float(data['z'])
            ])
            
            self.get_logger().info(f"Reasoning: {data.get('reasoning', 'N/A')}")
            return next_position
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.get_logger().error(f'Parse error: {e}')
            return None
    
    def publish_trajectory_setpoint(self, position):
        """Publish trajectory setpoint to PX4"""
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = [float(position[0]), float(position[1]), float(position[2])]
        msg.yaw = 0.0
        self.traj_setpoint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VoxlLLMPlanner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()