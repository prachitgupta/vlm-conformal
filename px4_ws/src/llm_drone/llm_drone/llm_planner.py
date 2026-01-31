#!/usr/bin/env python3
"""
LLM-based Trajectory Planner using OpenAI GPT-4
Save as: ros2_ws/src/px4_vision/px4_vision/llm_planner.py

Subscribes to:
  - /fmu/out/vehicle_odometry (drone position)
  - /camera/rgb (RGB image)
  - /camera/depth (Depth image)

Publishes to:
  - /llm/trajectory (LLM-generated trajectory)
  - /fmu/in/trajectory_setpoint (commands to PX4)

Compares with:
  - /mpc/trajectory (MPC ground truth)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import base64
import json
from openai import OpenAI
import os


class LLMTrajectoryPlanner(Node):
    """
    Uses GPT-4 Vision to generate trajectories based on RGB+Depth camera input
    """
    
    def __init__(self):
        super().__init__('llm_trajectory_planner')
        
        # OpenAI API setup
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.get_logger().error('OPENAI_API_KEY environment variable not set!')
            raise ValueError('Set OPENAI_API_KEY before running')
        
        self.client = OpenAI(api_key=api_key)
        
        # Load prompt from file
        prompt_file = os.path.join(
            os.path.dirname(__file__), 
            '../config/llm_prompt.txt'
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
        self.declare_parameter('goal_z', -2.0)
        self.declare_parameter('update_rate', 1.0)  # Hz
        self.declare_parameter('gpt_model', 'gpt-4o')  # gpt-4o has vision
        
        self.goal = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value,
            self.get_parameter('goal_z').value
        ])
        
        # State
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.rgb_image = None
        self.depth_image = None
        self.bridge = CvBridge()
        
        # Trajectory storage
        self.llm_trajectory = []
        self.mpc_trajectory = []
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/fmu/out/vehicle_odometry', self.odometry_callback, 10)
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth', self.depth_callback, 10)
        self.mpc_traj_sub = self.create_subscription(
            PoseStamped, '/mpc/trajectory', self.mpc_trajectory_callback, 10)
        
        # Publishers
        self.llm_traj_pub = self.create_publisher(
            PoseStamped, '/llm/trajectory', 10)
        self.cmd_vel_pub = self.create_publisher(
            TwistStamped, '/fmu/in/setpoint_velocity', 10)
        
        # Timer for LLM queries
        update_period = 1.0 / self.get_parameter('update_rate').value
        self.timer = self.create_timer(update_period, self.plan_trajectory)
        
        self.get_logger().info('LLM Trajectory Planner initialized')
        self.get_logger().info(f'Goal: {self.goal}')
    
    def _get_default_prompt(self):
        """Fallback prompt if file not found"""
        return """You are a drone motion planning system. Generate safe trajectories."""
    
    def rgb_callback(self, msg):
        """Store latest RGB image"""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'RGB conversion error: {e}')
    
    def depth_callback(self, msg):
        """Store latest depth image"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            self.get_logger().error(f'Depth conversion error: {e}')
    
    def odometry_callback(self, msg):
        """Update current state"""
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
    
    def mpc_trajectory_callback(self, msg):
        """Store MPC trajectory for comparison"""
        mpc_point = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.mpc_trajectory.append(mpc_point)
    
    def encode_image(self, image):
        """Encode image to base64 for OpenAI API"""
        # Resize to reduce tokens
        image_small = cv2.resize(image, (512, 384))
        _, buffer = cv2.imencode('.jpg', image_small)
        return base64.b64encode(buffer).decode('utf-8')
    
    def create_depth_visualization(self):
        """Create colored depth map for better LLM understanding"""
        if self.depth_image is None:
            return None
        
        # Normalize depth to 0-255
        depth_norm = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        
        return depth_colored
    
    def plan_trajectory(self):
        """Query LLM for next trajectory point"""
        
        # Check if we have all data
        if self.rgb_image is None or self.depth_image is None:
            self.get_logger().warn('Waiting for camera data...', throttle_duration_sec=2.0)
            return
        
        # Check if goal reached
        dist_to_goal = np.linalg.norm(self.goal - self.current_position)
        if dist_to_goal < 0.5:
            self.get_logger().info('Goal reached!', throttle_duration_sec=1.0)
            self.publish_zero_velocity()
            return
        
        # Query LLM
        try:
            next_position = self.query_llm()
            
            if next_position is not None:
                # Store trajectory
                self.llm_trajectory.append(next_position)
                
                # Publish trajectory
                traj_msg = PoseStamped()
                traj_msg.header.stamp = self.get_clock().now().to_msg()
                traj_msg.header.frame_id = 'map'
                traj_msg.pose.position.x = next_position[0]
                traj_msg.pose.position.y = next_position[1]
                traj_msg.pose.position.z = next_position[2]
                self.llm_traj_pub.publish(traj_msg)
                
                # Compute velocity command
                velocity = self.compute_velocity_command(next_position)
                self.publish_velocity(velocity)
                
                # Compute error if MPC data available
                if len(self.mpc_trajectory) > 0:
                    error = self.compute_trajectory_error()
                    self.get_logger().info(
                        f'LLM vs MPC error: {error:.3f}m',
                        throttle_duration_sec=1.0
                    )
                
        except Exception as e:
            self.get_logger().error(f'LLM query failed: {e}')
    
    def query_llm(self):
        """Query OpenAI GPT-4 Vision for next waypoint"""
        
        # Encode images
        rgb_base64 = self.encode_image(self.rgb_image)
        depth_colored = self.create_depth_visualization()
        depth_base64 = self.encode_image(depth_colored)
        
        # Create user message
        user_message = f"""
Current State:
- Position (NED): [{self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f}] meters
- Velocity: [{self.current_velocity[0]:.2f}, {self.current_velocity[1]:.2f}, {self.current_velocity[2]:.2f}] m/s
- Goal (NED): [{self.goal[0]:.2f}, {self.goal[1]:.2f}, {self.goal[2]:.2f}] meters
- Distance to goal: {np.linalg.norm(self.goal - self.current_position):.2f} meters

Images:
1. RGB Camera View (what the drone sees)
2. Depth Map (closer objects are red/yellow, farther objects are blue)

Task: Generate the next safe waypoint (x, y, z) in NED coordinates.
Respond ONLY with valid JSON: {{"x": float, "y": float, "z": float, "reasoning": "brief explanation"}}
"""
        
        # Call GPT-4 Vision
        response = self.client.chat.completions.create(
            model=self.get_parameter('gpt_model').value,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{rgb_base64}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{depth_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        # Parse response
        content = response.choices[0].message.content
        self.get_logger().info(f'LLM response: {content}')
        
        # Extract JSON
        try:
            # Try to find JSON in response
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
            self.get_logger().error(f'Failed to parse LLM response: {e}')
            return None
    
    def compute_velocity_command(self, target_position):
        """Convert target position to velocity command"""
        direction = target_position - self.current_position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            max_speed = 1.5  # m/s
            speed = min(max_speed, distance)
            velocity = (direction / distance) * speed
        else:
            velocity = np.zeros(3)
        
        return velocity
    
    def publish_velocity(self, velocity):
        """Publish velocity command"""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(velocity[0])
        msg.twist.linear.y = float(velocity[1])
        msg.twist.linear.z = float(velocity[2])
        self.cmd_vel_pub.publish(msg)
    
    def publish_zero_velocity(self):
        """Stop the drone"""
        self.publish_velocity(np.zeros(3))
    
    def compute_trajectory_error(self):
        """
        Compute supremum (max) error between LLM and MPC trajectories
        Returns: max Euclidean distance between corresponding points
        """
        if len(self.llm_trajectory) == 0 or len(self.mpc_trajectory) == 0:
            return 0.0
        
        # Align trajectories (use minimum length)
        min_len = min(len(self.llm_trajectory), len(self.mpc_trajectory))
        
        errors = []
        for i in range(min_len):
            llm_point = self.llm_trajectory[i]
            mpc_point = self.mpc_trajectory[i]
            error = np.linalg.norm(llm_point - mpc_point)
            errors.append(error)
        
        # Supremum (maximum error)
        sup_error = max(errors) if errors else 0.0
        
        return sup_error


def main(args=None):
    rclpy.init(args=args)
    
    node = LLMTrajectoryPlanner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()