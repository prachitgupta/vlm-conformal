#!/usr/bin/env python3
"""
Model Predictive Control (MPC) Vision-Based Obstacle Avoidance
Uses depth camera for obstacle detection and avoidance
Ready for deployment to real hardware (Starling 2)

Dependencies:
pip install rclpy mavsdk opencv-python numpy cvxpy scipy

Author: Vision MPC Controller
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed, PositionNedYaw
import cvxpy as cp
from scipy.spatial import distance
import threading

try:
    from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode, VehicleOdometry
    HAS_PX4_MSGS = True
except ImportError:
    HAS_PX4_MSGS = False


class MPCVisionController(Node):
    """
    MPC Controller with Vision-based Obstacle Avoidance
    """
    
    def __init__(self):
        super().__init__('mpc_vision_controller')
        
        # Parameters
        self.declare_parameter('goal_x', 15.0)
        self.declare_parameter('goal_y', 10.0)
        self.declare_parameter('goal_z', -2.0)  # NED frame (negative = up)
        self.declare_parameter('prediction_horizon', 10)
        self.declare_parameter('control_horizon', 5)
        self.declare_parameter('dt', 0.2)
        self.declare_parameter('max_velocity', 2.0)
        self.declare_parameter('obstacle_threshold', 2.5)  # meters
        self.declare_parameter('safety_distance', 1.5)  # meters
        
        self.goal = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value,
            self.get_parameter('goal_z').value
        ])
        
        self.N = self.get_parameter('prediction_horizon').value
        self.M = self.get_parameter('control_horizon').value
        self.dt = self.get_parameter('dt').value
        self.v_max = self.get_parameter('max_velocity').value
        self.obs_threshold = self.get_parameter('obstacle_threshold').value
        self.safety_dist = self.get_parameter('safety_distance').value
        
        # State
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.obstacles = []  # List of (x, y, z, radius) in NED frame
        self.bridge = CvBridge()
        self.warned_fallback_pub = False
        self.warned_solver = False

        # Prefer conic solvers for SOC constraints (norm <= vmax)
        self.available_solvers = cp.installed_solvers()
        if 'ECOS' in self.available_solvers:
            self.conic_solver = cp.ECOS
        elif 'SCS' in self.available_solvers:
            self.conic_solver = cp.SCS
        else:
            self.conic_solver = None
            self.get_logger().error(
                'No conic solver available to CVXPY. '
                'Install ECOS or SCS (python3-ecos or python3-scs).'
            )

        # PX4 uORB -> ROS 2 typically uses best-effort, volatile QoS
        self.odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        
        # Subscribers
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth', self.depth_callback, 10)
        if HAS_PX4_MSGS:
            self.odom_sub = self.create_subscription(
                VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, self.odom_qos)
        else:
            self.odom_sub = self.create_subscription(
                Odometry, '/fmu/out/vehicle_odometry', self.odometry_callback, self.odom_qos)
        
        # Publishers
        self.mpc_traj_pub = self.create_publisher(
            PoseStamped, '/mpc/trajectory', 10)
        if HAS_PX4_MSGS:
            self.traj_setpoint_pub = self.create_publisher(
                TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
            self.offboard_mode_pub = self.create_publisher(
                OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
            self.cmd_vel_pub = None
        else:
            self.traj_setpoint_pub = None
            self.offboard_mode_pub = None
            self.cmd_vel_pub = self.create_publisher(
                TwistStamped, '/fmu/in/setpoint_velocity', 10)
        
        # Vision data
        self.depth_image = None
        self.camera_K = np.array([
            [320, 0, 320],
            [0, 320, 240],
            [0, 0, 1]
        ])  # Default camera intrinsics
        
        # MPC Timer
        self.mpc_timer = self.create_timer(self.dt, self.mpc_control_loop)
        if HAS_PX4_MSGS:
            self.offboard_timer = self.create_timer(0.1, self.publish_offboard_heartbeat)
        
        # MAVSDK drone
        self.drone = None
        self.offboard_active = False
        
        self.get_logger().info('MPC Vision Controller initialized')
        self.get_logger().info(f'Goal: {self.goal}')
        if HAS_PX4_MSGS:
            self.get_logger().info('Publishing PX4 trajectory setpoints to /fmu/in/trajectory_setpoint')
        else:
            self.get_logger().warn(
                'px4_msgs not found in Python env; falling back to /fmu/in/setpoint_velocity'
            )
    
    def depth_callback(self, msg):
        """Process depth camera data and detect obstacles"""
        try:
            # Convert depth image
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            
            # Detect obstacles from depth
            self.detect_obstacles_from_depth()
            
        except Exception as e:
            self.get_logger().error(f'Depth conversion error: {e}')
    
    def detect_obstacles_from_depth(self):
        """
        Detect obstacles from depth image
        Projects depth points to 3D space and clusters them
        """
        if self.depth_image is None:
            return
        
        h, w = self.depth_image.shape
        obstacles = []
        
        # Sample depth image (reduce computation)
        step = 20
        for v in range(0, h, step):
            for u in range(0, w, step):
                depth = self.depth_image[v, u]
                
                # Filter valid depth readings
                if depth < 0.5 or depth > self.obs_threshold or np.isnan(depth):
                    continue
                
                # Project to 3D camera frame
                x_cam = (u - self.camera_K[0, 2]) * depth / self.camera_K[0, 0]
                y_cam = (v - self.camera_K[1, 2]) * depth / self.camera_K[1, 1]
                z_cam = depth
                
                # Transform to drone body frame (camera pointing forward)
                # Camera is mounted 0.15m forward, rotated 0 deg
                x_body = z_cam
                y_body = -x_cam
                z_body = -y_cam
                
                # Transform to NED frame (world)
                # Simplified: assume drone is level
                x_ned = self.current_position[0] + x_body
                y_ned = self.current_position[1] + y_body
                z_ned = self.current_position[2] + z_body
                
                obstacles.append([x_ned, y_ned, z_ned])
        
        # Cluster obstacles
        if len(obstacles) > 0:
            obstacles = np.array(obstacles)
            self.obstacles = self.cluster_obstacles(obstacles)
            
            self.get_logger().info(
                f'Detected {len(self.obstacles)} obstacle clusters',
                throttle_duration_sec=2.0
            )
    
    def cluster_obstacles(self, points, radius=0.5):
        """
        Simple clustering of obstacle points
        Returns list of (x, y, z, radius) tuples
        """
        if len(points) == 0:
            return []
        
        clusters = []
        used = np.zeros(len(points), dtype=bool)
        
        for i in range(len(points)):
            if used[i]:
                continue
            
            # Find nearby points
            distances = distance.cdist([points[i]], points)[0]
            cluster_mask = distances < radius
            cluster_points = points[cluster_mask]
            
            # Mark as used
            used[cluster_mask] = True
            
            # Calculate cluster center and size
            center = np.mean(cluster_points, axis=0)
            max_dist = np.max(distance.cdist([center], cluster_points)[0])
            
            clusters.append((center[0], center[1], center[2], max_dist + 0.3))
        
        return clusters
    
    def odometry_callback(self, msg):
        """Update current position and velocity"""
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

    def vehicle_odometry_callback(self, msg):
        """Update current state from px4_msgs/VehicleOdometry (NED)."""
        self.current_position = np.array([
            float(msg.position[0]),
            float(msg.position[1]),
            float(msg.position[2]),
        ])
        self.current_velocity = np.array([
            float(msg.velocity[0]),
            float(msg.velocity[1]),
            float(msg.velocity[2]),
        ])
    
    def mpc_control_loop(self):
        """Main MPC control loop"""
        if np.linalg.norm(self.current_position) < 0.1:
            # Not initialized yet
            return
        
        # Check if goal reached
        dist_to_goal = np.linalg.norm(self.goal - self.current_position)
        if dist_to_goal < 0.5:
            self.get_logger().info('Goal reached!', throttle_duration_sec=1.0)
            self.publish_zero_velocity()
            return
        
        # Solve MPC
        optimal_velocity = self.solve_mpc()
        
        if optimal_velocity is not None:
            # Publish MPC trajectory point for comparison/visualization.
            target_position = self.current_position + optimal_velocity * self.dt
            self.publish_mpc_trajectory(target_position)

            # Send command on the same PX4 topics as llm_planner.
            if HAS_PX4_MSGS:
                self.publish_trajectory_setpoint(target_position)
            else:
                self.publish_velocity(optimal_velocity)
            
            self.get_logger().info(
                f'Pos: [{self.current_position[0]:.2f}, '
                f'{self.current_position[1]:.2f}, '
                f'{self.current_position[2]:.2f}], '
                f'Vel: [{optimal_velocity[0]:.2f}, '
                f'{optimal_velocity[1]:.2f}, '
                f'{optimal_velocity[2]:.2f}], '
                f'Dist: {dist_to_goal:.2f}m',
                throttle_duration_sec=0.5
            )
    
    def solve_mpc(self):
        """
        Solve MPC optimization problem
        Minimize: distance to goal + control effort
        Subject to: velocity limits, obstacle avoidance
        """
        # State: [x, y, z]
        # Control: [vx, vy, vz]
        
        # Decision variables
        x = cp.Variable((3, self.N + 1))  # States
        u = cp.Variable((3, self.M))      # Control inputs
        
        # Objective function
        cost = 0
        Q = np.diag([10.0, 10.0, 10.0])   # State cost
        R = np.diag([0.1, 0.1, 0.1])      # Control cost
        
        for k in range(self.N):
            # Distance to goal
            cost += cp.quad_form(x[:, k] - self.goal, Q)
            
            # Control effort (if within horizon)
            if k < self.M:
                cost += cp.quad_form(u[:, k], R)
        
        # Terminal cost
        cost += cp.quad_form(x[:, self.N] - self.goal, Q * 10)
        
        # Constraints
        constraints = []
        
        # Initial condition
        constraints.append(x[:, 0] == self.current_position)
        
        # Dynamics (simple integrator model)
        for k in range(self.M):
            constraints.append(x[:, k+1] == x[:, k] + u[:, k] * self.dt)
        
        # Continue with last control input
        for k in range(self.M, self.N):
            constraints.append(
                x[:, k+1] == x[:, k] + u[:, self.M-1] * self.dt
            )
        
        # Velocity limits
        for k in range(self.M):
            constraints.append(cp.norm(u[:, k], 2) <= self.v_max)
        
        # Obstacle avoidance
        for obs_x, obs_y, obs_z, obs_r in self.obstacles:
            obs_pos = np.array([obs_x, obs_y, obs_z])
            
            for k in range(1, self.N + 1):
                # Linear approximation of obstacle constraint
                # Distance from obstacle center
                current_to_obs = obs_pos - self.current_position
                dist = np.linalg.norm(current_to_obs)
                
                if dist < self.obs_threshold:
                    # Add constraint
                    safe_dist = obs_r + self.safety_dist
                    
                    # Linearized constraint
                    constraints.append(
                        current_to_obs @ (x[:, k] - obs_pos) >= safe_dist * dist
                    )
        
        # Solve optimization
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            if self.conic_solver is None:
                if not self.warned_solver:
                    self.get_logger().error(
                        f'CVXPY conic solver missing. Installed solvers: {self.available_solvers}'
                    )
                    self.warned_solver = True
                return None

            problem.solve(solver=self.conic_solver, verbose=False, warm_start=True)
            
            if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
                # Return first control action
                return u[:, 0].value
            else:
                self.get_logger().warn(f'MPC solve status: {problem.status}')
                # Fallback: move towards goal at reduced speed
                direction = self.goal - self.current_position
                dist = np.linalg.norm(direction)
                if dist > 0:
                    return (direction / dist) * min(self.v_max * 0.5, dist)
                
        except Exception as e:
            self.get_logger().error(f'MPC solve error: {e}')
            return None
    
    def publish_velocity(self, velocity):
        """Publish velocity command"""
        if self.cmd_vel_pub is None:
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(velocity[0])
        msg.twist.linear.y = float(velocity[1])
        msg.twist.linear.z = float(velocity[2])
        self.cmd_vel_pub.publish(msg)

        if not self.warned_fallback_pub:
            self.get_logger().warn(
                'Using /fmu/in/setpoint_velocity fallback. '
                'Install px4_msgs to publish /fmu/in/trajectory_setpoint.'
            )
            self.warned_fallback_pub = True

    def publish_offboard_heartbeat(self):
        """Publish OffboardControlMode heartbeat required by PX4 offboard control."""
        if self.offboard_mode_pub is None:
            return
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        self.offboard_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, target_position):
        """Publish position setpoint directly to PX4 trajectory setpoint topic."""
        if self.traj_setpoint_pub is None:
            return
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = [
            float(target_position[0]),
            float(target_position[1]),
            float(target_position[2]),
        ]
        msg.yaw = 0.0
        self.traj_setpoint_pub.publish(msg)

    def publish_mpc_trajectory(self, position):
        """Publish MPC next position for comparison with LLM trajectory."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        self.mpc_traj_pub.publish(msg)
    
    def publish_zero_velocity(self):
        """Stop the drone"""
        if HAS_PX4_MSGS:
            self.publish_trajectory_setpoint(self.current_position)
        else:
            self.publish_velocity(np.zeros(3))


def main(args=None):
    rclpy.init(args=args)
    
    controller = MPCVisionController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
