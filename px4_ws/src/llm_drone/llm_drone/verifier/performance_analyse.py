#!/usr/bin/env python3
"""
Performance Analyzer: LLM vs MPC Conformity Analysis
Computes comprehensive metrics including sup(||LLM - MPC||)

Subscribes to:
  - /llm/trajectory (LLM waypoints)
  - /mpc/trajectory (MPC ground truth)
  - /qvio (actual vehicle position)

Outputs:
  - /performance/conformity_score (sup error)
  - /performance/metrics (JSON with all metrics)
  - performance_report.json (file output)

Usage:
  ros2 run px4_vision performance_analyzer
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, String
import numpy as np
from collections import deque
import json
import time
from datetime import datetime


class PerformanceAnalyzer(Node):
    
    def __init__(self):
        super().__init__('performance_analyzer')
        
        # Parameters
        self.declare_parameter('buffer_size', 1000)
        self.declare_parameter('output_file', 'performance_report.json')
        self.declare_parameter('goal_x', 15.0)
        self.declare_parameter('goal_y', 10.0)
        self.declare_parameter('goal_z', 2.0)
        
        buffer_size = self.get_parameter('buffer_size').value
        self.output_file = self.get_parameter('output_file').value
        
        self.goal = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value,
            self.get_parameter('goal_z').value
        ])
        
        # Data storage
        self.llm_trajectory = deque(maxlen=buffer_size)
        self.mpc_trajectory = deque(maxlen=buffer_size)
        self.vehicle_trajectory = deque(maxlen=buffer_size)
        
        self.llm_timestamps = deque(maxlen=buffer_size)
        self.mpc_timestamps = deque(maxlen=buffer_size)
        
        # Mission timing
        self.mission_start_time = None
        self.mission_end_time = None
        self.llm_reached_goal = False
        self.mpc_reached_goal = False
        
        # Subscribers
        self.llm_sub = self.create_subscription(
            PoseStamped, '/llm/trajectory', self.llm_callback, 10)
        self.mpc_sub = self.create_subscription(
            PoseStamped, '/mpc/trajectory', self.mpc_callback, 10)
        self.vehicle_sub = self.create_subscription(
            Odometry, '/qvio', self.vehicle_callback, 10)
        
        # Publishers
        self.conformity_pub = self.create_publisher(
            Float64, '/performance/conformity_score', 10)
        self.metrics_pub = self.create_publisher(
            String, '/performance/metrics', 10)
        
        # Analysis timer
        self.timer = self.create_timer(1.0, self.analyze_and_publish)
        
        self.get_logger().info('Performance Analyzer initialized')
        self.get_logger().info(f'Output file: {self.output_file}')
    
    def llm_callback(self, msg):
        """Store LLM trajectory point"""
        point = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.llm_trajectory.append(point)
        self.llm_timestamps.append(self.get_clock().now().nanoseconds / 1e9)
        
        # Start mission timer
        if self.mission_start_time is None:
            self.mission_start_time = time.time()
        
        # Check if goal reached
        if not self.llm_reached_goal:
            dist = np.linalg.norm(point - self.goal)
            if dist < 0.5:
                self.llm_reached_goal = True
                self.get_logger().info('LLM reached goal!')
    
    def mpc_callback(self, msg):
        """Store MPC trajectory point"""
        point = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.mpc_trajectory.append(point)
        self.mpc_timestamps.append(self.get_clock().now().nanoseconds / 1e9)
        
        # Check if goal reached
        if not self.mpc_reached_goal:
            dist = np.linalg.norm(point - self.goal)
            if dist < 0.5:
                self.mpc_reached_goal = True
                self.mission_end_time = time.time()
                self.get_logger().info('MPC reached goal!')
    
    def vehicle_callback(self, msg):
        """Store actual vehicle position"""
        point = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.vehicle_trajectory.append(point)
    
    def analyze_and_publish(self):
        """Compute and publish all metrics"""
        
        if len(self.llm_trajectory) == 0 or len(self.mpc_trajectory) == 0:
            return
        
        # Compute all metrics
        metrics = self.compute_all_metrics()
        
        # Publish conformity score (sup error)
        conformity_msg = Float64()
        conformity_msg.data = metrics['conformity_score']
        self.conformity_pub.publish(conformity_msg)
        
        # Publish full metrics as JSON
        metrics_msg = String()
        metrics_msg.data = json.dumps(metrics, indent=2)
        self.metrics_pub.publish(metrics_msg)
        
        # Log summary
        self.get_logger().info(
            f"Conformity Score (sup error): {metrics['conformity_score']:.3f}m | "
            f"Mean Error: {metrics['trajectory_error']['mean']:.3f}m | "
            f"Points: {metrics['trajectory_error']['num_points']}",
            throttle_duration_sec=2.0
        )
        
        # Save to file if mission complete
        if self.llm_reached_goal and self.mpc_reached_goal:
            self.save_report(metrics)
    
    def compute_all_metrics(self):
        """Compute comprehensive performance metrics"""
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'mission_info': self.compute_mission_info(),
            'trajectory_error': self.compute_trajectory_errors(),
            'path_metrics': self.compute_path_metrics(),
            'efficiency_metrics': self.compute_efficiency_metrics(),
            'conformity_score': 0.0  # Will be set below
        }
        
        # Conformity score is the supremum (max) of trajectory error
        metrics['conformity_score'] = metrics['trajectory_error']['supremum']
        
        return metrics
    
    def compute_mission_info(self):
        """Mission metadata"""
        mission_time = None
        if self.mission_start_time and self.mission_end_time:
            mission_time = self.mission_end_time - self.mission_start_time
        
        return {
            'goal': self.goal.tolist(),
            'llm_reached_goal': self.llm_reached_goal,
            'mpc_reached_goal': self.mpc_reached_goal,
            'mission_duration_sec': mission_time,
            'llm_trajectory_points': len(self.llm_trajectory),
            'mpc_trajectory_points': len(self.mpc_trajectory),
            'vehicle_trajectory_points': len(self.vehicle_trajectory)
        }
    
    def compute_trajectory_errors(self):
        """
        Compute trajectory conformity metrics
        Primary metric: sup(||LLM(t) - MPC(t)||) for all t
        """
        if len(self.llm_trajectory) == 0 or len(self.mpc_trajectory) == 0:
            return {
                'supremum': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'p95': 0.0,
                'num_points': 0
            }
        
        # Convert to arrays
        llm_traj = np.array(list(self.llm_trajectory))
        mpc_traj = np.array(list(self.mpc_trajectory))
        
        # Align trajectories (use minimum length)
        min_len = min(len(llm_traj), len(mpc_traj))
        llm_aligned = llm_traj[:min_len]
        mpc_aligned = mpc_traj[:min_len]
        
        # Compute pointwise errors
        errors = np.linalg.norm(llm_aligned - mpc_aligned, axis=1)
        
        return {
            'supremum': float(np.max(errors)),  # This is the conformity score
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'median': float(np.median(errors)),
            'p95': float(np.percentile(errors, 95)),
            'min': float(np.min(errors)),
            'num_points': int(min_len),
            'error_distribution': {
                '<0.5m': int(np.sum(errors < 0.5)),
                '0.5-1.0m': int(np.sum((errors >= 0.5) & (errors < 1.0))),
                '1.0-2.0m': int(np.sum((errors >= 1.0) & (errors < 2.0))),
                '>2.0m': int(np.sum(errors >= 2.0))
            }
        }
    
    def compute_path_metrics(self):
        """Compute path length and smoothness"""
        
        llm_length = self.compute_path_length(self.llm_trajectory)
        mpc_length = self.compute_path_length(self.mpc_trajectory)
        vehicle_length = self.compute_path_length(self.vehicle_trajectory)
        
        llm_smoothness = self.compute_path_smoothness(self.llm_trajectory)
        mpc_smoothness = self.compute_path_smoothness(self.mpc_trajectory)
        
        return {
            'llm': {
                'path_length_m': llm_length,
                'smoothness': llm_smoothness
            },
            'mpc': {
                'path_length_m': mpc_length,
                'smoothness': mpc_smoothness
            },
            'vehicle_actual': {
                'path_length_m': vehicle_length
            },
            'path_length_ratio': llm_length / mpc_length if mpc_length > 0 else 0
        }
    
    def compute_path_length(self, trajectory):
        """Compute total path length"""
        if len(trajectory) < 2:
            return 0.0
        
        traj = np.array(list(trajectory))
        diffs = np.diff(traj, axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
        return float(np.sum(lengths))
    
    def compute_path_smoothness(self, trajectory):
        """
        Compute path smoothness (inverse of total curvature)
        Lower is smoother
        """
        if len(trajectory) < 3:
            return 0.0
        
        traj = np.array(list(trajectory))
        
        # Compute acceleration (second derivative)
        velocities = np.diff(traj, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        # Sum of acceleration magnitudes
        curvature = np.sum(np.linalg.norm(accelerations, axis=1))
        
        return float(curvature)
    
    def compute_efficiency_metrics(self):
        """Compute efficiency metrics"""
        
        # Theoretical optimal distance (straight line to goal)
        if len(self.llm_trajectory) > 0:
            start = np.array(list(self.llm_trajectory))[0]
            optimal_distance = np.linalg.norm(self.goal - start)
        else:
            optimal_distance = 0.0
        
        llm_length = self.compute_path_length(self.llm_trajectory)
        mpc_length = self.compute_path_length(self.mpc_trajectory)
        
        return {
            'optimal_distance_m': float(optimal_distance),
            'llm_efficiency': optimal_distance / llm_length if llm_length > 0 else 0,
            'mpc_efficiency': optimal_distance / mpc_length if mpc_length > 0 else 0,
            'llm_path_optimality': llm_length / optimal_distance if optimal_distance > 0 else 0,
            'mpc_path_optimality': mpc_length / optimal_distance if optimal_distance > 0 else 0
        }
    
    def compute_conformity_grade(self, conformity_score):
        """
        Grade the conformity score
        Lower is better (closer to MPC ground truth)
        """
        if conformity_score < 0.5:
            return 'A+ (Excellent)'
        elif conformity_score < 1.0:
            return 'A (Very Good)'
        elif conformity_score < 2.0:
            return 'B (Good)'
        elif conformity_score < 3.0:
            return 'C (Acceptable)'
        elif conformity_score < 5.0:
            return 'D (Poor)'
        else:
            return 'F (Fail)'
    
    def save_report(self, metrics):
        """Save comprehensive report to file"""
        
        # Add grade
        conformity_score = metrics['conformity_score']
        grade = self.compute_conformity_grade(conformity_score)
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'node_name': self.get_name(),
                'ros2_distro': 'foxy'
            },
            'performance_metrics': metrics,
            'conformity_assessment': {
                'score': conformity_score,
                'grade': grade,
                'interpretation': self.interpret_conformity_score(conformity_score)
            },
            'trajectories': {
                'llm': [p.tolist() for p in self.llm_trajectory],
                'mpc': [p.tolist() for p in self.mpc_trajectory],
                'vehicle': [p.tolist() for p in self.vehicle_trajectory]
            }
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.get_logger().info(f'Performance report saved to {self.output_file}')
        self.get_logger().info(f'Conformity Score: {conformity_score:.3f}m ({grade})')
    
    def interpret_conformity_score(self, score):
        """Provide interpretation of conformity score"""
        if score < 0.5:
            return "LLM trajectory matches MPC ground truth excellently. Deployment ready."
        elif score < 1.0:
            return "LLM trajectory closely follows MPC. Minor deviations acceptable."
        elif score < 2.0:
            return "LLM trajectory shows good conformity with noticeable differences."
        elif score < 3.0:
            return "LLM trajectory deviates significantly from MPC. Review required."
        elif score < 5.0:
            return "LLM trajectory poorly matches MPC. Substantial tuning needed."
        else:
            return "LLM trajectory fails to conform to MPC ground truth. Not deployment ready."


def main(args=None):
    rclpy.init(args=args)
    
    node = PerformanceAnalyzer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()