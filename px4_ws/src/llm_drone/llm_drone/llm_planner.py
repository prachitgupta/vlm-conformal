#!/usr/bin/env python3
"""
LLM-based Trajectory Planner using open-source LLMs (Qwen/Llama)
Save as: ros2_ws/src/px4_vision/px4_vision/llm_planner.py

Subscribes to:
  - /fmu/out/vehicle_odometry (drone position)
  - /depth_camera/points (PointCloud2)

Publishes to:
  - /llm/trajectory (LLM-generated trajectory)
  - /fmu/in/trajectory_setpoint (commands to PX4)

Compares with:
  - /mpc/trajectory (MPC ground truth)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import base64
import json
from openai import OpenAI
import os
import urllib.request
import urllib.error
from ament_index_python.packages import get_package_share_directory

try:
    from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode, VehicleOdometry
    HAS_PX4_MSGS = True
except ImportError:
    HAS_PX4_MSGS = False


class LLMTrajectoryPlanner(Node):
    """
    Uses an open-source LLM to generate trajectories from depth camera input
    """
    
    def __init__(self):
        super().__init__('llm_trajectory_planner')
        
        # OpenAI setup (disabled for now)
        # api_key = self._load_openai_key()
        # if not api_key:
        #     self.get_logger().error('OpenAI API key not found in env or key file!')
        #     raise ValueError('Set OPENAI_API_KEY or provide /home/prachit/api_key.txt')
        # self.client = OpenAI(api_key=api_key)
        
        # Load prompt from file
        prompt_file = self._resolve_prompt_file()
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
        self.declare_parameter('llm_provider', 'llama')  # qwen or llama
        self.declare_parameter('ollama_url', 'http://localhost:11434/api/generate')
        self.declare_parameter('qwen_model', 'qwen2.5:7b-instruct')
        self.declare_parameter('llama_model', 'llama3.2:latest')
        
        self.goal = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value,
            self.get_parameter('goal_z').value
        ])
        
        # State
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.depth_image = None
        self.last_environment_vector = None
        self.bridge = CvBridge()
        self.warned_fallback_pub = False
        
        # Trajectory storage
        self.llm_trajectory = []
        self.mpc_trajectory = []
        
        # Subscribers
        if HAS_PX4_MSGS:
            self.odom_sub = self.create_subscription(
                VehicleOdometry,
                '/fmu/out/vehicle_odometry',
                self.vehicle_odometry_callback,
                qos_profile_sensor_data,
            )
        else:
            self.odom_sub = self.create_subscription(
                Odometry,
                '/fmu/out/vehicle_odometry',
                self.odometry_callback,
                qos_profile_sensor_data,
            )
        self.depth_sub = self.create_subscription(
            PointCloud2,
            '/depth_camera/points',
            self.depth_points_callback,
            qos_profile_sensor_data,
        )
        self.mpc_traj_sub = self.create_subscription(
            PoseStamped, '/mpc/trajectory', self.mpc_trajectory_callback, 10)
        
        # Publishers
        self.llm_traj_pub = self.create_publisher(
            PoseStamped, '/llm/trajectory', 10)
        # PX4 setpoint publishers (preferred for Gazebo PX4 offboard)
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
        
        # Timer for LLM queries
        update_period = 1.0 / self.get_parameter('update_rate').value
        self.timer = self.create_timer(update_period, self.plan_trajectory)
        if HAS_PX4_MSGS:
            self.offboard_timer = self.create_timer(0.1, self.publish_offboard_heartbeat)
        
        self.get_logger().info('LLM Trajectory Planner initialized')
        self.get_logger().info(f'Goal: {self.goal}')
        if HAS_PX4_MSGS:
            self.get_logger().info('Publishing PX4 trajectory setpoints to /fmu/in/trajectory_setpoint')
        else:
            self.get_logger().warn(
                'px4_msgs not found in Python env; falling back to /fmu/in/setpoint_velocity'
            )
    
    def _get_default_prompt(self):
        """Fallback prompt if file not found"""
        return """You are a drone motion planning system. Generate safe trajectories."""

    def _resolve_prompt_file(self):
        """Resolve prompt path for both source and installed package layouts."""
        local_path = os.path.join(os.path.dirname(__file__), '../config/llm_prompt.txt')
        local_path = os.path.abspath(local_path)
        if os.path.exists(local_path):
            return local_path

        share_dir = get_package_share_directory('llm_drone')
        share_path = os.path.join(share_dir, 'config', 'llm_prompt.txt')
        return share_path
    
    def _load_openai_key(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return api_key.strip()

        key_path = os.getenv('LLM_API_KEY_FILE', '/home/prachit/api_key.txt')
        try:
            with open(key_path, 'r') as f:
                text = f.read()
        except FileNotFoundError:
            self.get_logger().warn(f'API key file not found: {key_path}')
            return None

        openai_key, claude_key = self._parse_key_file(text)
        if claude_key:
            # Placeholder for future Claude usage; do not log the key.
            self.claude_api_key = claude_key
        if openai_key:
            return openai_key

        self.get_logger().warn('OpenAI key not found in key file')
        return None

    def _parse_key_file(self, text):
        """
        Parse a text file containing OpenAI and Claude keys.
        Supported formats:
        - OPENAI_API_KEY=...
        - CLAUDE_API_KEY=... or ANTHROPIC_API_KEY=...
        - openai: ... / claude: ...
        - Two-line fallback (first OpenAI, second Claude)
        """
        openai_key = None
        claude_key = None

        def _clean(val):
            return val.strip().strip('"').strip("'")

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            upper = line.upper()
            if 'OPENAI' in upper:
                parts = line.split('=', 1) if '=' in line else line.split(':', 1)
                if len(parts) == 2:
                    openai_key = _clean(parts[1])
                    continue
            if 'CLAUDE' in upper or 'ANTHROPIC' in upper:
                parts = line.split('=', 1) if '=' in line else line.split(':', 1)
                if len(parts) == 2:
                    claude_key = _clean(parts[1])
                    continue

        if not openai_key or not claude_key:
            raw_lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith('#')]
            if len(raw_lines) >= 1 and not openai_key:
                openai_key = _clean(raw_lines[0])
            if len(raw_lines) >= 2 and not claude_key:
                claude_key = _clean(raw_lines[1])

        return openai_key, claude_key

    def depth_points_callback(self, msg):
        """Store latest depth from PointCloud2; also build a depth image when possible."""
        try:
            depth_img = self._pointcloud_to_depth_image(msg)
            if depth_img is not None:
                self.depth_image = depth_img
        except Exception as e:
            self.get_logger().error(f'PointCloud2 conversion error: {e}')

    def _pointcloud_to_depth_image(self, msg):
        """Convert an organized PointCloud2 into a depth image (uses z as depth)."""
        if msg.height <= 0 or msg.width <= 0:
            return None

        points = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=False)
        if points is None or points.size == 0:
            return None

        if msg.height > 1 and msg.width > 1 and points.shape[0] == msg.height * msg.width:
            points = points.reshape((msg.height, msg.width, 3))
            depth = points[:, :, 2].astype(np.float32)
            depth[np.isinf(depth)] = np.nan
            return depth

        # Fallback for unorganized clouds: build a 1-row depth "image".
        depth = points[:, 2].astype(np.float32)
        depth[np.isinf(depth)] = np.nan
        return depth.reshape((1, -1))
    
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

    def build_environment_vector(self):
        """
        Build numerical vector v from odometry + depth for motion planning.
        """
        if self.depth_image is None:
            return None

        depth = self.depth_image
        valid_depth = depth[np.isfinite(depth)]
        valid_depth = valid_depth[(valid_depth > 0.2) & (valid_depth < 20.0)]

        depth_valid_fraction = float(valid_depth.size / depth.size) if depth.size > 0 else 0.0
        global_min = float(np.min(valid_depth)) if valid_depth.size > 0 else 20.0
        global_mean = float(np.mean(valid_depth)) if valid_depth.size > 0 else 20.0

        h, w = depth.shape
        sector_bounds = {
            "far_left": (0, int(0.2 * w)),
            "left": (int(0.2 * w), int(0.4 * w)),
            "center": (int(0.4 * w), int(0.6 * w)),
            "right": (int(0.6 * w), int(0.8 * w)),
            "far_right": (int(0.8 * w), w),
        }

        sector_min = {}
        sector_mean = {}
        for name, (start, end) in sector_bounds.items():
            patch = depth[:, start:end]
            patch_valid = patch[np.isfinite(patch)]
            patch_valid = patch_valid[(patch_valid > 0.2) & (patch_valid < 20.0)]
            sector_min[name] = float(np.min(patch_valid)) if patch_valid.size > 0 else 20.0
            sector_mean[name] = float(np.mean(patch_valid)) if patch_valid.size > 0 else 20.0

        nearest_sector = min(sector_min, key=sector_min.get)
        nearest_distance = float(sector_min[nearest_sector])
        if nearest_distance < 1.5:
            clearance_status = "blocked"
        elif nearest_distance < 3.0:
            clearance_status = "caution"
        else:
            clearance_status = "clear"

        goal_delta = self.goal - self.current_position
        dist_to_goal = float(np.linalg.norm(goal_delta))
        speed = float(np.linalg.norm(self.current_velocity))
        heading_to_goal_xy = float(np.arctan2(goal_delta[1], goal_delta[0]))

        vector = {
            "current_position_ned_m": [float(x) for x in self.current_position],
            "current_velocity_mps": [float(v) for v in self.current_velocity],
            "goal_position_ned_m": [float(g) for g in self.goal],
            "goal_delta_m": [float(d) for d in goal_delta],
            "distance_to_goal_m": dist_to_goal,
            "speed_mps": speed,
            "heading_to_goal_xy_rad": heading_to_goal_xy,
            "depth_features": {
                "valid_fraction": depth_valid_fraction,
                "global_min_m": global_min,
                "global_mean_m": global_mean,
                "sector_min_m": sector_min,
                "sector_mean_m": sector_mean,
                "nearest_obstacle_sector": nearest_sector,
                "nearest_obstacle_distance_m": nearest_distance,
                "clearance_status": clearance_status,
            },
        }
        self.last_environment_vector = vector
        return vector

    def translate_vector_to_nlp(self, vector):
        """
        Fixed translator T: v -> NLP(v).
        Converts numerical environment vector into deterministic natural language.
        """
        if vector is None:
            return "Environment summary unavailable: no depth data."

        p = vector["current_position_ned_m"]
        vel = vector["current_velocity_mps"]
        g = vector["goal_position_ned_m"]
        d = vector["goal_delta_m"]
        depth = vector["depth_features"]
        mins = depth["sector_min_m"]

        lateral_hint = "prefer center progress"
        side_clear = mins["right"] + mins["far_right"]
        side_left = mins["left"] + mins["far_left"]
        if side_clear > side_left + 0.8:
            lateral_hint = "right side is clearer"
        elif side_left > side_clear + 0.8:
            lateral_hint = "left side is clearer"

        return (
            "Environment section (T(v)):\n"
            f"- Current position NED is ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) m.\n"
            f"- Current velocity is ({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}) m/s (speed {vector['speed_mps']:.2f} m/s).\n"
            f"- Goal position NED is ({g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}) m.\n"
            f"- Goal delta from current state is ({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f}) m with distance {vector['distance_to_goal_m']:.2f} m.\n"
            f"- Depth validity fraction is {depth['valid_fraction']:.2f}. Global nearest obstacle distance is {depth['global_min_m']:.2f} m.\n"
            f"- Sector minimum distances [far_left, left, center, right, far_right] are "
            f"[{mins['far_left']:.2f}, {mins['left']:.2f}, {mins['center']:.2f}, {mins['right']:.2f}, {mins['far_right']:.2f}] m.\n"
            f"- Nearest obstacle sector is {depth['nearest_obstacle_sector']} at {depth['nearest_obstacle_distance_m']:.2f} m "
            f"and clearance status is {depth['clearance_status']}.\n"
            f"- Lateral planning hint: {lateral_hint}."
        )
    
    def plan_trajectory(self):
        """Query LLM for next trajectory point"""
        
        # Check if we have all data
        if self.depth_image is None:
            self.get_logger().warn('Waiting for depth camera data...', throttle_duration_sec=2.0)
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
                
                # Publish control command to PX4
                if HAS_PX4_MSGS:
                    self.publish_trajectory_setpoint(next_position)
                else:
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
        """Query open-source LLM (Qwen/Llama) for next waypoint"""
        
        env_vector = self.build_environment_vector()
        env_text = self.translate_vector_to_nlp(env_vector)

        # Compose final prompt = fixed prompt + translated environment block + vector
        user_message = self.compose_final_prompt(env_vector, env_text)

        # Open-source model call (Qwen enabled by default)
        content = self.query_open_source_model(user_message)
        self.get_logger().info(f'LLM response: {content}')
        
        # Extract JSON
        try:
            # Try to find JSON in response
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]
            data = json.loads(json_str)
            
            if isinstance(data, dict) and "waypoints" in data:
                waypoints = data.get("waypoints", [])
                selected_idx = int(data.get("selected_waypoint_index", 0))
                if not waypoints:
                    raise ValueError("Empty waypoints list")
                if selected_idx < 0 or selected_idx >= len(waypoints):
                    raise ValueError("selected_waypoint_index out of range")

                wp = waypoints[selected_idx]
                next_position = np.array([
                    float(wp['x']),
                    float(wp['y']),
                    float(wp['z'])
                ])
            else:
                # Backward-compatible fallback schema
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

    def query_open_source_model(self, user_message):
        """
        Query Ollama-compatible endpoint with either Qwen or Llama prompt style.
        Qwen is currently selected by default.
        """
        provider = str(self.get_parameter('llm_provider').value).strip().lower()

        # Prompt snippet for Llama:
        llama_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{self.system_prompt}\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_message}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

        # Prompt snippet for Qwen (ChatML):
        qwen_prompt = (
            "<|im_start|>system\n"
            f"{self.system_prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_message}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # For now keep Qwen active as the open-source model in use.
        # model_name = self.get_parameter('llama_model').value
        model_name = self.get_parameter('qwen_model').value

        if provider == 'llama':
            prompt = llama_prompt
            model_name = self.get_parameter('llama_model').value
        else:
            prompt = qwen_prompt

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 300,
            },
        }

        request = urllib.request.Request(
            url=str(self.get_parameter('ollama_url').value),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        url = str(self.get_parameter('ollama_url').value)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
            return parsed.get("response", "")
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = "<no response body>"
            self.get_logger().error(
                "Open-source model request failed "
                f"(status={e.code}, url={url}, model={model_name}): {error_body}"
            )
            raise
        except urllib.error.URLError as e:
            self.get_logger().error(
                "Open-source model request failed "
                f"(url={url}, model={model_name}): {e}"
            )
            raise

    def compose_final_prompt(self, env_vector, env_text):
        """Build final prompt from fixed instructions + dynamic environment section."""
        return f"""
Use the fixed planning policy exactly as defined in the system prompt.

Dynamic Environment Section (T(v)):
{env_text}

Numerical Environment Vector v:
{json.dumps(env_vector, indent=2)}

Sensor attachments:
1) Depth map visualization (red/yellow close, blue far)

Return only one JSON object with keys:
- waypoints
- selected_waypoint_index
- reasoning
"""
    
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
    
    def publish_zero_velocity(self):
        """Stop the drone"""
        if HAS_PX4_MSGS:
            # Hold current position when using trajectory setpoints.
            self.publish_trajectory_setpoint(self.current_position)
        else:
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

    node = None
    try:
        node = LLMTrajectoryPlanner()
        rclpy.spin(node)
    except rclpy.executors.ExternalShutdownException:
        # Context has already been shutdown externally.
        pass
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
