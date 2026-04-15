#!/usr/bin/env python3
"""
LLM-based Trajectory Planner using local Qwen, vLLM/TGIS (OpenAI-compatible), or OpenAI cloud API
Save as: ros2_ws/src/px4_vision/px4_vision/llm_planner.py

Subscribes to:
  - /fmu/out/vehicle_odometry (drone position)
  - /depth_camera (Image, for LLM visualization/features + obstacle detection)

Publishes to:
  - /llm/trajectory (LLM-generated trajectory)
  - /fmu/in/trajectory_setpoint (commands to PX4)

Compares with:
  - /mpc/trajectory (MPC ground truth)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import TwistStamped, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
from collections import deque
from pathlib import Path as FilesystemPath
from pydantic import BaseModel
import threading
import time
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    import instructor
except ImportError:
    instructor = None
import os
import urllib.request
import urllib.error
from llm_drone.verifier.cli_goal import parse_goal_overrides
from llm_drone.llm.dataset_pipeline_common import (
    PROMPT_MODE_ENV_NL_ONLY,
    build_prompt_from_scene,
)
from llm_drone.llm.llm_prompt_common import (
    DEPTH_HFOV_DEG,
    DEPTH_VFOV_DEG,
    DEPTH_MIN_M,
    DEPTH_MAX_M,
    OBS_FIFO_LEN,
    MPC_DEPTH_SAMPLE_COUNT,
    DATASET_PROMPT_FILENAME,
    DEFAULT_SYSTEM_PROMPT,
    build_environment_vector,
    load_system_prompt_from_path,
    resolve_prompt_file,
)

try:
    from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode, VehicleOdometry
    HAS_PX4_MSGS = True
except ImportError:
    HAS_PX4_MSGS = False


LLM_WAYPOINT_COUNT = 5
DEPTH_FILTER_MIN_FORWARD_M = 0.35
DEPTH_FILTER_MAX_FORWARD_M = 12.0
DEPTH_FILTER_MAX_LATERAL_M = 4.0
DEPTH_FILTER_MIN_BELOW_BODY_M = -0.25
DEPTH_FILTER_MAX_BELOW_BODY_M = 2.5
DEPTH_FILTER_FLOOR_CLEARANCE_M = 0.20
LATEST_PROMPT_FILENAME = "variant_X.txt"


class Waypoint2D(BaseModel):
    x: float
    y: float


class PlanTrace(BaseModel):
    observation: str
    reasoning: str


class PlannerOutput(BaseModel):
    waypoints: list[Waypoint2D]
    plan_trace: PlanTrace


class LocalObstacleMap:
    """FIFO local obstacle cache matching MPC local planner semantics."""

    def __init__(self, maxlen=OBS_FIFO_LEN):
        self._buf = []
        self._maxlen = int(maxlen)

    def update(self, points_ned):
        if points_ned is None or points_ned.shape[0] == 0:
            return
        for pt in points_ned:
            self._buf.append(np.array(pt, dtype=float))
        if len(self._buf) > self._maxlen:
            self._buf = self._buf[-self._maxlen:]

    def nearest(self, x_ref):
        if not self._buf:
            return None
        pts = np.array(self._buf, dtype=float)
        dists = np.linalg.norm(pts - np.asarray(x_ref, dtype=float)[None, :3], axis=1)
        return pts[int(np.argmin(dists))]

    def snapshot(self, max_points=None):
        if not self._buf:
            return np.zeros((0, 3), dtype=float)
        pts = np.array(self._buf, dtype=float)
        if max_points is not None and pts.shape[0] > max_points:
            pts = pts[-int(max_points):]
        return pts


class DepthToObstacles:
    """Depth image to obstacle points using the same back-projection and transforms as MPC."""

    def __init__(self, hfov_deg=DEPTH_HFOV_DEG, vfov_deg=DEPTH_VFOV_DEG, n_sample=MPC_DEPTH_SAMPLE_COUNT):
        self.hfov = np.deg2rad(float(hfov_deg))
        self.vfov = np.deg2rad(float(vfov_deg))
        self.n_sample = int(n_sample)
        self.rotation_body_camera = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=float)
        self.translation_body_camera_m = np.array([0.13233, 0.0, 0.26078], dtype=float)

    def depth_to_body_frame(self, depth_img):
        if depth_img is None or depth_img.ndim != 2:
            return np.zeros((0, 3), dtype=float)

        H, W = depth_img.shape
        fy = (H / 2.0) / np.tan(self.vfov / 2.0)
        fx = (W / 2.0) / np.tan(self.hfov / 2.0)
        cx, cy = W / 2.0, H / 2.0

        total = H * W
        if total <= 0:
            return np.zeros((0, 3), dtype=float)
        idx = np.random.choice(total, min(self.n_sample, total), replace=False)
        rows, cols = np.unravel_index(idx, (H, W))
        depths = depth_img[rows, cols]

        valid = (depths > DEPTH_MIN_M) & (depths < DEPTH_MAX_M) & np.isfinite(depths)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=float)
        rows = rows[valid]
        cols = cols[valid]
        depths = depths[valid]

        xc = (cols - cx) / fx * depths
        yc = (rows - cy) / fy * depths
        zc = depths
        pts_cam = np.column_stack([xc, yc, zc])
        return (self.rotation_body_camera @ pts_cam.T).T + self.translation_body_camera_m

    def filter_body_points(self, pts_body):
        if pts_body is None or pts_body.shape[0] == 0:
            return np.zeros((0, 3), dtype=float)

        pts = np.asarray(pts_body, dtype=float)
        forward = pts[:, 0]
        lateral = pts[:, 1]
        down = pts[:, 2]

        mask = np.isfinite(pts).all(axis=1)
        mask &= forward >= DEPTH_FILTER_MIN_FORWARD_M
        mask &= forward <= DEPTH_FILTER_MAX_FORWARD_M
        mask &= np.abs(lateral) <= DEPTH_FILTER_MAX_LATERAL_M
        mask &= down >= DEPTH_FILTER_MIN_BELOW_BODY_M
        mask &= down <= DEPTH_FILTER_MAX_BELOW_BODY_M

        camera_height_above_ground_m = max(0.0, -float(self.translation_body_camera_m[2]))
        if camera_height_above_ground_m > DEPTH_FILTER_FLOOR_CLEARANCE_M:
            mask &= down <= camera_height_above_ground_m - DEPTH_FILTER_FLOOR_CLEARANCE_M

        filtered = pts[mask]
        if filtered.shape[0] == 0:
            return np.zeros((0, 3), dtype=float)
        return filtered

    @staticmethod
    def body_to_spatial(pts_body, rotation_ned_body, pos_ned):
        if pts_body is None or pts_body.shape[0] == 0:
            return np.zeros((0, 3), dtype=float)
        return (rotation_ned_body @ pts_body.T).T + np.asarray(pos_ned, dtype=float)[None, :]


class LLMTrajectoryPlanner(Node):
    """
    Uses local Qwen (Ollama), vLLM/TGIS OpenAI-compatible API, or OpenAI cloud API.
    """
    
    def __init__(self, goal_override=None):
        super().__init__('llm_trajectory_planner')
        
        # OpenAI client is created lazily only when provider=openai.
        self.client = None
        self.structured_client = None

        # Parameters
        self.declare_parameter('goal_x', 35.0)
        self.declare_parameter('goal_y', 3.0)
        self.declare_parameter('goal_z', -2.5)
        self.declare_parameter('update_rate', 1.0)  # Hz
        # Supported providers: qwen (Ollama), vllm/tgis (OpenAI-compatible server), openai (cloud).
        self.declare_parameter('llm_provider', 'vllm')  # qwen, vllm/tgis, or openai
        self.declare_parameter('ollama_url', 'http://localhost:11434/api/generate')
        self.declare_parameter('qwen_model', 'qwen2.5:3b')
        self.declare_parameter('openai_model', 'gpt-4o-mini')
        self.declare_parameter('vllm_url', 'http://localhost:8000/v1/chat/completions')
        self.declare_parameter('vllm_model', 'Qwen/Qwen2.5-1.5B-Instruct')
        self.declare_parameter('vllm_api_key', 'token-abc123')
        self.declare_parameter('vllm_temperature', 0.3)
        self.declare_parameter('vllm_max_tokens', 400)
        self.declare_parameter('goal_frame', 'ned')  # gazebo(ENU) or ned
        self.declare_parameter('depth_obstacle_samples', MPC_DEPTH_SAMPLE_COUNT)
        self.declare_parameter('max_velocity_mps', 15.0)
        self.declare_parameter('waypoint_acceptance_radius_m', 0.5)
        self.declare_parameter('use_legacy_goto_execution', False)
        self.declare_parameter('prompt_file', str((FilesystemPath(__file__).resolve().parents[2] / 'config' / LATEST_PROMPT_FILENAME).resolve()))

        prompt_file_override = str(self.get_parameter('prompt_file').value or '').strip()
        prompt_file = self._resolve_prompt_file(prompt_file_override)
        self.system_prompt = load_system_prompt_from_path(prompt_file)
        if prompt_file.exists():
            self.get_logger().info(f'Loaded system prompt from: {prompt_file}')
        else:
            self.get_logger().warn(
                f'Prompt file not found: {prompt_file}; using built-in default prompt'
            )

        goal_input = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value,
            self.get_parameter('goal_z').value
        ])
        goal_frame = str(self.get_parameter('goal_frame').value).strip().lower()
        self.goal = self.convert_goal_to_ned(goal_input, goal_frame)
        if goal_override is not None:
            self.goal = np.array(goal_override, dtype=float)

        self.state_lock = threading.Lock()
        self.execution_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # State
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.rotation_ned_body = np.eye(3)
        self.depth_image = None
        self.latest_depth_obstacles_ned = np.zeros((0, 3), dtype=float)
        self.last_environment_vector = None
        self.bridge = CvBridge()
        self.warned_fallback_pub = False
        self.depth_to_obstacles = DepthToObstacles(
            n_sample=int(self.get_parameter('depth_obstacle_samples').value)
        )
        self.local_obstacle_map = LocalObstacleMap(OBS_FIFO_LEN)
        
        # Trajectory storage
        self.llm_trajectory = []
        self.mpc_trajectory = []
        self.pending_waypoints = deque()
        self.active_waypoint = None
        self.command_target = None
        self.rollout_counter = 0
        self.latest_rollout_id = 0
        self.applied_rollout_id = 0
        self.latest_rollout_waypoints = []
        self.completed_waypoints_in_rollout = 0
        self.active_waypoint_batch_index = None
        
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
            Image,
            '/depth_camera',
            self.depth_image_callback,
            qos_profile_sensor_data,
        )
        self.mpc_traj_sub = self.create_subscription(
            PoseStamped, '/mpc/trajectory', self.mpc_trajectory_callback, 10)
        
        # Publishers
        self.llm_traj_pub = self.create_publisher(
            PoseStamped, '/llm/trajectory', 10)
        self.llm_traj_seq_pub = self.create_publisher(
            Path, '/llm/trajectory_sequence', 10)
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
        
        self.command_timer = self.create_timer(0.1, self.publish_active_command)
        if HAS_PX4_MSGS:
            self.offboard_timer = self.create_timer(0.1, self.publish_offboard_heartbeat)

        self.llm_thread = threading.Thread(
            target=self._llm_worker_loop,
            name='llm_query_worker',
            daemon=True,
        )
        self.tracking_thread = threading.Thread(
            target=self._tracking_worker_loop,
            name='waypoint_tracking_worker',
            daemon=True,
        )
        self.llm_thread.start()
        self.tracking_thread.start()
        
        self.get_logger().info('LLM Trajectory Planner initialized')
        self.get_logger().info(f'Goal: {self.goal}')
        provider = str(self.get_parameter('llm_provider').value).strip().lower()
        if provider == 'openai':
            active_model = str(self.get_parameter('openai_model').value)
            backend_desc = 'OpenAI cloud API'
        elif provider in ('vllm', 'tgis'):
            active_model = str(self.get_parameter('vllm_model').value)
            backend_desc = 'vLLM/TGIS OpenAI-compatible API'
        else:
            active_model = str(self.get_parameter('qwen_model').value)
            backend_desc = 'Local Qwen via Ollama'
        self.get_logger().info(
            f'Active LLM backend: provider={provider} ({backend_desc}), model={active_model}'
        )
        self.get_logger().info(
            'Obstacle perception source: /depth_camera (current depth frame projected to NED)'
        )
        if HAS_PX4_MSGS:
            self.get_logger().info('Publishing PX4 trajectory setpoints to /fmu/in/trajectory_setpoint')
        else:
            self.get_logger().warn(
                'px4_msgs not found in Python env; falling back to /fmu/in/setpoint_velocity'
            )
    
    def _get_default_prompt(self):
        """Fallback prompt if file not found"""
        return DEFAULT_SYSTEM_PROMPT

    def _resolve_prompt_file(self, prompt_file_override=None):
        """Resolve prompt path for both source and installed package layouts."""
        if prompt_file_override:
            return FilesystemPath(prompt_file_override).expanduser().resolve()
        return resolve_prompt_file(LATEST_PROMPT_FILENAME)
    
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

        openai_key = self._parse_openai_key_file(text)
        if openai_key:
            return openai_key

        self.get_logger().warn('OpenAI key not found in key file')
        return None

    def _parse_openai_key_file(self, text):
        """Parse OpenAI API key from a text file."""
        openai_key = None

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

        if not openai_key:
            raw_lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith('#')]
            if len(raw_lines) >= 1:
                openai_key = _clean(raw_lines[0])

        return openai_key

    def depth_image_callback(self, msg):
        """Store latest depth image and update MPC-consistent local obstacle map from depth."""
        try:
            if msg.encoding in ('16UC1', 'mono16'):
                depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                depth_image = depth_mm.astype(np.float32) / 1000.0
            else:
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Depth image conversion error: {e}')
            return

        with self.state_lock:
            self.depth_image = depth_image
            rotation_ned_body = np.array(self.rotation_ned_body, dtype=float, copy=True)
            current_position = np.array(self.current_position, dtype=float, copy=True)

        # Mirror MPC obstacle detection pipeline: depth -> body points -> NED -> FIFO local map.
        pts_body = self.depth_to_obstacles.depth_to_body_frame(depth_image)
        pts_body = self.depth_to_obstacles.filter_body_points(pts_body)
        pts_ned = self.depth_to_obstacles.body_to_spatial(
            pts_body,
            rotation_ned_body,
            current_position,
        )
        with self.state_lock:
            self.latest_depth_obstacles_ned = pts_ned
            self.local_obstacle_map.update(pts_ned)
    
    def odometry_callback(self, msg):
        """Update current state"""
        current_position = np.array([
            msg.pose.pose.position.y,
            msg.pose.pose.position.x,
            -msg.pose.pose.position.z
        ])
        current_velocity = np.array([
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.x,
            -msg.twist.twist.linear.z
        ])
        q = msg.pose.pose.orientation
        rotation_enu_body = self.quaternion_to_rotation_matrix(
            q.w, q.x, q.y, q.z
        )
        rotation_enu_to_ned = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=float,
        )
        with self.state_lock:
            self.current_position = current_position
            self.current_velocity = current_velocity
            self.rotation_ned_body = rotation_enu_to_ned @ rotation_enu_body

    def vehicle_odometry_callback(self, msg):
        """Update current state from px4_msgs/VehicleOdometry (NED)."""
        current_position = np.array([
            float(msg.position[0]),
            float(msg.position[1]),
            float(msg.position[2]),
        ])
        current_velocity = np.array([
            float(msg.velocity[0]),
            float(msg.velocity[1]),
            float(msg.velocity[2]),
        ])
        rotation_ned_body = self.quaternion_to_rotation_matrix(
            float(msg.q[0]),
            float(msg.q[1]),
            float(msg.q[2]),
            float(msg.q[3]),
        )
        with self.state_lock:
            self.current_position = current_position
            self.current_velocity = current_velocity
            self.rotation_ned_body = rotation_ned_body

    @staticmethod
    def quaternion_to_rotation_matrix(w, x, y, z):
        """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n < 1e-9:
            return np.eye(3)
        w, x, y, z = w / n, x / n, y / n, z / n
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=float)

    @staticmethod
    def convert_goal_to_ned(goal, goal_frame):
        """Convert user goal to local NED; gazebo/map assumed ENU."""
        g = np.array(goal, dtype=float)
        if goal_frame in ('gazebo', 'gazebo_enu', 'enu', 'map'):
            return np.array([g[1], g[0], -g[2]], dtype=float)
        return g

    def mpc_trajectory_callback(self, msg):
        """Store MPC trajectory for comparison"""
        mpc_point = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        with self.execution_lock:
            self.mpc_trajectory.append(mpc_point)
    
    def create_depth_visualization(self):
        """Create colored depth map for better LLM understanding"""
        with self.state_lock:
            if self.depth_image is None:
                return None
            depth_image = np.array(self.depth_image, copy=True)

        if depth_image is None:
            return None
        
        # Normalize depth to 0-255
        depth_norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        
        return depth_colored

    def build_environment_vector(self):
        """
        Build numerical vector v from odometry + depth for motion planning.
        """
        with self.state_lock:
            current_position = np.array(self.current_position, dtype=float, copy=True)
            current_velocity = np.array(self.current_velocity, dtype=float, copy=True)
            depth_image = None if self.depth_image is None else np.array(self.depth_image, copy=True)
            latest_depth_obstacles_ned = np.array(self.latest_depth_obstacles_ned, dtype=float, copy=True)
            local_obstacle_snapshot = self.local_obstacle_map.snapshot()

        vector = build_environment_vector(
            position_ned=current_position,
            velocity_ned=current_velocity,
            goal_ned=self.goal,
            depth_image=depth_image,
            latest_depth_obstacles_ned=latest_depth_obstacles_ned,
            local_obstacle_snapshot=local_obstacle_snapshot,
            depth_sample_count=int(self.depth_to_obstacles.n_sample),
            waypoint_count=LLM_WAYPOINT_COUNT,
        )
        self.last_environment_vector = vector
        return vector

    def build_live_prompt(self):
        """Build the exact user prompt contract used in dataset generation/training."""
        with self.state_lock:
            current_position = np.asarray(self.current_position, dtype=float).copy()
            current_velocity = np.asarray(self.current_velocity, dtype=float).copy()
            depth_image = None if self.depth_image is None else np.array(self.depth_image, copy=True)
            latest_depth_obstacles_ned = np.array(self.latest_depth_obstacles_ned, dtype=float, copy=True)
            local_obstacle_snapshot = self.local_obstacle_map.snapshot()

        scene = {
            'position': current_position,
            'velocity': current_velocity,
            'depth_image': depth_image,
            'latest_depth_obstacles_ned': latest_depth_obstacles_ned,
            'local_obstacle_snapshot': local_obstacle_snapshot,
        }
        prompt_text, env_vector, env_text = build_prompt_from_scene(
            goal_ned=np.asarray(self.goal, dtype=float),
            scene=scene,
            system_prompt=self.system_prompt,
            prompt_mode=PROMPT_MODE_ENV_NL_ONLY,
            waypoint_count=LLM_WAYPOINT_COUNT,
            depth_sample_count=int(self.depth_to_obstacles.n_sample),
        )
        if prompt_text is None or env_vector is None or env_text is None:
            raise RuntimeError('Failed to build live prompt from current scene')
        self.last_environment_vector = env_vector
        return prompt_text.strip(), env_vector, env_text

    def _publish_debug_target_waypoint(self, target_position):
        """Publish the currently active LLM waypoint for visualization/debugging."""
        next_position = np.asarray(target_position, dtype=float)
        with self.execution_lock:
            self.llm_trajectory.append(next_position)
        traj_msg = PoseStamped()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.header.frame_id = 'ned'
        traj_msg.pose.position.x = float(next_position[0])
        traj_msg.pose.position.y = float(next_position[1])
        traj_msg.pose.position.z = float(next_position[2])
        traj_msg.pose.orientation.w = 1.0
        self.llm_traj_pub.publish(traj_msg)

    def _target_reached(self, target_position):
        """Match dataset replay semantics: hold each waypoint until the drone reaches it."""
        target = np.asarray(target_position, dtype=float)
        with self.state_lock:
            current_position = np.array(self.current_position, dtype=float, copy=True)
        xy_error = float(np.linalg.norm(
            np.asarray(target[:2], dtype=float) - np.asarray(current_position[:2], dtype=float)
        ))
        z_error = abs(float(target[2] - current_position[2]))
        acceptance_radius = float(self.get_parameter('waypoint_acceptance_radius_m').value)
        return xy_error <= acceptance_radius and z_error <= 0.50

    def _has_depth_image(self):
        with self.state_lock:
            return self.depth_image is not None

    def _goal_reached(self):
        with self.state_lock:
            current_position = np.array(self.current_position, dtype=float, copy=True)
        dist_to_goal = np.linalg.norm(self.goal - current_position)
        return dist_to_goal < 0.5

    def _hold_current_position(self):
        with self.state_lock:
            hold_target = np.array(self.current_position, dtype=float, copy=True)
        with self.execution_lock:
            self.active_waypoint = None
            self.active_waypoint_batch_index = None
            self.command_target = hold_target
            self.pending_waypoints.clear()
        return hold_target

    def _llm_worker_loop(self):
        period = max(0.05, 1.0 / float(self.get_parameter('update_rate').value))
        while not self.stop_event.is_set():
            cycle_started = time.monotonic()
            try:
                self._run_llm_cycle()
            except Exception as e:
                self.get_logger().error(f'LLM worker loop failed: {e}')
            sleep_s = max(0.0, period - (time.monotonic() - cycle_started))
            self.stop_event.wait(sleep_s)

    def _run_llm_cycle(self):
        """Background LLM loop that continuously refreshes the latest waypoint rollout."""
        if not self._has_depth_image():
            self.get_logger().warn('Waiting for depth camera data...', throttle_duration_sec=2.0)
            return

        if self._goal_reached():
            self.get_logger().info('Goal reached!', throttle_duration_sec=1.0)
            self._hold_current_position()
            return

        output = self.query_llm()
        if output is None:
            return

        llm_waypoints = self.extract_waypoints(output)
        if not llm_waypoints:
            return

        stamp_msg = self.get_clock().now().to_msg()
        with self.execution_lock:
            self.rollout_counter += 1
            self.latest_rollout_id = self.rollout_counter
            self.latest_rollout_waypoints = [
                np.array(wp, dtype=float, copy=True) for wp in llm_waypoints
            ]
            rollout_id = self.latest_rollout_id

        self.publish_llm_trajectory_sequence(llm_waypoints, stamp_msg)
        self.get_logger().info(
            f'Queued rollout {rollout_id} with {len(llm_waypoints)} waypoints from latest PlannerOutput'
        )

    def _tracking_worker_loop(self):
        period = 0.1
        while not self.stop_event.is_set():
            cycle_started = time.monotonic()
            try:
                self._run_tracking_cycle()
            except Exception as e:
                self.get_logger().error(f'Waypoint tracking loop failed: {e}')
            sleep_s = max(0.0, period - (time.monotonic() - cycle_started))
            self.stop_event.wait(sleep_s)

    def _run_tracking_cycle(self):
        """Consume the most recent waypoint rollout without blocking on LLM inference."""
        if self._goal_reached():
            self.get_logger().info('Goal reached!', throttle_duration_sec=1.0)
            self._hold_current_position()
            return

        rollout_loaded = None
        reached = None
        reached_idx = None
        next_waypoint = None
        next_waypoint_idx = None
        with self.state_lock:
            current_position_hold = np.array(self.current_position, dtype=float, copy=True)

        with self.execution_lock:
            if self.latest_rollout_id > self.applied_rollout_id and self.latest_rollout_waypoints:
                self.pending_waypoints = deque(
                    np.array(wp, dtype=float, copy=True)
                    for wp in self.latest_rollout_waypoints
                )
                self.active_waypoint = None
                self.active_waypoint_batch_index = None
                self.completed_waypoints_in_rollout = 0
                self.applied_rollout_id = self.latest_rollout_id
                rollout_loaded = (self.applied_rollout_id, len(self.pending_waypoints))

            active_waypoint = None if self.active_waypoint is None else np.array(
                self.active_waypoint, dtype=float, copy=True
            )

        if active_waypoint is not None and self._target_reached(active_waypoint):
            with self.execution_lock:
                if self.active_waypoint is not None:
                    reached = np.array(self.active_waypoint, dtype=float, copy=True)
                    reached_idx = self.active_waypoint_batch_index
                    self.completed_waypoints_in_rollout += 1
                    self.active_waypoint = None
                    self.active_waypoint_batch_index = None
                    self.command_target = np.array(reached, dtype=float, copy=True)

        with self.execution_lock:
            if self.active_waypoint is None and self.pending_waypoints:
                self.active_waypoint = np.array(self.pending_waypoints.popleft(), dtype=float, copy=True)
                self.active_waypoint_batch_index = self.completed_waypoints_in_rollout + 1
                self.command_target = np.array(self.active_waypoint, dtype=float, copy=True)
                next_waypoint = np.array(self.active_waypoint, dtype=float, copy=True)
                next_waypoint_idx = self.active_waypoint_batch_index
            elif self.active_waypoint is None and self.command_target is None:
                self.command_target = np.array(current_position_hold, dtype=float, copy=True)

            active_waypoint = None if self.active_waypoint is None else np.array(
                self.active_waypoint, dtype=float, copy=True
            )
            active_waypoint_idx = self.active_waypoint_batch_index
            applied_rollout_id = self.applied_rollout_id

        if rollout_loaded is not None:
            rollout_id, rollout_len = rollout_loaded
            self.get_logger().info(
                f'Loaded updated rollout {rollout_id} with {rollout_len} waypoints for tracking'
            )

        if reached is not None:
            self.get_logger().info(
                f'Reached rollout {applied_rollout_id} waypoint {reached_idx}/{LLM_WAYPOINT_COUNT} '
                f'({reached[0]:.2f}, {reached[1]:.2f}, {reached[2]:.2f})'
            )

        if next_waypoint is not None:
            self._publish_debug_target_waypoint(next_waypoint)
            self.get_logger().info(
                f'Tracking rollout {applied_rollout_id} waypoint '
                f'{next_waypoint_idx}/{LLM_WAYPOINT_COUNT} '
                f'({next_waypoint[0]:.2f}, {next_waypoint[1]:.2f}, {next_waypoint[2]:.2f})'
            )

        if active_waypoint is None:
            return

        with self.state_lock:
            current_position = np.array(self.current_position, dtype=float, copy=True)
        target = np.asarray(active_waypoint, dtype=float)
        xy_error = float(np.linalg.norm(target[:2] - np.asarray(current_position[:2], dtype=float)))
        z_error = abs(float(target[2] - current_position[2]))
        self.get_logger().info(
            f'Waiting for drone to reach rollout {applied_rollout_id} waypoint '
            f'{active_waypoint_idx}/{LLM_WAYPOINT_COUNT}: '
            f'target=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}), '
            f'current=({current_position[0]:.2f}, {current_position[1]:.2f}, {current_position[2]:.2f}), '
            f'xy_error={xy_error:.2f}m, z_error={z_error:.2f}m',
            throttle_duration_sec=1.0,
        )

        error = self.compute_trajectory_error()
        if error > 0.0:
            self.get_logger().info(
                f'LLM vs MPC error: {error:.3f}m',
                throttle_duration_sec=1.0
            )
    
    def _get_openai_client(self):
        """Initialize OpenAI client on first use."""
        if self.client is not None:
            return self.client
        if OpenAI is None:
            raise RuntimeError('openai package is not installed in this Python environment')

        api_key = self._load_openai_key()
        if not api_key:
            raise RuntimeError('OpenAI API key not found. Set OPENAI_API_KEY or LLM_API_KEY_FILE')

        self.client = OpenAI(api_key=api_key)
        return self.client

    def _get_structured_client(self, provider: str):
        """Initialize an instructor-wrapped OpenAI-compatible client on first use."""
        if self.structured_client is not None:
            return self.structured_client
        if instructor is None:
            raise RuntimeError('instructor package is not installed in this Python environment')
        if OpenAI is None:
            raise RuntimeError('openai package is not installed in this Python environment')

        provider = str(provider).strip().lower()
        if provider == 'openai':
            base_client = self._get_openai_client()
        elif provider in ('vllm', 'tgis'):
            api_key = os.getenv('VLLM_API_KEY', str(self.get_parameter('vllm_api_key').value).strip())
            if not api_key:
                raise RuntimeError('vLLM API key is empty. Set VLLM_API_KEY or parameter vllm_api_key')
            base_url = str(self.get_parameter('vllm_url').value).strip()
            if base_url.endswith('/chat/completions'):
                base_url = base_url[: -len('/chat/completions')]
            base_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        else:
            raise RuntimeError(f'Instructor structured output is not configured for provider={provider}')

        self.structured_client = instructor.from_openai(base_client)
        return self.structured_client

    def query_llm(self):
        """Query the configured LLM backend and return a validated PlannerOutput."""
        user_message, env_vector, env_text = self.build_live_prompt()

        provider = str(self.get_parameter('llm_provider').value).strip().lower()
        if provider == 'openai':
            output = self.query_structured_model(user_message, provider)
        elif provider in ('vllm', 'tgis'):
            output = self.query_structured_model(user_message, provider)
        else:
            content = self.query_qwen_model(user_message)
            self.get_logger().info(f'Raw LLM response:\n{content}')
            try:
                start = content.find('{')
                end = content.rfind('}') + 1
                output = PlannerOutput.model_validate_json(content[start:end])
            except Exception as e:
                self.get_logger().error(f'Failed to parse LLM response: {e}')
                return None

        return output

    def extract_waypoints(self, output):
        """Extract the fixed 5-waypoint rollout from a PlannerOutput object."""
        if output is None:
            return None
        try:
            if len(output.waypoints) != LLM_WAYPOINT_COUNT:
                raise ValueError(
                    f"Expected exactly {LLM_WAYPOINT_COUNT} waypoints, got {len(output.waypoints)}"
                )

            llm_waypoints = []
            for i, pt in enumerate(output.waypoints):
                wp_xy = np.array([float(pt.x), float(pt.y)], dtype=float)
                if not np.all(np.isfinite(wp_xy)):
                    raise ValueError(f"waypoint {i} contains non-finite x/y values")

                wp_xyz = np.array([
                    float(wp_xy[0]),
                    float(wp_xy[1]),
                    float(self.goal[2]),
                ], dtype=float)
                if not np.all(np.isfinite(wp_xyz)):
                    raise ValueError(f"waypoint {i} contains non-finite values")
                llm_waypoints.append(wp_xyz)

            self.get_logger().info(f"Observation: {output.plan_trace.observation}")
            self.get_logger().info(f"Reasoning: {output.plan_trace.reasoning}")
            return llm_waypoints
        except (AttributeError, ValueError, TypeError) as e:
            self.get_logger().error(f'Failed to validate structured planner output: {e}')
            return None

    def publish_llm_trajectory_sequence(self, waypoints_ned, stamp_msg):
        """Publish the full LLM waypoint rollout used for comparison and debugging."""
        if waypoints_ned is None:
            return
        path = Path()
        path.header.stamp = stamp_msg
        path.header.frame_id = 'ned'
        for wp in waypoints_ned:
            pose = PoseStamped()
            pose.header.stamp = stamp_msg
            pose.header.frame_id = 'ned'
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = float(wp[2])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.llm_traj_seq_pub.publish(path)

    def query_openai_model(self, user_message):
        """Query OpenAI cloud model (kept as optional backend)."""
        client = self._get_openai_client()
        model_name = str(self.get_parameter('openai_model').value)
        prompt = (
            f"{self.system_prompt}\n\n"
            f"{user_message}\n\n"
            "Return only one JSON object."
        )
        try:
            response = client.responses.create(
                model=model_name,
                input=prompt,
            )
            return response.output_text
        except Exception as e:
            self.get_logger().error(f'OpenAI request failed (model={model_name}): {e}')
            raise

    def query_structured_model(self, user_message, provider):
        """Query an OpenAI-compatible backend and extract PlannerOutput via instructor."""
        client = self._get_structured_client(provider)
        if provider == 'openai':
            model_name = str(self.get_parameter('openai_model').value)
            temperature = 0.1
            max_tokens = 400
        else:
            model_name = str(self.get_parameter('vllm_model').value).strip()
            temperature = float(self.get_parameter('vllm_temperature').value)
            max_tokens = int(self.get_parameter('vllm_max_tokens').value)

        try:
            return client.chat.completions.create(
                model=model_name,
                response_model=PlannerOutput,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            self.get_logger().error(
                f'Structured request failed (provider={provider}, model={model_name}): {e}'
            )
            raise

    def query_qwen_model(self, user_message):
        """Query local Qwen model through Ollama-compatible endpoint."""
        qwen_prompt = (
            "<|im_start|>system\n"
            f"{self.system_prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_message}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        model_name = self.get_parameter('qwen_model').value

        payload = {
            "model": model_name,
            "prompt": qwen_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
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
                "Local Qwen request failed "
                f"(status={e.code}, url={url}, model={model_name}): {error_body}"
            )
            raise
        except urllib.error.URLError as e:
            self.get_logger().error(
                "Local Qwen request failed "
                f"(url={url}, model={model_name}): {e}"
            )
            raise

    def compute_velocity_command(self, target_position):
        """Convert target position to velocity command"""
        with self.state_lock:
            current_position = np.array(self.current_position, dtype=float, copy=True)
        direction = target_position - current_position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            max_speed = float(self.get_parameter('max_velocity_mps').value)
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

    def publish_active_command(self):
        """Continuously stream the currently active waypoint/setpoint to PX4.

        The planning timer runs slowly because it may block on an LLM request,
        but PX4 OFFBOARD control behaves much better when the current setpoint is
        refreshed at a steady rate. This timer decouples control streaming from
        LLM query latency.
        """
        with self.execution_lock:
            target_position = None if self.command_target is None else np.array(
                self.command_target, dtype=float, copy=True
            )

        if target_position is None:
            return

        if HAS_PX4_MSGS:
            self.publish_trajectory_setpoint(target_position)
        else:
            velocity = self.compute_velocity_command(target_position)
            self.publish_velocity(velocity)

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
            with self.state_lock:
                current_position = np.array(self.current_position, dtype=float, copy=True)
            self.publish_trajectory_setpoint(current_position)
        else:
            self.publish_velocity(np.zeros(3))
    
    def compute_trajectory_error(self):
        """
        Compute supremum (max) error between LLM and MPC trajectories
        Returns: max Euclidean distance between corresponding points
        """
        with self.execution_lock:
            llm_trajectory = [np.array(pt, dtype=float, copy=True) for pt in self.llm_trajectory]
            mpc_trajectory = [np.array(pt, dtype=float, copy=True) for pt in self.mpc_trajectory]

        if len(llm_trajectory) == 0 or len(mpc_trajectory) == 0:
            return 0.0
        
        # Align trajectories (use minimum length)
        min_len = min(len(llm_trajectory), len(mpc_trajectory))
        
        errors = []
        for i in range(min_len):
            llm_point = llm_trajectory[i]
            mpc_point = mpc_trajectory[i]
            error = np.linalg.norm(llm_point - mpc_point)
            errors.append(error)
        
        # Supremum (maximum error)
        sup_error = max(errors) if errors else 0.0
        
        return sup_error

    def destroy_node(self):
        self.stop_event.set()
        for worker in (getattr(self, 'llm_thread', None), getattr(self, 'tracking_thread', None)):
            if worker is not None and worker.is_alive():
                worker.join(timeout=1.0)
        return super().destroy_node()


def main(args=None):
    goal_override, ros_args = parse_goal_overrides(args)
    rclpy.init(args=ros_args)

    node = None
    try:
        node = LLMTrajectoryPlanner(goal_override=goal_override)
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
