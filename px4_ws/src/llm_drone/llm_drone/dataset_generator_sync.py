#!/usr/bin/env python3
"""
Synchronized dataset generator for MPC imitation data.

Supported label modes:
1) local_mpc_prediction_window:
   prompt at the solver scene timestamp -> 5 open-loop waypoints from one solved MPC plan
2) executed_committed_waypoint_window:
   prompt at the solver scene timestamp -> 5 committed waypoints that actually became active

This keeps the prompt aligned to the scene the solver saw, while allowing you to choose
between local open-loop predictions and actual executed committed waypoints.
"""

from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path as FsPath

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as RosPath
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from llm_drone.eval import EvalResult, evaluate_response
from llm_drone.llm_prompt_common import (
    DATASET_PROMPT_FILENAME,
    DEPTH_HFOV_DEG,
    DEPTH_VFOV_DEG,
    DEPTH_MIN_M,
    DEPTH_MAX_M,
    OBS_FIFO_LEN,
    MPC_DEPTH_SAMPLE_COUNT,
    build_environment_vector,
    compose_final_prompt,
    load_system_prompt,
    translate_vector_to_nlp,
)

try:
    from px4_msgs.msg import VehicleOdometry
    HAS_PX4_MSGS = True
except ImportError:
    HAS_PX4_MSGS = False

LABEL_WAYPOINT_COUNT = 5
LOCAL_PREDICTION_SEQUENCE_TOPIC = '/mpc/local_prediction_sequence'
FRESH_COMMITTED_WAYPOINT_TOPIC = '/mpc/committed_waypoint_fresh'
EXECUTED_COMMITTED_WAYPOINT_TOPIC = '/mpc/committed_waypoint_executed'
LABEL_MODE_LOCAL_PREDICTION = 'local_mpc_prediction_window'
LABEL_MODE_EXECUTED = 'executed_committed_waypoint_window'
DEPTH_FILTER_MIN_FORWARD_M = 0.35
DEPTH_FILTER_MAX_FORWARD_M = 12.0
DEPTH_FILTER_MAX_LATERAL_M = 4.0
DEPTH_FILTER_MIN_BELOW_BODY_M = -0.25
DEPTH_FILTER_MAX_BELOW_BODY_M = 2.5
DEPTH_FILTER_FLOOR_CLEARANCE_M = 0.20


class LocalObstacleMap:
    def __init__(self, maxlen: int = OBS_FIFO_LEN):
        self._buf: list[np.ndarray] = []
        self._maxlen = int(maxlen)

    def update(self, points_ned: np.ndarray) -> None:
        if points_ned is None or points_ned.shape[0] == 0:
            return
        for pt in points_ned:
            self._buf.append(np.array(pt, dtype=float))
        if len(self._buf) > self._maxlen:
            self._buf = self._buf[-self._maxlen:]

    def nearest(self, x_ref: np.ndarray) -> np.ndarray | None:
        if not self._buf:
            return None
        pts = np.array(self._buf, dtype=float)
        dists = np.linalg.norm(pts - np.asarray(x_ref, dtype=float)[None, :3], axis=1)
        return pts[int(np.argmin(dists))]

    def snapshot(self, max_points: int | None = None) -> np.ndarray:
        if not self._buf:
            return np.zeros((0, 3), dtype=float)
        pts = np.array(self._buf, dtype=float)
        if max_points is not None and pts.shape[0] > max_points:
            pts = pts[-int(max_points):]
        return pts


class DepthToObstacles:
    def __init__(
        self,
        hfov_deg: float = DEPTH_HFOV_DEG,
        vfov_deg: float = DEPTH_VFOV_DEG,
        n_sample: int = MPC_DEPTH_SAMPLE_COUNT,
    ):
        self.hfov = np.deg2rad(float(hfov_deg))
        self.vfov = np.deg2rad(float(vfov_deg))
        self.n_sample = int(n_sample)
        self.rotation_body_camera = np.array(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=float
        )
        self.translation_body_camera_m = np.array([0.13233, 0.0, 0.26078], dtype=float)

    def depth_to_body_frame(self, depth_img: np.ndarray) -> np.ndarray:
        if depth_img is None or depth_img.ndim != 2:
            return np.zeros((0, 3), dtype=float)
        h, w = depth_img.shape
        total = h * w
        if total <= 0:
            return np.zeros((0, 3), dtype=float)

        fy = (h / 2.0) / np.tan(self.vfov / 2.0)
        fx = (w / 2.0) / np.tan(self.hfov / 2.0)
        cx, cy = w / 2.0, h / 2.0

        idx = np.random.choice(total, min(self.n_sample, total), replace=False)
        rows, cols = np.unravel_index(idx, (h, w))
        depths = depth_img[rows, cols]
        valid = (depths > DEPTH_MIN_M) & (depths < DEPTH_MAX_M) & np.isfinite(depths)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=float)
        rows, cols, depths = rows[valid], cols[valid], depths[valid]

        xc = (cols - cx) / fx * depths
        yc = (rows - cy) / fy * depths
        zc = depths
        pts_cam = np.column_stack([xc, yc, zc])
        return (self.rotation_body_camera @ pts_cam.T).T + self.translation_body_camera_m

    def filter_body_points(self, pts_body: np.ndarray) -> np.ndarray:
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
    def body_to_spatial(pts_body: np.ndarray, rotation_ned_body: np.ndarray, pos_ned: np.ndarray) -> np.ndarray:
        if pts_body is None or pts_body.shape[0] == 0:
            return np.zeros((0, 3), dtype=float)
        return (rotation_ned_body @ pts_body.T).T + np.asarray(pos_ned, dtype=float)[None, :]


class SyncDatasetGenerator(Node):
    def __init__(self):
        super().__init__('sync_dataset_generator')

        self.declare_parameter('goal_x', 35.0)
        self.declare_parameter('goal_y', 3.0)
        self.declare_parameter('goal_z', 2.5)
        self.declare_parameter('goal_frame', 'ned')
        self.declare_parameter('output_csv', '../dataset/dataset.csv')
        self.declare_parameter('label_mode', LABEL_MODE_LOCAL_PREDICTION)
        self.declare_parameter('max_odom_sync_dt_s', 0.10)   # depth->odom pairing tolerance
        self.declare_parameter('max_label_sync_dt_s', 0.20)  # label->scene pairing tolerance
        self.declare_parameter('odom_buffer_size', 400)
        self.declare_parameter('scene_buffer_size', 400)
        self.declare_parameter('waypoint_buffer_size', 400)
        self.declare_parameter('depth_obstacle_samples', MPC_DEPTH_SAMPLE_COUNT)
        self.declare_parameter('eval_dt_s', 0.1)
        self.declare_parameter('eval_v_max_mps', 15.0)
        self.declare_parameter('eval_a_max_mps2', 10.0)
        self.declare_parameter('eval_safety_radius_m', 0.5)

        self.goal = self.convert_goal_to_ned(
            np.array([
                self.get_parameter('goal_x').value,
                self.get_parameter('goal_y').value,
                self.get_parameter('goal_z').value,
            ], dtype=float),
            str(self.get_parameter('goal_frame').value),
        )
        self.max_odom_sync_dt_s = float(self.get_parameter('max_odom_sync_dt_s').value)
        self.max_label_sync_dt_s = float(self.get_parameter('max_label_sync_dt_s').value)
        self.output_csv = str(self.get_parameter('output_csv').value)
        self.label_mode = str(self.get_parameter('label_mode').value).strip().lower()
        self.eval_dt_s = float(self.get_parameter('eval_dt_s').value)
        self.eval_v_max_mps = float(self.get_parameter('eval_v_max_mps').value)
        self.eval_a_max_mps2 = float(self.get_parameter('eval_a_max_mps2').value)
        self.eval_safety_radius_m = float(self.get_parameter('eval_safety_radius_m').value)
        if self.label_mode not in (LABEL_MODE_LOCAL_PREDICTION, LABEL_MODE_EXECUTED):
            self.get_logger().warn(
                f'Unknown label_mode={self.label_mode!r}; falling back to {LABEL_MODE_EXECUTED}.'
            )
            self.label_mode = LABEL_MODE_EXECUTED
        self.odom_buffer = deque(maxlen=int(self.get_parameter('odom_buffer_size').value))
        self.scene_buffer = deque(maxlen=int(self.get_parameter('scene_buffer_size').value))
        self.executed_waypoint_buffer = deque(maxlen=int(self.get_parameter('waypoint_buffer_size').value))
        self.pending_anchors = deque()

        self.bridge = CvBridge()
        self.depth_to_obstacles = DepthToObstacles(
            n_sample=int(self.get_parameter('depth_obstacle_samples').value)
        )
        self.local_obstacle_map = LocalObstacleMap(OBS_FIFO_LEN)
        self.system_prompt = load_system_prompt(DATASET_PROMPT_FILENAME)

        self.rows_written = 0
        self.labels_seen = 0
        self.labels_dropped = 0

        # Subscribers
        if HAS_PX4_MSGS:
            self.create_subscription(
                VehicleOdometry,
                '/fmu/out/vehicle_odometry',
                self.vehicle_odometry_callback,
                qos_profile_sensor_data,
            )
        else:
            self.create_subscription(
                Odometry,
                '/fmu/out/vehicle_odometry',
                self.odometry_callback,
                qos_profile_sensor_data,
            )
        self.create_subscription(
            Image,
            '/depth_camera',
            self.depth_image_callback,
            qos_profile_sensor_data,
        )
        if self.label_mode == LABEL_MODE_LOCAL_PREDICTION:
            self.create_subscription(
                RosPath,
                LOCAL_PREDICTION_SEQUENCE_TOPIC,
                self.local_prediction_callback,
                10,
            )
        else:
            self.create_subscription(
                PoseStamped,
                FRESH_COMMITTED_WAYPOINT_TOPIC,
                self.fresh_committed_waypoint_callback,
                10,
            )
            self.create_subscription(
                PoseStamped,
                EXECUTED_COMMITTED_WAYPOINT_TOPIC,
                self.executed_committed_waypoint_callback,
                10,
            )

        self._init_csv()
        self.get_logger().info(f'Sync dataset generator started. output_csv={self.output_csv}')
        self.get_logger().info(f'label_mode={self.label_mode}')
        if self.label_mode == LABEL_MODE_LOCAL_PREDICTION:
            self.get_logger().info(
                f'Prediction topic: {LOCAL_PREDICTION_SEQUENCE_TOPIC} (5-waypoint local MPC prediction)'
            )
        else:
            self.get_logger().info(
                f'Anchor topic: {FRESH_COMMITTED_WAYPOINT_TOPIC} (solve request timestamp)'
            )
            self.get_logger().info(
                f'Executed label topic: {EXECUTED_COMMITTED_WAYPOINT_TOPIC} (actual committed waypoints)'
            )

    def _init_csv(self) -> None:
        out_path = FsPath(self.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (not out_path.exists()) or out_path.stat().st_size == 0
        expected_header_cols = [
            'anchor_stamp_s',
            'scene_stamp_s',
            'sync_dt_ms',
            'label_mode',
            'prompt',
            'label_waypoints_json',
        ]
        self._csv_fp = open(out_path, 'a', newline='')
        self._csv_writer = csv.writer(self._csv_fp)
        if write_header:
            header = list(expected_header_cols)
            for i in range(LABEL_WAYPOINT_COUNT):
                header += [f'wp{i}_x', f'wp{i}_y', f'wp{i}_z']
            self._csv_writer.writerow(header)
            self._csv_fp.flush()
        else:
            try:
                with out_path.open('r') as fp:
                    first_line = fp.readline()
            except OSError:
                first_line = ''
            if not all(col in first_line for col in expected_header_cols):
                self.get_logger().warn(
                    'Existing output_csv appears to use an older schema. '
                    'Use a fresh output file for the selected label_mode.'
                )

    @staticmethod
    def stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def px4_stamp_us_to_sec(stamp_us: int | float) -> float:
        return float(stamp_us) * 1e-6

    def _now_s(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    @staticmethod
    def quaternion_to_rotation_matrix(w, x, y, z):
        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n < 1e-9:
            return np.eye(3, dtype=float)
        w, x, y, z = w / n, x / n, y / n, z / n
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=float)

    @staticmethod
    def convert_goal_to_ned(goal, goal_frame):
        g = np.array(goal, dtype=float)
        gf = str(goal_frame).strip().lower()
        if gf in ('gazebo', 'gazebo_enu', 'enu', 'map'):
            return np.array([g[1], g[0], -g[2]], dtype=float)
        return g

    def _find_nearest_by_stamp(self, buf: deque, stamp_s: float, max_dt_s: float):
        if not buf:
            return None, None
        best = None
        best_dt = None
        for item in buf:
            dt = abs(float(item['stamp_s']) - float(stamp_s))
            if best_dt is None or dt < best_dt:
                best, best_dt = item, dt
        if best_dt is None or best_dt > max_dt_s:
            return None, best_dt
        return best, best_dt

    def odometry_callback(self, msg: Odometry) -> None:
        # Match by local receipt time so all streams share one clock domain,
        # even when publishers use mixed wall/sim/PX4 time bases.
        stamp_s = self._now_s()
        q = msg.pose.pose.orientation
        self.odom_buffer.append({
            'stamp_s': stamp_s,
            'position': np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=float),
            'velocity': np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z], dtype=float),
            'rotation_ned_body': self.quaternion_to_rotation_matrix(q.w, q.x, q.y, q.z),
        })

    def vehicle_odometry_callback(self, msg: VehicleOdometry) -> None:
        # PX4 timestamps are often boot-relative and do not share a clock domain
        # with ROS header stamps from depth images and planner labels. Use the ROS
        # receipt time so all sync buffers operate on one clock.
        stamp_s = self._now_s()
        self.odom_buffer.append({
            'stamp_s': stamp_s,
            'position': np.array([float(msg.position[0]), float(msg.position[1]), float(msg.position[2])], dtype=float),
            'velocity': np.array([float(msg.velocity[0]), float(msg.velocity[1]), float(msg.velocity[2])], dtype=float),
            'rotation_ned_body': self.quaternion_to_rotation_matrix(
                float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3])
            ),
        })

    def depth_image_callback(self, msg: Image) -> None:
        try:
            if msg.encoding in ('16UC1', 'mono16'):
                depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                depth_image = depth_mm.astype(np.float32) / 1000.0
            else:
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().warn(f'Depth decode error: {e}')
            return

        # Use receipt time here as well; depth header stamps may be in sim time
        # while the local node clock is in wall time, or vice versa.
        depth_stamp_s = self._now_s()
        odom, dt = self._find_nearest_by_stamp(self.odom_buffer, depth_stamp_s, self.max_odom_sync_dt_s)
        if odom is None:
            self.get_logger().warn(
                f'No odom match for depth frame (dt={dt if dt is not None else -1:.3f}s)',
                throttle_duration_sec=2.0,
            )
            return

        pts_body = self.depth_to_obstacles.depth_to_body_frame(depth_image)
        pts_body = self.depth_to_obstacles.filter_body_points(pts_body)
        pts_ned = self.depth_to_obstacles.body_to_spatial(
            pts_body,
            odom['rotation_ned_body'],
            odom['position'],
        )
        self.local_obstacle_map.update(pts_ned)

        self.scene_buffer.append({
            'stamp_s': depth_stamp_s,
            'position': np.array(odom['position'], copy=True),
            'velocity': np.array(odom['velocity'], copy=True),
            'rotation_ned_body': np.array(odom['rotation_ned_body'], copy=True),
            'depth_image': np.array(depth_image, copy=True),
            'latest_depth_obstacles_ned': np.array(pts_ned, copy=True),
            'local_obstacle_snapshot': self.local_obstacle_map.snapshot(),
        })

    def fresh_committed_waypoint_callback(self, msg: PoseStamped) -> None:
        self.labels_seen += 1
        self.pending_anchors.append({
            'stamp_s': self._now_s(),
        })
        self._flush_ready_samples()

    def local_prediction_callback(self, msg: RosPath) -> None:
        self.labels_seen += 1
        anchor_stamp_s = self._now_s()
        if len(msg.poses) < LABEL_WAYPOINT_COUNT:
            self.labels_dropped += 1
            self.get_logger().warn(
                f'Dropped label: expected {LABEL_WAYPOINT_COUNT} prediction waypoints, got {len(msg.poses)}'
            )
            return

        label_waypoints = []
        for pose in msg.poses[:LABEL_WAYPOINT_COUNT]:
            label_waypoints.append([
                float(pose.pose.position.x),
                float(pose.pose.position.y),
                float(pose.pose.position.z),
            ])
        self._write_sample(anchor_stamp_s, label_waypoints, LABEL_MODE_LOCAL_PREDICTION)

    def executed_committed_waypoint_callback(self, msg: PoseStamped) -> None:
        self.executed_waypoint_buffer.append({
            'stamp_s': self._now_s(),
            'position': np.array([
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ], dtype=float),
        })
        self._flush_ready_samples()

    def _executed_window_for_anchor(self, anchor_stamp_s: float):
        if not self.executed_waypoint_buffer:
            return 'waiting', None

        executed = list(self.executed_waypoint_buffer)
        active_idx = None
        for idx, item in enumerate(executed):
            if float(item['stamp_s']) <= float(anchor_stamp_s):
                active_idx = idx
            else:
                break

        if active_idx is None:
            return 'missing_past', None

        end_idx = active_idx + LABEL_WAYPOINT_COUNT
        if end_idx > len(executed):
            return 'waiting', None

        return 'ready', executed[active_idx:end_idx]

    def _flush_ready_samples(self) -> None:
        while self.pending_anchors:
            anchor = self.pending_anchors[0]
            anchor_stamp_s = float(anchor['stamp_s'])
            window_status, executed_window = self._executed_window_for_anchor(anchor_stamp_s)
            if window_status == 'waiting':
                return

            self.pending_anchors.popleft()
            if window_status == 'missing_past':
                self.labels_dropped += 1
                self.get_logger().warn(
                    f'Dropped label: no executed committed waypoint was active at anchor time '
                    f'{anchor_stamp_s:.3f}s',
                    throttle_duration_sec=1.0,
                )
                continue

            label_waypoints = [item['position'].tolist() for item in executed_window]
            self._write_sample(anchor_stamp_s, label_waypoints, LABEL_MODE_EXECUTED)

    def _write_sample(self, anchor_stamp_s: float, label_waypoints: list[list[float]], label_mode: str) -> None:
        scene, dt = self._find_nearest_by_stamp(
            self.scene_buffer,
            anchor_stamp_s,
            self.max_label_sync_dt_s,
        )
        if scene is None:
            self.labels_dropped += 1
            self.get_logger().warn(
                f'Dropped label: no scene match within {self.max_label_sync_dt_s:.3f}s '
                f'(nearest_dt={dt if dt is not None else -1:.3f}s)',
                throttle_duration_sec=1.0,
            )
            return

        env_vector = self.build_environment_vector_from_scene(scene)
        if env_vector is None:
            self.labels_dropped += 1
            self.get_logger().warn('Dropped label: environment vector unavailable')
            return
        env_text = translate_vector_to_nlp(env_vector)
        user_prompt = env_text.strip()
        label_response = self._build_label_response(label_waypoints, label_mode, scene)

        row = [
            f'{anchor_stamp_s:.9f}',
            f'{scene["stamp_s"]:.9f}',
            f'{1000.0 * abs(anchor_stamp_s - scene["stamp_s"]):.3f}',
            label_mode,
            user_prompt,
            json.dumps(label_response, separators=(',', ':')),
        ]
        for p in label_waypoints:
            row += [p[0], p[1], p[2]]
        self._csv_writer.writerow(row)
        self._csv_fp.flush()
        self.rows_written += 1

        self.get_logger().info(
            f'Wrote sample #{self.rows_written} | mode={label_mode} | '
            f'sync_dt={1000.0 * abs(anchor_stamp_s - scene["stamp_s"]):.1f} ms'
        )

    def _build_label_response(self, label_waypoints: list[list[float]], label_mode: str, scene: dict) -> dict:
        base_response = {
            'waypoints': [
                {
                    'x': float(p[0]),
                    'y': float(p[1]),
                    'z': float(p[2]),
                }
                for p in label_waypoints
            ],
        }
        eval_result = evaluate_response(
            current_position_ned=np.asarray(scene['position'], dtype=float),
            goal_ned=np.asarray(self.goal, dtype=float),
            llm_response_text=json.dumps(base_response, separators=(',', ':')),
            obstacle_points_ned=np.asarray(scene['local_obstacle_snapshot'], dtype=float),
            dt=self.eval_dt_s,
            v_max=self.eval_v_max_mps,
            a_max=self.eval_a_max_mps2,
            safety_radius=self.eval_safety_radius_m,
        )
        mode_reasoning = {
            LABEL_MODE_LOCAL_PREDICTION: 'This is the 5-step MPC local prediction aligned to the scene timestamp.',
            LABEL_MODE_EXECUTED: 'This is the 5-step committed MPC rollout that was actually executed after the solve.',
        }
        base_response['reasoning'] = self._build_label_reasoning(
            eval_result,
            mode_reasoning.get(
                label_mode,
                'This is the 5-step MPC waypoint rollout aligned to the prompt scene.',
            ),
        )
        return base_response

    @staticmethod
    def _build_label_reasoning(eval_result: EvalResult, prefix: str) -> str:
        metrics = eval_result.metrics or {}
        progress_selected = float(metrics.get('progress_selected_m', 0.0))
        progress_final = float(metrics.get('progress_final_m', 0.0))
        max_speed = float(metrics.get('max_speed_mps', 0.0))
        max_accel = float(metrics.get('max_accel_mps2', 0.0))
        min_clearance = metrics.get('min_clearance_m', None)
        clearance_is_proxy = bool(metrics.get('clearance_is_proxy', True))
        monotone_goal_progress = bool(metrics.get('monotone_goal_progress', False))

        if clearance_is_proxy or min_clearance is None:
            clearance_text = 'Obstacle clearance could not be measured from a local obstacle snapshot.'
        else:
            clearance_text = f'The rollout maintains about {float(min_clearance):.2f} m minimum obstacle clearance.'

        status_text = (
            'It passes the evaluator checks for format, kinematics, goal progress, and safety.'
            if eval_result.passed else
            'It is kept as the MPC label, but the evaluator flagged one or more checks that should be reviewed.'
        )
        monotone_text = (
            'Goal distance decreases monotonically across the 5-waypoint sequence.'
            if monotone_goal_progress else
            'Goal distance does not decrease monotonically across the full 5-waypoint sequence.'
        )
        warning_text = f' Primary warning: {eval_result.warnings[0]}' if eval_result.warnings else ''

        return (
            f'{prefix} The rollout reduces goal distance by {progress_selected:.2f} m immediately and by {progress_final:.2f} m by the final waypoint. '
            f'The rollout reaches max speed {max_speed:.2f} m/s and max acceleration {max_accel:.2f} m/s^2. '
            f'{clearance_text} {monotone_text} {status_text}{warning_text}'
        )

    def build_environment_vector_from_scene(self, scene: dict) -> dict | None:
        return build_environment_vector(
            position_ned=scene['position'],
            velocity_ned=scene['velocity'],
            goal_ned=self.goal,
            depth_image=scene['depth_image'],
            latest_depth_obstacles_ned=scene['latest_depth_obstacles_ned'],
            local_obstacle_snapshot=scene['local_obstacle_snapshot'],
            depth_sample_count=self.depth_to_obstacles.n_sample,
            waypoint_count=LABEL_WAYPOINT_COUNT,
        )

    def destroy_node(self):
        try:
            if hasattr(self, '_csv_fp') and self._csv_fp:
                self._csv_fp.flush()
                self._csv_fp.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SyncDatasetGenerator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.get_logger().info(
                f'Stopped. rows_written={node.rows_written}, labels_seen={node.labels_seen}, labels_dropped={node.labels_dropped}'
            )
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
