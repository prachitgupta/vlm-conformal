#!/usr/bin/env python3
"""
Synchronized dataset generator for LLM fine-tuning (Step 3 pipeline).

For each timestamped MPC label message on /mpc/trajectory_sequence (nav_msgs/Path):
1) find nearest synchronized scene snapshot (odom + depth)
2) build prompt using the same environment-vector/prompt format as llm_planner.py
3) write one CSV row with prompt + 5-waypoint MPC label sequence

This fixes time alignment by anchoring on the MPC label timestamp and matching buffered
sensor/state data within configurable tolerances.
"""

from __future__ import annotations

import csv
import json
import os
from collections import deque
from pathlib import Path as FsPath

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry, Path as RosPath
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

try:
    from px4_msgs.msg import VehicleOdometry
    HAS_PX4_MSGS = True
except ImportError:
    HAS_PX4_MSGS = False


DEPTH_HFOV_DEG = 72.995
DEPTH_VFOV_DEG = 58.053
DEPTH_MIN_M = 0.2
DEPTH_MAX_M = 19.1
OBS_FIFO_LEN = 200
MPC_DEPTH_SAMPLE_COUNT = 500
LABEL_WAYPOINT_COUNT = 5


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
        self.declare_parameter('output_csv', '/tmp/llm_sync_dataset.csv')
        self.declare_parameter('max_odom_sync_dt_s', 0.10)   # depth->odom pairing tolerance
        self.declare_parameter('max_label_sync_dt_s', 0.20)  # label->scene pairing tolerance
        self.declare_parameter('odom_buffer_size', 400)
        self.declare_parameter('scene_buffer_size', 400)
        self.declare_parameter('depth_obstacle_samples', MPC_DEPTH_SAMPLE_COUNT)

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
        self.odom_buffer = deque(maxlen=int(self.get_parameter('odom_buffer_size').value))
        self.scene_buffer = deque(maxlen=int(self.get_parameter('scene_buffer_size').value))

        self.bridge = CvBridge()
        self.depth_to_obstacles = DepthToObstacles(
            n_sample=int(self.get_parameter('depth_obstacle_samples').value)
        )
        self.local_obstacle_map = LocalObstacleMap(OBS_FIFO_LEN)
        self.system_prompt = self._load_system_prompt()

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
        self.create_subscription(
            RosPath,
            '/mpc/trajectory_sequence',
            self.mpc_trajectory_sequence_callback,
            10,
        )

        self._init_csv()
        self.get_logger().info(f'Sync dataset generator started. output_csv={self.output_csv}')
        self.get_logger().info('Anchor topic: /mpc/trajectory_sequence (5-waypoint labels)')

    def _load_system_prompt(self) -> str:
        local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/llm_prompt.txt'))
        if os.path.exists(local_path):
            with open(local_path, 'r') as f:
                return f.read()
        share_dir = get_package_share_directory('llm_drone')
        share_path = os.path.join(share_dir, 'config', 'llm_prompt.txt')
        try:
            with open(share_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            self.get_logger().warn(f'Prompt file not found: {share_path}')
            return "You are a drone motion planning system. Generate safe trajectories."

    def _init_csv(self) -> None:
        out_path = FsPath(self.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (not out_path.exists()) or out_path.stat().st_size == 0
        self._csv_fp = open(out_path, 'a', newline='')
        self._csv_writer = csv.writer(self._csv_fp)
        if write_header:
            header = [
                'label_stamp_s',
                'scene_stamp_s',
                'sync_dt_ms',
                'prompt',
                'mpc_waypoints_json',
            ]
            for i in range(LABEL_WAYPOINT_COUNT):
                header += [f'wp{i}_x', f'wp{i}_y', f'wp{i}_z']
            self._csv_writer.writerow(header)
            self._csv_fp.flush()

    @staticmethod
    def stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def px4_stamp_us_to_sec(stamp_us: int | float) -> float:
        return float(stamp_us) * 1e-6

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
        stamp_s = self.stamp_to_sec(msg.header.stamp)
        q = msg.pose.pose.orientation
        self.odom_buffer.append({
            'stamp_s': stamp_s,
            'position': np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=float),
            'velocity': np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z], dtype=float),
            'rotation_ned_body': self.quaternion_to_rotation_matrix(q.w, q.x, q.y, q.z),
        })

    def vehicle_odometry_callback(self, msg: VehicleOdometry) -> None:
        stamp_us = int(msg.timestamp_sample if getattr(msg, 'timestamp_sample', 0) else msg.timestamp)
        stamp_s = self.px4_stamp_us_to_sec(stamp_us)
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

        depth_stamp_s = self.stamp_to_sec(msg.header.stamp)
        odom, dt = self._find_nearest_by_stamp(self.odom_buffer, depth_stamp_s, self.max_odom_sync_dt_s)
        if odom is None:
            self.get_logger().warn(
                f'No odom match for depth frame (dt={dt if dt is not None else -1:.3f}s)',
                throttle_duration_sec=2.0,
            )
            return

        pts_body = self.depth_to_obstacles.depth_to_body_frame(depth_image)
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

    def mpc_trajectory_sequence_callback(self, msg: RosPath) -> None:
        self.labels_seen += 1
        label_stamp_s = self.stamp_to_sec(msg.header.stamp)
        if len(msg.poses) < LABEL_WAYPOINT_COUNT:
            self.labels_dropped += 1
            self.get_logger().warn(
                f'Dropped label: expected {LABEL_WAYPOINT_COUNT} waypoints, got {len(msg.poses)}'
            )
            return

        scene, dt = self._find_nearest_by_stamp(self.scene_buffer, label_stamp_s, self.max_label_sync_dt_s)
        if scene is None:
            self.labels_dropped += 1
            self.get_logger().warn(
                f'Dropped label: no scene match within {self.max_label_sync_dt_s:.3f}s '
                f'(nearest_dt={dt if dt is not None else -1:.3f}s)',
                throttle_duration_sec=1.0,
            )
            return

        label_waypoints = []
        for pose in msg.poses[:LABEL_WAYPOINT_COUNT]:
            label_waypoints.append([
                float(pose.pose.position.x),
                float(pose.pose.position.y),
                float(pose.pose.position.z),
            ])

        env_vector = self.build_environment_vector_from_scene(scene)
        env_text = self.translate_vector_to_nlp(env_vector)
        user_prompt = self.compose_final_prompt(env_vector, env_text)

        row = [
            f'{label_stamp_s:.9f}',
            f'{scene["stamp_s"]:.9f}',
            f'{1000.0 * abs(label_stamp_s - scene["stamp_s"]):.3f}',
            user_prompt,
            json.dumps(label_waypoints),
        ]
        for p in label_waypoints:
            row += [p[0], p[1], p[2]]
        self._csv_writer.writerow(row)
        self._csv_fp.flush()
        self.rows_written += 1

        self.get_logger().info(
            f'Wrote sample #{self.rows_written} | sync_dt={1000.0 * abs(label_stamp_s - scene["stamp_s"]):.1f} ms'
        )

    def build_environment_vector_from_scene(self, scene: dict) -> dict:
        depth = scene['depth_image']
        pos = scene['position']
        vel = scene['velocity']
        latest_obs = scene['latest_depth_obstacles_ned']
        local_obs_snapshot = scene['local_obstacle_snapshot']

        valid_depth = depth[np.isfinite(depth)]
        valid_depth = valid_depth[(valid_depth > DEPTH_MIN_M) & (valid_depth < 20.0)]
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
            patch_valid = patch_valid[(patch_valid > DEPTH_MIN_M) & (patch_valid < 20.0)]
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

        goal_delta = self.goal - pos
        dist_to_goal = float(np.linalg.norm(goal_delta))
        speed = float(np.linalg.norm(vel))
        heading_to_goal_xy = float(np.arctan2(goal_delta[1], goal_delta[0]))

        if local_obs_snapshot is not None and local_obs_snapshot.shape[0] > 0:
            rel = local_obs_snapshot - pos[None, :]
            dists = np.linalg.norm(rel, axis=1)
            idx = int(np.argmin(dists))
            nearest_dist_ned = float(dists[idx])
            nearest_point_ned = [float(x) for x in local_obs_snapshot[idx]]
            nearest_vec_ned = [float(x) for x in rel[idx]]
        else:
            nearest_dist_ned = 20.0
            nearest_point_ned = [float(pos[0]), float(pos[1]), float(pos[2])]
            nearest_vec_ned = [20.0, 0.0, 0.0]

        latest_obs_count = int(latest_obs.shape[0]) if isinstance(latest_obs, np.ndarray) and latest_obs.ndim == 2 else 0
        local_obs_count = int(local_obs_snapshot.shape[0]) if isinstance(local_obs_snapshot, np.ndarray) and local_obs_snapshot.ndim == 2 else 0

        return {
            "current_position_ned_m": [float(x) for x in pos],
            "current_velocity_mps": [float(v) for v in vel],
            "goal_position_ned_m": [float(g) for g in self.goal],
            "goal_delta_m": [float(d) for d in goal_delta],
            "distance_to_goal_m": dist_to_goal,
            "speed_mps": speed,
            "heading_to_goal_xy_rad": heading_to_goal_xy,
            "obstacle_features_ned": {
                "pipeline": "mpc_consistent_depth_to_body_to_ned_fifo_local_map",
                "depth_camera_params": {
                    "hfov_deg": DEPTH_HFOV_DEG,
                    "vfov_deg": DEPTH_VFOV_DEG,
                    "depth_min_m": DEPTH_MIN_M,
                    "depth_max_m": DEPTH_MAX_M,
                    "sample_count_per_frame_target": int(self.depth_to_obstacles.n_sample),
                },
                "latest_depth_frame_obstacle_points_ned_count": latest_obs_count,
                "local_obstacle_fifo_count": local_obs_count,
                "nearest_obstacle_distance_ned_m": nearest_dist_ned,
                "nearest_obstacle_position_ned_m": nearest_point_ned,
                "nearest_obstacle_relative_ned_m": nearest_vec_ned,
                "mpc_nearest_obstacle_eq11_x_min_ned_m": nearest_point_ned,
            },
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

    @staticmethod
    def translate_vector_to_nlp(vector: dict) -> str:
        p = vector["current_position_ned_m"]
        vel = vector["current_velocity_mps"]
        g = vector["goal_position_ned_m"]
        d = vector["goal_delta_m"]
        obs_ned = vector["obstacle_features_ned"]
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
            f"- MPC-consistent obstacle pipeline: {obs_ned['pipeline']}.\n"
            f"- Current depth frame obstacle points (NED): {obs_ned['latest_depth_frame_obstacle_points_ned_count']}; "
            f"FIFO local obstacle map size: {obs_ned['local_obstacle_fifo_count']} points. "
            f"Nearest obstacle in NED is at ({obs_ned['nearest_obstacle_position_ned_m'][0]:.2f}, "
            f"{obs_ned['nearest_obstacle_position_ned_m'][1]:.2f}, {obs_ned['nearest_obstacle_position_ned_m'][2]:.2f}) m, "
            f"relative vector ({obs_ned['nearest_obstacle_relative_ned_m'][0]:.2f}, "
            f"{obs_ned['nearest_obstacle_relative_ned_m'][1]:.2f}, {obs_ned['nearest_obstacle_relative_ned_m'][2]:.2f}) m, "
            f"distance {obs_ned['nearest_obstacle_distance_ned_m']:.2f} m.\n"
            f"- Depth validity fraction is {depth['valid_fraction']:.2f}. Global nearest obstacle distance is {depth['global_min_m']:.2f} m.\n"
            f"- Sector minimum distances [far_left, left, center, right, far_right] are "
            f"[{mins['far_left']:.2f}, {mins['left']:.2f}, {mins['center']:.2f}, {mins['right']:.2f}, {mins['far_right']:.2f}] m.\n"
            f"- Nearest obstacle sector is {depth['nearest_obstacle_sector']} at {depth['nearest_obstacle_distance_m']:.2f} m "
            f"and clearance status is {depth['clearance_status']}.\n"
            f"- Lateral planning hint: {lateral_hint}."
        )

    @staticmethod
    def compose_final_prompt(env_vector: dict, env_text: str) -> str:
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
""".strip() + "\n"

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
