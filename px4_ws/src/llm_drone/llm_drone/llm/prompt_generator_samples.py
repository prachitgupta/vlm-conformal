#!/usr/bin/env python3
"""
ROS-backed prompt recorder for cloud-LLM experimentation.

Records a timestamped prompt .txt every N seconds (default 2s) using real-time:
  - /depth_camera (sensor_msgs/Image)
  - /fmu/out/vehicle_odometry (px4_msgs/VehicleOdometry or nav_msgs/Odometry fallback)

Prompt structure mirrors the active planner:
  1) system prompt loaded from the active 2D prompt file
  2) deterministic natural-language environment summary T(v)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from llm_drone.llm.llm_prompt_common import (
    DEPTH_HFOV_DEG,
    DEPTH_VFOV_DEG,
    DEPTH_MIN_M,
    DEPTH_MAX_M,
    OBS_FIFO_LEN,
    MPC_DEPTH_SAMPLE_COUNT,
    LLM_PLANNER_PROMPT_FILENAME,
    build_environment_vector,
    compose_user_prompt,
    load_system_prompt_from_path,
    resolve_prompt_file,
    serialize_prompt_bundle,
    translate_vector_to_nlp,
)

try:
    from px4_msgs.msg import VehicleOdometry
    HAS_PX4_MSGS = True
except ImportError:
    HAS_PX4_MSGS = False

DEFAULT_PROMPT_FILE = resolve_prompt_file(LLM_PLANNER_PROMPT_FILENAME)
DEFAULT_OUTPUT_DIR = (Path(__file__).resolve().parent / "../config/test2_prompt_samples").resolve()
LLM_WAYPOINT_COUNT = 5


class LocalObstacleMap:
    """FIFO local obstacle cache matching MPC local planner semantics."""

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
    """Depth image to obstacle points using the same back-projection/transforms as llm_planner.py."""

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
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=float,
        )
        self.translation_body_camera_m = np.array([0.13233, 0.0, 0.26078], dtype=float)

    def depth_to_body_frame(self, depth_img: np.ndarray) -> np.ndarray:
        if depth_img is None or depth_img.ndim != 2:
            return np.zeros((0, 3), dtype=float)
        h, w = depth_img.shape
        fy = (h / 2.0) / np.tan(self.vfov / 2.0)
        fx = (w / 2.0) / np.tan(self.hfov / 2.0)
        cx, cy = w / 2.0, h / 2.0

        total = h * w
        if total <= 0:
            return np.zeros((0, 3), dtype=float)

        idx = np.random.choice(total, min(self.n_sample, total), replace=False)
        rows, cols = np.unravel_index(idx, (h, w))
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

    @staticmethod
    def body_to_spatial(pts_body: np.ndarray, rotation_ned_body: np.ndarray, pos_ned: np.ndarray) -> np.ndarray:
        if pts_body is None or pts_body.shape[0] == 0:
            return np.zeros((0, 3), dtype=float)
        return (rotation_ned_body @ pts_body.T).T + np.asarray(pos_ned, dtype=float)[None, :]

class LivePromptRecorder(Node):
    def __init__(
        self,
        system_prompt: str,
        out_dir: Path,
        interval_sec: float = 2.0,
        goal_xyz: tuple[float, float, float] = (35.0, 3.0, 2.5),
        goal_frame: str = "ned",
    ):
        super().__init__("live_prompt_recorder")
        self.system_prompt = system_prompt
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.interval_sec = max(0.2, float(interval_sec))
        self.goal = self.convert_goal_to_ned(np.array(goal_xyz, dtype=float), goal_frame)

        self.bridge = CvBridge()
        self.depth_to_obstacles = DepthToObstacles()
        self.local_obstacle_map = LocalObstacleMap(OBS_FIFO_LEN)

        self.depth_image: np.ndarray | None = None
        self.current_position = np.zeros(3, dtype=float)
        self.current_velocity = np.zeros(3, dtype=float)
        self.rotation_ned_body = np.eye(3, dtype=float)
        self.latest_depth_obstacles_ned = np.zeros((0, 3), dtype=float)
        self.odom_received = False
        self.depth_received = False
        self.write_count = 0

        if HAS_PX4_MSGS:
            self.create_subscription(
                VehicleOdometry,
                "/fmu/out/vehicle_odometry",
                self.vehicle_odometry_callback,
                qos_profile_sensor_data,
            )
        else:
            self.create_subscription(
                Odometry,
                "/fmu/out/vehicle_odometry",
                self.odometry_callback,
                qos_profile_sensor_data,
            )
        self.create_subscription(
            Image,
            "/depth_camera",
            self.depth_image_callback,
            qos_profile_sensor_data,
        )

        self.create_timer(self.interval_sec, self.record_prompt_tick)
        self.get_logger().info(
            f"Live prompt recorder started | interval={self.interval_sec:.1f}s | out_dir={self.out_dir}"
        )
        self.get_logger().info("Topics: /depth_camera and /fmu/out/vehicle_odometry")

    @staticmethod
    def quaternion_to_rotation_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n < 1e-9:
            return np.eye(3, dtype=float)
        w, x, y, z = w / n, x / n, y / n, z / n
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )

    @staticmethod
    def convert_goal_to_ned(goal: np.ndarray, goal_frame: str) -> np.ndarray:
        g = np.array(goal, dtype=float)
        gf = str(goal_frame).strip().lower()
        if gf in ("gazebo", "gazebo_enu", "enu", "map"):
            return np.array([g[1], g[0], -g[2]], dtype=float)
        return g

    def odometry_callback(self, msg: Odometry) -> None:
        self.current_position = np.array(
            [msg.pose.pose.position.y, msg.pose.pose.position.x, -msg.pose.pose.position.z],
            dtype=float,
        )
        self.current_velocity = np.array(
            [msg.twist.twist.linear.y, msg.twist.twist.linear.x, -msg.twist.twist.linear.z],
            dtype=float,
        )
        q = msg.pose.pose.orientation
        rotation_enu_body = self.quaternion_to_rotation_matrix(q.w, q.x, q.y, q.z)
        rotation_enu_to_ned = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=float,
        )
        self.rotation_ned_body = rotation_enu_to_ned @ rotation_enu_body
        self.odom_received = True

    def vehicle_odometry_callback(self, msg: "VehicleOdometry") -> None:
        self.current_position = np.array([float(msg.position[0]), float(msg.position[1]), float(msg.position[2])], dtype=float)
        self.current_velocity = np.array([float(msg.velocity[0]), float(msg.velocity[1]), float(msg.velocity[2])], dtype=float)
        self.rotation_ned_body = self.quaternion_to_rotation_matrix(
            float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3])
        )
        self.odom_received = True

    def depth_image_callback(self, msg: Image) -> None:
        try:
            if msg.encoding in ("16UC1", "mono16"):
                depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                self.depth_image = depth_mm.astype(np.float32) / 1000.0
            else:
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().warn(f"Depth image conversion error: {e}")
            return

        pts_body = self.depth_to_obstacles.depth_to_body_frame(self.depth_image)
        pts_ned = self.depth_to_obstacles.body_to_spatial(pts_body, self.rotation_ned_body, self.current_position)
        self.latest_depth_obstacles_ned = pts_ned
        self.local_obstacle_map.update(pts_ned)
        self.depth_received = True

    def build_environment_vector(self) -> dict | None:
        return build_environment_vector(
            position_ned=self.current_position,
            velocity_ned=self.current_velocity,
            goal_ned=self.goal,
            depth_image=self.depth_image,
            latest_depth_obstacles_ned=self.latest_depth_obstacles_ned,
            local_obstacle_snapshot=self.local_obstacle_map.snapshot(),
            depth_sample_count=self.depth_to_obstacles.n_sample,
            waypoint_count=LLM_WAYPOINT_COUNT,
        )

    def record_prompt_tick(self) -> None:
        if not self.depth_received:
            self.get_logger().warn("Waiting for /depth_camera...", throttle_duration_sec=5.0)
            return
        if not self.odom_received:
            self.get_logger().warn("Waiting for /fmu/out/vehicle_odometry...", throttle_duration_sec=5.0)
            return

        env_vector = self.build_environment_vector()
        if env_vector is None:
            self.get_logger().warn("Environment vector unavailable", throttle_duration_sec=5.0)
            return

        env_text = translate_vector_to_nlp(env_vector)
        user_prompt = compose_user_prompt(env_text)

        now = datetime.now(timezone.utc)
        ts_utc = now.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        stamp_for_file = now.strftime("%Y%m%dT%H%M%S_%fZ")
        out_path = self.out_dir / f"{stamp_for_file}.txt"

        text = (
            f"timestamp_utc: {ts_utc}\n\n"
            f"{serialize_prompt_bundle(self.system_prompt, user_prompt)}"
        )
        out_path.write_text(text)
        self.write_count += 1
        self.get_logger().info(f"[{self.write_count}] wrote prompt {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record live timestamped prompt .txt files from ROS topics.")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE, help="Path to system prompt file")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for timestamped .txt prompts")
    parser.add_argument("--interval-sec", type=float, default=2.0, help="Prompt recording interval in seconds")
    parser.add_argument("--goal-x", type=float, default=35.0)
    parser.add_argument("--goal-y", type=float, default=3.0)
    parser.add_argument("--goal-z", type=float, default=2.5)
    parser.add_argument("--goal-frame", type=str, default="ned", help="ned or gazebo/map/enu")
    args = parser.parse_args()

    system_prompt = load_system_prompt_from_path(args.prompt_file)
    rclpy.init()

    node = LivePromptRecorder(
        system_prompt=system_prompt,
        out_dir=args.out_dir,
        interval_sec=args.interval_sec,
        goal_xyz=(args.goal_x, args.goal_y, args.goal_z),
        goal_frame=args.goal_frame,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Stopped. Total prompts written: {node.write_count}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
