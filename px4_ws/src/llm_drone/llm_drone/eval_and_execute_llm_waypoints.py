#!/usr/bin/env python3
"""
Evaluate a cloud-LLM waypoint response and execute it in Gazebo/PX4 only if it passes.

Workflow:
1) Subscribe to odometry + depth image
2) Build a minimal environment state (current pose, goal, local obstacle snapshot)
3) Parse and evaluate an LLM JSON response containing 5 waypoints
4) If pass, publish the waypoint sequence to PX4 trajectory setpoint topic

This script is intended for homework-style validation of "next 5 waypoints" outputs
before commanding the simulated drone.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

try:
    from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode, VehicleOdometry
    HAS_PX4_MSGS = True
except ImportError:
    HAS_PX4_MSGS = False


DEFAULT_LLM_RESPONSE = (
    '{"waypoints":[{"x":2.7,"y":0.8,"z":-2.2},{"x":3.6,"y":1.7,"z":-2.25},'
    '{"x":4.8,"y":2.2,"z":-2.3},{"x":6.1,"y":2.1,"z":-2.35},{"x":7.4,"y":1.8,'
    '"z":-2.4}],"selected_waypoint_index":0,"reasoning":"Center sector is blocked nearby, '
    'so sidestep through the clearer right sectors with a smooth 5-point arc, then start '
    'merging back toward the goal while gently descending."}'
)

# Keep these aligned with llm_planner.py / prompt_generator_samples.py
DEPTH_HFOV_DEG = 72.995
DEPTH_VFOV_DEG = 58.053
DEPTH_MIN_M = 0.2
DEPTH_MAX_M = 19.1
OBS_FIFO_LEN = 200
DEPTH_SAMPLE_COUNT = 500


@dataclass
class EvalResult:
    passed: bool
    format_pass: bool
    kinematic_pass: bool
    progress_pass: bool
    safety_pass: bool
    metrics: dict[str, Any]
    errors: list[str]
    warnings: list[str]
    parsed: dict[str, Any] | None = None


class LocalObstacleMap:
    """FIFO local obstacle cache in NED frame."""

    def __init__(self, maxlen: int = OBS_FIFO_LEN):
        self._buf: list[np.ndarray] = []
        self._maxlen = int(maxlen)

    def update(self, points_ned: np.ndarray) -> None:
        if points_ned is None or points_ned.size == 0:
            return
        for pt in points_ned:
            self._buf.append(np.array(pt, dtype=float))
        if len(self._buf) > self._maxlen:
            self._buf = self._buf[-self._maxlen:]

    def snapshot(self, max_points: int | None = None) -> np.ndarray:
        if not self._buf:
            return np.zeros((0, 3), dtype=float)
        pts = np.array(self._buf, dtype=float)
        if max_points is not None and pts.shape[0] > max_points:
            pts = pts[-int(max_points):]
        return pts


class DepthToObstacles:
    """Depth image -> obstacle points in body frame / NED frame."""

    def __init__(self, hfov_deg=DEPTH_HFOV_DEG, vfov_deg=DEPTH_VFOV_DEG, n_sample=DEPTH_SAMPLE_COUNT):
        self.hfov = np.deg2rad(float(hfov_deg))
        self.vfov = np.deg2rad(float(vfov_deg))
        self.n_sample = int(n_sample)
        self.rotation_body_camera = np.array(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=float,
        )
        self.translation_body_camera_m = np.array([0.13233, 0.0, 0.26078], dtype=float)

    def depth_to_body_frame(self, depth_img: np.ndarray | None) -> np.ndarray:
        if depth_img is None or depth_img.ndim != 2:
            return np.zeros((0, 3), dtype=float)

        h, w = depth_img.shape
        if h <= 0 or w <= 0:
            return np.zeros((0, 3), dtype=float)

        fy = (h / 2.0) / np.tan(self.vfov / 2.0)
        fx = (w / 2.0) / np.tan(self.hfov / 2.0)
        cx, cy = w / 2.0, h / 2.0

        total = h * w
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
        if pts_body is None or pts_body.size == 0:
            return np.zeros((0, 3), dtype=float)
        return (rotation_ned_body @ pts_body.T).T + np.asarray(pos_ned, dtype=float)[None, :]


def quaternion_to_rotation_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def convert_goal_to_ned(goal_xyz: np.ndarray, goal_frame: str) -> np.ndarray:
    g = np.array(goal_xyz, dtype=float)
    gf = str(goal_frame).strip().lower()
    if gf in ("gazebo", "gazebo_enu", "enu", "map"):
        return np.array([g[1], g[0], -g[2]], dtype=float)
    return g


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _seg_point_distance(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = np.asarray(p, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def parse_llm_response(text: str) -> dict[str, Any]:
    s = text.find("{")
    e = text.rfind("}") + 1
    if s < 0 or e <= s:
        raise ValueError("No JSON object found in response")
    data = json.loads(text[s:e])
    if "waypoints" not in data:
        raise ValueError("Missing 'waypoints'")

    waypoints_raw = data["waypoints"]
    if not isinstance(waypoints_raw, list) or len(waypoints_raw) != 5:
        raise ValueError("'waypoints' must be a list of exactly 5 items")

    selected_idx = int(data.get("selected_waypoint_index", 0))
    if selected_idx < 0 or selected_idx >= 5:
        raise ValueError("'selected_waypoint_index' out of range")

    waypoints = []
    for i, wp in enumerate(waypoints_raw):
        x = float(wp["x"])
        y = float(wp["y"])
        z = float(wp["z"])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise ValueError(f"Waypoint {i} contains non-finite values")
        waypoints.append(np.array([x, y, z], dtype=float))

    return {
        "waypoints": waypoints,
        "selected_waypoint_index": selected_idx,
        "reasoning": data.get("reasoning", ""),
        "raw": data,
    }


def evaluate_response(
    current_position_ned: np.ndarray,
    goal_ned: np.ndarray,
    llm_response_text: str,
    obstacle_points_ned: np.ndarray | None,
    dt: float,
    v_max: float,
    a_max: float,
    safety_radius: float,
) -> EvalResult:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        parsed = parse_llm_response(llm_response_text)
    except Exception as e:
        return EvalResult(
            passed=False,
            format_pass=False,
            kinematic_pass=False,
            progress_pass=False,
            safety_pass=False,
            metrics={},
            errors=[f"parse_error: {e}"],
            warnings=[],
            parsed=None,
        )

    p0 = np.asarray(current_position_ned, dtype=float)
    goal = np.asarray(goal_ned, dtype=float)
    waypoints = parsed["waypoints"]
    idx = int(parsed["selected_waypoint_index"])
    p_sel = waypoints[idx]

    seq = [p0] + waypoints
    seg_lengths = [_dist(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
    speeds = [d / max(dt, 1e-6) for d in seg_lengths]
    accels = [
        abs(speeds[i + 1] - speeds[i]) / max(dt, 1e-6) for i in range(len(speeds) - 1)
    ]
    kinematic_pass = (max(speeds) if speeds else 0.0) <= v_max and (max(accels) if accels else 0.0) <= a_max
    if not kinematic_pass:
        warnings.append(
            f"Kinematic violation: max_speed={max(speeds) if speeds else 0.0:.2f} m/s (limit {v_max:.2f}), "
            f"max_accel={max(accels) if accels else 0.0:.2f} m/s^2 (limit {a_max:.2f})"
        )

    d0 = _dist(p0, goal)
    d_sel = _dist(p_sel, goal)
    d5 = _dist(waypoints[-1], goal)
    progress_selected = d0 - d_sel
    progress_final = d0 - d5
    progress_pass = (progress_selected > 0.0) and (progress_final > 0.0)
    if not progress_pass:
        warnings.append(
            f"Goal progress violation: selected_progress={progress_selected:.2f} m, "
            f"final_progress={progress_final:.2f} m"
        )

    min_clearance = None
    clearance_violations = 0
    safety_pass = True
    if obstacle_points_ned is not None and obstacle_points_ned.size > 0:
        min_clearance = float("inf")
        for i in range(len(seq) - 1):
            a = seq[i]
            b = seq[i + 1]
            for obs in obstacle_points_ned:
                d = _seg_point_distance(a, b, obs)
                if d < min_clearance:
                    min_clearance = d
                if d < safety_radius:
                    clearance_violations += 1
        safety_pass = bool(min_clearance >= safety_radius)
        if not safety_pass:
            warnings.append(
                f"Safety violation: min_clearance={min_clearance:.2f} m < safety_radius={safety_radius:.2f} m "
                f"(clearance_violations={clearance_violations})"
            )
    else:
        warnings.append("Safety check degraded: no obstacle snapshot available, geometric clearance not validated")

    turn_sum = 0.0
    for i in range(1, len(seq) - 1):
        u = seq[i] - seq[i - 1]
        v = seq[i + 1] - seq[i]
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu > 1e-8 and nv > 1e-8:
            c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
            turn_sum += math.acos(c)

    metrics = {
        "selected_waypoint_index": idx,
        "distance_to_goal_before_m": d0,
        "distance_to_goal_selected_m": d_sel,
        "distance_to_goal_final_m": d5,
        "progress_selected_m": progress_selected,
        "progress_final_m": progress_final,
        "segment_lengths_m": seg_lengths,
        "speeds_mps": speeds,
        "accels_mps2": accels,
        "max_speed_mps": max(speeds) if speeds else 0.0,
        "max_accel_mps2": max(accels) if accels else 0.0,
        "min_clearance_m": min_clearance,
        "clearance_violations": clearance_violations,
        "turn_angle_sum_rad": turn_sum,
        "reasoning": parsed.get("reasoning", ""),
    }

    passed = bool(kinematic_pass and progress_pass and safety_pass)
    return EvalResult(
        passed=passed,
        format_pass=True,
        kinematic_pass=bool(kinematic_pass),
        progress_pass=bool(progress_pass),
        safety_pass=bool(safety_pass),
        metrics=metrics,
        errors=errors,
        warnings=warnings,
        parsed=parsed,
    )


class EvalAndExecuteNode(Node):
    def __init__(self):
        super().__init__("eval_and_execute_llm_waypoints")

        self.declare_parameter("llm_response", DEFAULT_LLM_RESPONSE)
        # Match mpc_local_planner.py default target exactly:
        # self._goal_ned = np.array([20.0, 0.0, -5.0])
        self.declare_parameter("goal_x", 20.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_z", -5.0)
        self.declare_parameter("goal_frame", "ned")
        self.declare_parameter("eval_dt", 1.0)
        self.declare_parameter("eval_v_max", 3.0)
        self.declare_parameter("eval_a_max", 4.0)
        self.declare_parameter("eval_safety_radius", 1.0)
        self.declare_parameter("acceptance_radius_m", 0.6)
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("execute_selected_only", False)
        self.declare_parameter("max_obstacle_points_eval", 200)

        goal_input = np.array([
            float(self.get_parameter("goal_x").value),
            float(self.get_parameter("goal_y").value),
            float(self.get_parameter("goal_z").value),
        ], dtype=float)
        self.goal_ned = convert_goal_to_ned(goal_input, str(self.get_parameter("goal_frame").value))

        self.bridge = CvBridge()
        self.depth_to_obstacles = DepthToObstacles()
        self.local_obstacle_map = LocalObstacleMap()

        self.current_position = np.zeros(3, dtype=float)
        self.current_velocity = np.zeros(3, dtype=float)
        self.rotation_ned_body = np.eye(3, dtype=float)
        self.depth_image: np.ndarray | None = None
        self.odom_received = False
        self.depth_received = False

        self.evaluated = False
        self.execution_armed = False
        self.exec_waypoints: list[np.ndarray] = []
        self.exec_index = 0

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

        self.create_subscription(Image, "/depth_camera", self.depth_callback, qos_profile_sensor_data)
        self.llm_traj_pub = self.create_publisher(PoseStamped, "/llm/trajectory", 10)

        if HAS_PX4_MSGS:
            self.traj_setpoint_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10)
            self.offboard_mode_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
            self.cmd_vel_pub = None
            self.offboard_timer = self.create_timer(0.1, self.publish_offboard_heartbeat)
        else:
            self.traj_setpoint_pub = None
            self.offboard_mode_pub = None
            self.cmd_vel_pub = self.create_publisher(TwistStamped, "/fmu/in/setpoint_velocity", 10)

        self.eval_timer = self.create_timer(0.5, self.try_evaluate_once)
        period = 1.0 / max(float(self.get_parameter("publish_rate_hz").value), 0.5)
        self.exec_timer = self.create_timer(period, self.execution_loop)

        self.get_logger().info("Eval-and-execute node started.")
        self.get_logger().info(f"Goal NED: {self.goal_ned.tolist()}")
        self.get_logger().info(f"px4_msgs available: {HAS_PX4_MSGS}")

    def odometry_callback(self, msg: Odometry) -> None:
        self.current_position = np.array([
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            float(msg.pose.pose.position.z),
        ], dtype=float)
        self.current_velocity = np.array([
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.linear.y),
            float(msg.twist.twist.linear.z),
        ], dtype=float)
        q = msg.pose.pose.orientation
        self.rotation_ned_body = quaternion_to_rotation_matrix(float(q.w), float(q.x), float(q.y), float(q.z))
        self.odom_received = True

    def vehicle_odometry_callback(self, msg: VehicleOdometry) -> None:
        self.current_position = np.array([float(msg.position[0]), float(msg.position[1]), float(msg.position[2])], dtype=float)
        self.current_velocity = np.array([float(msg.velocity[0]), float(msg.velocity[1]), float(msg.velocity[2])], dtype=float)
        self.rotation_ned_body = quaternion_to_rotation_matrix(float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3]))
        self.odom_received = True

    def depth_callback(self, msg: Image) -> None:
        try:
            if msg.encoding in ("16UC1", "mono16"):
                depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                self.depth_image = depth_mm.astype(np.float32) / 1000.0
            else:
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f"Depth image conversion error: {e}")
            return

        pts_body = self.depth_to_obstacles.depth_to_body_frame(self.depth_image)
        pts_ned = self.depth_to_obstacles.body_to_spatial(pts_body, self.rotation_ned_body, self.current_position)
        self.local_obstacle_map.update(pts_ned)
        self.depth_received = True

    def try_evaluate_once(self) -> None:
        if self.evaluated:
            return
        if not (self.odom_received and self.depth_received):
            self.get_logger().info("Waiting for odometry + depth before evaluation...", throttle_duration_sec=2.0)
            return

        llm_response_text = str(self.get_parameter("llm_response").value)
        obstacle_snapshot = self.local_obstacle_map.snapshot(
            max_points=int(self.get_parameter("max_obstacle_points_eval").value)
        )

        res = evaluate_response(
            current_position_ned=self.current_position,
            goal_ned=self.goal_ned,
            llm_response_text=llm_response_text,
            obstacle_points_ned=obstacle_snapshot,
            dt=float(self.get_parameter("eval_dt").value),
            v_max=float(self.get_parameter("eval_v_max").value),
            a_max=float(self.get_parameter("eval_a_max").value),
            safety_radius=float(self.get_parameter("eval_safety_radius").value),
        )
        self.evaluated = True

        self.get_logger().info(f"EVAL PASS: {res.passed}")
        self.get_logger().info(
            f"Checks: format={res.format_pass}, progress={res.progress_pass}, "
            f"kinematic={res.kinematic_pass}, safety={res.safety_pass}"
        )
        if res.errors:
            for err in res.errors:
                self.get_logger().error(err)
        if res.warnings:
            for warn in res.warnings:
                self.get_logger().warn(warn)
        self.get_logger().info(f"Metrics: {json.dumps(res.metrics, indent=2)}")

        if res.parsed is None:
            self.get_logger().warn("Evaluation parse failed. Waypoints will NOT be published to PX4.")
            return

        parsed = res.parsed
        waypoints = parsed["waypoints"]
        selected_idx = int(parsed["selected_waypoint_index"])
        selected_only = bool(self.get_parameter("execute_selected_only").value)
        self.exec_waypoints = [waypoints[selected_idx]] if selected_only else list(waypoints)
        self.exec_index = 0
        self.execution_armed = True
        mode = "selected waypoint only" if selected_only else "all 5 waypoints sequentially"
        if res.passed:
            self.get_logger().warn(f"Evaluation passed. Arming execution ({mode}).")
        else:
            self.get_logger().warn(
                f"Evaluation flagged issues, but executing anyway ({mode}). Review warnings above."
            )

    def execution_loop(self) -> None:
        if not self.execution_armed:
            return
        if self.exec_index >= len(self.exec_waypoints):
            self.get_logger().info("Finished publishing all waypoints.")
            self.execution_armed = False
            return

        target = self.exec_waypoints[self.exec_index]
        self.publish_llm_pose(target)

        if HAS_PX4_MSGS:
            self.publish_trajectory_setpoint(target)
        else:
            self.publish_velocity_fallback(target)

        dist = _dist(self.current_position, target)
        if dist <= float(self.get_parameter("acceptance_radius_m").value):
            self.get_logger().info(f"Reached waypoint {self.exec_index + 1}/{len(self.exec_waypoints)} (dist={dist:.2f} m)")
            self.exec_index += 1

    def publish_llm_pose(self, target: np.ndarray) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(target[0])
        msg.pose.position.y = float(target[1])
        msg.pose.position.z = float(target[2])
        self.llm_traj_pub.publish(msg)

    def publish_trajectory_setpoint(self, target: np.ndarray) -> None:
        if self.traj_setpoint_pub is None:
            return
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = [float(target[0]), float(target[1]), float(target[2])]
        msg.yaw = 0.0
        self.traj_setpoint_pub.publish(msg)

    def publish_offboard_heartbeat(self) -> None:
        if self.offboard_mode_pub is None:
            return
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        self.offboard_mode_pub.publish(msg)

    def publish_velocity_fallback(self, target: np.ndarray) -> None:
        if self.cmd_vel_pub is None:
            return
        delta = target - self.current_position
        d = float(np.linalg.norm(delta))
        v = np.zeros(3, dtype=float)
        if d > 1e-6:
            vmax = min(1.5, d)
            v = delta / d * vmax
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(v[0])
        msg.twist.linear.y = float(v[1])
        msg.twist.linear.z = float(v[2])
        self.cmd_vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = EvalAndExecuteNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
