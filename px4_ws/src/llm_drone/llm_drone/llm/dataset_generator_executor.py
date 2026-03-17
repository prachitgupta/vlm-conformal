#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as RosPath
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from llm_drone.llm.dataset_pipeline_common import (
    DatasetCsvWriter,
    GROUND_TRUTH_LABEL_WINDOW_TOPIC,
    LABEL_MODE_OFFLINE_GROUND_TRUTH,
    LABEL_WAYPOINT_COUNT,
    PROMPT_MODE_ENV_NL_ONLY,
    SceneSynchronizer,
    build_label_response,
    build_prompt_from_scene,
    quaternion_to_rotation_matrix,
)
from llm_drone.llm.llm_prompt_common import DATASET_PROMPT_FILENAME, MPC_DEPTH_SAMPLE_COUNT, load_system_prompt
from llm_drone.llm.offline_ground_truth_support import (
    ScenarioManifest,
    extract_future_waypoints,
    gazebo_enu_to_ned,
    project_pose_to_reference_path,
    trajectory_artifact_goal_z_ned,
    trajectory_artifact_path_ned,
)

DEFAULT_OUTPUT_CSV = str((Path(__file__).resolve().parents[2] / 'dataset' / 'offline_ground_truth_dataset.csv').resolve())

try:
    from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleOdometry
    HAS_PX4_MSGS = True
except ImportError:
    HAS_PX4_MSGS = False


class DatasetGeneratorExecuter(Node):
    def __init__(self):
        super().__init__('dataset_generator_executor')
        if not HAS_PX4_MSGS:
            raise RuntimeError('px4_msgs is required for dataset_generator_executor')

        self.declare_parameter('trajectory_json', '')
        self.declare_parameter('output_csv', DEFAULT_OUTPUT_CSV)
        self.declare_parameter('prompt_mode', PROMPT_MODE_ENV_NL_ONLY)
        self.declare_parameter('start_tolerance_m', 0.7)
        self.declare_parameter('waypoint_acceptance_radius_m', 0.5)
        self.declare_parameter('playback_speed_scale', 1.0)
        self.declare_parameter('max_odom_sync_dt_s', 0.10)
        self.declare_parameter('odom_buffer_size', 400)
        self.declare_parameter('scene_buffer_size', 400)
        self.declare_parameter('depth_obstacle_samples', MPC_DEPTH_SAMPLE_COUNT)
        self.declare_parameter('eval_dt_s', 0.1)
        self.declare_parameter('eval_v_max_mps', 15.0)
        self.declare_parameter('eval_a_max_mps2', 8.0)
        self.declare_parameter('eval_safety_radius_m', 1.5)
        self.declare_parameter('target_yaw_rad', float('nan'))

        trajectory_json = str(self.get_parameter('trajectory_json').value).strip()
        if not trajectory_json:
            raise ValueError('trajectory_json parameter is required')

        payload = json.loads(Path(trajectory_json).expanduser().resolve().read_text())
        self.scenario = payload.get('scenario', {})
        self.trajectory = payload.get('trajectory', payload)
        self.scenario_manifest = (
            ScenarioManifest.from_dict(self.scenario)
            if isinstance(self.scenario, dict) and self.scenario
            else None
        )
        self.reference_path_ned = trajectory_artifact_path_ned(self.trajectory)
        if self.reference_path_ned.shape[0] < 2:
            raise ValueError('trajectory_json must contain at least 2 reference points')
        self._normalize_offline_reference_to_local_frame()

        self.trajectory_dt_s = float(self.trajectory.get('dt_s', 0.1))
        playback_speed_scale = max(1e-3, float(self.get_parameter('playback_speed_scale').value))
        self.replay_dt_s = self.trajectory_dt_s / playback_speed_scale
        self.goal_z_ned = float(trajectory_artifact_goal_z_ned(self.trajectory))
        self.goal_ned = np.array(self.reference_path_ned[-1], dtype=float, copy=True)
        self.goal_ned[2] = self.goal_z_ned

        self.prompt_mode = str(self.get_parameter('prompt_mode').value).strip().lower()
        self.system_prompt = load_system_prompt(DATASET_PROMPT_FILENAME)
        self.start_tolerance_m = float(self.get_parameter('start_tolerance_m').value)
        self.waypoint_acceptance_radius_m = float(self.get_parameter('waypoint_acceptance_radius_m').value)
        self.target_yaw_rad = float(self.get_parameter('target_yaw_rad').value)
        self.eval_dt_s = float(self.get_parameter('eval_dt_s').value)
        self.eval_v_max_mps = float(self.get_parameter('eval_v_max_mps').value)
        self.eval_a_max_mps2 = float(self.get_parameter('eval_a_max_mps2').value)
        self.eval_safety_radius_m = float(self.get_parameter('eval_safety_radius_m').value)

        self.scene_sync = SceneSynchronizer(
            logger=self.get_logger(),
            now_s_fn=self._now_s,
            odom_buffer_size=int(self.get_parameter('odom_buffer_size').value),
            scene_buffer_size=int(self.get_parameter('scene_buffer_size').value),
            max_odom_sync_dt_s=float(self.get_parameter('max_odom_sync_dt_s').value),
            depth_obstacle_samples=int(self.get_parameter('depth_obstacle_samples').value),
        )
        self.csv_writer = DatasetCsvWriter(str(self.get_parameter('output_csv').value), logger=self.get_logger())

        self.current_position_ned: np.ndarray | None = None
        self.current_velocity_ned = np.zeros(3, dtype=float)
        self.current_yaw_rad = 0.0

        self.command_index = 0
        self.last_projection_index = 0
        self.last_written_anchor_index = -1
        self.replay_started = False
        self.replay_complete = False
        self.rows_written = 0

        # This is the simple "camera mover" state:
        # 1) hold one PX4 target waypoint,
        # 2) wait until the drone reaches it,
        # 3) wait until one synced scene at that waypoint is written,
        # 4) then move to the next waypoint.
        self.waiting_for_waypoint_capture = False
        self.capture_anchor_index = -1
        self.capture_confirmed_for_index = -1

        self._offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self._setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self._label_window_pub = self.create_publisher(RosPath, GROUND_TRUTH_LABEL_WINDOW_TOPIC, 10)

        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.vehicle_odometry_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            '/depth_camera',
            self.depth_image_callback,
            qos_profile_sensor_data,
        )
        self.create_timer(0.05, self._publish_offboard_heartbeat)
        self.create_timer(self.replay_dt_s, self._replay_step)

        self.get_logger().info(
            'Offline replay executor ready. '
            f'trajectory_json={trajectory_json} output_csv={self.csv_writer.output_path} '
            f'replay_dt_s={self.replay_dt_s:.3f} '
            f'waypoint_acceptance_radius_m={self.waypoint_acceptance_radius_m:.2f}'
        )

    def _normalize_offline_reference_to_local_frame(self) -> None:
        if self.scenario_manifest is None:
            return

        start_pose = self.scenario_manifest.start_pose_enu
        start_x_enu = float(start_pose.x)
        start_y_enu = float(start_pose.y)
        start_z_enu = float(start_pose.z)
        start_ned = gazebo_enu_to_ned(start_x_enu, start_y_enu, start_z_enu)
        xy_offset_ned = np.array([float(start_ned[0]), float(start_ned[1])], dtype=float)

        # PX4 local odometry is reported in a local NED frame rooted near the
        # vehicle home/start position, while offline trajectories are generated
        # in absolute world coordinates. Shift the offline assets into that
        # local frame so replay can start for translated worlds as well.
        self.reference_path_ned[:, 0] -= xy_offset_ned[0]
        self.reference_path_ned[:, 1] -= xy_offset_ned[1]

        self.scenario_manifest.start_pose_enu.x = 0.0
        self.scenario_manifest.start_pose_enu.y = 0.0
        self.scenario_manifest.goal_pose_enu.x -= start_x_enu
        self.scenario_manifest.goal_pose_enu.y -= start_y_enu
        if self.scenario_manifest.spawn_pose_enu is not None:
            self.scenario_manifest.spawn_pose_enu.x -= start_x_enu
            self.scenario_manifest.spawn_pose_enu.y -= start_y_enu

        for obstacle in self.scenario_manifest.obstacles:
            obstacle.position.x -= start_x_enu
            obstacle.position.y -= start_y_enu

        self.get_logger().info(
            'Normalized offline reference path to local PX4 frame '
            f'using start offset NED=({xy_offset_ned[0]:.2f}, {xy_offset_ned[1]:.2f})'
        )

    def _now_s(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    @staticmethod
    def _quaternion_to_yaw(w: float, x: float, y: float, z: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def vehicle_odometry_callback(self, msg: VehicleOdometry) -> None:
        position = np.array([float(msg.position[0]), float(msg.position[1]), float(msg.position[2])], dtype=float)
        velocity = np.array([float(msg.velocity[0]), float(msg.velocity[1]), float(msg.velocity[2])], dtype=float)
        rotation_ned_body = quaternion_to_rotation_matrix(
            float(msg.q[0]),
            float(msg.q[1]),
            float(msg.q[2]),
            float(msg.q[3]),
        )
        self.current_position_ned = position
        self.current_velocity_ned = velocity
        self.current_yaw_rad = self._quaternion_to_yaw(
            float(msg.q[0]),
            float(msg.q[1]),
            float(msg.q[2]),
            float(msg.q[3]),
        )
        self.scene_sync.push_odometry(
            position=position,
            velocity=velocity,
            rotation_ned_body=rotation_ned_body,
        )

    def depth_image_callback(self, msg: Image) -> None:
        scene = self.scene_sync.push_depth_msg(msg)
        if scene is None or not self.replay_started:
            return
        if not self.waiting_for_waypoint_capture:
            return
        self._maybe_write_sample(scene)

    def _publish_offboard_heartbeat(self) -> None:
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self._offboard_pub.publish(msg)

    def _find_replay_start_index(self) -> tuple[int, float] | None:
        if self.current_position_ned is None:
            return None
        anchor_index = project_pose_to_reference_path(self.reference_path_ned, self.current_position_ned)
        distance_to_path = float(np.linalg.norm(
            np.asarray(self.current_position_ned[:2], dtype=float) -
            np.asarray(self.reference_path_ned[anchor_index, :2], dtype=float)
        ))
        if distance_to_path > self.start_tolerance_m:
            return None
        return int(anchor_index), float(distance_to_path)

    def _current_target(self) -> np.ndarray:
        target = np.array(self.reference_path_ned[self.command_index], dtype=float, copy=True)
        target[2] = self.goal_z_ned
        return target

    def _publish_target_waypoint(self, target_ned: np.ndarray) -> None:
        # Keep this simple: publish the current reference waypoint directly as a
        # PX4 position target and let PX4's internal position controller attain it.
        sp = TrajectorySetpoint()
        sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        sp.position[0] = float(target_ned[0])
        sp.position[1] = float(target_ned[1])
        sp.position[2] = float(target_ned[2])
        sp.velocity[0] = float('nan')
        sp.velocity[1] = float('nan')
        sp.velocity[2] = float('nan')
        sp.acceleration[0] = float('nan')
        sp.acceleration[1] = float('nan')
        sp.acceleration[2] = float('nan')
        sp.yaw = float(self.target_yaw_rad) if math.isfinite(self.target_yaw_rad) else float('nan')
        sp.yawspeed = float('nan')
        self._setpoint_pub.publish(sp)

    def _target_reached(self, target_ned: np.ndarray) -> bool:
        if self.current_position_ned is None:
            return False
        xy_error = float(np.linalg.norm(
            np.asarray(target_ned[:2], dtype=float) - np.asarray(self.current_position_ned[:2], dtype=float)
        ))
        z_error = abs(float(target_ned[2] - self.current_position_ned[2]))
        return xy_error <= self.waypoint_acceptance_radius_m and z_error <= 0.50

    def _advance_after_capture(self) -> None:
        if self.command_index >= self.reference_path_ned.shape[0] - 1:
            self.replay_complete = True
            self.get_logger().info('Replay reached the end of the reference trajectory')
            return
        self.command_index += 1
        self.waiting_for_waypoint_capture = False
        self.capture_anchor_index = -1

    def _build_label_waypoints(self, anchor_index: int) -> list[list[float]]:
        label_waypoints = extract_future_waypoints(
            self.reference_path_ned,
            anchor_index,
            count=LABEL_WAYPOINT_COUNT,
        )
        if len(label_waypoints) >= LABEL_WAYPOINT_COUNT:
            return label_waypoints

        goal_waypoint = np.array(self.goal_ned, dtype=float)
        while len(label_waypoints) < LABEL_WAYPOINT_COUNT:
            label_waypoints.append([float(goal_waypoint[0]), float(goal_waypoint[1]), float(goal_waypoint[2])])
        return label_waypoints

    def _replay_step(self) -> None:
        if self.replay_complete:
            return

        if not self.replay_started:
            start_info = self._find_replay_start_index()
            if start_info is None:
                return
            start_index, distance_to_path = start_info
            self.replay_started = True
            self.command_index = start_index
            self.last_projection_index = start_index
            self.get_logger().info(
                f'Replay started at path index {start_index} '
                f'(distance_to_path={distance_to_path:.2f} m)'
            )

        target_ned = self._current_target()
        self._publish_target_waypoint(target_ned)

        if self.waiting_for_waypoint_capture:
            return

        if not self._target_reached(target_ned):
            return

        self.waiting_for_waypoint_capture = True
        self.capture_anchor_index = int(self.command_index)
        self.get_logger().info(
            f'Waypoint {self.command_index} reached. Waiting for synced scene capture + label confirmation.'
        )

    def _maybe_write_sample(self, scene: dict) -> None:
        projected_anchor_index = project_pose_to_reference_path(
            self.reference_path_ned,
            np.asarray(scene['position'], dtype=float),
            start_index=self.last_projection_index,
            search_ahead=120,
        )
        if projected_anchor_index < self.capture_anchor_index:
            return

        # Once the vehicle is at waypoint i, write exactly one dataset row tied
        # to that waypoint (or slightly ahead if odometry projects there already).
        anchor_index = max(int(projected_anchor_index), int(self.capture_anchor_index))
        if anchor_index <= self.last_written_anchor_index:
            return

        label_waypoints = self._build_label_waypoints(anchor_index)

        prompt, env_vector, _ = build_prompt_from_scene(
            goal_ned=self.goal_ned,
            scene=scene,
            system_prompt=self.system_prompt,
            prompt_mode=self.prompt_mode,
            waypoint_count=LABEL_WAYPOINT_COUNT,
            depth_sample_count=self.scene_sync.depth_to_obstacles.n_sample,
        )
        if prompt is None or env_vector is None:
            return

        label_response = build_label_response(
            label_waypoints=label_waypoints,
            label_mode=LABEL_MODE_OFFLINE_GROUND_TRUTH,
            scene=scene,
            goal_ned=self.goal_ned,
            evaluation_start_ned=np.asarray(self.reference_path_ned[anchor_index], dtype=float),
            obstacle_points_ned=np.zeros((0, 3), dtype=float),
            obstacle_specs=None if self.scenario_manifest is None else self.scenario_manifest.obstacles,
            eval_dt_s=self.eval_dt_s,
            eval_v_max_mps=self.eval_v_max_mps,
            eval_a_max_mps2=self.eval_a_max_mps2,
            eval_safety_radius_m=self.eval_safety_radius_m,
        )
        stamp_s = float(scene['stamp_s'])
        self.csv_writer.write_sample(
            anchor_stamp_s=stamp_s,
            scene_stamp_s=stamp_s,
            label_mode=LABEL_MODE_OFFLINE_GROUND_TRUTH,
            prompt=prompt,
            label_response=label_response,
            label_waypoints=label_waypoints,
        )
        self.last_projection_index = max(self.last_projection_index, anchor_index)
        self.last_written_anchor_index = anchor_index
        self.capture_confirmed_for_index = int(self.capture_anchor_index)
        self.rows_written = self.csv_writer.rows_written
        self._publish_label_window(label_waypoints)
        self.get_logger().info(
            'Waypoint capture confirmed: '
            f'waypoint_index={self.capture_confirmed_for_index} '
            f'anchor_index={anchor_index} '
            f'label_waypoints={len(label_waypoints)} '
            f'scene_stamp_s={stamp_s:.3f} '
            f'rows_written={self.rows_written}'
        )
        self._advance_after_capture()

    def _publish_label_window(self, label_waypoints: list[list[float]]) -> None:
        stamp = self.get_clock().now().to_msg()
        path = RosPath()
        path.header.stamp = stamp
        path.header.frame_id = 'ned'
        for waypoint in label_waypoints:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = 'ned'
            pose.pose.position.x = float(waypoint[0])
            pose.pose.position.y = float(waypoint[1])
            pose.pose.position.z = float(waypoint[2])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self._label_window_pub.publish(path)

    def destroy_node(self):
        try:
            self.csv_writer.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DatasetGeneratorExecuter()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
