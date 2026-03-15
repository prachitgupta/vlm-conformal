#!/usr/bin/env python3

from __future__ import annotations

import json
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
    PROMPT_MODE_FULL_BUNDLE,
    SceneSynchronizer,
    build_label_response,
    build_prompt_from_scene,
    quaternion_to_rotation_matrix,
)
from llm_drone.llm.llm_prompt_common import DATASET_PROMPT_FILENAME, MPC_DEPTH_SAMPLE_COUNT, load_system_prompt
from llm_drone.llm.offline_ground_truth_support import (
    extract_future_waypoints,
    project_pose_to_reference_path,
    trajectory_artifact_goal_z_ned,
    trajectory_artifact_path_ned,
)

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
        self.declare_parameter('output_csv', '../dataset/offline_ground_truth_dataset.csv')
        self.declare_parameter('prompt_mode', PROMPT_MODE_FULL_BUNDLE)
        self.declare_parameter('start_tolerance_m', 0.50)
        self.declare_parameter('playback_speed_scale', 1.0)
        self.declare_parameter('max_odom_sync_dt_s', 0.10)
        self.declare_parameter('odom_buffer_size', 400)
        self.declare_parameter('scene_buffer_size', 400)
        self.declare_parameter('depth_obstacle_samples', MPC_DEPTH_SAMPLE_COUNT)
        self.declare_parameter('eval_dt_s', 0.1)
        self.declare_parameter('eval_v_max_mps', 15.0)
        self.declare_parameter('eval_a_max_mps2', 10.0)
        self.declare_parameter('eval_safety_radius_m', 0.5)
        self.declare_parameter('z_hold_kp', 1.5)
        self.declare_parameter('z_hold_max_mps', 2.0)

        trajectory_json = str(self.get_parameter('trajectory_json').value).strip()
        if not trajectory_json:
            raise ValueError('trajectory_json parameter is required')
        payload = json.loads(Path(trajectory_json).expanduser().resolve().read_text())
        self.scenario = payload.get('scenario', {})
        self.trajectory = payload.get('trajectory', payload)
        self.reference_path_ned = trajectory_artifact_path_ned(self.trajectory)
        if self.reference_path_ned.shape[0] < LABEL_WAYPOINT_COUNT + 1:
            raise ValueError('trajectory_json must contain at least 6 reference points')

        self.trajectory_dt_s = float(self.trajectory.get('dt_s', 0.1))
        self.goal_z_ned = float(trajectory_artifact_goal_z_ned(self.trajectory))
        self.goal_ned = np.array(self.reference_path_ned[-1], dtype=float, copy=True)
        self.goal_ned[2] = self.goal_z_ned
        self.prompt_mode = str(self.get_parameter('prompt_mode').value).strip().lower()
        self.system_prompt = load_system_prompt(DATASET_PROMPT_FILENAME)
        self.eval_dt_s = float(self.get_parameter('eval_dt_s').value)
        self.eval_v_max_mps = float(self.get_parameter('eval_v_max_mps').value)
        self.eval_a_max_mps2 = float(self.get_parameter('eval_a_max_mps2').value)
        self.eval_safety_radius_m = float(self.get_parameter('eval_safety_radius_m').value)
        self.start_tolerance_m = float(self.get_parameter('start_tolerance_m').value)
        self.z_hold_kp = float(self.get_parameter('z_hold_kp').value)
        self.z_hold_max_mps = float(self.get_parameter('z_hold_max_mps').value)
        playback_speed_scale = max(1e-3, float(self.get_parameter('playback_speed_scale').value))

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
        self.command_index = 0
        self.last_projection_index = 0
        self.last_written_anchor_index = -1
        self.replay_started = False
        self.replay_complete = False
        self.rows_written = 0

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
        self.create_timer(self.trajectory_dt_s / playback_speed_scale, self._replay_step)

        self.get_logger().info(
            f'Offline replay executor ready. trajectory_json={trajectory_json} output_csv={self.csv_writer.output_path}'
        )

    def _now_s(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

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
        self.scene_sync.push_odometry(
            position=position,
            velocity=velocity,
            rotation_ned_body=rotation_ned_body,
        )

    def depth_image_callback(self, msg: Image) -> None:
        scene = self.scene_sync.push_depth_msg(msg)
        if scene is None or not self.replay_started:
            return
        self._maybe_write_sample(scene)

    def _publish_offboard_heartbeat(self) -> None:
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = True
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self._offboard_pub.publish(msg)

    def _reference_velocity(self, index: int) -> np.ndarray:
        idx = min(max(int(index), 0), self.reference_path_ned.shape[0] - 1)
        if idx >= self.reference_path_ned.shape[0] - 1:
            if self.reference_path_ned.shape[0] < 2:
                return np.zeros(3, dtype=float)
            delta = self.reference_path_ned[-1] - self.reference_path_ned[-2]
            velocity = np.asarray(delta, dtype=float) / max(self.trajectory_dt_s, 1e-6)
            velocity[2] = 0.0
            return velocity
        delta = self.reference_path_ned[idx + 1] - self.reference_path_ned[idx]
        velocity = np.asarray(delta, dtype=float) / max(self.trajectory_dt_s, 1e-6)
        velocity[2] = 0.0
        return velocity

    def _z_hold_velocity(self) -> float:
        if self.current_position_ned is None:
            return 0.0
        z_error = float(self.goal_z_ned - self.current_position_ned[2])
        return float(np.clip(self.z_hold_kp * z_error, -self.z_hold_max_mps, self.z_hold_max_mps))

    def _publish_setpoint(self, position_ned: np.ndarray, velocity_ned: np.ndarray) -> None:
        sp = TrajectorySetpoint()
        sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        sp.position[0] = float(position_ned[0])
        sp.position[1] = float(position_ned[1])
        sp.position[2] = float(self.goal_z_ned)
        sp.velocity[0] = float(velocity_ned[0])
        sp.velocity[1] = float(velocity_ned[1])
        sp.velocity[2] = self._z_hold_velocity()
        sp.acceleration[0] = float('nan')
        sp.acceleration[1] = float('nan')
        sp.acceleration[2] = float('nan')
        sp.yaw = float('nan')
        sp.yawspeed = float('nan')
        self._setpoint_pub.publish(sp)

    def _replay_step(self) -> None:
        target = np.array(self.reference_path_ned[self.command_index], dtype=float, copy=True)
        velocity = self._reference_velocity(self.command_index)
        self._publish_setpoint(target, velocity)

        if self.current_position_ned is None:
            return
        if not self.replay_started:
            distance_to_start = float(np.linalg.norm(self.current_position_ned - self.reference_path_ned[0]))
            if distance_to_start <= self.start_tolerance_m:
                self.replay_started = True
                self.get_logger().info(
                    f'Replay started at path index 0 (distance_to_start={distance_to_start:.2f} m)'
                )
            return

        if self.command_index < self.reference_path_ned.shape[0] - 1:
            self.command_index += 1
            return

        if not self.replay_complete:
            self.replay_complete = True
            self.get_logger().info('Replay reached the end of the reference trajectory')

    def _maybe_write_sample(self, scene: dict) -> None:
        anchor_index = project_pose_to_reference_path(
            self.reference_path_ned,
            np.asarray(scene['position'], dtype=float),
            start_index=self.last_projection_index,
            search_ahead=120,
        )
        self.last_projection_index = max(self.last_projection_index, anchor_index)
        if anchor_index <= self.last_written_anchor_index:
            return

        label_waypoints = extract_future_waypoints(
            self.reference_path_ned,
            anchor_index,
            count=LABEL_WAYPOINT_COUNT,
        )
        if len(label_waypoints) < LABEL_WAYPOINT_COUNT:
            return

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
        self.last_written_anchor_index = anchor_index
        self.rows_written = self.csv_writer.rows_written
        self._publish_label_window(label_waypoints)
        self.get_logger().info(
            f'Wrote offline sample #{self.rows_written} | anchor_index={anchor_index} | label_mode={LABEL_MODE_OFFLINE_GROUND_TRUTH}'
        )

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
