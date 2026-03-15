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

from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as RosPath
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from llm_drone.llm.dataset_pipeline_common import (
    DatasetCsvWriter,
    EXECUTED_COMMITTED_WAYPOINT_TOPIC,
    FRESH_COMMITTED_WAYPOINT_TOPIC,
    LABEL_MODE_EXECUTED,
    LABEL_MODE_LOCAL_PREDICTION,
    LABEL_WAYPOINT_COUNT,
    LOCAL_PREDICTION_SEQUENCE_TOPIC,
    PROMPT_MODE_ENV_NL_ONLY,
    PROMPT_MODE_FULL_BUNDLE,
    PROMPT_MODE_VALUES,
    SceneSynchronizer,
    build_label_response,
    build_prompt_from_scene,
    convert_goal_to_ned,
    quaternion_to_rotation_matrix,
)
from llm_drone.llm.llm_prompt_common import DATASET_PROMPT_FILENAME, MPC_DEPTH_SAMPLE_COUNT, load_system_prompt

try:
    from px4_msgs.msg import VehicleOdometry
    HAS_PX4_MSGS = True
except ImportError:
    HAS_PX4_MSGS = False


class SyncDatasetGenerator(Node):
    def __init__(self):
        super().__init__('sync_dataset_generator')

        self.declare_parameter('goal_x', 35.0)
        self.declare_parameter('goal_y', 3.0)
        self.declare_parameter('goal_z', 2.5)
        self.declare_parameter('goal_frame', 'ned')
        self.declare_parameter('output_csv', '../dataset/dataset.csv')
        self.declare_parameter('label_mode', LABEL_MODE_LOCAL_PREDICTION)
        self.declare_parameter('prompt_mode', PROMPT_MODE_FULL_BUNDLE)
        self.declare_parameter('max_odom_sync_dt_s', 0.10)
        self.declare_parameter('max_label_sync_dt_s', 0.20)
        self.declare_parameter('odom_buffer_size', 400)
        self.declare_parameter('scene_buffer_size', 400)
        self.declare_parameter('waypoint_buffer_size', 400)
        self.declare_parameter('depth_obstacle_samples', MPC_DEPTH_SAMPLE_COUNT)
        self.declare_parameter('eval_dt_s', 0.1)
        self.declare_parameter('eval_v_max_mps', 15.0)
        self.declare_parameter('eval_a_max_mps2', 10.0)
        self.declare_parameter('eval_safety_radius_m', 0.5)

        self.goal = convert_goal_to_ned(
            np.array([
                self.get_parameter('goal_x').value,
                self.get_parameter('goal_y').value,
                self.get_parameter('goal_z').value,
            ], dtype=float),
            str(self.get_parameter('goal_frame').value),
        )
        self.output_csv = str(self.get_parameter('output_csv').value)
        self.label_mode = str(self.get_parameter('label_mode').value).strip().lower()
        self.prompt_mode = str(self.get_parameter('prompt_mode').value).strip().lower()
        self.max_label_sync_dt_s = float(self.get_parameter('max_label_sync_dt_s').value)
        self.eval_dt_s = float(self.get_parameter('eval_dt_s').value)
        self.eval_v_max_mps = float(self.get_parameter('eval_v_max_mps').value)
        self.eval_a_max_mps2 = float(self.get_parameter('eval_a_max_mps2').value)
        self.eval_safety_radius_m = float(self.get_parameter('eval_safety_radius_m').value)

        if self.label_mode not in (LABEL_MODE_LOCAL_PREDICTION, LABEL_MODE_EXECUTED):
            self.get_logger().warn(
                f'Unknown label_mode={self.label_mode!r}; falling back to {LABEL_MODE_EXECUTED}.'
            )
            self.label_mode = LABEL_MODE_EXECUTED
        if self.prompt_mode not in PROMPT_MODE_VALUES:
            self.get_logger().warn(
                f'Unknown prompt_mode={self.prompt_mode!r}; falling back to {PROMPT_MODE_FULL_BUNDLE}.'
            )
            self.prompt_mode = PROMPT_MODE_FULL_BUNDLE

        self.system_prompt = load_system_prompt(DATASET_PROMPT_FILENAME)
        self.scene_sync = SceneSynchronizer(
            logger=self.get_logger(),
            now_s_fn=self._now_s,
            odom_buffer_size=int(self.get_parameter('odom_buffer_size').value),
            scene_buffer_size=int(self.get_parameter('scene_buffer_size').value),
            max_odom_sync_dt_s=float(self.get_parameter('max_odom_sync_dt_s').value),
            depth_obstacle_samples=int(self.get_parameter('depth_obstacle_samples').value),
        )
        self.csv_writer = DatasetCsvWriter(self.output_csv, logger=self.get_logger())
        self.executed_waypoint_buffer = deque(maxlen=int(self.get_parameter('waypoint_buffer_size').value))
        self.pending_anchors = deque()

        self.rows_written = 0
        self.labels_seen = 0
        self.labels_dropped = 0

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

        self.get_logger().info(f'Sync dataset generator started. output_csv={self.output_csv}')
        self.get_logger().info(f'label_mode={self.label_mode} prompt_mode={self.prompt_mode}')
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

    def _now_s(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def odometry_callback(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        self.scene_sync.push_odometry(
            position=np.array(
                [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
                dtype=float,
            ),
            velocity=np.array(
                [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
                dtype=float,
            ),
            rotation_ned_body=quaternion_to_rotation_matrix(q.w, q.x, q.y, q.z),
        )

    def vehicle_odometry_callback(self, msg: VehicleOdometry) -> None:
        self.scene_sync.push_odometry(
            position=np.array([float(msg.position[0]), float(msg.position[1]), float(msg.position[2])], dtype=float),
            velocity=np.array([float(msg.velocity[0]), float(msg.velocity[1]), float(msg.velocity[2])], dtype=float),
            rotation_ned_body=quaternion_to_rotation_matrix(
                float(msg.q[0]),
                float(msg.q[1]),
                float(msg.q[2]),
                float(msg.q[3]),
            ),
        )

    def depth_image_callback(self, msg: Image) -> None:
        self.scene_sync.push_depth_msg(msg)

    def fresh_committed_waypoint_callback(self, msg: PoseStamped) -> None:
        del msg
        self.labels_seen += 1
        self.pending_anchors.append({'stamp_s': self._now_s()})
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

        label_waypoints = [
            [
                float(pose.pose.position.x),
                float(pose.pose.position.y),
                float(pose.pose.position.z),
            ]
            for pose in msg.poses[:LABEL_WAYPOINT_COUNT]
        ]
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
        scene, dt = self.scene_sync.match_scene(anchor_stamp_s, self.max_label_sync_dt_s)
        if scene is None:
            self.labels_dropped += 1
            self.get_logger().warn(
                f'Dropped label: no scene match within {self.max_label_sync_dt_s:.3f}s '
                f'(nearest_dt={dt if dt is not None else -1:.3f}s)',
                throttle_duration_sec=1.0,
            )
            return

        prompt, env_vector, _ = build_prompt_from_scene(
            goal_ned=self.goal,
            scene=scene,
            system_prompt=self.system_prompt,
            prompt_mode=self.prompt_mode,
            waypoint_count=LABEL_WAYPOINT_COUNT,
            depth_sample_count=self.scene_sync.depth_to_obstacles.n_sample,
        )
        if env_vector is None or prompt is None:
            self.labels_dropped += 1
            self.get_logger().warn('Dropped label: environment vector unavailable')
            return

        label_response = build_label_response(
            label_waypoints=label_waypoints,
            label_mode=label_mode,
            scene=scene,
            goal_ned=self.goal,
            eval_dt_s=self.eval_dt_s,
            eval_v_max_mps=self.eval_v_max_mps,
            eval_a_max_mps2=self.eval_a_max_mps2,
            eval_safety_radius_m=self.eval_safety_radius_m,
        )
        self.csv_writer.write_sample(
            anchor_stamp_s=anchor_stamp_s,
            scene_stamp_s=float(scene['stamp_s']),
            label_mode=label_mode,
            prompt=prompt,
            label_response=label_response,
            label_waypoints=label_waypoints,
        )
        self.rows_written = self.csv_writer.rows_written
        self.get_logger().info(
            f'Wrote sample #{self.rows_written} | mode={label_mode} | '
            f'sync_dt={1000.0 * abs(anchor_stamp_s - scene["stamp_s"]):.1f} ms'
        )

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
