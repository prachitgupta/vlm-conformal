#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

try:
    from px4_msgs.msg import GotoSetpoint, VehicleOdometry
    HAS_PX4_MSGS = True
except ImportError:
    HAS_PX4_MSGS = False

from llm_drone.llm.offline_ground_truth_support import (
    project_pose_to_reference_path,
    trajectory_artifact_goal_z_ned,
    trajectory_artifact_path_ned,
)


class TrajectoryGotoDebug(Node):
    def __init__(self) -> None:
        super().__init__('trajectory_goto_debug')
        if not HAS_PX4_MSGS:
            raise RuntimeError('px4_msgs is required for trajectory_goto_debug')

        self.declare_parameter('trajectory_json', '')
        self.declare_parameter('start_tolerance_m', 0.5)
        self.declare_parameter('acceptance_radius_m', 0.75)
        self.declare_parameter('max_horizontal_speed_mps', 15.0)
        self.declare_parameter('max_vertical_speed_mps', 2.0)
        self.declare_parameter('heading_rad', float('nan'))

        trajectory_json = str(self.get_parameter('trajectory_json').value).strip()
        if not trajectory_json:
            raise ValueError('trajectory_json parameter is required')

        payload = json.loads(Path(trajectory_json).expanduser().resolve().read_text())
        trajectory = payload.get('trajectory', payload)
        self.reference_path_ned = trajectory_artifact_path_ned(trajectory)
        if self.reference_path_ned.shape[0] == 0:
            raise ValueError('trajectory_json does not contain a reference_path_ned')

        self.goal_z_ned = float(trajectory_artifact_goal_z_ned(trajectory))
        self.start_tolerance_m = float(self.get_parameter('start_tolerance_m').value)
        self.acceptance_radius_m = float(self.get_parameter('acceptance_radius_m').value)
        self.max_horizontal_speed_mps = float(self.get_parameter('max_horizontal_speed_mps').value)
        self.max_vertical_speed_mps = float(self.get_parameter('max_vertical_speed_mps').value)
        self.heading_rad = float(self.get_parameter('heading_rad').value)

        self.current_position_ned: np.ndarray | None = None
        self.command_index = 0
        self.started = False
        self.complete = False
        self._last_wait_log_s = -1.0
        self._last_track_log_s = -1.0
        self._last_publish_log_s = -1.0

        self._goto_pub = self.create_publisher(GotoSetpoint, '/fmu/in/goto_setpoint', 10)
        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self._vehicle_odometry_callback,
            qos_profile_sensor_data,
        )
        self.create_timer(0.1, self._tick)

        self.get_logger().info(
            f'Trajectory goto debug ready. trajectory_json={trajectory_json} '
            f'waypoints={self.reference_path_ned.shape[0]}'
        )

    def _vehicle_odometry_callback(self, msg: VehicleOdometry) -> None:
        self.current_position_ned = np.array(
            [float(msg.position[0]), float(msg.position[1]), float(msg.position[2])],
            dtype=float,
        )

    def _current_target(self) -> np.ndarray:
        target = np.array(self.reference_path_ned[self.command_index], dtype=float, copy=True)
        target[2] = self.goal_z_ned
        return target

    def _xy_distance_to(self, target_ned: np.ndarray) -> float | None:
        if self.current_position_ned is None:
            return None
        return float(np.linalg.norm(
            np.asarray(target_ned[:2], dtype=float) - np.asarray(self.current_position_ned[:2], dtype=float)
        ))

    def _log_waiting_to_start(self, anchor_index: int, anchor_ned: np.ndarray, xy_distance: float) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s - self._last_wait_log_s < 1.0:
            return
        self._last_wait_log_s = now_s
        self.get_logger().info(
            'Waiting to start goto replay: '
            f'odom_xy=({self.current_position_ned[0]:.2f}, {self.current_position_ned[1]:.2f}) '
            f'closest_ref_index={anchor_index} '
            f'closest_ref_xy=({anchor_ned[0]:.2f}, {anchor_ned[1]:.2f}) '
            f'xy_error={xy_distance:.2f} m '
            f'start_tolerance={self.start_tolerance_m:.2f} m'
        )

    def _log_tracking_target(self, target_ned: np.ndarray, xy_distance: float) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s - self._last_track_log_s < 1.0:
            return
        self._last_track_log_s = now_s
        self.get_logger().info(
            'Tracking goto target: '
            f'index={self.command_index} '
            f'odom_xy=({self.current_position_ned[0]:.2f}, {self.current_position_ned[1]:.2f}) '
            f'target_xy=({target_ned[0]:.2f}, {target_ned[1]:.2f}) '
            f'xy_error={xy_distance:.2f} m '
            f'acceptance_radius={self.acceptance_radius_m:.2f} m'
        )

    def _maybe_start(self) -> bool:
        if self.current_position_ned is None:
            return False
        anchor_index = project_pose_to_reference_path(self.reference_path_ned, self.current_position_ned)
        anchor_ned = np.array(self.reference_path_ned[anchor_index], dtype=float, copy=True)
        xy_distance = float(np.linalg.norm(
            np.asarray(self.current_position_ned[:2], dtype=float) -
            np.asarray(anchor_ned[:2], dtype=float)
        ))
        if xy_distance > self.start_tolerance_m:
            self._log_waiting_to_start(anchor_index, anchor_ned, xy_distance)
            return False

        self.started = True
        self.command_index = anchor_index
        self.get_logger().info(
            f'Started goto replay at index {anchor_index} (xy_distance={xy_distance:.2f} m)'
        )
        return True

    def _publish_goto(self, target_ned: np.ndarray) -> None:
        msg = GotoSetpoint()
        msg.timestamp = 0
        msg.position[0] = float(target_ned[0])
        msg.position[1] = float(target_ned[1])
        msg.position[2] = float(target_ned[2])
        if np.isfinite(self.heading_rad):
            msg.heading = float(self.heading_rad)
            msg.flag_control_heading = True
        else:
            msg.heading = 0.0
            msg.flag_control_heading = False
        msg.max_horizontal_speed = float(self.max_horizontal_speed_mps)
        msg.max_vertical_speed = float(self.max_vertical_speed_mps)
        msg.max_heading_rate = 0.0
        msg.flag_set_max_horizontal_speed = True
        msg.flag_set_max_vertical_speed = True
        msg.flag_set_max_heading_rate = False
        self._goto_pub.publish(msg)
        self._log_published_goto(target_ned)

    def _log_published_goto(self, target_ned: np.ndarray) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s - self._last_publish_log_s < 1.0:
            return
        self._last_publish_log_s = now_s
        heading_text = 'nan' if not np.isfinite(self.heading_rad) else f'{self.heading_rad:.2f}'
        self.get_logger().info(
            'Publishing GotoSetpoint: '
            f'index={self.command_index} '
            f'target_ned=({target_ned[0]:.2f}, {target_ned[1]:.2f}, {target_ned[2]:.2f}) '
            f'max_horiz_speed={self.max_horizontal_speed_mps:.2f} m/s '
            f'max_vert_speed={self.max_vertical_speed_mps:.2f} m/s '
            f'heading_rad={heading_text}'
        )

    def _tick(self) -> None:
        if self.complete:
            return
        if not self.started and not self._maybe_start():
            return

        target = self._current_target()
        self._publish_goto(target)
        xy_distance = self._xy_distance_to(target)
        if xy_distance is None or xy_distance > self.acceptance_radius_m:
            if xy_distance is not None:
                self._log_tracking_target(target, xy_distance)
            return

        self.get_logger().info(
            f'Reached waypoint {self.command_index + 1}/{self.reference_path_ned.shape[0]} '
            f'(xy_distance={xy_distance:.2f} m)'
        )
        if self.command_index < self.reference_path_ned.shape[0] - 1:
            self.command_index += 1
            return

        self.complete = True
        self.get_logger().info('Completed trajectory goto debug replay')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = TrajectoryGotoDebug()
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
