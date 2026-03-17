#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from px4_msgs.msg import VehicleStatus
import rclpy
from rclpy.node import Node as RclpyNode

from llm_drone.llm.offline_ground_truth_support import (
    project_pose_to_reference_path,
    trajectory_artifact_goal_z_ned,
    trajectory_artifact_path_ned,
)


def _import_px4_ros2():
    required_attrs = ('Node', 'components', 'control', 'odometry')

    def _has_required_api(module) -> bool:
        return all(hasattr(module, attr) for attr in required_attrs)

    try:
        import px4_ros2  # type: ignore
        if _has_required_api(px4_ros2):
            return px4_ros2
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    local_src = repo_root.parent / 'px4-ros2-interface-lib' / 'px4_ros2_py' / 'src'
    if local_src.exists() and str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))

    try:
        import importlib

        sys.modules.pop('px4_ros2', None)
        px4_ros2 = importlib.import_module('px4_ros2')  # type: ignore
        if _has_required_api(px4_ros2):
            return px4_ros2
        raise RuntimeError(
            'px4_ros2 was imported, but it does not expose the required mode-control API '
            f'({", ".join(required_attrs)}).'
        )
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            'px4_ros2 could not be imported with the required API. Build/install px4_ros2_py '
            f'or keep {local_src} available on PYTHONPATH.'
        ) from exc


px4_ros2 = _import_px4_ros2()


class TrajectoryGotoMode(px4_ros2.components.ModeBase):
    def __init__(
        self,
        node,
        trajectory_json: str,
        start_tolerance_m: float,
        acceptance_radius_m: float,
        max_horizontal_speed_mps: float,
        max_vertical_speed_mps: float,
        heading_rad: float,
    ) -> None:
        super().__init__(
            node=node,
            mode_name='Trajectory Goto',
            replace_internal_mode=int(VehicleStatus.NAVIGATION_STATE_POSCTL),
        )
        self._node = node
        self._goto = px4_ros2.control.MulticopterGotoSetpointType(self)
        self._local_position = px4_ros2.odometry.LocalPosition(self)

        payload = json.loads(Path(trajectory_json).expanduser().resolve().read_text())
        trajectory = payload.get('trajectory', payload)
        self.reference_path_ned = trajectory_artifact_path_ned(trajectory)
        if self.reference_path_ned.shape[0] == 0:
            raise ValueError('trajectory_json does not contain a reference_path_ned')

        self.goal_z_ned = float(trajectory_artifact_goal_z_ned(trajectory))
        self.start_tolerance_m = float(start_tolerance_m)
        self.acceptance_radius_m = float(acceptance_radius_m)
        self.max_horizontal_speed_mps = float(max_horizontal_speed_mps)
        self.max_vertical_speed_mps = float(max_vertical_speed_mps)
        self.heading_rad = float(heading_rad)

        self.command_index = 0
        self.started = False
        self._done = False
        self._last_wait_log_s = 0.0
        self._last_track_log_s = 0.0
        self._last_publish_log_s = 0.0

        self._node.get_logger().info(
            f'Trajectory goto mode ready. trajectory_json={trajectory_json} '
            f'waypoints={self.reference_path_ned.shape[0]} '
            'Activate Position mode to hand waypoint following to PX4 goto smoothing.'
        )

    def on_activate(self):
        self.started = False
        self._done = False
        self.command_index = 0
        self._node.get_logger().info('Trajectory goto mode activated')

    def on_deactivate(self):
        self._node.get_logger().info('Trajectory goto mode deactivated')

    def _now_s(self) -> float:
        return time.monotonic()

    def _position_ned(self) -> np.ndarray | None:
        if not self._local_position.position_xy_valid() or not self._local_position.position_z_valid():
            return None
        return np.asarray(self._local_position.position_ned(), dtype=float)

    def _target_ned(self) -> np.ndarray:
        target = np.array(self.reference_path_ned[self.command_index], dtype=float, copy=True)
        target[2] = self.goal_z_ned
        return target

    def _publish_target(self, target_ned: np.ndarray) -> None:
        heading = None if not math.isfinite(self.heading_rad) else float(self.heading_rad)
        # Let PX4 generate the jerk-limited motion; this mode only updates the active goal.
        self._goto.update(
            tuple(float(v) for v in target_ned),
            heading=heading,
            max_horizontal_speed=float(self.max_horizontal_speed_mps),
            max_vertical_speed=float(self.max_vertical_speed_mps),
        )
        now_s = self._now_s()
        if now_s - self._last_publish_log_s >= 1.0:
            self._last_publish_log_s = now_s
            heading_text = 'nan' if heading is None else f'{heading:.2f}'
            self._node.get_logger().info(
                'Publishing mode goto target: '
                f'index={self.command_index} '
                f'target_ned=({target_ned[0]:.2f}, {target_ned[1]:.2f}, {target_ned[2]:.2f}) '
                f'max_horiz_speed={self.max_horizontal_speed_mps:.2f} m/s '
                f'max_vert_speed={self.max_vertical_speed_mps:.2f} m/s '
                f'heading_rad={heading_text}'
            )

    def _log_waiting_to_start(self, position_ned: np.ndarray, anchor_index: int, anchor_ned: np.ndarray, xy_distance: float) -> None:
        now_s = self._now_s()
        if now_s - self._last_wait_log_s < 1.0:
            return
        self._last_wait_log_s = now_s
        self._node.get_logger().info(
            'Waiting to start mode goto replay: '
            f'odom_xy=({position_ned[0]:.2f}, {position_ned[1]:.2f}) '
            f'closest_ref_index={anchor_index} '
            f'closest_ref_xy=({anchor_ned[0]:.2f}, {anchor_ned[1]:.2f}) '
            f'xy_error={xy_distance:.2f} m '
            f'start_tolerance={self.start_tolerance_m:.2f} m'
        )

    def _log_tracking_target(self, position_ned: np.ndarray, target_ned: np.ndarray, xy_distance: float) -> None:
        now_s = self._now_s()
        if now_s - self._last_track_log_s < 1.0:
            return
        self._last_track_log_s = now_s
        self._node.get_logger().info(
            'Tracking mode goto target: '
            f'index={self.command_index} '
            f'odom_xy=({position_ned[0]:.2f}, {position_ned[1]:.2f}) '
            f'target_xy=({target_ned[0]:.2f}, {target_ned[1]:.2f}) '
            f'xy_error={xy_distance:.2f} m '
            f'acceptance_radius={self.acceptance_radius_m:.2f} m'
        )

    def update_setpoint(self, dt_s: float):
        del dt_s
        position_ned = self._position_ned()
        if position_ned is None:
            return

        if self._done:
            self._publish_target(self._target_ned())
            return

        if not self.started:
            anchor_index = project_pose_to_reference_path(self.reference_path_ned, position_ned)
            anchor_ned = np.array(self.reference_path_ned[anchor_index], dtype=float, copy=True)
            xy_distance = float(np.linalg.norm(position_ned[:2] - anchor_ned[:2]))
            if xy_distance > self.start_tolerance_m:
                # Hold the current position until the vehicle is placed close to the replay path.
                hold_target = np.array([position_ned[0], position_ned[1], self.goal_z_ned], dtype=float)
                self._publish_target(hold_target)
                self._log_waiting_to_start(position_ned, anchor_index, anchor_ned, xy_distance)
                return

            self.started = True
            self.command_index = int(anchor_index)
            self._node.get_logger().info(
                f'Started mode goto replay at index {anchor_index} (xy_distance={xy_distance:.2f} m)'
            )

        target_ned = self._target_ned()
        self._publish_target(target_ned)
        xy_distance = float(np.linalg.norm(position_ned[:2] - target_ned[:2]))
        if xy_distance > self.acceptance_radius_m:
            self._log_tracking_target(position_ned, target_ned, xy_distance)
            return

        self._node.get_logger().info(
            f'Reached waypoint {self.command_index + 1}/{self.reference_path_ned.shape[0]} '
            f'(xy_distance={xy_distance:.2f} m)'
        )
        if self.command_index < self.reference_path_ned.shape[0] - 1:
            self.command_index += 1
            return

        self._done = True
        self.completed(px4_ros2.components.Result.success)
        self._node.get_logger().info('Completed trajectory goto mode replay')


def main(args=None) -> None:
    rclpy.init(args=args)
    param_node = RclpyNode('trajectory_goto_mode_debug_params')
    param_node.declare_parameter('trajectory_json', '')
    param_node.declare_parameter('start_tolerance_m', 0.5)
    param_node.declare_parameter('acceptance_radius_m', 0.75)
    param_node.declare_parameter('max_horizontal_speed_mps', 15.0)
    param_node.declare_parameter('max_vertical_speed_mps', 2.0)
    param_node.declare_parameter('heading_rad', float('nan'))

    trajectory_json = str(param_node.get_parameter('trajectory_json').value).strip()
    start_tolerance_m = float(param_node.get_parameter('start_tolerance_m').value)
    acceptance_radius_m = float(param_node.get_parameter('acceptance_radius_m').value)
    max_horizontal_speed_mps = float(param_node.get_parameter('max_horizontal_speed_mps').value)
    max_vertical_speed_mps = float(param_node.get_parameter('max_vertical_speed_mps').value)
    heading_rad = float(param_node.get_parameter('heading_rad').value)
    param_node.destroy_node()
    rclpy.shutdown()

    if not trajectory_json:
        raise ValueError('trajectory_json parameter is required')

    node = px4_ros2.Node('trajectory_goto_mode_debug', debug_output=True)
    mode = TrajectoryGotoMode(
        node=node,
        trajectory_json=trajectory_json,
        start_tolerance_m=start_tolerance_m,
        acceptance_radius_m=acceptance_radius_m,
        max_horizontal_speed_mps=max_horizontal_speed_mps,
        max_vertical_speed_mps=max_vertical_speed_mps,
        heading_rad=heading_rad,
    )
    if not mode.register():
        raise RuntimeError('Failed to register trajectory goto mode')
    node.spin()


if __name__ == '__main__':
    main()
