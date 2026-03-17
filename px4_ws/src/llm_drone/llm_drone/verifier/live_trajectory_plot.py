#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data

from llm_drone.llm.offline_ground_truth_support import (
    trajectory_artifact_goal_z_ned,
    trajectory_artifact_path_ned,
)

try:
    from nav_msgs.msg import Odometry
except ImportError:  # pragma: no cover
    Odometry = None

try:
    from px4_msgs.msg import VehicleOdometry
except ImportError:  # pragma: no cover
    VehicleOdometry = None


def load_reference_path(trajectory_json: str) -> np.ndarray:
    payload = json.loads(Path(trajectory_json).expanduser().resolve().read_text())
    trajectory = payload.get('trajectory', payload)
    path_ned = np.asarray(trajectory_artifact_path_ned(trajectory), dtype=float)
    if path_ned.size == 0:
        raise ValueError('trajectory_json does not contain a reference_path_ned')

    path_ned = np.atleast_2d(path_ned)
    path_ned[:, 2] = float(trajectory_artifact_goal_z_ned(trajectory))
    return path_ned


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Live 3D reference-vs-tracking plot from a trajectory JSON and odometry topic.'
    )
    parser.add_argument('trajectory_json', help='Path to the trajectory JSON file')
    parser.add_argument('--topic', default='/fmu/out/vehicle_odometry', help='Odometry topic to subscribe to')
    parser.add_argument(
        '--msg-type',
        choices=('px4', 'nav'),
        default='px4',
        help='Odometry message type: px4=px4_msgs/VehicleOdometry, nav=nav_msgs/Odometry',
    )
    parser.add_argument('--goal-tolerance', type=float, default=0.5, help='Distance threshold for goal reached')
    parser.add_argument('--update-hz', type=float, default=10.0, help='Plot refresh rate')
    return parser


def main(args=None) -> None:
    parsed = make_parser().parse_args(args=args)
    reference_path = load_reference_path(parsed.trajectory_json)

    if parsed.msg_type == 'px4' and VehicleOdometry is None:
        raise RuntimeError('px4_msgs is required for --msg-type px4')
    if parsed.msg_type == 'nav' and Odometry is None:
        raise RuntimeError('nav_msgs is required for --msg-type nav')

    rclpy.init(args=None)
    node = rclpy.create_node('live_trajectory_plot')

    tracked_positions: list[np.ndarray] = []
    goal = np.asarray(reference_path[-1], dtype=float)
    goal_reached = False

    def px4_callback(msg: VehicleOdometry) -> None:
        tracked_positions.append(np.array(msg.position[:3], dtype=float))

    def nav_callback(msg: Odometry) -> None:
        pos = msg.pose.pose.position
        tracked_positions.append(np.array([pos.x, pos.y, pos.z], dtype=float))

    if parsed.msg_type == 'px4':
        node.create_subscription(VehicleOdometry, parsed.topic, px4_callback, qos_profile_sensor_data)
    else:
        node.create_subscription(Odometry, parsed.topic, nav_callback, qos_profile_sensor_data)

    plt.ion()
    fig, ax = plt.subplots()

    while rclpy.ok() and plt.fignum_exists(fig.number):
        rclpy.spin_once(node, timeout_sec=1.0 / max(parsed.update_hz, 1e-3))

        ax.cla()
        ax.plot(reference_path[:, 0], reference_path[:, 1], color='red', label='reference')
        if tracked_positions:
            tracked = np.vstack(tracked_positions)
            ax.plot(tracked[:, 0], tracked[:, 1], color='blue', label='tracking')
            current = tracked[-1]
            if not goal_reached and np.linalg.norm(current - goal) <= parsed.goal_tolerance:
                goal_reached = True
                print('Goal reached')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Reference vs Drone Tracking')
        ax.axis('equal')
        ax.legend(loc='upper right')
        plt.pause(0.001)

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    plt.ioff()


if __name__ == '__main__':
    main()
