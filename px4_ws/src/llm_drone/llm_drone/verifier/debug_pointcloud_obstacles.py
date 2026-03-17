#!/usr/bin/env python3
"""
PointCloud obstacle debug node.

Subscribes to:
  - sensor_msgs/msg/PointCloud2 (depth cloud)
  - vehicle odometry (for body->NED transform)

Applies the same manual camera->body->NED transform used by MPC vision controller
and logs all detected clustered obstacles. It does not publish control commands.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2 as pc2
import numpy as np
from scipy.spatial import distance

try:
    from px4_msgs.msg import VehicleOdometry
    HAS_PX4_MSGS = True
except ImportError:
    HAS_PX4_MSGS = False


class PointcloudObstacleDebugger(Node):
    def __init__(self):
        super().__init__('pointcloud_obstacle_debugger')

        self.declare_parameter('pointcloud_topic', '/depth_camera/points')
        self.declare_parameter('obstacle_threshold', 2.5)
        self.declare_parameter('cluster_radius', 0.5)
        self.declare_parameter('cluster_padding', 0.3)
        self.declare_parameter('log_throttle_s', 1.0)

        self.pointcloud_topic = str(self.get_parameter('pointcloud_topic').value)
        self.obs_threshold = float(self.get_parameter('obstacle_threshold').value)
        self.cluster_radius = float(self.get_parameter('cluster_radius').value)
        self.cluster_padding = float(self.get_parameter('cluster_padding').value)
        self.log_throttle_s = float(self.get_parameter('log_throttle_s').value)

        self.current_position = np.zeros(3, dtype=float)
        self.rotation_ned_body = np.eye(3, dtype=float)

        # Same extrinsics as MPC controller.
        self.rotation_body_camera = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=float,
        )
        self.translation_body_camera_m = np.array([0.13233, 0.0, 0.26078], dtype=float)

        self.depth_sub = self.create_subscription(
            PointCloud2, self.pointcloud_topic, self.pointcloud_callback, 10
        )

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        if HAS_PX4_MSGS:
            self.odom_sub = self.create_subscription(
                VehicleOdometry,
                '/fmu/out/vehicle_odometry',
                self.vehicle_odometry_callback,
                odom_qos,
            )
        else:
            self.odom_sub = self.create_subscription(
                Odometry, '/fmu/out/vehicle_odometry', self.odometry_callback, odom_qos
            )

        self.get_logger().info(f'Pointcloud obstacle debugger on: {self.pointcloud_topic}')
        self.get_logger().info(
            'This node only logs obstacles. It does not publish any movement commands.'
        )

    @staticmethod
    def quaternion_to_rotation_matrix(w, x, y, z):
        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n < 1e-9:
            return np.eye(3)
        w, x, y, z = w / n, x / n, y / n, z / n
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )

    def odometry_callback(self, msg):
        self.current_position = np.array(
            [msg.pose.pose.position.y, msg.pose.pose.position.x, -msg.pose.pose.position.z],
            dtype=float,
        )
        q = msg.pose.pose.orientation
        rotation_enu_body = self.quaternion_to_rotation_matrix(q.w, q.x, q.y, q.z)
        rotation_enu_to_ned = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=float,
        )
        self.rotation_ned_body = rotation_enu_to_ned @ rotation_enu_body

    def vehicle_odometry_callback(self, msg):
        self.current_position = np.array(
            [float(msg.position[0]), float(msg.position[1]), float(msg.position[2])],
            dtype=float,
        )
        self.rotation_ned_body = self.quaternion_to_rotation_matrix(
            float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3])
        )

    def cluster_obstacles(self, points):
        if len(points) == 0:
            return []
        points = np.asarray(points, dtype=float)
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) == 0:
            return []

        clusters = []
        used = np.zeros(len(points), dtype=bool)
        for i in range(len(points)):
            if used[i]:
                continue
            distances = distance.cdist([points[i]], points)[0]
            distances = np.where(np.isfinite(distances), distances, np.inf)
            mask = distances < self.cluster_radius
            cluster_points = points[mask]
            if len(cluster_points) == 0:
                used[i] = True
                continue
            used[mask] = True

            center = np.mean(cluster_points, axis=0)
            if not np.all(np.isfinite(center)):
                continue
            max_dist = np.max(distance.cdist([center], cluster_points)[0])
            clusters.append((center[0], center[1], center[2], max_dist + self.cluster_padding))
        return clusters

    def pointcloud_callback(self, msg):
        points_ned = []
        valid_points = 0

        for x_cam, y_cam, z_cam in pc2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True
        ):
            valid_points += 1
            if not (np.isfinite(x_cam) and np.isfinite(y_cam) and np.isfinite(z_cam)):
                continue
            point_cam = np.array([x_cam, y_cam, z_cam], dtype=float)
            point_body = self.rotation_body_camera @ point_cam + self.translation_body_camera_m
            if not np.all(np.isfinite(point_body)):
                continue
            point_ned = self.current_position + self.rotation_ned_body @ point_body
            if not np.all(np.isfinite(point_ned)):
                continue
            if np.linalg.norm(point_ned - self.current_position) > self.obs_threshold:
                continue
            points_ned.append(point_ned.tolist())

        if points_ned:
            cloud = np.array(points_ned, dtype=float)
            obstacles = self.cluster_obstacles(cloud)
        else:
            obstacles = []

        self.get_logger().info(
            f'Cloud points={valid_points}, obstacle_clusters={len(obstacles)}',
            throttle_duration_sec=self.log_throttle_s,
        )
        if obstacles:
            formatted = ', '.join(
                f'({ox:.2f},{oy:.2f},{oz:.2f},r={orad:.2f})' for ox, oy, oz, orad in obstacles
            )
            self.get_logger().info(
                f'Detected obstacles: {formatted}',
                throttle_duration_sec=self.log_throttle_s,
            )


def main(args=None):
    rclpy.init(args=args)
    node = PointcloudObstacleDebugger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
