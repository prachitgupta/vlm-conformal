#!/usr/bin/env python3
"""
Publish a simple figure-8 trajectory to PX4 using TrajectorySetpoint.

This node assumes PX4 is already armed and in offboard mode, or that another
tool is handling mode transitions. It publishes:
  - /fmu/in/offboard_control_mode
  - /fmu/in/trajectory_setpoint

Frame: local NED
  x: North
  y: East
  z: Down (negative altitude means above the origin)
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint


class Figure8TrajectorySetpoints(Node):
    def __init__(self) -> None:
        super().__init__('figure8_trajectory_setpoints')

        self.declare_parameter('center_x', 0.0)
        self.declare_parameter('center_y', 0.0)
        self.declare_parameter('center_z', -3.0)
        self.declare_parameter('radius_x', 6.0)
        self.declare_parameter('radius_y', 3.0)
        self.declare_parameter('period_sec', 20.0)
        self.declare_parameter('update_rate_hz', 20.0)
        self.declare_parameter('hold_before_start_sec', 2.0)

        self.center_x = float(self.get_parameter('center_x').value)
        self.center_y = float(self.get_parameter('center_y').value)
        self.center_z = float(self.get_parameter('center_z').value)
        self.radius_x = max(0.1, float(self.get_parameter('radius_x').value))
        self.radius_y = max(0.1, float(self.get_parameter('radius_y').value))
        self.period_sec = max(1.0, float(self.get_parameter('period_sec').value))
        self.update_rate_hz = max(2.0, float(self.get_parameter('update_rate_hz').value))
        self.hold_before_start_sec = max(0.0, float(self.get_parameter('hold_before_start_sec').value))

        self._w = 2.0 * math.pi / self.period_sec
        self._start_time = self.get_clock().now()

        px4_qos = QoSProfile(
        reliability = ReliabilityPolicy.BEST_EFFORT,
        durability  = DurabilityPolicy.TRANSIENT_LOCAL,
        history     = HistoryPolicy.KEEP_LAST,
        depth       = 1
        )
        sensor_qos = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            history     = HistoryPolicy.KEEP_LAST,
            depth       = 1
        )

        self._offboard_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            px4_qos,
        )
        self._sp_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            px4_qos,
        )

        #self.create_timer(0.1, self.publish_offboard_heartbeat)
        self.create_timer(1.0 / self.update_rate_hz, self.publish_trajectory_setpoint)

        self.get_logger().info(
            'Figure-8 setpoint node started '
            f'center=({self.center_x:.2f},{self.center_y:.2f},{self.center_z:.2f}) '
            f'radius=({self.radius_x:.2f},{self.radius_y:.2f}) '
            f'period={self.period_sec:.2f}s'
        )

    def publish_offboard_heartbeat(self) -> None:
        msg = OffboardControlMode()
        #msg.timestamp = self._now_us()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self._offboard_pub.publish(msg)

    def publish_trajectory_setpoint(self) -> None:
        t = (self.get_clock().now() - self._start_time).nanoseconds * 1e-9

        if t < self.hold_before_start_sec:
            x = self.center_x
            y = self.center_y
            vx = 0.0
            vy = 0.0
        else:
            tau = t - self.hold_before_start_sec
            wt = self._w * tau
            x = self.center_x + self.radius_x * math.sin(wt)
            y = self.center_y + 0.5 * self.radius_y * math.sin(2.0 * wt)
            vx = self.radius_x * self._w * math.cos(wt)
            vy = self.radius_y * self._w * math.cos(2.0 * wt)

        msg = TrajectorySetpoint()
        #msg.timestamp = self._now_us()
        msg.position[0] = 0.0
        msg.position[1] = 20.0
        msg.position[2] = -3.0
        msg.velocity[0] = 0.0
        msg.velocity[1] = 3.0
        msg.velocity[2] = 0.0
        msg.acceleration[0] = float('nan')
        msg.acceleration[1] = float('nan')
        msg.acceleration[2] = float('nan')
        msg.yaw = float('nan')
        msg.yawspeed = float('nan')
        self._sp_pub.publish(msg)

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds // 1000)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Figure8TrajectorySetpoints()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
