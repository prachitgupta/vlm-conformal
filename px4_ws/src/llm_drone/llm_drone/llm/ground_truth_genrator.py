#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as RosPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from llm_drone.llm.dataset_pipeline_common import GROUND_TRUTH_REFERENCE_PATH_TOPIC
from llm_drone.llm.offline_ground_truth_support import (
    DEFAULT_FIXED_ALTITUDE_M,
    DEFAULT_GRID_RESOLUTION_M,
    DEFAULT_LOOKAHEAD_M,
    DEFAULT_MAX_ACCEL_MPS2,
    DEFAULT_MAX_SPEED_MPS,
    DEFAULT_PATH_DT_S,
    DEFAULT_PLANNER_CLEARANCE_M,
    DEFAULT_ROUTE_STEP_M,
    DEFAULT_SCENARIO_PREFIX,
    DEFAULT_SPAWN_POSE_ENU,
    DEFAULT_START_POSE_ENU,
    PoseENU,
    ScenarioManifest,
    build_trajectory_artifact,
    env_file_text,
    generate_procedural_scenarios,
    generate_reference_trajectory,
    scenario_manifest_from_json,
    scenario_manifests_from_sdf_inputs,
    trajectory_artifact_path_ned,
    write_json,
)

DEFAULT_TEMPLATE_WORLD_PATH = '/home/prachit/PX4-Autopilot/Tools/simulation/gz/worlds/obstacle_avoidance.sdf'
DEFAULT_OUTPUT_ROOT = str((Path(__file__).resolve().parents[2] / 'generated' / 'offline_ground_truth').resolve())


class GroundTruthGenerator(Node):
    def __init__(self):
        super().__init__('ground_truth_genrator')

        self.declare_parameter('manifest_json', '')
        self.declare_parameter('world_sdf_path', '')
        self.declare_parameter('world_sdf_glob', '')
        self.declare_parameter('scenario_count', 5)
        self.declare_parameter('seed', 42)
        self.declare_parameter('template_world_path', DEFAULT_TEMPLATE_WORLD_PATH)
        self.declare_parameter('output_root', DEFAULT_OUTPUT_ROOT)
        self.declare_parameter('manifest_output_dir', '')
        self.declare_parameter('trajectory_output_dir', '')
        self.declare_parameter('world_output_dir', '')
        self.declare_parameter('env_output_dir', '')
        self.declare_parameter('publish_scenario_id', '')
        self.declare_parameter('fixed_altitude_m', DEFAULT_FIXED_ALTITUDE_M)
        self.declare_parameter('start_pose_enu', list(DEFAULT_START_POSE_ENU))
        self.declare_parameter('spawn_pose_enu', list(DEFAULT_SPAWN_POSE_ENU))
        self.declare_parameter('goal_pose_enu', [float('nan')] * 4)
        self.declare_parameter('min_random_obstacles', 6)
        self.declare_parameter('max_random_obstacles', 10)
        self.declare_parameter('grid_resolution_m', DEFAULT_GRID_RESOLUTION_M)
        self.declare_parameter('planner_clearance_m', DEFAULT_PLANNER_CLEARANCE_M)
        self.declare_parameter('dt_s', DEFAULT_PATH_DT_S)
        self.declare_parameter('route_step_m', DEFAULT_ROUTE_STEP_M)
        self.declare_parameter('lookahead_distance_m', DEFAULT_LOOKAHEAD_M)
        self.declare_parameter('max_speed_mps', DEFAULT_MAX_SPEED_MPS)
        self.declare_parameter('max_accel_mps2', DEFAULT_MAX_ACCEL_MPS2)

        self.output_root = Path(str(self.get_parameter('output_root').value)).expanduser().resolve()
        self.manifest_output_dir = self._resolve_dir('manifest_output_dir', self.output_root / 'manifests')
        self.trajectory_output_dir = self._resolve_dir('trajectory_output_dir', self.output_root / 'trajectories')
        self.env_output_dir = self._resolve_dir('env_output_dir', self.output_root / 'env')
        self.template_world_path = Path(str(self.get_parameter('template_world_path').value)).expanduser().resolve()

        self.generated_artifacts: list[dict] = []
        scenarios = self._load_or_generate_scenarios()
        if not scenarios:
            raise RuntimeError('No scenarios available for ground-truth generation')

        for manifest in scenarios:
            trajectory = generate_reference_trajectory(
                manifest,
                grid_resolution_m=float(self.get_parameter('grid_resolution_m').value),
                planner_clearance_m=float(self.get_parameter('planner_clearance_m').value),
                dt_s=float(self.get_parameter('dt_s').value),
                route_step_m=float(self.get_parameter('route_step_m').value),
                lookahead_distance_m=float(self.get_parameter('lookahead_distance_m').value),
                max_speed_mps=float(self.get_parameter('max_speed_mps').value),
                max_accel_mps2=float(self.get_parameter('max_accel_mps2').value),
            )
            artifact = build_trajectory_artifact(manifest, trajectory)
            manifest_path = self.manifest_output_dir / f'{manifest.world_name}.json'
            trajectory_path = self.trajectory_output_dir / f'{manifest.world_name}.json'
            env_path = self.env_output_dir / f'{manifest.world_name}.env'

            write_json(manifest_path, manifest.as_dict())
            write_json(trajectory_path, artifact)
            env_path.parent.mkdir(parents=True, exist_ok=True)
            env_path.write_text(env_file_text(manifest))

            self.generated_artifacts.append({
                'manifest': manifest,
                'trajectory': trajectory,
                'manifest_path': manifest_path,
                'trajectory_path': trajectory_path,
                'env_path': env_path,
            })
            self.get_logger().info(
                f'Generated {manifest.world_name}: manifest={manifest_path} trajectory={trajectory_path} env={env_path}'
            )

        publish_scenario_id = str(self.get_parameter('publish_scenario_id').value).strip()
        selected = self._select_artifact(publish_scenario_id)
        self._selected_path = self._build_path_msg(selected['trajectory'])
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._path_pub = self.create_publisher(RosPath, GROUND_TRUTH_REFERENCE_PATH_TOPIC, qos)
        self.create_timer(1.0, self._publish_reference_path)
        self.get_logger().info(
            f'Publishing reference path for {selected["manifest"].world_name} on {GROUND_TRUTH_REFERENCE_PATH_TOPIC}'
        )

    def _resolve_dir(self, parameter_name: str, default_path: Path) -> Path:
        configured = str(self.get_parameter(parameter_name).value).strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return default_path.resolve()

    def _load_or_generate_scenarios(self) -> list[ScenarioManifest]:
        fixed_altitude_m = float(self.get_parameter('fixed_altitude_m').value)
        start_pose = self._pose_parameter('start_pose_enu', DEFAULT_START_POSE_ENU, fixed_altitude_m)
        spawn_pose = self._pose_parameter('spawn_pose_enu', DEFAULT_SPAWN_POSE_ENU, None)
        goal_pose = self._optional_pose_parameter('goal_pose_enu', fixed_altitude_m)

        world_sdf_path = str(self.get_parameter('world_sdf_path').value).strip()
        world_sdf_glob = str(self.get_parameter('world_sdf_glob').value).strip()
        if world_sdf_path or world_sdf_glob:
            scenarios = scenario_manifests_from_sdf_inputs(
                world_sdf_path=world_sdf_path,
                world_sdf_glob=world_sdf_glob,
                fixed_altitude_m=fixed_altitude_m,
                default_start_pose_enu=start_pose,
                default_spawn_pose_enu=spawn_pose,
                default_goal_pose_enu=goal_pose,
            )
            self.get_logger().info(
                f'Loaded {len(scenarios)} scenario manifest(s) from existing SDF world file(s)'
            )
            return scenarios

        manifest_json = str(self.get_parameter('manifest_json').value).strip()
        if manifest_json:
            scenarios = scenario_manifest_from_json(manifest_json)
            self.get_logger().info(f'Loaded {len(scenarios)} scenario manifest(s) from {manifest_json}')
            return scenarios

        scenario_count = int(self.get_parameter('scenario_count').value)
        seed = int(self.get_parameter('seed').value)
        scenarios = generate_procedural_scenarios(
            count=scenario_count,
            seed=seed,
            fixed_altitude_m=fixed_altitude_m,
            min_random_obstacles=int(self.get_parameter('min_random_obstacles').value),
            max_random_obstacles=int(self.get_parameter('max_random_obstacles').value),
            planner_clearance_m=float(self.get_parameter('planner_clearance_m').value),
        )
        self.get_logger().info(
            f'Procedurally generated {len(scenarios)} scenario(s) with prefix {DEFAULT_SCENARIO_PREFIX} and seed {seed}'
        )
        return scenarios

    def _pose_parameter(
        self,
        parameter_name: str,
        default_values: tuple[float, float, float, float] | list[float],
        override_z: float | None,
    ) -> PoseENU:
        values = list(self.get_parameter(parameter_name).value)
        if len(values) < 4:
            values = list(default_values)
        x, y, z, yaw = (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
        if override_z is not None:
            z = float(override_z)
        return PoseENU(x, y, z, yaw)

    def _optional_pose_parameter(self, parameter_name: str, override_z: float | None) -> PoseENU | None:
        values = list(self.get_parameter(parameter_name).value)
        if len(values) < 4 or not all(math.isfinite(float(value)) for value in values[:4]):
            return None
        x, y, z, yaw = (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
        if override_z is not None:
            z = float(override_z)
        return PoseENU(x, y, z, yaw)

    def _select_artifact(self, scenario_id: str) -> dict:
        if not scenario_id:
            return self.generated_artifacts[0]
        for artifact in self.generated_artifacts:
            manifest = artifact['manifest']
            if manifest.scenario_id == scenario_id or manifest.world_name == scenario_id:
                return artifact
        self.get_logger().warn(
            f'publish_scenario_id={scenario_id!r} not found; publishing first generated artifact instead.'
        )
        return self.generated_artifacts[0]

    def _build_path_msg(self, trajectory: dict) -> RosPath:
        path = RosPath()
        path.header.frame_id = 'ned'
        points_ned = trajectory_artifact_path_ned(trajectory)
        for point in points_ned:
            pose = PoseStamped()
            pose.header.frame_id = 'ned'
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = float(point[2])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        return path

    def _publish_reference_path(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self._selected_path.header.stamp = stamp
        for pose in self._selected_path.poses:
            pose.header.stamp = stamp
        self._path_pub.publish(self._selected_path)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GroundTruthGenerator()
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
