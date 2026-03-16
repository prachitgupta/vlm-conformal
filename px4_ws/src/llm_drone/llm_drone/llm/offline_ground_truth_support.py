from __future__ import annotations

import json
import math
import heapq
import glob
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
try:
    import cvxpy as cp
except Exception:
    cp = None

DEFAULT_FIXED_ALTITUDE_M = 2.5
DEFAULT_GRID_RESOLUTION_M = 0.25
DEFAULT_PLANNER_CLEARANCE_M = 1.5
DEFAULT_PATH_DT_S = 0.1
DEFAULT_ROUTE_STEP_M = 0.5
DEFAULT_BOUNDARY_SPACING_M = 0.3
DEFAULT_LOOKAHEAD_M = 1.5
DEFAULT_MAX_SPEED_MPS = 15
DEFAULT_MAX_ACCEL_MPS2 = 8
DEFAULT_GOAL_TOLERANCE_M = 0.35
DEFAULT_SCENARIO_PREFIX = 'obstacle_avoidance'
DEFAULT_SPAWN_POSE_ENU = (0.0, 0.0, 1.0, 0.0)
DEFAULT_START_POSE_ENU = (0.0, 0.0, DEFAULT_FIXED_ALTITUDE_M, 0.0)
DEFAULT_GLOBAL_OPT_ITERS = 7
DEFAULT_GLOBAL_OPT_TRUST_M = 1.25
DEFAULT_RESAMPLE_SPEED_FRACTION = 0.75
DEFAULT_GLOBAL_OPT_MAX_SLACK_M = 1e-3
DEFAULT_GLOBAL_OPT_TOTAL_SLACK_M = 5e-3


@dataclass
class PoseENU:
    x: float
    y: float
    z: float
    yaw: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def as_dict(self) -> dict[str, float]:
        return {
            'x': float(self.x),
            'y': float(self.y),
            'z': float(self.z),
            'yaw': float(self.yaw),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PoseENU':
        return cls(
            x=float(data['x']),
            y=float(data['y']),
            z=float(data['z']),
            yaw=float(data.get('yaw', 0.0)),
        )


@dataclass
class ObstacleSpec:
    name: str
    kind: str
    position: PoseENU
    radius: float | None = None
    length: float | None = None
    size: tuple[float, float, float] | None = None
    static: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'name': self.name,
            'type': self.kind,
            'position': self.position.as_dict(),
            'static': bool(self.static),
        }
        if self.radius is not None:
            payload['radius'] = float(self.radius)
        if self.length is not None:
            payload['length'] = float(self.length)
        if self.size is not None:
            payload['size'] = [float(v) for v in self.size]
        if self.metadata:
            payload['metadata'] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ObstacleSpec':
        size = data.get('size', None)
        return cls(
            name=str(data['name']),
            kind=str(data.get('type', data.get('kind', 'box'))).strip().lower(),
            position=PoseENU.from_dict(data['position']),
            radius=None if data.get('radius') is None else float(data['radius']),
            length=None if data.get('length') is None else float(data['length']),
            size=None if size is None else tuple(float(v) for v in size),
            static=bool(data.get('static', True)),
            metadata=dict(data.get('metadata', {})),
        )

    def half_extents_xy(self) -> tuple[float, float]:
        if self.kind == 'box' and self.size is not None:
            return float(self.size[0]) / 2.0, float(self.size[1]) / 2.0
        if self.kind == 'cylinder' and self.radius is not None:
            return float(self.radius), float(self.radius)
        raise ValueError(f'Unsupported obstacle shape for {self.name}: {self.kind!r}')

    def signed_distance_xy(self, point_xy: np.ndarray) -> float:
        point_xy = np.asarray(point_xy, dtype=float)
        center = np.array([self.position.x, self.position.y], dtype=float)
        local = point_xy - center
        yaw = float(self.position.yaw)
        if abs(yaw) > 1e-9:
            c = math.cos(-yaw)
            s = math.sin(-yaw)
            local = np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]], dtype=float)

        if self.kind == 'cylinder':
            if self.radius is None:
                raise ValueError(f'Cylinder obstacle missing radius: {self.name}')
            return float(np.linalg.norm(local) - float(self.radius))

        if self.kind == 'box':
            if self.size is None:
                raise ValueError(f'Box obstacle missing size: {self.name}')
            hx = float(self.size[0]) / 2.0
            hy = float(self.size[1]) / 2.0
            dx = abs(local[0]) - hx
            dy = abs(local[1]) - hy
            outside = np.linalg.norm(np.maximum([dx, dy], 0.0))
            inside = min(max(dx, dy), 0.0)
            return float(outside + inside)

        raise ValueError(f'Unsupported obstacle type: {self.kind!r}')

    def contains_xy(self, point_xy: np.ndarray, inflation_m: float = 0.0) -> bool:
        return self.signed_distance_xy(point_xy) <= float(inflation_m)

    def boundary_points_xy(self, spacing_m: float = DEFAULT_BOUNDARY_SPACING_M) -> np.ndarray:
        spacing_m = max(0.05, float(spacing_m))
        center = np.array([self.position.x, self.position.y], dtype=float)
        yaw = float(self.position.yaw)
        if self.kind == 'cylinder':
            if self.radius is None:
                raise ValueError(f'Cylinder obstacle missing radius: {self.name}')
            circumference = max(2.0 * math.pi * float(self.radius), spacing_m)
            count = max(12, int(math.ceil(circumference / spacing_m)))
            angles = np.linspace(0.0, 2.0 * math.pi, num=count, endpoint=False)
            return np.column_stack([
                center[0] + float(self.radius) * np.cos(angles),
                center[1] + float(self.radius) * np.sin(angles),
            ])

        if self.kind == 'box':
            if self.size is None:
                raise ValueError(f'Box obstacle missing size: {self.name}')
            hx = float(self.size[0]) / 2.0
            hy = float(self.size[1]) / 2.0
            nx = max(2, int(math.ceil((2.0 * hx) / spacing_m)))
            ny = max(2, int(math.ceil((2.0 * hy) / spacing_m)))
            top = np.column_stack([np.linspace(-hx, hx, num=nx), np.full(nx, hy)])
            bottom = np.column_stack([np.linspace(-hx, hx, num=nx), np.full(nx, -hy)])
            left = np.column_stack([np.full(ny, -hx), np.linspace(-hy, hy, num=ny)])
            right = np.column_stack([np.full(ny, hx), np.linspace(-hy, hy, num=ny)])
            local = np.vstack([top, bottom, left, right])
            if abs(yaw) > 1e-9:
                c = math.cos(yaw)
                s = math.sin(yaw)
                rot = np.array([[c, -s], [s, c]], dtype=float)
                local = (rot @ local.T).T
            return local + center[None, :]

        raise ValueError(f'Unsupported obstacle type: {self.kind!r}')

    def max_extent_xy(self) -> float:
        hx, hy = self.half_extents_xy()
        return float(max(hx, hy))


@dataclass
class ScenarioManifest:
    scenario_id: str
    world_name: str
    start_pose_enu: PoseENU
    goal_pose_enu: PoseENU
    fixed_altitude_m: float
    obstacles: list[ObstacleSpec]
    spawn_pose_enu: PoseENU | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            'scenario_id': self.scenario_id,
            'world_name': self.world_name,
            'start_pose_enu': self.start_pose_enu.as_dict(),
            'goal_pose_enu': self.goal_pose_enu.as_dict(),
            'fixed_altitude_m': float(self.fixed_altitude_m),
            'spawn_pose_enu': None if self.spawn_pose_enu is None else self.spawn_pose_enu.as_dict(),
            'obstacles': [obstacle.as_dict() for obstacle in self.obstacles],
            'metadata': dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ScenarioManifest':
        start_key = 'start_pose_enu' if 'start_pose_enu' in data else 'start_pose'
        goal_key = 'goal_pose_enu' if 'goal_pose_enu' in data else 'goal_pose'
        return cls(
            scenario_id=str(data['scenario_id']),
            world_name=str(data.get('world_name', f'{DEFAULT_SCENARIO_PREFIX}_{data["scenario_id"]}')),
            start_pose_enu=PoseENU.from_dict(data[start_key]),
            goal_pose_enu=PoseENU.from_dict(data[goal_key]),
            fixed_altitude_m=float(data.get('fixed_altitude_m', DEFAULT_FIXED_ALTITUDE_M)),
            obstacles=[ObstacleSpec.from_dict(obstacle) for obstacle in data.get('obstacles', [])],
            spawn_pose_enu=None if data.get('spawn_pose_enu') is None else PoseENU.from_dict(data['spawn_pose_enu']),
            metadata=dict(data.get('metadata', {})),
        )


@dataclass
class OccupancyGrid2D:
    occupancy: np.ndarray
    resolution_m: float
    x_min: float
    y_min: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupancy.shape

    @property
    def x_max(self) -> float:
        return float(self.x_min + self.resolution_m * (self.occupancy.shape[1] - 1))

    @property
    def y_max(self) -> float:
        return float(self.y_min + self.resolution_m * (self.occupancy.shape[0] - 1))

    def world_to_grid(self, point_xy: np.ndarray) -> tuple[int, int]:
        x, y = float(point_xy[0]), float(point_xy[1])
        col = int(round((x - self.x_min) / self.resolution_m))
        row = int(round((y - self.y_min) / self.resolution_m))
        return row, col

    def grid_to_world(self, rc: tuple[int, int]) -> np.ndarray:
        row, col = int(rc[0]), int(rc[1])
        return np.array([
            self.x_min + float(col) * self.resolution_m,
            self.y_min + float(row) * self.resolution_m,
        ], dtype=float)

    def in_bounds(self, rc: tuple[int, int]) -> bool:
        row, col = int(rc[0]), int(rc[1])
        return 0 <= row < self.occupancy.shape[0] and 0 <= col < self.occupancy.shape[1]

    def is_occupied(self, rc: tuple[int, int]) -> bool:
        row, col = int(rc[0]), int(rc[1])
        return bool(self.occupancy[row, col])

    def is_occupied_xy(self, point_xy: np.ndarray) -> bool:
        rc = self.world_to_grid(point_xy)
        if not self.in_bounds(rc):
            return True
        return self.is_occupied(rc)


def gazebo_enu_to_ned(x_enu: float, y_enu: float, z_enu: float) -> np.ndarray:
    return np.array([float(y_enu), float(x_enu), -float(z_enu)], dtype=float)


def ned_to_gazebo_enu(x_ned: float, y_ned: float, z_ned: float) -> np.ndarray:
    return np.array([float(y_ned), float(x_ned), -float(z_ned)], dtype=float)


def scenario_manifest_from_json(path: Path | str) -> list[ScenarioManifest]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        return [ScenarioManifest.from_dict(data)]
    if isinstance(data, list):
        return [ScenarioManifest.from_dict(item) for item in data]
    raise ValueError(f'Unsupported manifest JSON payload in {path}')


def _parse_pose_element(pose_text: str | None) -> PoseENU:
    if pose_text is None or not str(pose_text).strip():
        return PoseENU(0.0, 0.0, 0.0, 0.0)
    values = [float(token) for token in str(pose_text).split()]
    values = values[:6] + [0.0] * max(0, 6 - len(values))
    return PoseENU(values[0], values[1], values[2], values[5])


def _first_geometry_element(model: ET.Element) -> ET.Element | None:
    for search_path in ('.//collision/geometry', './/visual/geometry'):
        geometry = model.find(search_path)
        if geometry is not None:
            return geometry
    return None


def _obstacle_spec_from_model(model: ET.Element) -> ObstacleSpec | None:
    model_name = str(model.get('name', 'unnamed_model'))
    if model_name.lower() in {
        'ground_plane',
        'start_marker',
        'start_pose',
        'start',
        'spawn_marker',
        'spawn_pose',
        'spawn',
        'goal_marker',
        'goal_pose',
        'goal',
    }:
        return None

    geometry = _first_geometry_element(model)
    if geometry is None:
        return None

    pose = _parse_pose_element(model.findtext('pose'))
    static_text = str(model.findtext('static', 'true')).strip().lower()
    static = static_text in {'1', 'true', 'yes'}

    cylinder = geometry.find('cylinder')
    if cylinder is not None:
        radius_text = cylinder.findtext('radius')
        length_text = cylinder.findtext('length')
        if radius_text is None or length_text is None:
            return None
        return ObstacleSpec(
            name=model_name,
            kind='cylinder',
            position=pose,
            radius=float(radius_text),
            length=float(length_text),
            static=static,
        )

    box = geometry.find('box')
    if box is not None:
        size_text = box.findtext('size')
        if size_text is None:
            return None
        size_values = [float(token) for token in size_text.split()]
        if len(size_values) != 3:
            return None
        return ObstacleSpec(
            name=model_name,
            kind='box',
            position=pose,
            size=(size_values[0], size_values[1], size_values[2]),
            static=static,
        )

    return None


def _find_named_pose(
    model_poses: dict[str, PoseENU],
    candidate_names: tuple[str, ...],
) -> PoseENU | None:
    lowered = {name.lower(): pose for name, pose in model_poses.items()}
    for candidate in candidate_names:
        pose = lowered.get(candidate.lower())
        if pose is not None:
            return pose
    return None


def _infer_goal_pose_from_sdf_models(
    *,
    model_poses: dict[str, PoseENU],
    obstacles: list[ObstacleSpec],
    fixed_altitude_m: float,
    fallback_start_pose: PoseENU,
) -> tuple[PoseENU, str]:
    named_goal = _find_named_pose(model_poses, ('goal_marker', 'goal_pose', 'goal'))
    if named_goal is not None:
        return PoseENU(named_goal.x, named_goal.y, fixed_altitude_m, named_goal.yaw), 'named_goal_model'

    gate_left = _find_named_pose(model_poses, ('goal_gate_left',))
    gate_right = _find_named_pose(model_poses, ('goal_gate_right',))
    if gate_left is not None and gate_right is not None:
        return (
            PoseENU(
                x=0.5 * (gate_left.x + gate_right.x),
                y=0.5 * (gate_left.y + gate_right.y),
                z=float(fixed_altitude_m),
                yaw=0.5 * (gate_left.yaw + gate_right.yaw),
            ),
            'goal_gate_midpoint',
        )

    if obstacles:
        furthest_x = max(obstacle.position.x + obstacle.max_extent_xy() for obstacle in obstacles)
        return PoseENU(
            x=float(furthest_x + 3.0),
            y=float(fallback_start_pose.y),
            z=float(fixed_altitude_m),
            yaw=0.0,
        ), 'furthest_obstacle_plus_margin'

    return PoseENU(33.0, 0.0, float(fixed_altitude_m), 0.0), 'default_goal'


def scenario_manifest_from_sdf(
    path: Path | str,
    *,
    fixed_altitude_m: float = DEFAULT_FIXED_ALTITUDE_M,
    default_start_pose_enu: PoseENU | None = None,
    default_spawn_pose_enu: PoseENU | None = None,
    default_goal_pose_enu: PoseENU | None = None,
) -> ScenarioManifest:
    sdf_path = Path(path).expanduser().resolve()
    root = ET.parse(str(sdf_path)).getroot()
    world = root.find('world')
    if world is None:
        raise ValueError(f'Could not find <world> in SDF: {sdf_path}')

    model_poses: dict[str, PoseENU] = {}
    obstacles: list[ObstacleSpec] = []
    for model in world.findall('model'):
        model_name = str(model.get('name', 'unnamed_model'))
        model_poses[model_name] = _parse_pose_element(model.findtext('pose'))
        obstacle = _obstacle_spec_from_model(model)
        if obstacle is not None:
            obstacles.append(obstacle)

    start_pose = default_start_pose_enu or PoseENU(*DEFAULT_START_POSE_ENU)
    spawn_pose = default_spawn_pose_enu or PoseENU(*DEFAULT_SPAWN_POSE_ENU)
    named_start = _find_named_pose(model_poses, ('start_marker', 'start_pose', 'start'))
    named_spawn = _find_named_pose(model_poses, ('spawn_marker', 'spawn_pose', 'spawn'))
    if named_start is not None:
        start_pose = PoseENU(named_start.x, named_start.y, float(fixed_altitude_m), named_start.yaw)
    if named_spawn is not None:
        spawn_pose = PoseENU(named_spawn.x, named_spawn.y, named_spawn.z, named_spawn.yaw)

    if default_goal_pose_enu is not None:
        goal_pose = PoseENU(
            default_goal_pose_enu.x,
            default_goal_pose_enu.y,
            float(fixed_altitude_m),
            default_goal_pose_enu.yaw,
        )
        goal_source = 'parameter_override'
    else:
        goal_pose, goal_source = _infer_goal_pose_from_sdf_models(
            model_poses=model_poses,
            obstacles=obstacles,
            fixed_altitude_m=fixed_altitude_m,
            fallback_start_pose=start_pose,
        )

    return ScenarioManifest(
        scenario_id=sdf_path.stem,
        world_name=sdf_path.stem,
        start_pose_enu=PoseENU(start_pose.x, start_pose.y, float(fixed_altitude_m), start_pose.yaw),
        goal_pose_enu=goal_pose,
        fixed_altitude_m=float(fixed_altitude_m),
        obstacles=obstacles,
        spawn_pose_enu=spawn_pose,
        metadata={
            'source_world_sdf': str(sdf_path),
            'sdf_world_name': str(world.get('name', sdf_path.stem)),
            'goal_source': goal_source,
            'start_source': 'named_start_model' if named_start is not None else 'default_start_pose',
            'spawn_source': 'named_spawn_model' if named_spawn is not None else 'default_spawn_pose',
        },
    )


def scenario_manifests_from_sdf_inputs(
    *,
    world_sdf_path: str = '',
    world_sdf_glob: str = '',
    fixed_altitude_m: float = DEFAULT_FIXED_ALTITUDE_M,
    default_start_pose_enu: PoseENU | None = None,
    default_spawn_pose_enu: PoseENU | None = None,
    default_goal_pose_enu: PoseENU | None = None,
) -> list[ScenarioManifest]:
    candidate_paths: list[Path] = []
    if world_sdf_path.strip():
        for token in world_sdf_path.split(','):
            resolved = Path(token.strip()).expanduser()
            if resolved.is_dir():
                candidate_paths.extend(sorted(resolved.glob('*.sdf')))
            elif resolved.exists():
                candidate_paths.append(resolved.resolve())
    if world_sdf_glob.strip():
        candidate_paths.extend(sorted(Path(path_str).resolve() for path_str in glob.glob(world_sdf_glob.strip())))

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path_obj in candidate_paths:
        resolved = path_obj.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(resolved)

    return [
        scenario_manifest_from_sdf(
            path_obj,
            fixed_altitude_m=fixed_altitude_m,
            default_start_pose_enu=default_start_pose_enu,
            default_spawn_pose_enu=default_spawn_pose_enu,
            default_goal_pose_enu=default_goal_pose_enu,
        )
        for path_obj in unique_paths
    ]


def write_json(path: Path | str, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + '\n')


def default_lane_walls() -> list[ObstacleSpec]:
    return [
        ObstacleSpec(
            name='lane_left_wall',
            kind='box',
            position=PoseENU(16.0, 6.0, 2.5, 0.0),
            size=(36.0, 0.5, 5.0),
        ),
        ObstacleSpec(
            name='lane_right_wall',
            kind='box',
            position=PoseENU(16.0, -6.0, 2.5, 0.0),
            size=(36.0, 0.5, 5.0),
        ),
    ]


def _candidate_goal_pose(rng: np.random.Generator, fixed_altitude_m: float) -> PoseENU:
    return PoseENU(
        x=float(rng.uniform(31.0, 34.0)),
        y=float(rng.uniform(-2.5, 2.5)),
        z=float(fixed_altitude_m),
        yaw=0.0,
    )


def _random_obstacle(
    rng: np.random.Generator,
    *,
    idx: int,
    fixed_altitude_m: float,
) -> ObstacleSpec:
    obstacle_type = 'cylinder' if rng.random() < 0.6 else 'box'
    x = float(rng.uniform(6.0, 29.0))
    y = float(rng.uniform(-4.25, 4.25))
    if obstacle_type == 'cylinder':
        radius = float(rng.uniform(0.3, 0.6))
        length = float(rng.uniform(3.0, 4.2))
        return ObstacleSpec(
            name=f'proc_cyl_{idx:02d}',
            kind='cylinder',
            position=PoseENU(x, y, max(fixed_altitude_m * 0.65, length / 2.0), 0.0),
            radius=radius,
            length=length,
        )

    size_x = float(rng.uniform(0.8, 1.6))
    size_y = float(rng.uniform(0.6, 1.4))
    size_z = float(rng.uniform(2.5, 4.0))
    yaw = float(rng.uniform(-0.45, 0.45))
    return ObstacleSpec(
        name=f'proc_box_{idx:02d}',
        kind='box',
        position=PoseENU(x, y, max(fixed_altitude_m * 0.65, size_z / 2.0), yaw),
        size=(size_x, size_y, size_z),
    )


def _is_valid_candidate(
    candidate: ObstacleSpec,
    placed: list[ObstacleSpec],
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    *,
    clearance_m: float,
    min_separation_m: float,
) -> bool:
    center = np.array([candidate.position.x, candidate.position.y], dtype=float)
    if np.linalg.norm(center - start_xy) < clearance_m:
        return False
    if np.linalg.norm(center - goal_xy) < clearance_m:
        return False
    for obstacle in placed:
        other_center = np.array([obstacle.position.x, obstacle.position.y], dtype=float)
        min_dist = obstacle.max_extent_xy() + candidate.max_extent_xy() + float(min_separation_m)
        if np.linalg.norm(center - other_center) < min_dist:
            return False
    return True


def generate_procedural_scenarios(
    *,
    count: int,
    seed: int,
    fixed_altitude_m: float = DEFAULT_FIXED_ALTITUDE_M,
    min_random_obstacles: int = 6,
    max_random_obstacles: int = 10,
    planner_clearance_m: float = DEFAULT_PLANNER_CLEARANCE_M,
) -> list[ScenarioManifest]:
    rng = np.random.default_rng(int(seed))
    scenarios: list[ScenarioManifest] = []

    for idx in range(int(count)):
        scenario_id = f'{idx:03d}'
        world_name = f'{DEFAULT_SCENARIO_PREFIX}_{scenario_id}'
        start_pose = PoseENU(*DEFAULT_START_POSE_ENU)
        spawn_pose = PoseENU(*DEFAULT_SPAWN_POSE_ENU)
        goal_pose = _candidate_goal_pose(rng, fixed_altitude_m)
        start_xy = np.array([start_pose.x, start_pose.y], dtype=float)
        goal_xy = np.array([goal_pose.x, goal_pose.y], dtype=float)

        obstacles = default_lane_walls()
        placed = list(obstacles)
        obstacle_count = int(rng.integers(min_random_obstacles, max_random_obstacles + 1))
        for obstacle_idx in range(obstacle_count):
            for _attempt in range(300):
                candidate = _random_obstacle(
                    rng,
                    idx=obstacle_idx + idx * max_random_obstacles,
                    fixed_altitude_m=fixed_altitude_m,
                )
                if _is_valid_candidate(
                    candidate,
                    placed,
                    start_xy,
                    goal_xy,
                    clearance_m=max(2.5, planner_clearance_m + 1.0),
                    min_separation_m=0.6,
                ):
                    obstacles.append(candidate)
                    placed.append(candidate)
                    break
            else:
                raise RuntimeError(f'Failed to place obstacle {obstacle_idx} for scenario {scenario_id}')

        scenarios.append(
            ScenarioManifest(
                scenario_id=scenario_id,
                world_name=world_name,
                start_pose_enu=start_pose,
                goal_pose_enu=goal_pose,
                fixed_altitude_m=float(fixed_altitude_m),
                obstacles=obstacles,
                spawn_pose_enu=spawn_pose,
                metadata={
                    'seed': int(seed),
                    'procedural_index': int(idx),
                    'random_obstacle_count': int(obstacle_count),
                },
            )
        )

    return scenarios


def _obstacle_bbox_xy(obstacle: ObstacleSpec, inflation_m: float = 0.0) -> tuple[float, float, float, float]:
    center = np.array([obstacle.position.x, obstacle.position.y], dtype=float)
    if obstacle.kind == 'cylinder':
        radius = float(obstacle.radius or 0.0) + float(inflation_m)
        return (
            float(center[0] - radius),
            float(center[0] + radius),
            float(center[1] - radius),
            float(center[1] + radius),
        )

    if obstacle.kind == 'box':
        if obstacle.size is None:
            raise ValueError(f'Box obstacle missing size: {obstacle.name}')
        hx = float(obstacle.size[0]) / 2.0 + float(inflation_m)
        hy = float(obstacle.size[1]) / 2.0 + float(inflation_m)
        corners = np.array([
            [-hx, -hy],
            [-hx, hy],
            [hx, -hy],
            [hx, hy],
        ], dtype=float)
        yaw = float(obstacle.position.yaw)
        if abs(yaw) > 1e-9:
            c = math.cos(yaw)
            s = math.sin(yaw)
            rot = np.array([[c, -s], [s, c]], dtype=float)
            corners = (rot @ corners.T).T
        corners = corners + center[None, :]
        return (
            float(np.min(corners[:, 0])),
            float(np.max(corners[:, 0])),
            float(np.min(corners[:, 1])),
            float(np.max(corners[:, 1])),
        )

    raise ValueError(f'Unsupported obstacle type: {obstacle.kind!r}')


def compute_world_bounds_xy(manifest: ScenarioManifest, margin_m: float = 3.0) -> tuple[float, float, float, float]:
    xs = [manifest.start_pose_enu.x, manifest.goal_pose_enu.x]
    ys = [manifest.start_pose_enu.y, manifest.goal_pose_enu.y]
    for obstacle in manifest.obstacles:
        x_min, x_max, y_min, y_max = _obstacle_bbox_xy(obstacle, 0.5)
        xs.extend([x_min, x_max])
        ys.extend([y_min, y_max])
    return (
        float(min(xs) - margin_m),
        float(max(xs) + margin_m),
        float(min(ys) - margin_m),
        float(max(ys) + margin_m),
    )


def build_planning_grid(
    manifest: ScenarioManifest,
    *,
    resolution_m: float = DEFAULT_GRID_RESOLUTION_M,
    inflation_m: float = DEFAULT_PLANNER_CLEARANCE_M,
) -> OccupancyGrid2D:
    x_min, x_max, y_min, y_max = compute_world_bounds_xy(manifest, margin_m=inflation_m + 1.0)
    cols = int(math.ceil((x_max - x_min) / resolution_m)) + 1
    rows = int(math.ceil((y_max - y_min) / resolution_m)) + 1
    occupancy = np.zeros((rows, cols), dtype=bool)
    grid = OccupancyGrid2D(occupancy=occupancy, resolution_m=float(resolution_m), x_min=x_min, y_min=y_min)

    for obstacle in manifest.obstacles:
        bbox = _obstacle_bbox_xy(obstacle, inflation_m=float(inflation_m))
        min_rc = grid.world_to_grid(np.array([bbox[0], bbox[2]], dtype=float))
        max_rc = grid.world_to_grid(np.array([bbox[1], bbox[3]], dtype=float))
        r0 = max(0, min(min_rc[0], max_rc[0]))
        r1 = min(rows - 1, max(min_rc[0], max_rc[0]))
        c0 = max(0, min(min_rc[1], max_rc[1]))
        c1 = min(cols - 1, max(min_rc[1], max_rc[1]))
        for row in range(r0, r1 + 1):
            y = grid.y_min + float(row) * grid.resolution_m
            for col in range(c0, c1 + 1):
                x = grid.x_min + float(col) * grid.resolution_m
                if obstacle.contains_xy(np.array([x, y], dtype=float), inflation_m=float(inflation_m)):
                    occupancy[row, col] = True

    return grid


def _nearest_free_cell(grid: OccupancyGrid2D, rc: tuple[int, int]) -> tuple[int, int]:
    if grid.in_bounds(rc) and not grid.is_occupied(rc):
        return rc
    queue = deque([rc])
    visited = {tuple(rc)}
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        row, col = queue.popleft()
        for dr, dc in neighbors:
            nxt = (row + dr, col + dc)
            if nxt in visited or not grid.in_bounds(nxt):
                continue
            if not grid.is_occupied(nxt):
                return nxt
            visited.add(nxt)
            queue.append(nxt)
    raise RuntimeError('Failed to find a free grid cell near the requested point')


def astar_path(grid: OccupancyGrid2D, start_xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    start_rc = _nearest_free_cell(grid, grid.world_to_grid(start_xy))
    goal_rc = _nearest_free_cell(grid, grid.world_to_grid(goal_xy))

    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))

    frontier: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(frontier, (0.0, start_rc))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_rc: None}
    g_score: dict[tuple[int, int], float] = {start_rc: 0.0}

    while frontier:
        _prio, current = heapq.heappop(frontier)
        if current == goal_rc:
            break

        for dr, dc, step_cost in neighbors:
            nxt = (current[0] + dr, current[1] + dc)
            if not grid.in_bounds(nxt) or grid.is_occupied(nxt):
                continue
            tentative = g_score[current] + step_cost
            if tentative >= g_score.get(nxt, float('inf')):
                continue
            came_from[nxt] = current
            g_score[nxt] = tentative
            heapq.heappush(frontier, (tentative + heuristic(nxt, goal_rc), nxt))

    if goal_rc not in came_from:
        raise RuntimeError('A* failed to find a path through the occupancy grid')

    path_rc = []
    node = goal_rc
    while node is not None:
        path_rc.append(node)
        node = came_from[node]
    path_rc.reverse()
    return np.vstack([grid.grid_to_world(rc) for rc in path_rc])


def point_is_collision_free(
    point_xy: np.ndarray,
    obstacles: list[ObstacleSpec],
    *,
    inflation_m: float = 0.0,
) -> bool:
    for obstacle in obstacles:
        if obstacle.contains_xy(point_xy, inflation_m=float(inflation_m)):
            return False
    return True


def segment_is_collision_free(
    a_xy: np.ndarray,
    b_xy: np.ndarray,
    obstacles: list[ObstacleSpec],
    *,
    inflation_m: float,
    sample_spacing_m: float,
) -> bool:
    segment = np.asarray(b_xy, dtype=float) - np.asarray(a_xy, dtype=float)
    distance = float(np.linalg.norm(segment))
    if distance < 1e-9:
        return point_is_collision_free(np.asarray(a_xy, dtype=float), obstacles, inflation_m=inflation_m)
    count = max(2, int(math.ceil(distance / max(0.05, sample_spacing_m))) + 1)
    for t in np.linspace(0.0, 1.0, num=count):
        point = np.asarray(a_xy, dtype=float) + t * segment
        if not point_is_collision_free(point, obstacles, inflation_m=inflation_m):
            return False
    return True


def simplify_polyline(
    path_xy: np.ndarray,
    obstacles: list[ObstacleSpec],
    *,
    inflation_m: float,
    sample_spacing_m: float = DEFAULT_GRID_RESOLUTION_M,
) -> np.ndarray:
    if path_xy.shape[0] <= 2:
        return np.array(path_xy, dtype=float, copy=True)

    simplified = [np.array(path_xy[0], dtype=float, copy=True)]
    idx = 0
    while idx < path_xy.shape[0] - 1:
        next_idx = idx + 1
        for candidate_idx in range(path_xy.shape[0] - 1, idx, -1):
            if segment_is_collision_free(
                path_xy[idx],
                path_xy[candidate_idx],
                obstacles,
                inflation_m=inflation_m,
                sample_spacing_m=sample_spacing_m,
            ):
                next_idx = candidate_idx
                break
        simplified.append(np.array(path_xy[next_idx], dtype=float, copy=True))
        idx = next_idx
    return np.vstack(simplified)


def densify_polyline(path_xy: np.ndarray, step_m: float = DEFAULT_ROUTE_STEP_M) -> np.ndarray:
    if path_xy.shape[0] <= 1:
        return np.array(path_xy, dtype=float, copy=True)
    step_m = max(0.05, float(step_m))
    dense = [np.array(path_xy[0], dtype=float, copy=True)]
    for idx in range(path_xy.shape[0] - 1):
        a = np.array(path_xy[idx], dtype=float, copy=True)
        b = np.array(path_xy[idx + 1], dtype=float, copy=True)
        segment = b - a
        distance = float(np.linalg.norm(segment))
        if distance < 1e-9:
            continue
        count = max(1, int(math.ceil(distance / step_m)))
        for step in range(1, count + 1):
            dense.append(a + (float(step) / float(count)) * segment)
    return np.vstack(dense)


def resample_polyline_for_dynamics(
    path_xy: np.ndarray,
    *,
    dt_s: float,
    route_step_m: float,
    max_speed_mps: float,
) -> np.ndarray:
    dynamic_step_m = max(0.05, float(max_speed_mps) * float(dt_s) * DEFAULT_RESAMPLE_SPEED_FRACTION)
    step_m = min(max(0.05, float(route_step_m)), dynamic_step_m)
    return densify_polyline(path_xy, step_m=step_m)


def obstacle_boundary_points_ned(
    manifest: ScenarioManifest,
    *,
    spacing_m: float = DEFAULT_BOUNDARY_SPACING_M,
) -> np.ndarray:
    pts = []
    for obstacle in manifest.obstacles:
        boundary_enu = obstacle.boundary_points_xy(spacing_m=spacing_m)
        if boundary_enu.size == 0:
            continue
        boundary_ned = np.column_stack([
            boundary_enu[:, 1],
            boundary_enu[:, 0],
        ])
        pts.append(boundary_ned)
    if not pts:
        return np.zeros((0, 2), dtype=float)
    return np.vstack(pts)


def _advance_along_polyline(
    current_xy: np.ndarray,
    route_xy: np.ndarray,
    route_index: int,
    distance_m: float,
) -> np.ndarray:
    remaining = float(distance_m)
    point = np.array(current_xy, dtype=float, copy=True)
    idx = int(route_index)
    while remaining > 1e-9 and idx < route_xy.shape[0] - 1:
        target = np.array(route_xy[idx + 1], dtype=float, copy=True)
        segment = target - point
        seg_len = float(np.linalg.norm(segment))
        if seg_len < 1e-9:
            idx += 1
            point = target
            continue
        if seg_len >= remaining:
            return point + (remaining / seg_len) * segment
        remaining -= seg_len
        idx += 1
        point = target
    return np.array(route_xy[-1], dtype=float, copy=True)


def _advance_path_index(route_xy: np.ndarray, start_index: int, lookahead_distance_m: float) -> int:
    cumulative = 0.0
    idx = int(start_index)
    while idx < route_xy.shape[0] - 1 and cumulative < float(lookahead_distance_m):
        cumulative += float(np.linalg.norm(route_xy[idx + 1] - route_xy[idx]))
        idx += 1
    return idx


def project_pose_to_reference_path(
    path_ned: np.ndarray,
    position_ned: np.ndarray,
    *,
    start_index: int = 0,
    search_ahead: int | None = None,
) -> int:
    if path_ned.shape[0] == 0:
        raise ValueError('Cannot project onto an empty path')
    start = max(0, int(start_index))
    end = path_ned.shape[0] if search_ahead is None else min(path_ned.shape[0], start + int(search_ahead))
    window = np.asarray(path_ned[start:end, :2], dtype=float)
    query = np.asarray(position_ned[:2], dtype=float)
    if window.shape[0] == 0:
        return int(path_ned.shape[0] - 1)
    dists = np.linalg.norm(window - query[None, :], axis=1)
    return int(start + int(np.argmin(dists)))


def extract_future_waypoints(path_ned: np.ndarray, anchor_index: int, count: int = 5) -> list[list[float]]:
    start = int(anchor_index) + 1
    end = start + int(count)
    if start < 0 or end > path_ned.shape[0]:
        return []
    return [[float(v) for v in point] for point in np.asarray(path_ned[start:end], dtype=float)]


def _estimate_path_state_guess(
    path_xy: np.ndarray,
    *,
    dt_s: float,
    max_speed_mps: float,
    max_accel_mps2: float,
) -> tuple[np.ndarray, np.ndarray]:
    path_xy = np.asarray(path_xy, dtype=float)
    if path_xy.shape[0] == 0:
        return np.zeros((0, 4), dtype=float), np.zeros((0, 2), dtype=float)
    state_guess = np.zeros((path_xy.shape[0], 4), dtype=float)
    state_guess[:, :2] = path_xy
    if path_xy.shape[0] == 1:
        return state_guess, np.zeros((0, 2), dtype=float)

    velocities = np.diff(path_xy, axis=0) / max(float(dt_s), 1e-6)
    np.clip(velocities, -float(max_speed_mps), float(max_speed_mps), out=velocities)
    state_guess[:-1, 2:4] = velocities
    state_guess[-1, 2:4] = 0.0

    controls = np.diff(state_guess[:, 2:4], axis=0) / max(float(dt_s), 1e-6)
    np.clip(controls, -float(max_accel_mps2), float(max_accel_mps2), out=controls)
    return state_guess, controls


def _path_length_m(path_xy: np.ndarray) -> float:
    path_xy = np.asarray(path_xy, dtype=float)
    if path_xy.shape[0] <= 1:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1)))


def _signed_distance_with_gradient_xy(
    obstacle: ObstacleSpec,
    point_xy: np.ndarray,
    *,
    eps: float = 1e-3,
) -> tuple[float, np.ndarray]:
    point_xy = np.asarray(point_xy, dtype=float)
    distance = float(obstacle.signed_distance_xy(point_xy))
    grad = np.zeros(2, dtype=float)
    for axis in range(2):
        step = np.zeros(2, dtype=float)
        step[axis] = float(eps)
        plus = float(obstacle.signed_distance_xy(point_xy + step))
        minus = float(obstacle.signed_distance_xy(point_xy - step))
        grad[axis] = (plus - minus) / (2.0 * float(eps))

    grad_norm = float(np.linalg.norm(grad))
    if grad_norm <= 1e-6:
        direction = point_xy - np.array([obstacle.position.x, obstacle.position.y], dtype=float)
        dir_norm = float(np.linalg.norm(direction))
        if dir_norm > 1e-6:
            grad = direction / dir_norm
        else:
            grad = np.array([1.0, 0.0], dtype=float)
    else:
        grad /= grad_norm
    return distance, grad


def min_signed_distance_to_obstacles(path_xy: np.ndarray, obstacles: list[ObstacleSpec]) -> float:
    path_xy = np.asarray(path_xy, dtype=float)
    if path_xy.shape[0] == 0 or not obstacles:
        return float('inf')
    min_distance = float('inf')
    for point_xy in path_xy:
        for obstacle in obstacles:
            min_distance = min(min_distance, float(obstacle.signed_distance_xy(point_xy)))
    return min_distance


def polyline_is_collision_free(
    path_xy: np.ndarray,
    obstacles: list[ObstacleSpec],
    *,
    inflation_m: float,
    sample_spacing_m: float,
) -> bool:
    path_xy = np.asarray(path_xy, dtype=float)
    if path_xy.shape[0] == 0:
        return True
    if not all(point_is_collision_free(point, obstacles, inflation_m=inflation_m) for point in path_xy):
        return False
    for idx in range(path_xy.shape[0] - 1):
        if not segment_is_collision_free(
            path_xy[idx],
            path_xy[idx + 1],
            obstacles,
            inflation_m=inflation_m,
            sample_spacing_m=sample_spacing_m,
        ):
            return False
    return True


def _solve_global_reference_path_xy(
    init_path_xy: np.ndarray,
    obstacles: list[ObstacleSpec],
    *,
    planner_clearance_m: float,
    dt_s: float,
    max_speed_mps: float,
    max_accel_mps2: float,
    grid_resolution_m: float,
    max_iters: int = DEFAULT_GLOBAL_OPT_ITERS,
    initial_trust_region_m: float = DEFAULT_GLOBAL_OPT_TRUST_M,
) -> tuple[np.ndarray, dict[str, Any]]:
    init_path_xy = np.asarray(init_path_xy, dtype=float)
    metadata: dict[str, Any] = {
        'planner_method': 'astar_initialization_global_qp',
        'solver': 'cvxpy' if cp is not None else 'astar_only_fallback',
        'iterations': 0,
        'accepted_iterations': 0,
        'status': 'astar_only_fallback',
        'max_slack_m': float('inf'),
        'total_slack_m': float('inf'),
    }
    if init_path_xy.shape[0] <= 1 or cp is None:
        return np.array(init_path_xy, dtype=float, copy=True), metadata

    num_points = int(init_path_xy.shape[0])
    warm_positions = np.array(init_path_xy, dtype=float, copy=True)
    warm_states, warm_controls = _estimate_path_state_guess(
        warm_positions,
        dt_s=dt_s,
        max_speed_mps=max_speed_mps,
        max_accel_mps2=max_accel_mps2,
    )
    best_positions = np.array(warm_positions, dtype=float, copy=True)
    best_clearance = min_signed_distance_to_obstacles(best_positions, obstacles)
    trust_region_m = max(float(initial_trust_region_m), max(0.25, float(grid_resolution_m)))
    target_clearance_m = float(planner_clearance_m)
    sample_spacing_m = max(0.05, min(float(grid_resolution_m), 0.25))
    slack_penalty = 2_000.0
    slack_max_tol = float(DEFAULT_GLOBAL_OPT_MAX_SLACK_M)
    slack_total_tol = float(DEFAULT_GLOBAL_OPT_TOTAL_SLACK_M)

    for iteration in range(int(max_iters)):
        metadata['iterations'] = int(iteration + 1)
        state = cp.Variable((num_points, 4))
        control = cp.Variable((num_points - 1, 2))
        slack = cp.Variable((num_points, len(obstacles)), nonneg=True) if obstacles else None

        constraints = [
            state[0, :2] == warm_positions[0],
            state[-1, :2] == warm_positions[-1],
            state[0, 2:4] == 0.0,
            state[-1, 2:4] == 0.0,
            cp.abs(state[:, 2:4]) <= float(max_speed_mps),
            cp.abs(state[:, :2] - warm_states[:, :2]) <= trust_region_m,
            cp.abs(state[:, 2:4] - warm_states[:, 2:4]) <= float(max_speed_mps),
        ]

        if num_points > 1:
            constraints.append(cp.abs(control - warm_controls) <= float(max_accel_mps2))

        for idx in range(num_points - 1):
            constraints += [
                state[idx + 1, :2] == state[idx, :2] + float(dt_s) * state[idx, 2:4],
                state[idx + 1, 2:4] == state[idx, 2:4] + float(dt_s) * control[idx],
                cp.abs(control[idx]) <= float(max_accel_mps2),
            ]

        for point_idx in range(1, num_points - 1):
            current_xy = warm_states[point_idx, :2]
            for obstacle_idx, obstacle in enumerate(obstacles):
                distance_now, gradient = _signed_distance_with_gradient_xy(obstacle, current_xy)
                linearized_distance = float(distance_now) + gradient @ (state[point_idx, :2] - current_xy)
                if slack is None:
                    constraints.append(linearized_distance >= target_clearance_m)
                else:
                    constraints.append(linearized_distance + slack[point_idx, obstacle_idx] >= target_clearance_m)

        objective_terms = [
            0.25 * cp.sum_squares(control),
            1.5 * cp.sum_squares(state[1:-1, :2] - init_path_xy[1:-1]),
            0.15 * cp.sum_squares(state[:, 2:4]),
            20.0 * cp.sum_squares(state[-1, 2:4]),
        ]
        if slack is not None:
            objective_terms.append(float(slack_penalty) * cp.sum(slack))
        problem = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)

        solved = False
        solver_status = 'not_run'
        for solver_name in (cp.OSQP, cp.SCS):
            try:
                problem.solve(solver=solver_name, warm_start=True, verbose=False)
                solver_status = str(problem.status)
                if state.value is not None and solver_status in {'optimal', 'optimal_inaccurate'}:
                    solved = True
                    break
            except Exception:
                solver_status = f'{solver_name}_failed'

        if not solved or state.value is None:
            metadata['status'] = str(solver_status)
            trust_region_m = max(0.25, trust_region_m * 0.5)
            continue

        candidate_states = np.asarray(state.value, dtype=float)
        candidate_controls = np.asarray(control.value, dtype=float) if control.value is not None else np.zeros((0, 2))
        candidate_positions = np.asarray(candidate_states[:, :2], dtype=float)
        candidate_clearance = min_signed_distance_to_obstacles(candidate_positions, obstacles)
        if slack is not None and slack.value is not None:
            candidate_slack = np.asarray(slack.value, dtype=float)
            max_slack = float(np.max(candidate_slack)) if candidate_slack.size > 0 else 0.0
            total_slack = float(np.sum(candidate_slack))
        else:
            max_slack = 0.0
            total_slack = 0.0
        collision_free = polyline_is_collision_free(
            candidate_positions,
            obstacles,
            inflation_m=max(0.0, planner_clearance_m - 0.05),
            sample_spacing_m=sample_spacing_m,
        )
        metadata['status'] = str(solver_status)
        metadata['max_slack_m'] = float(max_slack)
        metadata['total_slack_m'] = float(total_slack)

        slack_feasible = max_slack <= slack_max_tol and total_slack <= slack_total_tol

        if collision_free and slack_feasible:
            metadata['accepted_iterations'] = int(metadata['accepted_iterations']) + 1
            position_step = float(np.max(np.linalg.norm(candidate_positions - warm_positions, axis=1)))
            warm_positions = candidate_positions
            warm_states = candidate_states
            warm_controls = candidate_controls
            best_positions = np.array(candidate_positions, dtype=float, copy=True)
            best_clearance = candidate_clearance
            trust_region_m = min(max(0.5, trust_region_m * 1.15), 3.0)
            slack_penalty = max(2_000.0, slack_penalty * 1.25)
            if position_step < 5e-3:
                break
            continue

        if collision_free and not slack_feasible:
            metadata['status'] = 'rejected_nonzero_slack'
            slack_penalty *= 8.0
            trust_region_m = max(0.25, trust_region_m * 0.7)
            continue

        if not collision_free and total_slack <= slack_total_tol * 2.0:
            metadata['status'] = 'rejected_postcheck_collision'
            slack_penalty *= 6.0
            trust_region_m = max(0.25, trust_region_m * 0.6)
            continue

        slack_penalty *= 10.0
        trust_region_m = max(0.25, trust_region_m * 0.5)

    metadata['final_min_signed_distance_m'] = float(best_clearance)
    metadata['path_length_m'] = _path_length_m(best_positions)
    metadata['status'] = str(metadata['status'])
    return best_positions, metadata


def generate_reference_trajectory(
    manifest: ScenarioManifest,
    *,
    grid_resolution_m: float = DEFAULT_GRID_RESOLUTION_M,
    planner_clearance_m: float = DEFAULT_PLANNER_CLEARANCE_M,
    dt_s: float = DEFAULT_PATH_DT_S,
    route_step_m: float = DEFAULT_ROUTE_STEP_M,
    boundary_spacing_m: float = DEFAULT_BOUNDARY_SPACING_M,
    lookahead_distance_m: float = DEFAULT_LOOKAHEAD_M,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
    max_accel_mps2: float = DEFAULT_MAX_ACCEL_MPS2,
    max_obstacle_points: int = 12,
    goal_tolerance_m: float = DEFAULT_GOAL_TOLERANCE_M,
    max_steps: int = 800,
) -> dict[str, Any]:
    del boundary_spacing_m
    del lookahead_distance_m
    del max_obstacle_points
    del goal_tolerance_m
    del max_steps
    start_xy_enu = np.array([manifest.start_pose_enu.x, manifest.start_pose_enu.y], dtype=float)
    goal_xy_enu = np.array([manifest.goal_pose_enu.x, manifest.goal_pose_enu.y], dtype=float)
    grid = build_planning_grid(
        manifest,
        resolution_m=grid_resolution_m,
        inflation_m=planner_clearance_m,
    )
    coarse_xy_enu = astar_path(grid, start_xy_enu, goal_xy_enu)
    simplified_xy_enu = simplify_polyline(
        coarse_xy_enu,
        manifest.obstacles,
        inflation_m=planner_clearance_m,
        sample_spacing_m=grid_resolution_m,
    )
    dense_xy_enu = resample_polyline_for_dynamics(
        simplified_xy_enu,
        dt_s=dt_s,
        route_step_m=route_step_m,
        max_speed_mps=max_speed_mps,
    )
    if dense_xy_enu.shape[0] == 0:
        raise RuntimeError(f'Failed to build an A* initialization route for scenario {manifest.scenario_id}')

    fixed_altitude_m = float(manifest.fixed_altitude_m)
    optimized_xy_enu, optimization_metadata = _solve_global_reference_path_xy(
        dense_xy_enu,
        manifest.obstacles,
        planner_clearance_m=planner_clearance_m,
        dt_s=dt_s,
        max_speed_mps=max_speed_mps,
        max_accel_mps2=max_accel_mps2,
        grid_resolution_m=grid_resolution_m,
    )
    if optimized_xy_enu.shape[0] == 0:
        raise RuntimeError(f'Failed to optimize a reference path for scenario {manifest.scenario_id}')
    if not polyline_is_collision_free(
        optimized_xy_enu,
        manifest.obstacles,
        inflation_m=max(0.0, planner_clearance_m - 0.05),
        sample_spacing_m=max(0.05, min(grid_resolution_m, 0.25)),
    ):
        optimized_xy_enu = np.array(dense_xy_enu, dtype=float, copy=True)
        optimization_metadata['status'] = 'global_solve_rejected_fallback_to_astar_init'

    goal_z_ned = float(gazebo_enu_to_ned(manifest.goal_pose_enu.x, manifest.goal_pose_enu.y, fixed_altitude_m)[2])
    reference_ned_arr = np.column_stack([
        optimized_xy_enu[:, 1],
        optimized_xy_enu[:, 0],
        np.full(optimized_xy_enu.shape[0], goal_z_ned, dtype=float),
    ])
    reference_enu_arr = np.column_stack([
        optimized_xy_enu[:, 0],
        optimized_xy_enu[:, 1],
        np.full(optimized_xy_enu.shape[0], fixed_altitude_m, dtype=float),
    ])

    return {
        'scenario_id': manifest.scenario_id,
        'world_name': manifest.world_name,
        'dt_s': float(dt_s),
        'grid_resolution_m': float(grid_resolution_m),
        'planner_clearance_m': float(planner_clearance_m),
        'fixed_altitude_m': float(fixed_altitude_m),
        'goal_z_ned_m': float(goal_z_ned),
        'goal_z_enu_m': float(fixed_altitude_m),
        'optimization_metadata': optimization_metadata,
        'coarse_route_enu': [
            {
                'x': float(point[0]),
                'y': float(point[1]),
            }
            for point in simplified_xy_enu
        ],
        'reference_path_enu': [
            {
                't': float(idx) * float(dt_s),
                'x': float(point[0]),
                'y': float(point[1]),
            }
            for idx, point in enumerate(reference_enu_arr)
        ],
        'reference_path_ned': [
            {
                't': float(idx) * float(dt_s),
                'x': float(point[0]),
                'y': float(point[1]),
            }
            for idx, point in enumerate(reference_ned_arr)
        ],
    }


def trajectory_records_to_array(
    records: list[dict[str, Any]],
    *,
    frame: str,
    goal_z_m: float | None = None,
) -> np.ndarray:
    del frame
    return np.array([
        [float(item['x']), float(item['y']), float(item.get('z', 0.0 if goal_z_m is None else goal_z_m))]
        for item in records if isinstance(item, dict)
    ], dtype=float)


def trajectory_artifact_goal_z_ned(artifact: dict[str, Any]) -> float:
    if 'goal_z_ned_m' in artifact:
        return float(artifact['goal_z_ned_m'])

    path_ned = artifact.get('reference_path_ned', [])
    if path_ned and 'z' in path_ned[-1]:
        return float(path_ned[-1]['z'])

    scenario = artifact.get('scenario', {})
    goal_pose = scenario.get('goal_pose_enu', scenario.get('goal_pose', {}))
    if isinstance(goal_pose, dict) and 'z' in goal_pose:
        return float(-float(goal_pose['z']))

    return float(-artifact.get('fixed_altitude_m', DEFAULT_FIXED_ALTITUDE_M))


def trajectory_artifact_path_ned(artifact: dict[str, Any]) -> np.ndarray:
    goal_z_ned = trajectory_artifact_goal_z_ned(artifact)
    return trajectory_records_to_array(
        artifact.get('reference_path_ned', []),
        frame='ned',
        goal_z_m=goal_z_ned,
    )


def trajectory_artifact_timestamps(artifact: dict[str, Any]) -> np.ndarray:
    return np.array([float(item['t']) for item in artifact.get('reference_path_ned', [])], dtype=float)


def build_trajectory_artifact(manifest: ScenarioManifest, trajectory: dict[str, Any]) -> dict[str, Any]:
    return {
        'scenario': manifest.as_dict(),
        'trajectory': trajectory,
    }


def env_file_text(manifest: ScenarioManifest) -> str:
    spawn = manifest.spawn_pose_enu or manifest.start_pose_enu
    return (
        f'export PX4_GZ_WORLD={manifest.world_name}\n'
        'export PX4_SIM_MODEL="${PX4_SIM_MODEL:-gz_x500_depth}"\n'
        f'export PX4_GZ_MODEL_POSE="{spawn.x:.3f},{spawn.y:.3f},{spawn.z:.3f},0,0,{spawn.yaw:.3f}"\n'
    )


def _obstacle_to_model_element(obstacle: ObstacleSpec) -> ET.Element:
    model = ET.Element('model', {'name': obstacle.name})
    static = ET.SubElement(model, 'static')
    static.text = 'true' if obstacle.static else 'false'
    pose = ET.SubElement(model, 'pose')
    pose.text = (
        f'{float(obstacle.position.x):.6f} {float(obstacle.position.y):.6f} '
        f'{float(obstacle.position.z):.6f} 0 0 {float(obstacle.position.yaw):.6f}'
    )
    link = ET.SubElement(model, 'link', {'name': 'link'})
    collision = ET.SubElement(link, 'collision', {'name': 'collision'})
    visual = ET.SubElement(link, 'visual', {'name': 'visual'})

    for parent in (collision, visual):
        geometry = ET.SubElement(parent, 'geometry')
        if obstacle.kind == 'cylinder':
            cyl = ET.SubElement(geometry, 'cylinder')
            radius = ET.SubElement(cyl, 'radius')
            radius.text = f'{float(obstacle.radius):.6f}'
            length = ET.SubElement(cyl, 'length')
            length.text = f'{float(obstacle.length):.6f}'
        elif obstacle.kind == 'box':
            box = ET.SubElement(geometry, 'box')
            size = ET.SubElement(box, 'size')
            size.text = ' '.join(f'{float(value):.6f}' for value in obstacle.size or (1.0, 1.0, 1.0))
        else:
            raise ValueError(f'Unsupported obstacle type for SDF export: {obstacle.kind!r}')

    material = ET.SubElement(visual, 'material')
    ambient = ET.SubElement(material, 'ambient')
    diffuse = ET.SubElement(material, 'diffuse')
    if obstacle.kind == 'cylinder':
        ambient.text = '0.72 0.58 0.32 1'
        diffuse.text = '0.76 0.63 0.38 1'
    else:
        ambient.text = '0.46 0.52 0.58 1'
        diffuse.text = '0.58 0.64 0.70 1'
    return model


def render_world_sdf(manifest: ScenarioManifest, template_world_path: Path | str) -> str:
    template_tree = ET.parse(str(template_world_path))
    root = template_tree.getroot()
    world = root.find('world')
    if world is None:
        raise ValueError(f'Could not find <world> in template: {template_world_path}')
    world.set('name', manifest.world_name)

    for model in list(world.findall('model')):
        if model.get('name') != 'ground_plane':
            world.remove(model)

    for obstacle in manifest.obstacles:
        world.append(_obstacle_to_model_element(obstacle))

    xml_text = ET.tostring(root, encoding='unicode')
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_text
