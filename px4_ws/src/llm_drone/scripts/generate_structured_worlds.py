#!/usr/bin/env python3
"""Generate structured SDF worlds and simple 2D preview plots."""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llm_drone.llm.offline_ground_truth_support import scenario_manifest_from_sdf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_WORLD = REPO_ROOT / "worlds" / "run1.sdf"
DEFAULT_WORLD_DIR = REPO_ROOT / "worlds"
DEFAULT_PLOT_DIR = DEFAULT_WORLD_DIR / "plots"
DEFAULT_DISTRIBUTION_PATH = REPO_ROOT / "resource" / "world_generation_distribution.txt"
WORLD_NAME = "obstacle_avoidance"
FIXED_ALTITUDE_M = 2.5
MARKER_Z_M = 2.0
GOAL_MARKER_Z_M = 2.38
CYLINDER_LENGTH_M = 3.2
GOAL_GATE_HEIGHT_M = 5.0
GOAL_GATE_WIDTH_M = 2.0
GOAL_GATE_THICKNESS_M = 0.5
WORLD_X_MIN = -2.0
WORLD_X_MAX = 34.0


BUCKET_NAME_MAP = {
    "easy corridor": "easy_corridor",
    "offset-start/offset-goal": "offset_start_goal",
    "bottleneck": "bottleneck",
    "cluttered but traversable": "cluttered",
    "deceptive-gap": "deceptive_gap",
}


@dataclass(frozen=True)
class BucketShare:
    key: str
    label: str
    percent: int


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float
    length: float = CYLINDER_LENGTH_M


@dataclass(frozen=True)
class WorldConfig:
    bucket_key: str
    start_x: float
    start_y: float
    goal_x: float
    goal_y: float
    lane_half_width: float
    goal_gate_x: float
    goal_gate_half_gap: float
    obstacles: tuple[Obstacle, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-index", type=int, default=20)
    parser.add_argument("--end-index", type=int, default=220)
    parser.add_argument("--seed", type=int, default=20260317)
    parser.add_argument("--template-world", type=Path, default=DEFAULT_TEMPLATE_WORLD)
    parser.add_argument("--world-dir", type=Path, default=DEFAULT_WORLD_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--distribution-path", type=Path, default=DEFAULT_DISTRIBUTION_PATH)
    return parser.parse_args()


def parse_distribution(path: Path) -> list[BucketShare]:
    shares: list[BucketShare] = []
    pattern = re.compile(r"^-\s*(\d+)%\s+(.+?)\s+worlds\s*$", re.IGNORECASE)
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        percent = int(match.group(1))
        label = match.group(2).strip().lower()
        bucket_key = BUCKET_NAME_MAP.get(label)
        if bucket_key is None:
            continue
        shares.append(BucketShare(key=bucket_key, label=label, percent=percent))
    if not shares:
        raise ValueError(f"Could not parse world bucket percentages from {path}")
    return shares


def build_bucket_sequence(total_worlds: int, shares: list[BucketShare], rng: random.Random) -> list[str]:
    raw_counts = [total_worlds * share.percent / 100.0 for share in shares]
    counts = [math.floor(value) for value in raw_counts]
    remaining = total_worlds - sum(counts)
    ranked = sorted(
        range(len(shares)),
        key=lambda idx: (raw_counts[idx] - counts[idx], shares[idx].percent),
        reverse=True,
    )
    for idx in ranked[:remaining]:
        counts[idx] += 1

    sequence: list[str] = []
    for share, count in zip(shares, counts):
        sequence.extend([share.key] * count)
    rng.shuffle(sequence)
    return sequence


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def distance_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def obstacle_clear_of_points(
    obstacle: Obstacle,
    protected_points: list[tuple[float, float, float]],
    existing: list[Obstacle],
) -> bool:
    for px, py, clearance in protected_points:
        if distance_xy(obstacle.x, obstacle.y, px, py) < obstacle.radius + clearance:
            return False
    for other in existing:
        min_dist = obstacle.radius + other.radius + 0.8
        if distance_xy(obstacle.x, obstacle.y, other.x, other.y) < min_dist:
            return False
    return True


def make_obstacle(
    x: float,
    y: float,
    radius: float,
    lane_half_width: float,
    protected_points: list[tuple[float, float, float]],
    existing: list[Obstacle],
) -> Obstacle | None:
    if x < 2.5 or x > 28.8:
        return None
    y_limit = lane_half_width - radius - 0.45
    if y_limit <= 0.4:
        return None
    if abs(y) > y_limit:
        return None
    candidate = Obstacle(x=float(x), y=float(y), radius=float(radius))
    if not obstacle_clear_of_points(candidate, protected_points, existing):
        return None
    return candidate


def place_random_obstacles(
    *,
    rng: random.Random,
    count: int,
    lane_half_width: float,
    x_bounds: tuple[float, float],
    y_mode: str,
    protected_points: list[tuple[float, float, float]],
    existing: list[Obstacle] | None = None,
) -> list[Obstacle]:
    obstacles = list(existing or [])
    target_count = len(obstacles) + count
    attempts = 0
    while len(obstacles) < target_count and attempts < count * 250:
        attempts += 1
        x = rng.uniform(*x_bounds)
        radius = rng.uniform(0.32, 0.60)
        if y_mode == "edge":
            sign = rng.choice((-1.0, 1.0))
            y = sign * rng.uniform(lane_half_width * 0.48, lane_half_width * 0.78)
        elif y_mode == "offset":
            y = rng.uniform(-lane_half_width * 0.70, lane_half_width * 0.70)
            if abs(y) < 1.2:
                y += math.copysign(1.2, y if abs(y) > 1e-6 else rng.choice((-1.0, 1.0)))
        else:
            y = rng.uniform(-lane_half_width * 0.78, lane_half_width * 0.78)

        candidate = make_obstacle(
            x=x,
            y=y,
            radius=radius,
            lane_half_width=lane_half_width,
            protected_points=protected_points,
            existing=obstacles,
        )
        if candidate is not None:
            obstacles.append(candidate)
    return obstacles


def build_easy_corridor(rng: random.Random) -> WorldConfig:
    lane_half_width = rng.uniform(6.1, 6.8)
    start_y = rng.uniform(-0.8, 0.8)
    goal_y = rng.uniform(-0.8, 0.8)
    start_x = rng.uniform(0.3, 1.2)
    goal_x = rng.uniform(28.2, 30.0)
    protected = [(start_x, start_y, 4.3), (goal_x, goal_y, 3.1)]
    obstacles = place_random_obstacles(
        rng=rng,
        count=rng.randint(3, 5),
        lane_half_width=lane_half_width,
        x_bounds=(4.5, goal_x - 4.0),
        y_mode="edge",
        protected_points=protected,
    )
    return WorldConfig(
        bucket_key="easy_corridor",
        start_x=start_x,
        start_y=start_y,
        goal_x=goal_x,
        goal_y=goal_y,
        lane_half_width=lane_half_width,
        goal_gate_x=goal_x + 3.1,
        goal_gate_half_gap=2.7,
        obstacles=tuple(obstacles),
    )


def build_offset_start_goal(rng: random.Random) -> WorldConfig:
    lane_half_width = rng.uniform(5.6, 6.5)
    start_side = rng.choice((-1.0, 1.0))
    goal_side = -start_side if rng.random() < 0.75 else start_side
    start_y = start_side * rng.uniform(1.6, lane_half_width - 1.0)
    goal_y = goal_side * rng.uniform(1.4, lane_half_width - 1.1)
    start_x = rng.uniform(0.4, 1.4)
    goal_x = rng.uniform(28.4, 30.8)
    protected = [(start_x, start_y, 4.1), (goal_x, goal_y, 3.1)]
    obstacles = place_random_obstacles(
        rng=rng,
        count=rng.randint(4, 6),
        lane_half_width=lane_half_width,
        x_bounds=(4.8, goal_x - 4.1),
        y_mode="offset",
        protected_points=protected,
    )
    return WorldConfig(
        bucket_key="offset_start_goal",
        start_x=start_x,
        start_y=start_y,
        goal_x=goal_x,
        goal_y=goal_y,
        lane_half_width=lane_half_width,
        goal_gate_x=goal_x + 3.0,
        goal_gate_half_gap=2.6,
        obstacles=tuple(obstacles),
    )


def build_bottleneck(rng: random.Random) -> WorldConfig:
    lane_half_width = rng.uniform(5.8, 6.4)
    start_y = rng.uniform(-1.0, 1.0)
    goal_y = rng.uniform(-1.0, 1.0)
    start_x = rng.uniform(0.4, 1.2)
    goal_x = rng.uniform(28.0, 30.4)
    protected = [(start_x, start_y, 4.2), (goal_x, goal_y, 3.1)]
    obstacles: list[Obstacle] = []
    x_candidates = [rng.uniform(11.0, 14.0), rng.uniform(18.0, 22.5)]
    if rng.random() < 0.45:
        x_candidates = x_candidates[:1]
    for x_mid in x_candidates:
        gap_center = rng.uniform(-1.2, 1.2)
        gap_width = rng.uniform(2.4, 3.1)
        radius_a = rng.uniform(0.48, 0.72)
        radius_b = rng.uniform(0.48, 0.72)
        top_y = gap_center + gap_width * 0.5 + radius_a + 0.25
        bottom_y = gap_center - gap_width * 0.5 - radius_b - 0.25
        for y_pos, radius in ((top_y, radius_a), (bottom_y, radius_b)):
            candidate = make_obstacle(
                x=x_mid,
                y=y_pos,
                radius=radius,
                lane_half_width=lane_half_width,
                protected_points=protected,
                existing=obstacles,
            )
            if candidate is not None:
                obstacles.append(candidate)
    extra_count = max(2, rng.randint(2, 4))
    obstacles = place_random_obstacles(
        rng=rng,
        count=extra_count,
        lane_half_width=lane_half_width,
        x_bounds=(5.0, goal_x - 4.2),
        y_mode="offset",
        protected_points=protected,
        existing=obstacles,
    )
    return WorldConfig(
        bucket_key="bottleneck",
        start_x=start_x,
        start_y=start_y,
        goal_x=goal_x,
        goal_y=goal_y,
        lane_half_width=lane_half_width,
        goal_gate_x=goal_x + 3.2,
        goal_gate_half_gap=2.5,
        obstacles=tuple(obstacles),
    )


def build_cluttered(rng: random.Random) -> WorldConfig:
    lane_half_width = rng.uniform(5.9, 6.6)
    start_y = rng.uniform(-1.3, 1.3)
    goal_y = rng.uniform(-1.3, 1.3)
    start_x = rng.uniform(0.4, 1.3)
    goal_x = rng.uniform(28.0, 30.3)
    protected = [(start_x, start_y, 4.2), (goal_x, goal_y, 3.2)]
    obstacles = place_random_obstacles(
        rng=rng,
        count=rng.randint(7, 10),
        lane_half_width=lane_half_width,
        x_bounds=(4.6, goal_x - 4.0),
        y_mode="full",
        protected_points=protected,
    )
    return WorldConfig(
        bucket_key="cluttered",
        start_x=start_x,
        start_y=start_y,
        goal_x=goal_x,
        goal_y=goal_y,
        lane_half_width=lane_half_width,
        goal_gate_x=goal_x + 3.0,
        goal_gate_half_gap=2.6,
        obstacles=tuple(obstacles),
    )


def build_deceptive_gap(rng: random.Random) -> WorldConfig:
    lane_half_width = rng.uniform(5.7, 6.4)
    start_y = rng.uniform(-0.9, 0.9)
    goal_y = rng.uniform(-0.9, 0.9)
    start_x = rng.uniform(0.4, 1.2)
    goal_x = rng.uniform(28.2, 30.5)
    protected = [(start_x, start_y, 4.2), (goal_x, goal_y, 3.2)]
    obstacles: list[Obstacle] = []
    x_series = [rng.uniform(8.0, 10.0), rng.uniform(13.0, 15.0), rng.uniform(18.0, 21.0)]
    bias = rng.choice((-1.0, 1.0))
    y_series = [bias * 2.0, -bias * 1.0, bias * 0.5]
    for x_pos, y_pos in zip(x_series, y_series):
        radius = rng.uniform(0.45, 0.72)
        candidate = make_obstacle(
            x=x_pos,
            y=y_pos,
            radius=radius,
            lane_half_width=lane_half_width,
            protected_points=protected,
            existing=obstacles,
        )
        if candidate is not None:
            obstacles.append(candidate)
    obstacles = place_random_obstacles(
        rng=rng,
        count=rng.randint(4, 6),
        lane_half_width=lane_half_width,
        x_bounds=(4.5, goal_x - 4.2),
        y_mode="full",
        protected_points=protected,
        existing=obstacles,
    )
    return WorldConfig(
        bucket_key="deceptive_gap",
        start_x=start_x,
        start_y=start_y,
        goal_x=goal_x,
        goal_y=goal_y,
        lane_half_width=lane_half_width,
        goal_gate_x=goal_x + 3.1,
        goal_gate_half_gap=2.5,
        obstacles=tuple(obstacles),
    )


BUCKET_BUILDERS = {
    "easy_corridor": build_easy_corridor,
    "offset_start_goal": build_offset_start_goal,
    "bottleneck": build_bottleneck,
    "cluttered": build_cluttered,
    "deceptive_gap": build_deceptive_gap,
}


def ensure_world_valid(config: WorldConfig) -> None:
    if not config.obstacles:
        raise ValueError(f"{config.bucket_key}: generated no obstacles")
    if config.goal_x <= config.start_x + 8.0:
        raise ValueError(f"{config.bucket_key}: goal too close to start")
    if abs(config.start_y) >= config.lane_half_width - 0.8:
        raise ValueError(f"{config.bucket_key}: start too close to wall")
    if abs(config.goal_y) >= config.lane_half_width - 0.8:
        raise ValueError(f"{config.bucket_key}: goal too close to wall")


def set_pose(model: ET.Element, x: float, y: float, z: float, yaw: float = 0.0) -> None:
    pose = model.find("pose")
    if pose is None:
        pose = ET.SubElement(model, "pose")
    pose.text = f"{x:.6f} {y:.6f} {z:.6f} 0 0 {yaw:.6f}"


def set_box_size(model: ET.Element, size_xyz: tuple[float, float, float]) -> None:
    size_text = f"{size_xyz[0]:.6f} {size_xyz[1]:.6f} {size_xyz[2]:.6f}"
    for path in ("./link/collision/geometry/box/size", "./link/visual/geometry/box/size"):
        size_elem = model.find(path)
        if size_elem is not None:
            size_elem.text = size_text


def configure_cylinder_model(model: ET.Element, obstacle: Obstacle, name: str) -> None:
    model.set("name", name)
    set_pose(model, obstacle.x, obstacle.y, obstacle.length * 0.5, 0.0)
    for path in ("./link/collision/geometry/cylinder/radius", "./link/visual/geometry/cylinder/radius"):
        elem = model.find(path)
        if elem is not None:
            elem.text = f"{obstacle.radius:.6f}"
    for path in ("./link/collision/geometry/cylinder/length", "./link/visual/geometry/cylinder/length"):
        elem = model.find(path)
        if elem is not None:
            elem.text = f"{obstacle.length:.6f}"


def load_template_parts(template_world: Path) -> tuple[ET.ElementTree, dict[str, ET.Element]]:
    tree = ET.parse(str(template_world))
    world = tree.getroot().find("world")
    if world is None:
        raise ValueError(f"Template world missing <world>: {template_world}")
    parts: dict[str, ET.Element] = {}
    for model in world.findall("model"):
        name = model.get("name", "")
        if name in {
            "lane_left_wall",
            "lane_right_wall",
            "start_marker",
            "goal_marker",
            "goal_gate_left",
            "goal_gate_right",
        } or name.startswith("block_"):
            parts[name] = copy.deepcopy(model)
    if "block_01" not in parts:
        raise ValueError(f"Template world missing block_01 model: {template_world}")
    return tree, parts


def write_world_from_config(
    *,
    template_world: Path,
    output_path: Path,
    config: WorldConfig,
) -> None:
    base_tree, template_parts = load_template_parts(template_world)
    root = base_tree.getroot()
    world = root.find("world")
    if world is None:
        raise ValueError(f"Could not find <world> in template: {template_world}")
    world.set("name", WORLD_NAME)

    for model in list(world.findall("model")):
        name = model.get("name", "")
        if name.startswith("block_"):
            world.remove(model)

    lane_length = clamp(config.goal_gate_x - config.start_x + 3.0, 34.0, 40.0)
    lane_center_x = 0.5 * (config.start_x + config.goal_gate_x)
    lane_z = GOAL_GATE_HEIGHT_M * 0.5

    lane_left = world.find("./model[@name='lane_left_wall']")
    lane_right = world.find("./model[@name='lane_right_wall']")
    start_marker = world.find("./model[@name='start_marker']")
    goal_marker = world.find("./model[@name='goal_marker']")
    goal_gate_left = world.find("./model[@name='goal_gate_left']")
    goal_gate_right = world.find("./model[@name='goal_gate_right']")
    if None in {lane_left, lane_right, start_marker, goal_marker, goal_gate_left, goal_gate_right}:
        raise ValueError(f"Template world is missing required named models: {template_world}")

    set_pose(lane_left, lane_center_x, config.lane_half_width, lane_z)
    set_pose(lane_right, lane_center_x, -config.lane_half_width, lane_z)
    set_box_size(lane_left, (lane_length, 0.5, GOAL_GATE_HEIGHT_M))
    set_box_size(lane_right, (lane_length, 0.5, GOAL_GATE_HEIGHT_M))

    set_pose(start_marker, config.start_x, config.start_y, MARKER_Z_M)
    set_pose(goal_marker, config.goal_x, config.goal_y, GOAL_MARKER_Z_M)
    set_pose(goal_gate_left, config.goal_gate_x, config.goal_y + config.goal_gate_half_gap, lane_z)
    set_pose(goal_gate_right, config.goal_gate_x, config.goal_y - config.goal_gate_half_gap, lane_z)
    set_box_size(goal_gate_left, (GOAL_GATE_THICKNESS_M, GOAL_GATE_WIDTH_M, GOAL_GATE_HEIGHT_M))
    set_box_size(goal_gate_right, (GOAL_GATE_THICKNESS_M, GOAL_GATE_WIDTH_M, GOAL_GATE_HEIGHT_M))

    start_index = next(
        idx for idx, child in enumerate(list(world))
        if child.tag == "model" and child.get("name") == "start_marker"
    )
    block_template = template_parts["block_01"]
    insert_at = start_index
    for idx, obstacle in enumerate(config.obstacles, start=1):
        block = copy.deepcopy(block_template)
        configure_cylinder_model(block, obstacle, f"block_{idx:02d}")
        world.insert(insert_at, block)
        insert_at += 1

    ET.indent(base_tree, space="  ")
    output_path.write_text(
        ET.tostring(root, encoding="unicode"),
        encoding="utf-8",
    )


def plot_world(sdf_path: Path, plot_path: Path) -> None:
    manifest = scenario_manifest_from_sdf(sdf_path, fixed_altitude_m=FIXED_ALTITUDE_M)
    obstacle_points = [
        obstacle for obstacle in manifest.obstacles
        if obstacle.name.startswith("block_")
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    if obstacle_points:
        ax.scatter(
            [ob.position.x for ob in obstacle_points],
            [ob.position.y for ob in obstacle_points],
            c="#888888",
            s=48,
            label="Obstacles",
        )
    ax.scatter(
        [manifest.start_pose_enu.x],
        [manifest.start_pose_enu.y],
        c="#2ca02c",
        marker="*",
        s=220,
        label="Start",
    )
    ax.scatter(
        [manifest.goal_pose_enu.x],
        [manifest.goal_pose_enu.y],
        c="#d62728",
        marker="*",
        s=220,
        label="Goal",
    )
    ax.set_title(sdf_path.stem)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(WORLD_X_MIN, WORLD_X_MAX)
    ax.set_ylim(-8.0, 8.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def build_world_config(bucket_key: str, rng: random.Random) -> WorldConfig:
    builder = BUCKET_BUILDERS[bucket_key]
    for _ in range(25):
        config = builder(rng)
        ensure_world_valid(config)
        return config
    raise RuntimeError(f"Failed to generate a valid world for bucket {bucket_key}")


def main() -> int:
    args = parse_args()
    if args.end_index < args.start_index:
        raise ValueError("--end-index must be >= --start-index")

    args.world_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    world_indices = list(range(args.start_index, args.end_index + 1))
    shares = parse_distribution(args.distribution_path)
    bucket_sequence = build_bucket_sequence(len(world_indices), shares, rng)

    for world_idx, bucket_key in zip(world_indices, bucket_sequence):
        world_name = f"run{world_idx}"
        sdf_path = args.world_dir / f"{world_name}.sdf"
        plot_path = args.plot_dir / f"{world_name}.png"
        config = build_world_config(bucket_key, rng)
        write_world_from_config(
            template_world=args.template_world,
            output_path=sdf_path,
            config=config,
        )
        plot_world(sdf_path, plot_path)
        print(f"Generated {sdf_path.name} [{bucket_key}] and {plot_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
