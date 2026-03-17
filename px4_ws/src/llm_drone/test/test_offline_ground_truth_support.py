from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from llm_drone.llm.dataset_pipeline_common import (
    LABEL_MODE_OFFLINE_GROUND_TRUTH,
    PROMPT_MODE_FULL_BUNDLE,
    build_label_response,
    build_prompt_from_scene,
)
from llm_drone.llm.llm_prompt_common import DATASET_PROMPT_FILENAME, resolve_prompt_file, split_prompt_bundle
from llm_drone.llm.offline_ground_truth_support import (
    ObstacleSpec,
    PoseENU,
    ScenarioManifest,
    astar_path,
    build_planning_grid,
    default_lane_walls,
    env_file_text,
    extract_future_waypoints,
    gazebo_enu_to_ned,
    generate_reference_trajectory,
    ned_to_gazebo_enu,
    point_is_collision_free,
    project_pose_to_reference_path,
    render_world_sdf,
    scenario_manifest_from_sdf,
    trajectory_artifact_goal_z_ned,
    trajectory_artifact_path_ned,
)


def _simple_scene() -> dict:
    depth_image = np.full((8, 10), 5.0, dtype=np.float32)
    return {
        'position': np.array([0.0, 0.0, -2.5], dtype=float),
        'velocity': np.array([0.0, 0.0, 0.0], dtype=float),
        'depth_image': depth_image,
        'latest_depth_obstacles_ned': np.zeros((0, 3), dtype=float),
        'local_obstacle_snapshot': np.zeros((0, 3), dtype=float),
    }


def _make_scenario(obstacles: list[ObstacleSpec], goal_x: float = 8.0, goal_y: float = 0.0) -> ScenarioManifest:
    return ScenarioManifest(
        scenario_id='unit',
        world_name='unit_world',
        start_pose_enu=PoseENU(0.0, 0.0, 2.5, 0.0),
        goal_pose_enu=PoseENU(goal_x, goal_y, 2.5, 0.0),
        fixed_altitude_m=2.5,
        obstacles=obstacles,
        spawn_pose_enu=PoseENU(0.0, 0.0, 1.0, 0.0),
    )


def test_enu_ned_round_trip() -> None:
    point_enu = np.array([4.2, -1.7, 2.5], dtype=float)
    point_ned = gazebo_enu_to_ned(*point_enu)
    recovered_enu = ned_to_gazebo_enu(*point_ned)
    assert np.allclose(recovered_enu, point_enu)


def test_astar_path_avoids_inflated_blocker() -> None:
    scenario = _make_scenario([
        ObstacleSpec(
            name='blocker',
            kind='box',
            position=PoseENU(4.0, 0.0, 1.5, 0.0),
            size=(1.5, 2.0, 3.0),
        ),
    ])
    grid = build_planning_grid(scenario, resolution_m=0.25, inflation_m=0.8)
    path = astar_path(grid, np.array([0.0, 0.0], dtype=float), np.array([8.0, 0.0], dtype=float))
    assert path.shape[0] > 2
    assert all(point_is_collision_free(point, scenario.obstacles, inflation_m=0.8 - 1e-6) for point in path)


def test_project_pose_and_extract_future_waypoints() -> None:
    path_ned = np.array([
        [0.0, 0.0, -2.5],
        [1.0, 0.0, -2.5],
        [2.0, 0.0, -2.5],
        [3.0, 0.0, -2.5],
        [4.0, 0.0, -2.5],
        [5.0, 0.0, -2.5],
        [6.0, 0.0, -2.5],
    ], dtype=float)
    idx = project_pose_to_reference_path(path_ned, np.array([1.2, 0.1, -2.5], dtype=float))
    assert idx == 1
    forward_only_idx = project_pose_to_reference_path(
        path_ned,
        np.array([1.2, 0.1, -2.5], dtype=float),
        start_index=3,
    )
    assert forward_only_idx >= 3
    future = extract_future_waypoints(path_ned, idx, count=5)
    assert len(future) == 5
    assert np.allclose(future[0], [2.0, 0.0, -2.5])


def test_full_prompt_bundle_contains_system_prompt_and_user_only() -> None:
    prompt, env_vector, _env_text = build_prompt_from_scene(
        goal_ned=np.array([8.0, 0.0, -2.5], dtype=float),
        scene=_simple_scene(),
        system_prompt='SYSTEM POLICY',
        prompt_mode=PROMPT_MODE_FULL_BUNDLE,
    )
    assert env_vector is not None
    assert prompt is not None
    system_prompt, user_prompt = split_prompt_bundle(prompt, fallback_system_prompt='')
    assert system_prompt == 'SYSTEM POLICY'
    assert 'Environment section (T(v)):' in user_prompt
    assert 'Current position NED' in user_prompt
    assert 'Numerical Environment Vector v' not in prompt


def test_active_prompt_file_resolves_to_llm_prompt2d() -> None:
    prompt_path = resolve_prompt_file(DATASET_PROMPT_FILENAME)
    assert prompt_path.name == 'llm_prompt2d.txt'
    assert prompt_path.exists()


def test_build_label_response_outputs_2d_json() -> None:
    label_waypoints = [
        [1.0, 0.5, -2.5],
        [2.0, 0.8, -2.5],
        [3.0, 1.0, -2.5],
        [4.0, 1.1, -2.5],
        [5.0, 1.0, -2.5],
    ]
    label_response = build_label_response(
        label_waypoints=label_waypoints,
        label_mode=LABEL_MODE_OFFLINE_GROUND_TRUTH,
        scene=_simple_scene(),
        goal_ned=np.array([8.0, 0.0, -2.5], dtype=float),
        eval_dt_s=0.1,
        eval_v_max_mps=15.0,
        eval_a_max_mps2=10.0,
        eval_safety_radius_m=0.5,
    )
    assert list(label_response.keys()) == ['waypoints', 'reasoning']
    assert len(label_response['waypoints']) == 5
    assert set(label_response['waypoints'][0].keys()) == {'x', 'y'}
    assert 'z' not in label_response['waypoints'][0]
    assert label_response['reasoning'].startswith('Eval metrics:')


def test_trajectory_artifact_path_reconstructs_fixed_goal_z() -> None:
    trajectory = {
        'goal_z_ned_m': -2.5,
        'reference_path_ned': [
            {'t': 0.0, 'x': 0.0, 'y': 0.0},
            {'t': 0.1, 'x': 1.0, 'y': 0.5},
        ],
    }
    assert math.isclose(trajectory_artifact_goal_z_ned(trajectory), -2.5)
    points_ned = trajectory_artifact_path_ned(trajectory)
    assert points_ned.shape == (2, 3)
    assert np.allclose(points_ned[:, 2], -2.5)


def test_render_world_sdf_contains_unique_world_and_obstacles(tmp_path) -> None:
    template = tmp_path / 'template.sdf'
    template.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sdf version="1.9"><world name="template_world"><model name="ground_plane"/></world></sdf>\n'
    )
    scenario = _make_scenario(default_lane_walls())
    scenario.world_name = 'obstacle_avoidance_unit'
    xml_text = render_world_sdf(scenario, template)
    root = ET.fromstring(xml_text)
    world = root.find('world')
    assert world is not None
    assert world.get('name') == 'obstacle_avoidance_unit'
    models = {model.get('name') for model in world.findall('model')}
    assert 'ground_plane' in models
    assert 'lane_left_wall' in models
    assert 'lane_right_wall' in models


def test_scenario_manifest_from_sdf_extracts_obstacles_and_goal(tmp_path) -> None:
    sdf_path = tmp_path / 'custom_corridor.sdf'
    sdf_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sdf version="1.9"><world name="custom_world">'
        '<model name="ground_plane"><static>true</static><link name="link"><collision name="collision">'
        '<geometry><plane><normal>0 0 1</normal><size>1 1</size></plane></geometry>'
        '</collision></link></model>'
        '<model name="goal_gate_left"><pose>12 2 2.5 0 0 0</pose><link name="link"><collision name="collision">'
        '<geometry><box><size>0.5 2.0 5.0</size></box></geometry></collision></link></model>'
        '<model name="goal_gate_right"><pose>12 -2 2.5 0 0 0</pose><link name="link"><collision name="collision">'
        '<geometry><box><size>0.5 2.0 5.0</size></box></geometry></collision></link></model>'
        '<model name="pillar"><pose>6 1.5 1.6 0 0 0</pose><link name="link"><collision name="collision">'
        '<geometry><cylinder><radius>0.4</radius><length>3.2</length></cylinder></geometry>'
        '</collision></link></model>'
        '</world></sdf>\n'
    )

    manifest = scenario_manifest_from_sdf(sdf_path, fixed_altitude_m=2.5)

    assert manifest.scenario_id == 'custom_corridor'
    assert manifest.world_name == 'custom_corridor'
    assert manifest.metadata['sdf_world_name'] == 'custom_world'
    assert manifest.metadata['goal_source'] == 'goal_gate_midpoint'
    assert math.isclose(manifest.goal_pose_enu.x, 12.0)
    assert math.isclose(manifest.goal_pose_enu.y, 0.0)
    assert math.isclose(manifest.goal_pose_enu.z, 2.5)
    assert len(manifest.obstacles) == 3
    assert {obstacle.name for obstacle in manifest.obstacles} == {'goal_gate_left', 'goal_gate_right', 'pillar'}


def test_scenario_manifest_from_sdf_uses_start_xy_for_spawn_fallback(tmp_path) -> None:
    sdf_path = tmp_path / 'start_only_world.sdf'
    sdf_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sdf version="1.9"><world name="obstacle_avoidance">'
        '<model name="ground_plane"><static>true</static><link name="link"><collision name="collision">'
        '<geometry><plane><normal>0 0 1</normal><size>1 1</size></plane></geometry>'
        '</collision></link></model>'
        '<model name="start_marker"><pose>4.5 -1.25 2.6 0 0 0.4</pose><link name="link"><collision name="collision">'
        '<geometry><box><size>0.5 0.5 0.2</size></box></geometry></collision></link></model>'
        '<model name="goal_marker"><pose>12.0 1.0 2.5 0 0 0</pose><link name="link"><collision name="collision">'
        '<geometry><box><size>0.5 0.5 0.2</size></box></geometry></collision></link></model>'
        '</world></sdf>\n'
    )

    manifest = scenario_manifest_from_sdf(
        sdf_path,
        fixed_altitude_m=2.5,
        default_spawn_pose_enu=PoseENU(0.0, 0.0, 1.0, 0.0),
    )

    assert manifest.spawn_pose_enu is not None
    assert math.isclose(manifest.spawn_pose_enu.x, 4.5)
    assert math.isclose(manifest.spawn_pose_enu.y, -1.25)
    assert math.isclose(manifest.spawn_pose_enu.z, 1.0)
    assert math.isclose(manifest.spawn_pose_enu.yaw, 0.4)
    assert manifest.metadata['spawn_source'] == 'start_marker_xy_fallback'

    env_text = env_file_text(manifest)
    assert 'export PX4_GZ_WORLD=obstacle_avoidance' in env_text
    assert 'export PX4_GZ_MODEL_POSE="4.500,-1.250,1.000,0,0,0.400"' in env_text


def test_generate_reference_trajectory_smoke_empty_world() -> None:
    scenario = _make_scenario([])
    trajectory = generate_reference_trajectory(
        scenario,
        lookahead_distance_m=1.0,
        max_speed_mps=2.0,
        max_accel_mps2=4.0,
        max_steps=220,
    )
    assert 'z' not in trajectory['reference_path_enu'][0]
    assert math.isclose(float(trajectory['goal_z_enu_m']), 2.5)
    points = trajectory_artifact_path_ned(trajectory)
    assert points.shape[0] > 3
    points_enu_xy = np.array([[row['x'], row['y']] for row in trajectory['reference_path_enu']], dtype=float)
    assert np.linalg.norm(points_enu_xy[-1, :2] - np.array([8.0, 0.0], dtype=float)) <= 0.6
    assert np.allclose(points[:, 2], -2.5)


def test_generate_reference_trajectory_smoke_single_blocker() -> None:
    scenario = _make_scenario([
        ObstacleSpec(
            name='cyl_block',
            kind='cylinder',
            position=PoseENU(4.0, 0.0, 1.6, 0.0),
            radius=0.6,
            length=3.2,
        ),
    ])
    trajectory = generate_reference_trajectory(
        scenario,
        lookahead_distance_m=1.0,
        max_speed_mps=2.0,
        max_accel_mps2=4.0,
        max_steps=260,
    )
    assert 'z' not in trajectory['reference_path_ned'][0]
    points = trajectory_artifact_path_ned(trajectory)
    points_enu_xy = np.array([[row['x'], row['y']] for row in trajectory['reference_path_enu']], dtype=float)
    assert np.linalg.norm(points_enu_xy[-1, :2] - np.array([8.0, 0.0], dtype=float)) <= 0.6
    assert trajectory['optimization_metadata']['planner_method'] == 'astar_initialization_global_qp'
    assert trajectory['optimization_metadata']['accepted_iterations'] >= 1
    assert trajectory['optimization_metadata']['max_slack_m'] <= 1e-3
    assert all(point_is_collision_free(point, scenario.obstacles, inflation_m=0.1) for point in points_enu_xy)


def test_generate_reference_trajectory_smoke_multi_obstacle_corridor() -> None:
    obstacles = default_lane_walls() + [
        ObstacleSpec(
            name='left_block',
            kind='cylinder',
            position=PoseENU(3.0, 1.5, 1.6, 0.0),
            radius=0.5,
            length=3.2,
        ),
        ObstacleSpec(
            name='right_block',
            kind='box',
            position=PoseENU(5.2, -1.2, 1.5, 0.25),
            size=(1.2, 0.9, 3.0),
        ),
    ]
    scenario = _make_scenario(obstacles, goal_x=9.0, goal_y=1.0)
    trajectory = generate_reference_trajectory(
        scenario,
        lookahead_distance_m=1.0,
        max_speed_mps=2.0,
        max_accel_mps2=4.0,
        max_steps=320,
    )
    points = trajectory_artifact_path_ned(trajectory)
    points_enu_xy = np.array([[row['x'], row['y']] for row in trajectory['reference_path_enu']], dtype=float)
    assert np.linalg.norm(points_enu_xy[-1, :2] - np.array([9.0, 1.0], dtype=float)) <= 0.7
    assert trajectory['optimization_metadata']['planner_method'] == 'astar_initialization_global_qp'
    assert trajectory['optimization_metadata']['accepted_iterations'] >= 1
    assert trajectory['optimization_metadata']['max_slack_m'] <= 1e-3
    assert all(point_is_collision_free(point, scenario.obstacles, inflation_m=0.1) for point in points_enu_xy)
    assert math.isclose(float(np.min(points[:, 2])), -2.5)
