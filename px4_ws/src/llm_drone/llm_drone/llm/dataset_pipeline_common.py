from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path as FsPath
from typing import Any

import numpy as np
from cv_bridge import CvBridge

from llm_drone.llm.llm_prompt_common import (
    DEPTH_HFOV_DEG,
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    DEPTH_VFOV_DEG,
    MPC_DEPTH_SAMPLE_COUNT,
    OBS_FIFO_LEN,
    build_environment_vector,
    compose_user_prompt,
    compose_final_prompt,
    serialize_prompt_bundle,
    translate_vector_to_nlp,
)
from llm_drone.verifier.eval import EvalResult, evaluate_response

LABEL_WAYPOINT_COUNT = 5
LOCAL_PREDICTION_SEQUENCE_TOPIC = '/mpc/local_prediction_sequence'
FRESH_COMMITTED_WAYPOINT_TOPIC = '/mpc/committed_waypoint_fresh'
EXECUTED_COMMITTED_WAYPOINT_TOPIC = '/mpc/committed_waypoint_executed'
GROUND_TRUTH_REFERENCE_PATH_TOPIC = '/ground_truth/reference_path'
GROUND_TRUTH_LABEL_WINDOW_TOPIC = '/ground_truth/label_window'

LABEL_MODE_LOCAL_PREDICTION = 'local_mpc_prediction_window'
LABEL_MODE_EXECUTED = 'executed_committed_waypoint_window'
LABEL_MODE_OFFLINE_GROUND_TRUTH = 'offline_ground_truth_window'

PROMPT_MODE_ENV_NL_ONLY = 'env_nl_only'
PROMPT_MODE_ENV_NL_PLUS_JSON = 'env_nl_plus_json'
PROMPT_MODE_FULL_BUNDLE = 'full_bundle'
PROMPT_MODE_VALUES = (
    PROMPT_MODE_ENV_NL_ONLY,
    PROMPT_MODE_ENV_NL_PLUS_JSON,
    PROMPT_MODE_FULL_BUNDLE,
)

DEPTH_FILTER_MIN_FORWARD_M = 0.35
DEPTH_FILTER_MAX_FORWARD_M = 12.0
DEPTH_FILTER_MAX_LATERAL_M = 4.0
DEPTH_FILTER_MIN_BELOW_BODY_M = -0.25
DEPTH_FILTER_MAX_BELOW_BODY_M = 2.5
DEPTH_FILTER_FLOOR_CLEARANCE_M = 0.20


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def px4_stamp_us_to_sec(stamp_us: int | float) -> float:
    return float(stamp_us) * 1e-6


def quaternion_to_rotation_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return np.eye(3, dtype=float)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def convert_goal_to_ned(goal: np.ndarray | list[float] | tuple[float, float, float], goal_frame: str) -> np.ndarray:
    g = np.asarray(goal, dtype=float)
    gf = str(goal_frame).strip().lower()
    if gf in ('gazebo', 'gazebo_enu', 'enu', 'map'):
        return np.array([g[1], g[0], -g[2]], dtype=float)
    return np.array(g, dtype=float, copy=True)


class LocalObstacleMap:
    def __init__(self, maxlen: int = OBS_FIFO_LEN):
        self._buf: list[np.ndarray] = []
        self._maxlen = int(maxlen)

    def update(self, points_ned: np.ndarray) -> None:
        if points_ned is None or points_ned.shape[0] == 0:
            return
        for pt in points_ned:
            self._buf.append(np.array(pt, dtype=float))
        if len(self._buf) > self._maxlen:
            self._buf = self._buf[-self._maxlen:]

    def nearest(self, x_ref: np.ndarray) -> np.ndarray | None:
        if not self._buf:
            return None
        pts = np.array(self._buf, dtype=float)
        dists = np.linalg.norm(pts - np.asarray(x_ref, dtype=float)[None, :3], axis=1)
        return pts[int(np.argmin(dists))]

    def snapshot(self, max_points: int | None = None) -> np.ndarray:
        if not self._buf:
            return np.zeros((0, 3), dtype=float)
        pts = np.array(self._buf, dtype=float)
        if max_points is not None and pts.shape[0] > max_points:
            pts = pts[-int(max_points):]
        return pts


class DepthToObstacles:
    def __init__(
        self,
        hfov_deg: float = DEPTH_HFOV_DEG,
        vfov_deg: float = DEPTH_VFOV_DEG,
        n_sample: int = MPC_DEPTH_SAMPLE_COUNT,
    ):
        self.hfov = np.deg2rad(float(hfov_deg))
        self.vfov = np.deg2rad(float(vfov_deg))
        self.n_sample = int(n_sample)
        self.rotation_body_camera = np.array(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=float,
        )
        self.translation_body_camera_m = np.array([0.13233, 0.0, 0.26078], dtype=float)

    def depth_to_body_frame(self, depth_img: np.ndarray) -> np.ndarray:
        if depth_img is None or depth_img.ndim != 2:
            return np.zeros((0, 3), dtype=float)
        h, w = depth_img.shape
        total = h * w
        if total <= 0:
            return np.zeros((0, 3), dtype=float)

        fy = (h / 2.0) / np.tan(self.vfov / 2.0)
        fx = (w / 2.0) / np.tan(self.hfov / 2.0)
        cx, cy = w / 2.0, h / 2.0

        idx = np.random.choice(total, min(self.n_sample, total), replace=False)
        rows, cols = np.unravel_index(idx, (h, w))
        depths = depth_img[rows, cols]
        valid = (depths > DEPTH_MIN_M) & (depths < DEPTH_MAX_M) & np.isfinite(depths)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=float)
        rows, cols, depths = rows[valid], cols[valid], depths[valid]

        xc = (cols - cx) / fx * depths
        yc = (rows - cy) / fy * depths
        zc = depths
        pts_cam = np.column_stack([xc, yc, zc])
        return (self.rotation_body_camera @ pts_cam.T).T + self.translation_body_camera_m

    def filter_body_points(self, pts_body: np.ndarray) -> np.ndarray:
        if pts_body is None or pts_body.shape[0] == 0:
            return np.zeros((0, 3), dtype=float)

        pts = np.asarray(pts_body, dtype=float)
        forward = pts[:, 0]
        lateral = pts[:, 1]
        down = pts[:, 2]

        mask = np.isfinite(pts).all(axis=1)
        mask &= forward >= DEPTH_FILTER_MIN_FORWARD_M
        mask &= forward <= DEPTH_FILTER_MAX_FORWARD_M
        mask &= np.abs(lateral) <= DEPTH_FILTER_MAX_LATERAL_M
        mask &= down >= DEPTH_FILTER_MIN_BELOW_BODY_M
        mask &= down <= DEPTH_FILTER_MAX_BELOW_BODY_M

        camera_height_above_ground_m = max(0.0, -float(self.translation_body_camera_m[2]))
        if camera_height_above_ground_m > DEPTH_FILTER_FLOOR_CLEARANCE_M:
            mask &= down <= camera_height_above_ground_m - DEPTH_FILTER_FLOOR_CLEARANCE_M

        filtered = pts[mask]
        if filtered.shape[0] == 0:
            return np.zeros((0, 3), dtype=float)
        return filtered

    @staticmethod
    def body_to_spatial(pts_body: np.ndarray, rotation_ned_body: np.ndarray, pos_ned: np.ndarray) -> np.ndarray:
        if pts_body is None or pts_body.shape[0] == 0:
            return np.zeros((0, 3), dtype=float)
        return (rotation_ned_body @ pts_body.T).T + np.asarray(pos_ned, dtype=float)[None, :]


class SceneSynchronizer:
    def __init__(
        self,
        *,
        logger,
        now_s_fn,
        odom_buffer_size: int = 400,
        scene_buffer_size: int = 400,
        max_odom_sync_dt_s: float = 0.10,
        depth_obstacle_samples: int = MPC_DEPTH_SAMPLE_COUNT,
    ):
        self._logger = logger
        self._now_s_fn = now_s_fn
        self._max_odom_sync_dt_s = float(max_odom_sync_dt_s)
        self.odom_buffer = deque(maxlen=int(odom_buffer_size))
        self.scene_buffer = deque(maxlen=int(scene_buffer_size))
        self.bridge = CvBridge()
        self.depth_to_obstacles = DepthToObstacles(n_sample=int(depth_obstacle_samples))
        self.local_obstacle_map = LocalObstacleMap(OBS_FIFO_LEN)

    def _warn(self, message: str, throttle_duration_sec: float | None = None) -> None:
        try:
            if throttle_duration_sec is None:
                self._logger.warn(message)
            else:
                self._logger.warn(message, throttle_duration_sec=throttle_duration_sec)
        except TypeError:
            self._logger.warn(message)

    def now_s(self) -> float:
        return float(self._now_s_fn())

    @staticmethod
    def find_nearest_by_stamp(buf: deque, stamp_s: float, max_dt_s: float):
        if not buf:
            return None, None
        best = None
        best_dt = None
        for item in buf:
            dt = abs(float(item['stamp_s']) - float(stamp_s))
            if best_dt is None or dt < best_dt:
                best, best_dt = item, dt
        if best_dt is None or best_dt > max_dt_s:
            return None, best_dt
        return best, best_dt

    def push_odometry(
        self,
        *,
        position: np.ndarray,
        velocity: np.ndarray,
        rotation_ned_body: np.ndarray,
        stamp_s: float | None = None,
    ) -> dict[str, Any]:
        sample = {
            'stamp_s': float(self.now_s() if stamp_s is None else stamp_s),
            'position': np.array(position, dtype=float, copy=True),
            'velocity': np.array(velocity, dtype=float, copy=True),
            'rotation_ned_body': np.array(rotation_ned_body, dtype=float, copy=True),
        }
        self.odom_buffer.append(sample)
        return sample

    def decode_depth_image(self, msg) -> np.ndarray | None:
        try:
            if msg.encoding in ('16UC1', 'mono16'):
                depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                return depth_mm.astype(np.float32) / 1000.0
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as exc:
            self._warn(f'Depth decode error: {exc}')
            return None

    def push_depth_msg(self, msg):
        depth_image = self.decode_depth_image(msg)
        if depth_image is None:
            return None
        return self.push_depth_image(depth_image)

    def push_depth_image(self, depth_image: np.ndarray, stamp_s: float | None = None):
        depth_stamp_s = float(self.now_s() if stamp_s is None else stamp_s)
        odom, dt = self.find_nearest_by_stamp(self.odom_buffer, depth_stamp_s, self._max_odom_sync_dt_s)
        if odom is None:
            self._warn(
                f'No odom match for depth frame (dt={dt if dt is not None else -1:.3f}s)',
                throttle_duration_sec=2.0,
            )
            return None

        pts_body = self.depth_to_obstacles.depth_to_body_frame(depth_image)
        pts_body = self.depth_to_obstacles.filter_body_points(pts_body)
        pts_ned = self.depth_to_obstacles.body_to_spatial(
            pts_body,
            odom['rotation_ned_body'],
            odom['position'],
        )
        self.local_obstacle_map.update(pts_ned)

        scene = {
            'stamp_s': depth_stamp_s,
            'position': np.array(odom['position'], copy=True),
            'velocity': np.array(odom['velocity'], copy=True),
            'rotation_ned_body': np.array(odom['rotation_ned_body'], copy=True),
            'depth_image': np.array(depth_image, copy=True),
            'latest_depth_obstacles_ned': np.array(pts_ned, copy=True),
            'local_obstacle_snapshot': self.local_obstacle_map.snapshot(),
        }
        self.scene_buffer.append(scene)
        return scene

    def match_scene(self, anchor_stamp_s: float, max_label_sync_dt_s: float):
        return self.find_nearest_by_stamp(self.scene_buffer, anchor_stamp_s, max_label_sync_dt_s)


class DatasetCsvWriter:
    def __init__(self, output_csv: str | FsPath, *, logger=None, waypoint_count: int = LABEL_WAYPOINT_COUNT):
        self.output_path = FsPath(output_csv)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logger
        self._waypoint_count = int(waypoint_count)
        self.rows_written = 0
        self._fp = None
        self._writer = None
        self._open()

    def _open(self) -> None:
        write_header = (not self.output_path.exists()) or self.output_path.stat().st_size == 0
        expected_header_cols = [
            'anchor_stamp_s',
            'scene_stamp_s',
            'sync_dt_ms',
            'label_mode',
            'prompt',
            'label_waypoints_json',
        ]
        self._fp = open(self.output_path, 'a', newline='')
        self._writer = csv.writer(self._fp)
        if write_header:
            header = list(expected_header_cols)
            for idx in range(self._waypoint_count):
                header += [f'wp{idx}_x', f'wp{idx}_y', f'wp{idx}_z']
            self._writer.writerow(header)
            self._fp.flush()
            return

        try:
            with self.output_path.open('r') as fp:
                first_line = fp.readline()
        except OSError:
            first_line = ''
        if self._logger is not None and not all(col in first_line for col in expected_header_cols):
            self._logger.warn(
                'Existing output_csv appears to use an older schema. '
                'Use a fresh output file for the selected label_mode.'
            )

    def write_sample(
        self,
        *,
        anchor_stamp_s: float,
        scene_stamp_s: float,
        label_mode: str,
        prompt: str,
        label_response: dict[str, Any],
        label_waypoints: list[list[float]],
    ) -> None:
        row = [
            f'{float(anchor_stamp_s):.9f}',
            f'{float(scene_stamp_s):.9f}',
            f'{1000.0 * abs(float(anchor_stamp_s) - float(scene_stamp_s)):.3f}',
            str(label_mode),
            prompt,
            json.dumps(label_response, separators=(',', ':')),
        ]
        for waypoint in label_waypoints:
            row += [float(waypoint[0]), float(waypoint[1]), float(waypoint[2])]
        self._writer.writerow(row)
        self._fp.flush()
        self.rows_written += 1

    def close(self) -> None:
        if self._fp is None:
            return
        self._fp.flush()
        self._fp.close()
        self._fp = None
        self._writer = None


def build_prompt_from_scene(
    *,
    goal_ned: np.ndarray,
    scene: dict[str, Any],
    system_prompt: str,
    prompt_mode: str,
    waypoint_count: int = LABEL_WAYPOINT_COUNT,
    depth_sample_count: int = MPC_DEPTH_SAMPLE_COUNT,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    env_vector = build_environment_vector(
        position_ned=scene['position'],
        velocity_ned=scene['velocity'],
        goal_ned=goal_ned,
        depth_image=scene['depth_image'],
        latest_depth_obstacles_ned=scene['latest_depth_obstacles_ned'],
        local_obstacle_snapshot=scene['local_obstacle_snapshot'],
        depth_sample_count=depth_sample_count,
        waypoint_count=waypoint_count,
    )
    if env_vector is None:
        return None, None, None

    env_text = translate_vector_to_nlp(env_vector).strip()
    if prompt_mode == PROMPT_MODE_ENV_NL_ONLY:
        user_prompt = compose_user_prompt(env_text).strip()
        return user_prompt, env_vector, env_text

    user_prompt = compose_final_prompt(env_vector, env_text).strip()
    if prompt_mode == PROMPT_MODE_ENV_NL_PLUS_JSON:
        return user_prompt, env_vector, env_text

    system_prompt_text = str(system_prompt or '').strip()
    return serialize_prompt_bundle(system_prompt_text, user_prompt), env_vector, env_text

def build_label_reasoning(eval_result: EvalResult) -> str:
    metrics = eval_result.metrics or {}
    selected_idx = int(metrics.get('selected_waypoint_index', 0))
    distance_before = float(metrics.get('distance_to_goal_before_m', 0.0))
    distance_selected = float(metrics.get('distance_to_goal_selected_m', 0.0))
    distance_final = float(metrics.get('distance_to_goal_final_m', 0.0))
    progress_selected = float(metrics.get('progress_selected_m', 0.0))
    progress_final = float(metrics.get('progress_final_m', 0.0))
    monotone_goal_progress = bool(metrics.get('monotone_goal_progress', False))
    goal_reached_tolerance_m = float(metrics.get('goal_reached_tolerance_m', 0.35))
    goal_reached_before_sequence = bool(metrics.get('goal_reached_before_sequence', False))
    goal_reached_after_waypoint_index = metrics.get('goal_reached_after_waypoint_index', None)
    kinematic_waypoint_count = int(metrics.get('kinematic_waypoint_count', 0))
    v_max_limit = float(metrics.get('v_max_limit_mps', 15.0))
    a_max_limit = float(metrics.get('a_max_limit_mps2', 8.0))
    min_clearance = metrics.get('min_clearance_m', None)
    clearance_violations = int(metrics.get('clearance_violations', 0))
    clearance_is_proxy = bool(metrics.get('clearance_is_proxy', True))
    clearance_source = str(metrics.get('clearance_source', 'none'))
    clearance_text = 'clearance could not be measured geometrically'
    if not clearance_is_proxy and min_clearance is not None:
        clearance_source_text = 'manifest obstacles' if clearance_source == 'manifest_obstacles' else 'depth obstacles'
        clearance_text = (
            f'minimum clearance {float(min_clearance):.2f} m against {clearance_source_text} '
            f'with {clearance_violations} violating segment(s)'
        )

    kinematic_text = 'passes' if eval_result.kinematic_pass else 'fails'
    progress_text = 'passes' if eval_result.progress_pass else 'fails'
    safety_text = 'passes' if eval_result.safety_pass else 'fails'
    monotone_text = 'monotonic' if monotone_goal_progress else 'non-monotonic'
    if goal_reached_before_sequence:
        terminal_text = (
            f'Goal has already been reached at the reference anchor within {goal_reached_tolerance_m:.2f} m tolerance.'
        )
    elif goal_reached_after_waypoint_index is not None:
        terminal_text = (
            f'Goal will be reached in {int(goal_reached_after_waypoint_index)} waypoint(s) '
            f'within {goal_reached_tolerance_m:.2f} m tolerance.'
        )
    else:
        terminal_text = 'Goal is not reached within this label waypoint window.'

    return (
        f'Selected waypoint {selected_idx} moves the reference anchor from {distance_before:.2f} m to {distance_selected:.2f} m '
        f'from goal immediately and to {distance_final:.2f} m by the final waypoint, so progress is '
        f'{progress_selected:.2f} m selected / {progress_final:.2f} m final and {progress_text} '
        f'with {monotone_text} goal-distance reduction. Kinematics is evaluated on {kinematic_waypoint_count} unique '
        f'label waypoint(s), excluding consecutive duplicate terminal holds, and checks only that the trajectory stays '
        f'within the configured bounds speed <= {v_max_limit:.2f} m/s and acceleration <= {a_max_limit:.2f} m/s^2, '
        f'so kinematics {kinematic_text}. {terminal_text} Safety {safety_text}: {clearance_text}.'
    )


def build_label_response(
    *,
    label_waypoints: list[list[float]],
    label_mode: str,
    scene: dict[str, Any],
    goal_ned: np.ndarray,
    evaluation_start_ned: np.ndarray | None = None,
    obstacle_points_ned: np.ndarray | None = None,
    obstacle_specs=None,
    eval_dt_s: float,
    eval_v_max_mps: float,
    eval_a_max_mps2: float,
    eval_safety_radius_m: float,
) -> dict[str, Any]:
    eval_response = {
        'waypoints': [
            {'x': float(point[0]), 'y': float(point[1]), 'z': float(point[2])}
            for point in label_waypoints
        ],
    }
    eval_result = evaluate_response(
        current_position_ned=np.asarray(
            scene['position'] if evaluation_start_ned is None else evaluation_start_ned,
            dtype=float,
        ),
        goal_ned=np.asarray(goal_ned, dtype=float),
        llm_response_text=json.dumps(eval_response, separators=(',', ':')),
        obstacle_points_ned=np.asarray(
            scene['local_obstacle_snapshot'] if obstacle_points_ned is None else obstacle_points_ned,
            dtype=float,
        ),
        obstacle_specs=obstacle_specs,
        dt=float(eval_dt_s),
        v_max=float(eval_v_max_mps),
        a_max=float(eval_a_max_mps2),
        safety_radius=float(eval_safety_radius_m),
    )
    label_response = {
        'waypoints': [
            {'x': float(point[0]), 'y': float(point[1])}
            for point in label_waypoints
        ],
    }
    del label_mode
    label_response['reasoning'] = build_label_reasoning(eval_result)
    return label_response
