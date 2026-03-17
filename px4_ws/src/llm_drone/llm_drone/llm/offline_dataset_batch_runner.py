#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

from llm_drone.llm.offline_ground_truth_support import (
    DEFAULT_FIXED_ALTITUDE_M,
    DEFAULT_GRID_RESOLUTION_M,
    DEFAULT_LOOKAHEAD_M,
    DEFAULT_MAX_ACCEL_MPS2,
    DEFAULT_MAX_SPEED_MPS,
    DEFAULT_PATH_DT_S,
    DEFAULT_PLANNER_CLEARANCE_M,
    DEFAULT_ROUTE_STEP_M,
    DEFAULT_SPAWN_POSE_ENU,
    PoseENU,
    build_trajectory_artifact,
    env_file_text,
    generate_reference_trajectory,
    scenario_manifest_from_sdf,
    write_json,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORLD_DIR = PACKAGE_ROOT / 'worlds'
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / 'generated' / 'offline_ground_truth'
DEFAULT_PER_RUN_DATASET_DIR = PACKAGE_ROOT / 'dataset' / 'offline_ground_truth_runs'
DEFAULT_MERGED_CSV = PACKAGE_ROOT / 'dataset' / 'offline_ground_truth_dataset.csv'
DEFAULT_PX4_DIR = Path('/home/prachit/PX4-Autopilot')
DEFAULT_LAUNCH_SCRIPT = DEFAULT_PX4_DIR / 'Tools' / 'simulation' / 'gz' / 'launch_obstacle_avoidance_x500.sh'
MISSION_READY_MARKER = 'Drone is airborne and in OFFBOARD mode.'
DATASET_DONE_MARKER = 'Replay reached the end of the reference trajectory'
CLEANUP_WAIT_S = 3.0


@dataclass
class GeneratedArtifacts:
    world_path: Path
    manifest_path: Path
    trajectory_path: Path
    env_path: Path
    manifest: dict


class ManagedProcess:
    def __init__(
        self,
        *,
        name: str,
        cmd: list[str],
        env: dict[str, str],
        cwd: Path | None,
        log_path: Path,
    ) -> None:
        self.name = name
        self.cmd = [str(token) for token in cmd]
        self.env = dict(env)
        self.cwd = None if cwd is None else str(cwd)
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self._queue: Queue[str] = Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = self.log_path.open('w', encoding='utf-8')
        self.process = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def _reader() -> None:
            assert self.process is not None
            try:
                if self.process.stdout is None:
                    return
                for line in self.process.stdout:
                    log_fp.write(line)
                    log_fp.flush()
                    self._queue.put(line.rstrip('\n'))
            finally:
                log_fp.flush()
                log_fp.close()

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()

    def poll(self) -> int | None:
        if self.process is None:
            return None
        return self.process.poll()

    def wait_for_output(self, needle: str, timeout_s: float) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f'{self.name} exited early with code {self.process.returncode}')
            try:
                line = self._queue.get(timeout=0.25)
            except Empty:
                continue
            if needle in line:
                return
        raise TimeoutError(f'Timed out waiting for {self.name} to emit: {needle!r}')

    def stop(self, grace_s: float = 8.0, sig: int = signal.SIGINT) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + float(grace_s)
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                break
            time.sleep(0.2)
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    return
                self.process.wait(timeout=5.0)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Batch-generate offline dataset rows from SDF worlds using PX4/Gazebo simulation.',
    )
    parser.add_argument('--world-dir', default=str(DEFAULT_WORLD_DIR))
    parser.add_argument('--world-glob', default='*.sdf')
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--per-run-dataset-dir', default=str(DEFAULT_PER_RUN_DATASET_DIR))
    parser.add_argument('--merged-csv', default=str(DEFAULT_MERGED_CSV))
    parser.add_argument('--px4-dir', default=str(DEFAULT_PX4_DIR))
    parser.add_argument('--launch-script', default=str(DEFAULT_LAUNCH_SCRIPT))
    parser.add_argument('--px4-sim-model', default='gz_x500_depth')
    parser.add_argument('--depth-topic', default='/depth_camera')
    parser.add_argument('--max-worlds', type=int, default=0)
    parser.add_argument('--spawn-z', type=float, default=float(DEFAULT_SPAWN_POSE_ENU[2]))
    parser.add_argument('--fixed-altitude-m', type=float, default=DEFAULT_FIXED_ALTITUDE_M)
    parser.add_argument('--grid-resolution-m', type=float, default=DEFAULT_GRID_RESOLUTION_M)
    parser.add_argument('--planner-clearance-m', type=float, default=DEFAULT_PLANNER_CLEARANCE_M)
    parser.add_argument('--dt-s', type=float, default=DEFAULT_PATH_DT_S)
    parser.add_argument('--route-step-m', type=float, default=DEFAULT_ROUTE_STEP_M)
    parser.add_argument('--lookahead-distance-m', type=float, default=DEFAULT_LOOKAHEAD_M)
    parser.add_argument('--max-speed-mps', type=float, default=DEFAULT_MAX_SPEED_MPS)
    parser.add_argument('--max-accel-mps2', type=float, default=DEFAULT_MAX_ACCEL_MPS2)
    parser.add_argument('--mission-ready-timeout-s', type=float, default=180.0)
    parser.add_argument('--post-ready-settle-s', type=float, default=5.0)
    parser.add_argument('--dataset-timeout-s', type=float, default=900.0)
    parser.add_argument('--process-stop-timeout-s', type=float, default=10.0)
    parser.add_argument('--start-tolerance-m', type=float, default=0.7)
    parser.add_argument('--waypoint-acceptance-radius-m', type=float, default=0.5)
    parser.add_argument('--playback-speed-scale', type=float, default=1.0)
    parser.add_argument('--skip-existing-csv', action='store_true')
    parser.add_argument('--reset-merged-csv', action='store_true')
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--enable-gui', action='store_true')
    parser.add_argument('--enable-video-recording', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(argv)


def _iter_worlds(world_dir: Path, world_glob: str, max_worlds: int) -> list[Path]:
    worlds = sorted(world_dir.glob(world_glob))
    if max_worlds > 0:
        return worlds[:max_worlds]
    return worlds


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('ROS_LOG_DIR', '/tmp/roslog')
    Path(env['ROS_LOG_DIR']).mkdir(parents=True, exist_ok=True)
    return env


def _stale_process_patterns(px4_dir: Path) -> list[str]:
    return [
        str((px4_dir / 'build' / 'px4_sitl_default' / 'bin' / 'px4').resolve()),
        'gz sim',
        'MicroXRCEAgent udp4 -p 8888',
        'ros_gz_bridge parameter_bridge',
        'llm_drone mission_executor --sim',
        'llm_drone dataset_generator_executor',
    ]


def _matching_pids(pattern: str) -> list[int]:
    result = subprocess.run(
        ['pgrep', '-f', pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        return []
    pids: list[int] = []
    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        pids.append(pid)
    return pids


def _ensure_no_stale_sim_processes(px4_dir: Path) -> None:
    patterns = _stale_process_patterns(px4_dir)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        killed_any = False
        for pattern in patterns:
            for pid in _matching_pids(pattern):
                try:
                    os.kill(pid, sig)
                    killed_any = True
                except ProcessLookupError:
                    continue
        if not killed_any:
            return
        time.sleep(CLEANUP_WAIT_S if sig == signal.SIGTERM else 1.0)


def _generate_artifacts(world_path: Path, args: argparse.Namespace) -> GeneratedArtifacts:
    output_root = Path(args.output_root).expanduser().resolve()
    manifest_dir = output_root / 'manifests'
    trajectory_dir = output_root / 'trajectories'
    env_dir = output_root / 'env'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)

    default_spawn = PoseENU(0.0, 0.0, float(args.spawn_z), 0.0)
    manifest_obj = scenario_manifest_from_sdf(
        world_path,
        fixed_altitude_m=float(args.fixed_altitude_m),
        default_spawn_pose_enu=default_spawn,
        fallback_start_as_spawn_when_missing=True,
    )
    trajectory = generate_reference_trajectory(
        manifest_obj,
        grid_resolution_m=float(args.grid_resolution_m),
        planner_clearance_m=float(args.planner_clearance_m),
        dt_s=float(args.dt_s),
        route_step_m=float(args.route_step_m),
        lookahead_distance_m=float(args.lookahead_distance_m),
        max_speed_mps=float(args.max_speed_mps),
        max_accel_mps2=float(args.max_accel_mps2),
    )
    artifact = build_trajectory_artifact(manifest_obj, trajectory)

    manifest_path = manifest_dir / f'{world_path.stem}.json'
    trajectory_path = trajectory_dir / f'{world_path.stem}.json'
    env_path = env_dir / f'{world_path.stem}.env'
    write_json(manifest_path, manifest_obj.as_dict())
    write_json(trajectory_path, artifact)
    env_path.write_text(env_file_text(manifest_obj), encoding='utf-8')
    return GeneratedArtifacts(
        world_path=world_path,
        manifest_path=manifest_path,
        trajectory_path=trajectory_path,
        env_path=env_path,
        manifest=manifest_obj.as_dict(),
    )


def _merged_csv_path(args: argparse.Namespace) -> Path:
    return Path(args.merged_csv).expanduser().resolve()


def _run_csv_path(world_path: Path, args: argparse.Namespace) -> Path:
    out_dir = Path(args.per_run_dataset_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f'{world_path.stem}.csv'


def _merge_run_csv(run_csv: Path, merged_csv: Path) -> None:
    if not run_csv.exists():
        raise FileNotFoundError(f'Per-run CSV missing: {run_csv}')
    merged_csv.parent.mkdir(parents=True, exist_ok=True)

    with run_csv.open('r', newline='') as src_fp:
        reader = csv.reader(src_fp)
        rows = list(reader)
    if not rows:
        return

    write_header = (not merged_csv.exists()) or merged_csv.stat().st_size == 0
    with merged_csv.open('a', newline='') as dst_fp:
        writer = csv.writer(dst_fp)
        start_idx = 0
        if write_header:
            writer.writerow(rows[0])
            start_idx = 1
        else:
            start_idx = 1
        for row in rows[start_idx:]:
            writer.writerow(row)


def _spawn_pose_from_manifest(manifest: dict) -> PoseENU:
    spawn = manifest.get('spawn_pose_enu') or manifest.get('start_pose_enu')
    if spawn is None:
        raise ValueError('Manifest is missing both spawn_pose_enu and start_pose_enu')
    return PoseENU.from_dict(spawn)


def _gazebo_world_name_from_manifest(manifest: dict) -> str:
    metadata = manifest.get('metadata') or {}
    return str(metadata.get('sdf_world_name', manifest.get('world_name', 'obstacle_avoidance')))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _start_processes(
    *,
    artifacts: GeneratedArtifacts,
    run_csv: Path,
    args: argparse.Namespace,
) -> tuple[ManagedProcess, ManagedProcess, ManagedProcess, ManagedProcess, ManagedProcess]:
    _ensure_no_stale_sim_processes(Path(args.px4_dir).expanduser().resolve())

    spawn_pose = _spawn_pose_from_manifest(artifacts.manifest)
    gazebo_world_name = _gazebo_world_name_from_manifest(artifacts.manifest)
    world_run_dir = Path(args.output_root).expanduser().resolve() / 'batch_logs' / artifacts.world_path.stem
    env = _base_env()

    launcher_env = dict(env)
    launcher_env['WORLD_FILE'] = str(artifacts.world_path)
    launcher_env['PX4_GZ_WORLD'] = gazebo_world_name
    launcher_env['PX4_GZ_MODEL_POSE'] = (
        f'{spawn_pose.x:.3f},{spawn_pose.y:.3f},{spawn_pose.z:.3f},0,0,{spawn_pose.yaw:.3f}'
    )
    launcher_env['PX4_SIM_MODEL'] = str(args.px4_sim_model)
    launcher_env['ENABLE_GZ_GUI'] = '1' if args.enable_gui else '0'
    launcher_env['ENABLE_GZ_VIDEO_RECORDING'] = '1' if args.enable_video_recording else '0'

    launcher = ManagedProcess(
        name='px4_gazebo',
        cmd=[str(Path(args.launch_script).expanduser().resolve())],
        env=launcher_env,
        cwd=Path(args.px4_dir).expanduser().resolve(),
        log_path=world_run_dir / 'px4_gazebo.log',
    )
    xrce_agent = ManagedProcess(
        name='microxrce_agent',
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        env=env,
        cwd=None,
        log_path=world_run_dir / 'microxrce_agent.log',
    )
    bridge = ManagedProcess(
        name='ros_gz_bridge',
        cmd=[
            'ros2',
            'run',
            'ros_gz_bridge',
            'parameter_bridge',
            f'{args.depth_topic}@sensor_msgs/msg/Image[gz.msgs.Image',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        env=env,
        cwd=None,
        log_path=world_run_dir / 'ros_gz_bridge.log',
    )
    mission = ManagedProcess(
        name='mission_executor',
        cmd=['ros2', 'run', 'llm_drone', 'mission_executor', '--sim'],
        env=env,
        cwd=None,
        log_path=world_run_dir / 'mission_executor.log',
    )
    dataset = ManagedProcess(
        name='dataset_generator_executor',
        cmd=[
            'ros2',
            'run',
            'llm_drone',
            'dataset_generator_executor',
            '--ros-args',
            '-p',
            f'trajectory_json:={artifacts.trajectory_path}',
            '-p',
            f'output_csv:={run_csv}',
            '-p',
            f'start_tolerance_m:={args.start_tolerance_m}',
            '-p',
            f'waypoint_acceptance_radius_m:={args.waypoint_acceptance_radius_m}',
            '-p',
            f'playback_speed_scale:={args.playback_speed_scale}',
        ],
        env=env,
        cwd=None,
        log_path=world_run_dir / 'dataset_generator_executor.log',
    )

    launcher.start()
    time.sleep(2.0)
    xrce_agent.start()
    bridge.start()
    time.sleep(2.0)
    mission.start()
    return launcher, xrce_agent, bridge, mission, dataset


def _cleanup_processes(processes: list[ManagedProcess], stop_timeout_s: float, px4_dir: Path) -> None:
    for proc in reversed(processes):
        proc.stop(grace_s=stop_timeout_s)
    _ensure_no_stale_sim_processes(px4_dir)


def _run_single_world(world_path: Path, args: argparse.Namespace) -> None:
    run_csv = _run_csv_path(world_path, args)
    if args.skip_existing_csv and run_csv.exists() and run_csv.stat().st_size > 0:
        _log(f'[skip] {world_path.name}: per-run CSV already exists at {run_csv}')
        return

    _log(f'[ground-truth] {world_path.name}: generating manifest + trajectory JSON')
    artifacts = _generate_artifacts(world_path, args)

    if args.dry_run:
        spawn_pose = _spawn_pose_from_manifest(artifacts.manifest)
        _log(
            f'[dry-run] {world_path.name}: world={world_path} '
            f'gazebo_world={_gazebo_world_name_from_manifest(artifacts.manifest)} '
            f'spawn=({spawn_pose.x:.2f},{spawn_pose.y:.2f},{spawn_pose.z:.2f},{spawn_pose.yaw:.2f}) '
            f'trajectory={artifacts.trajectory_path} output_csv={run_csv}'
        )
        return

    processes: list[ManagedProcess] = []
    try:
        _log(f'[launch] {world_path.name}: starting PX4/Gazebo, bridge, and mission executor')
        launcher, xrce_agent, bridge, mission, dataset = _start_processes(
            artifacts=artifacts,
            run_csv=run_csv,
            args=args,
        )
        processes = [launcher, xrce_agent, bridge, mission, dataset]

        _log(f'[wait] {world_path.name}: waiting for OFFBOARD readiness')
        mission.wait_for_output(MISSION_READY_MARKER, timeout_s=float(args.mission_ready_timeout_s))
        if args.post_ready_settle_s > 0.0:
            _log(f'[wait] {world_path.name}: settling for {args.post_ready_settle_s:.1f}s')
            time.sleep(float(args.post_ready_settle_s))

        _log(f'[dataset] {world_path.name}: starting dataset replay')
        dataset.start()
        dataset.wait_for_output(DATASET_DONE_MARKER, timeout_s=float(args.dataset_timeout_s))
        _log(f'[done] {world_path.name}: replay finished, merging CSV')

        merged_csv = _merged_csv_path(args)
        _merge_run_csv(run_csv, merged_csv)
    finally:
        _cleanup_processes(
            [proc for proc in processes if proc.process is not None],
            args.process_stop_timeout_s,
            Path(args.px4_dir).expanduser().resolve(),
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    world_dir = Path(args.world_dir).expanduser().resolve()
    if not world_dir.exists():
        raise FileNotFoundError(f'World directory does not exist: {world_dir}')

    launch_script = Path(args.launch_script).expanduser().resolve()
    if not launch_script.exists():
        raise FileNotFoundError(f'Launch script does not exist: {launch_script}')

    merged_csv = _merged_csv_path(args)
    if args.reset_merged_csv and merged_csv.exists():
        merged_csv.unlink()

    worlds = _iter_worlds(world_dir, args.world_glob, args.max_worlds)
    if not worlds:
        _log(f'No SDF worlds matched {args.world_glob!r} under {world_dir}')
        return 0

    failures: list[tuple[Path, str]] = []
    for world_path in worlds:
        try:
            _run_single_world(world_path, args)
        except Exception as exc:
            failures.append((world_path, str(exc)))
            _log(f'[error] {world_path.name}: {exc}')
            if not args.continue_on_error:
                break

    if failures:
        _log('Failures:')
        for world_path, message in failures:
            _log(f'  - {world_path.name}: {message}')
        return 1

    _log('Batch dataset generation completed successfully.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
