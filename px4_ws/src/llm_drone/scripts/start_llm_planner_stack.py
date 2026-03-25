#!/usr/bin/env python3
"""Launch the full LLM planner stack in the same order we use manually.

This script is intentionally verbose and heavily commented because it is meant
to be edited by hand during experiments. The default flow is:

1. Launch the external PX4/Gazebo "master" script that brings up the simulator
   stack (Gazebo, PX4 SITL, MicroDDS/bridge tools, RQT if your script does so).
2. Wait until the PX4 stack is active enough that ROS odometry is flowing on
   `/fmu/out/vehicle_odometry`.
3. Start the ROS mission executor and wait until PX4 reports the vehicle is
   airborne and in OFFBOARD mode.
4. Start a vLLM server hosting the chosen model and wait until the HTTP health
   endpoint responds.
5. Once both the mission side and the vLLM side are ready, launch llm_planner.

If your lab setup changes, the easiest things to tweak are:
- DEFAULT_MASTER_COMMAND
- DEFAULT_MODEL_SPEC
- DEFAULT_MAX_MODEL_LEN
- the ros2 run command used for llm_planner
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


MISSION_READY_MARKER = "Drone is airborne and in OFFBOARD mode."
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MASTER_COMMAND = str(
    (Path.home() / "PX4-Autopilot" / "Tools" / "simulation" / "gz" / "launch_obstacle_avoidance_full_stack.sh").resolve()
)
DEFAULT_MODEL_SPEC = str(REPO_ROOT / "models" / "drone_planner_checkpoints12k" / "final_merged")
DEFAULT_SERVED_MODEL_NAME = "qwen25_7b_drone_planner"
DEFAULT_VLLM_PORT = 8000
DEFAULT_VLLM_HOST = "127.0.0.1"
DEFAULT_API_KEY = "token-abc123"
DEFAULT_MASTER_READY_MARKER = "Opening Terminal 3: MicroXRCEAgent"
DEFAULT_PX4_READY_TOPICS = (
    "/fmu/out/timesync_status",
    "/fmu/out/vehicle_status_v1",
    "/fmu/out/vehicle_odometry",
)
DEFAULT_ODOM_TOPIC = "/fmu/out/vehicle_odometry"
DEFAULT_MISSION_UDP_PORT = 14540
DEFAULT_GOAL_FRAME = "gazebo"
DEFAULT_GOAL_X = 29.0
DEFAULT_GOAL_Y = 0.0
DEFAULT_GOAL_Z = 2.5
# 2048 matches the training-time sequence budget in scripts/train.py and is
# large enough for the current llm_prompt2d.txt + environment text + JSON reply.
# You can raise this later if prompts grow, but using a much larger value than
# needed increases KV-cache memory pressure on the vLLM server.
DEFAULT_MAX_MODEL_LEN = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-command",
        default=os.environ.get("LLM_MASTER_COMMAND", "").strip() or DEFAULT_MASTER_COMMAND,
        help=(
            "Command that launches Gazebo/PX4/MicroDDS/rqt. "
            "Defaults to the PX4 full-stack obstacle avoidance launcher. "
            "Can also be overridden via LLM_MASTER_COMMAND."
        ),
    )
    parser.add_argument("--mission-timeout-s", type=float, default=240.0)
    parser.add_argument("--vllm-timeout-s", type=float, default=300.0)
    parser.add_argument("--shutdown-timeout-s", type=float, default=8.0)
    parser.add_argument(
        "--master-ready-marker",
        default=DEFAULT_MASTER_READY_MARKER,
        help=(
            "Log line from the master launcher that indicates PX4/Gazebo is far "
            "enough along to start mission_executor. The default matches the "
            "current launch_obstacle_avoidance_full_stack.sh output."
        ),
    )
    parser.add_argument("--master-ready-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--master-post-ready-delay-s",
        type=float,
        default=4.0,
        help="Extra settle time after the master ready marker before mission_executor starts.",
    )
    parser.add_argument(
        "--px4-ready-topics",
        default=",".join(DEFAULT_PX4_READY_TOPICS),
        help=(
            "Comma-separated PX4 ROS topics used as an early readiness gate before mission_executor. "
            "The launcher waits until any one of them appears on the real ROS graph."
        ),
    )
    parser.add_argument(
        "--px4-ready-timeout-s",
        type=float,
        default=120.0,
        help="How long to wait for an early PX4 ROS topic to appear before mission_executor is launched.",
    )
    parser.add_argument(
        "--odom-topic",
        default=DEFAULT_ODOM_TOPIC,
        help="ROS odometry topic that must produce a message before llm_planner is launched.",
    )
    parser.add_argument(
        "--odom-timeout-s",
        type=float,
        default=120.0,
        help="How long to wait for the odometry topic to produce a message before llm_planner starts.",
    )
    parser.add_argument(
        "--mission-udp-port",
        type=int,
        default=DEFAULT_MISSION_UDP_PORT,
        help="UDP port that mission_executor's MAVSDK client must be able to bind before launch.",
    )
    parser.add_argument(
        "--port-cleanup-timeout-s",
        type=float,
        default=10.0,
        help="How long to wait for stale MAVSDK/mission processes to release the UDP port.",
    )
    parser.add_argument(
        "--goal-frame",
        default=DEFAULT_GOAL_FRAME,
        help=(
            "Goal frame passed to llm_planner. Use 'gazebo' for ENU world-frame goals "
            "in simulation, or 'ned' if you are already supplying NED coordinates."
        ),
    )
    parser.add_argument("--goal-x", type=float, default=DEFAULT_GOAL_X)
    parser.add_argument("--goal-y", type=float, default=DEFAULT_GOAL_Y)
    parser.add_argument(
        "--goal-z",
        type=float,
        default=DEFAULT_GOAL_Z,
        help=(
            "Goal altitude in the selected goal frame. With the default "
            "--goal-frame gazebo, +2.3 becomes -2.3 in NED before querying the LLM "
            "and before publishing PX4 setpoints."
        ),
    )
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--vllm-host", default=DEFAULT_VLLM_HOST)
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument("--served-model-name", default=DEFAULT_SERVED_MODEL_NAME)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_SPEC,
        help=(
            "Model spec passed to `vllm serve`. This can be either a local path "
            "(default: the merged fine-tuned model under models/drone_planner_checkpoints/final_merged) "
            "or a remote Hugging Face model id such as Qwen/Qwen2.5-1.5B-Instruct."
        ),
    )
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--planner-prompt-file",
        default=os.environ.get("LLM_PLANNER_PROMPT_FILE", "").strip(),
        help=(
            "Optional path to a fixed prompt .txt for llm_planner. "
            "If unset, llm_planner uses its built-in default prompt path. "
            "Can also be overridden via LLM_PLANNER_PROMPT_FILE."
        ),
    )
    return parser.parse_args()


class ManagedProcess:
    """Small helper around subprocess.Popen with live prefixed log streaming.

    We use a lightweight process wrapper instead of tmux here so the script can:
    - stream all child logs into one terminal with a process prefix
    - wait for specific readiness messages
    - shut everything down together on Ctrl+C
    """

    def __init__(self, *, name: str, cmd: str, env: dict[str, str] | None = None) -> None:
        self.name = name
        self.cmd = cmd
        self.env = dict(env or {})
        self.process: subprocess.Popen[str] | None = None
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # Each child process gets a clean bash shell so we can pass normal shell
        # commands (including sourced environments) instead of rebuilding every
        # launch as a giant Python argv list.
        proc_env = os.environ.copy()
        proc_env.update(self.env)
        self.process = subprocess.Popen(
            ["bash", "-lc", self.cmd],
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def _reader() -> None:
            assert self.process is not None
            assert self.process.stdout is not None
            for line in self.process.stdout:
                text = line.rstrip("\n")
                with self._lock:
                    self._lines.append(text)
                print(f"[{self.name}] {text}", flush=True)

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()

    def has_line(self, needle: str) -> bool:
        with self._lock:
            return any(needle in line for line in self._lines)

    def wait_for_output(self, needle: str, timeout_s: float) -> None:
        # Used for mission_executor readiness. We purposely wait on a concrete
        # printed marker instead of a fixed sleep because PX4 startup timing can
        # vary a lot from run to run.
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"{self.name} exited early with code {self.process.returncode}")
            if self.has_line(needle):
                return
            time.sleep(0.25)
        raise TimeoutError(f"Timed out waiting for {self.name} to emit: {needle!r}")

    def stop(self, timeout_s: float) -> None:
        # Graceful first (SIGINT), then SIGTERM, then SIGKILL if the child hangs.
        # Since each child is started in its own session, killpg tears down the
        # whole process tree for that child launcher.
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                return
            time.sleep(0.2)
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


def ros_wrapped(command: str) -> str:
    """Wrap a command in the standard ROS 2 workspace environment.

    We use this for mission_executor and llm_planner so they always start with:
    - ROS Humble sourced
    - this px4_ws sourced
    - ROS_LOG_DIR set to /tmp/roslog
    - PYTHONUNBUFFERED=1 so Python print() calls stream immediately through the
      launcher instead of sitting in a stdio buffer while we wait on readiness
      markers
    """
    return (
        # Do NOT use `set -u` here. ROS setup scripts may reference optional
        # environment variables before assigning them, which causes immediate
        # failure under nounset. We still keep `-e` and `pipefail` for useful
        # error propagation.
        "set -eo pipefail\n"
        "source /opt/ros/humble/setup.bash\n"
        f"cd {WORKSPACE_ROOT}\n"
        "source install/setup.bash\n"
        "mkdir -p /tmp/roslog\n"
        "export ROS_LOG_DIR=/tmp/roslog\n"
        "export PYTHONUNBUFFERED=1\n"
        f"{command}\n"
    )


def workspace_wrapped(command: str) -> str:
    """Run a shell command from the px4_ws root without forcing ROS sourcing.

    The external PX4 full-stack launcher already handles its own environment,
    sleeps, and terminal spawning. Wrapping it in ROS setup was unnecessary and
    caused the `AMENT_TRACE_SETUP_FILES: unbound variable` failure on your server.
    """
    return (
        "set -eo pipefail\n"
        f"cd {WORKSPACE_ROOT}\n"
        f"{command}\n"
    )


def wait_for_http_ready(url: str, timeout_s: float) -> None:
    """Poll the vLLM health endpoint until the server is ready."""
    deadline = time.monotonic() + float(timeout_s)
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if 200 <= int(response.status) < 300:
                    return
        except urllib.error.URLError as exc:
            last_error = exc
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for vLLM readiness at {url} ({last_error})")


def parse_topic_csv(raw_topics: str) -> list[str]:
    return [topic.strip() for topic in str(raw_topics).split(',') if topic.strip()]


def wait_for_ros_topic_presence(topics: list[str], timeout_s: float) -> str:
    """Wait until any requested ROS topic appears on the real ROS graph."""
    if not topics:
        raise ValueError('At least one PX4 readiness topic must be provided')

    probe_cmd = ros_wrapped('ros2 topic list --no-daemon')
    deadline = time.monotonic() + float(timeout_s)
    last_error = ''
    last_px4_topics: list[str] = []
    while time.monotonic() < deadline:
        probe = subprocess.run(
            ["bash", "-lc", probe_cmd],
            text=True,
            capture_output=True,
        )
        if probe.returncode == 0:
            seen_topics = {line.strip() for line in probe.stdout.splitlines() if line.strip()}
            for topic in topics:
                if topic in seen_topics:
                    return topic
            last_px4_topics = sorted(topic for topic in seen_topics if topic.startswith('/fmu/'))
            last_error = ''
        else:
            last_error = (probe.stderr or probe.stdout or '').strip()
        time.sleep(1.0)

    if last_px4_topics:
        preview = ', '.join(last_px4_topics[:10])
        raise TimeoutError(
            'Timed out waiting for PX4 ROS topics to appear. '
            f'Wanted one of {topics!r}; last seen PX4 topics: {preview}'
        )
    raise TimeoutError(
        'Timed out waiting for PX4 ROS topics to appear. '
        f'Wanted one of {topics!r}; last error: {last_error or "no /fmu topics seen"}'
    )


def wait_for_ros_topic_message(topic: str, timeout_s: float) -> None:
    """Wait until a ROS topic produces at least one message."""
    probe_cmd = ros_wrapped(
        f"timeout 5 ros2 topic echo --once {topic} >/tmp/start_llm_planner_stack_odom_probe.log 2>&1"
    )
    deadline = time.monotonic() + float(timeout_s)
    last_exit_code = None
    last_probe_log = ''
    while time.monotonic() < deadline:
        probe = subprocess.run(
            ["bash", "-lc", probe_cmd],
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        last_exit_code = probe.returncode
        if probe.returncode == 0:
            return
        try:
            last_probe_log = Path('/tmp/start_llm_planner_stack_odom_probe.log').read_text().strip()
        except FileNotFoundError:
            last_probe_log = ''
        time.sleep(1.0)
    suffix = f'; last probe log: {last_probe_log}' if last_probe_log else ''
    raise TimeoutError(
        f"Timed out waiting for ROS topic {topic!r} to emit a message "
        f"(last probe exit code {last_exit_code}{suffix})"
    )


def kill_matching_processes(patterns: list[str]) -> None:
    """Best-effort cleanup for stale mission/MAVSDK processes.

    We only do this immediately before launching mission_executor. These are the
    two process families that most often keep UDP 14540 bound after a failed or
    manually interrupted run:
    - ros2 run llm_drone mission_executor
    - mavsdk_server
    """
    for pattern in patterns:
        subprocess.run(
            ["pkill", "-f", pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def udp_port_is_bindable(port: int) -> bool:
    """Return True if we can bind the MAVSDK UDP port locally right now."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", int(port)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def ensure_udp_port_free(port: int, timeout_s: float) -> None:
    """Wait until the mission UDP port is free, failing loudly if not.

    Binding here is only a probe. We immediately close the socket again. The
    real consumer will be MAVSDK inside mission_executor, but this check catches
    stale processes before we start another takeoff cycle.
    """
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if udp_port_is_bindable(port):
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"UDP port {port} is still in use after cleanup. "
        "A stale mission_executor or mavsdk_server is likely still running."
    )


def main() -> int:
    args = parse_args()
    if not args.master_command:
        raise SystemExit(
            "No master command provided. Pass --master-command '<your launcher>' "
            "or set LLM_MASTER_COMMAND."
        )

    # `--model` supports two cases:
    # 1) local merged checkpoint path from train.py
    # 2) a normal model id from Hugging Face / local cache, e.g. Qwen/Qwen2.5-1.5B-Instruct
    model_spec_raw = str(args.model).strip()
    if not model_spec_raw:
        raise ValueError("--model must not be empty")
    local_model_candidate = Path(model_spec_raw).expanduser()
    if local_model_candidate.exists():
        model_spec = str(local_model_candidate.resolve())
    else:
        model_spec = model_spec_raw

    planner_prompt_file = ""
    planner_prompt_raw = str(args.planner_prompt_file).strip()
    if planner_prompt_raw:
        planner_prompt_candidate = Path(planner_prompt_raw).expanduser()
        if not planner_prompt_candidate.exists():
            raise FileNotFoundError(
                f"--planner-prompt-file does not exist: {planner_prompt_candidate}"
            )
        planner_prompt_file = str(planner_prompt_candidate.resolve())

    vllm_url = f"http://{args.vllm_host}:{args.vllm_port}/v1/chat/completions"
    vllm_health_url = f"http://{args.vllm_host}:{args.vllm_port}/health"

    # 1) Master process: your existing PX4/Gazebo full-stack launcher.
    #    This is the main integration point with your local sim setup.
    #    We intentionally do NOT ROS-wrap this anymore because the script already
    #    manages its own startup environment.
    master = ManagedProcess(
        name="master",
        cmd=workspace_wrapped(args.master_command),
    )
    # 2) Mission executor: arms, takes off, and holds OFFBOARD until the planner
    #    starts publishing setpoints.
    mission = ManagedProcess(
        name="mission_executor",
        cmd=ros_wrapped("ros2 run llm_drone mission_executor --sim"),
    )
    planner_tokens = [
        "ros2",
        "run",
        "llm_drone",
        "llm",
        "--ros-args",
        "-p",
        f"goal_x:={args.goal_x}",
        "-p",
        f"goal_y:={args.goal_y}",
        "-p",
        f"goal_z:={args.goal_z}",
        "-p",
        f"goal_frame:={args.goal_frame}",
        "-p",
        "llm_provider:=vllm",
        "-p",
        f"vllm_url:={vllm_url}",
        "-p",
        f"vllm_model:={args.served_model_name}",
        "-p",
        f"vllm_api_key:={args.api_key}",
    ]
    if planner_prompt_file:
        planner_tokens.extend(["-p", f"prompt_file:={planner_prompt_file}"])

    # 3) Planner process: this stays dormant until both mission and vLLM are
    #    confirmed ready.
    planner = ManagedProcess(
        name="llm_planner",
        cmd=ros_wrapped(shlex.join(planner_tokens)),
    )
    # 4) vLLM process: serves either:
    #    - the merged fine-tuned model produced by train.py, or
    #    - a standard HF model id passed through --model
    #
    #    We keep this outside the ROS wrapper because it does not need ROS.
    vllm = ManagedProcess(
        name="vllm",
        cmd=(
            f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} "
            f"vllm serve {model_spec} "
            f"--served-model-name {args.served_model_name} "
            f"--host 0.0.0.0 "
            f"--port {args.vllm_port} "
            f"--dtype {args.dtype} "
            f"--api-key {args.api_key} "
            f"--gpu-memory-utilization {args.gpu_memory_utilization:.2f} "
            f"--max-model-len {args.max_model_len}"
        ),
    )

    processes = [planner, mission, vllm, master]
    try:
        print("[startup] launching PX4/Gazebo master stack", flush=True)
        print(f"[startup] master command: {args.master_command}", flush=True)
        print(f"[startup] vLLM model spec: {model_spec}", flush=True)
        if planner_prompt_file:
            print(f"[startup] planner prompt file: {planner_prompt_file}", flush=True)
        master.start()

        # Do not start mission_executor immediately. Wait until the master
        # launcher prints a known progress marker that indicates Gazebo is up and
        # the script has reached the MicroXRCEAgent stage. This mirrors the
        # intent of offline_dataset_batch_runner's ordered launcher startup.
        print(
            f"[startup] waiting for master readiness marker: {args.master_ready_marker}",
            flush=True,
        )
        master.wait_for_output(args.master_ready_marker, timeout_s=args.master_ready_timeout_s)
        if args.master_post_ready_delay_s > 0.0:
            print(
                f"[startup] master marker seen; settling for {args.master_post_ready_delay_s:.1f}s",
                flush=True,
            )
            time.sleep(args.master_post_ready_delay_s)

        px4_ready_topics = parse_topic_csv(args.px4_ready_topics)

        # mission_executor talks to PX4 via MAVSDK rather than ROS, so the
        # early gate only needs to prove that PX4 ROS outputs have started to
        # appear. We keep the stricter odometry-message check for just before
        # llm_planner launches.
        print(
            "[startup] waiting for PX4 ROS topics to appear: "
            f"{', '.join(px4_ready_topics)}",
            flush=True,
        )
        matched_ready_topic = wait_for_ros_topic_presence(
            px4_ready_topics,
            timeout_s=args.px4_ready_timeout_s,
        )
        print(f"[startup] detected PX4 ROS topic: {matched_ready_topic}", flush=True)

        # Before we launch mission_executor, clean up any stale MAVSDK-side
        # processes from previous runs and make sure the UDP port it needs is
        # genuinely free. Without this guard, a leftover mission_executor can
        # keep controlling the vehicle and the new one will fail with
        # "bind error: Address in use".
        print(
            f"[startup] cleaning stale mission/MAVSDK processes and checking UDP {args.mission_udp_port}",
            flush=True,
        )
        kill_matching_processes(
            [
                "ros2 run llm_drone mission_executor",
                "llm_drone/lib/llm_drone/mission_executor",
                "mavsdk_server",
            ]
        )
        ensure_udp_port_free(args.mission_udp_port, timeout_s=args.port_cleanup_timeout_s)

        # Only after PX4 is active and odometry is live do we start
        # mission_executor.
        print("[startup] launching mission_executor", flush=True)
        mission.start()

        print(f"[startup] waiting for mission readiness marker: {MISSION_READY_MARKER}", flush=True)
        mission.wait_for_output(MISSION_READY_MARKER, timeout_s=args.mission_timeout_s)

        # Bring up the model server only after the vehicle is already airborne
        # and stable in OFFBOARD. This keeps vLLM startup completely out of the
        # critical path for takeoff and avoids GPU/model initialization noise
        # interfering with sim bring-up experiments.
        print("[startup] launching vLLM server", flush=True)
        vllm.start()

        print(f"[startup] waiting for vLLM health endpoint: {vllm_health_url}", flush=True)
        wait_for_http_ready(vllm_health_url, timeout_s=args.vllm_timeout_s)

        print(
            f"[startup] waiting for planner odometry topic to produce data: {args.odom_topic}",
            flush=True,
        )
        wait_for_ros_topic_message(args.odom_topic, timeout_s=args.odom_timeout_s)

        print("[startup] mission + vLLM are ready; launching llm_planner", flush=True)
        planner.start()

        if planner.process is None:
            raise RuntimeError("Planner failed to start")
        return_code = planner.process.wait()
        return return_code if return_code is not None else 0
    except KeyboardInterrupt:
        print("[startup] interrupted; shutting down managed processes", flush=True)
        return 130
    finally:
        for proc in processes:
            proc.stop(timeout_s=args.shutdown_timeout_s)


if __name__ == "__main__":
    raise SystemExit(main())
