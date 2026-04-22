#!/usr/bin/env python3
"""Launch Gazebo/PX4, wait for FMU data, then start vLLM and llm_planner_goto_cpp."""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MASTER_COMMAND = str(
    (Path.home() / "PX4-Autopilot" / "Tools" / "simulation" / "gz" / "launch_obstacle_avoidance_full_stack.sh").resolve()
)
DEFAULT_MODEL_SPEC = str(REPO_ROOT / "models" / "drone_planner_check")
DEFAULT_SERVED_MODEL_NAME = "qwen25_7b_drone_planner"
DEFAULT_VLLM_PORT = 8000
DEFAULT_VLLM_HOST = "172.22.224.93"
DEFAULT_API_KEY = "token-abc123"
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_PROMPT_FILE = str(REPO_ROOT / "config" / "variant_X.txt")
DEFAULT_MASTER_READY_MARKER = "Opening Terminal 3: MicroXRCEAgent"
DEFAULT_PX4_READY_TOPICS = (
    "/fmu/out/timesync_status",
    "/fmu/out/vehicle_status_v1",
    "/fmu/out/vehicle_odometry",
)
DEFAULT_ODOM_TOPIC = "/fmu/out/vehicle_odometry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-command",
        default=os.environ.get("LLM_MASTER_COMMAND", "").strip() or DEFAULT_MASTER_COMMAND,
    )
    parser.add_argument("--master-ready-marker", default=DEFAULT_MASTER_READY_MARKER)
    parser.add_argument("--master-ready-timeout-s", type=float, default=120.0)
    parser.add_argument("--master-post-ready-delay-s", type=float, default=4.0)
    parser.add_argument("--px4-ready-topics", default=",".join(DEFAULT_PX4_READY_TOPICS))
    parser.add_argument("--px4-ready-timeout-s", type=float, default=120.0)
    parser.add_argument("--odom-topic", default=DEFAULT_ODOM_TOPIC)
    parser.add_argument("--odom-timeout-s", type=float, default=120.0)
    parser.add_argument("--shutdown-timeout-s", type=float, default=8.0)
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--model", default=DEFAULT_MODEL_SPEC)
    parser.add_argument("--served-model-name", default=DEFAULT_SERVED_MODEL_NAME)
    parser.add_argument("--llm-backend", choices=("vllm", "openai"), default="vllm")
    parser.add_argument("--vllm-host", default=DEFAULT_VLLM_HOST)
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--openai-url", default=DEFAULT_OPENAI_URL)
    parser.add_argument(
        "--openai-api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--goal-frame", default="gazebo")
    parser.add_argument("--goal-x", type=float, default=29.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-z", type=float, default=2.5)
    parser.add_argument("--planner-prompt-file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--vllm-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--start-local-vllm",
        action="store_true",
        help="Start vLLM on this machine. By default the script uses an already-running remote vLLM server.",
    )
    return parser.parse_args()


class ManagedProcess:
    def __init__(self, *, name: str, cmd: str, env: dict[str, str] | None = None) -> None:
        self.name = name
        self.cmd = cmd
        self.env = dict(env or {})
        self.process: subprocess.Popen[str] | None = None
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
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

    def wait_for_output(self, needle: str, timeout_s: float) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            with self._lock:
                if any(needle in line for line in self._lines):
                    return
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"{self.name} exited early with code {self.process.returncode}")
            time.sleep(0.25)
        raise TimeoutError(f"Timed out waiting for {self.name} to emit: {needle!r}")

    def stop(self, timeout_s: float) -> None:
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
    return (
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
    return (
        "set -eo pipefail\n"
        f"cd {WORKSPACE_ROOT}\n"
        f"{command}\n"
    )


def wait_for_http_ready(url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + float(timeout_s)
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if 200 <= int(response.status) < 300:
                    return
        except urllib.error.URLError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for vLLM readiness at {url} ({last_error})")


def parse_topic_csv(raw_topics: str) -> list[str]:
    return [topic.strip() for topic in str(raw_topics).split(",") if topic.strip()]


def wait_for_ros_topic_presence(topics: list[str], timeout_s: float) -> str:
    if not topics:
        raise ValueError("At least one PX4 readiness topic must be provided")

    probe_cmd = ros_wrapped("ros2 topic list --no-daemon")
    deadline = time.monotonic() + float(timeout_s)
    last_error = ""
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
            last_px4_topics = sorted(topic for topic in seen_topics if topic.startswith("/fmu/"))
            last_error = ""
        else:
            last_error = (probe.stderr or probe.stdout or "").strip()
        time.sleep(1.0)

    if last_px4_topics:
        preview = ", ".join(last_px4_topics[:10])
        raise TimeoutError(
            "Timed out waiting for PX4 ROS topics to appear. "
            f"Wanted one of {topics!r}; last seen PX4 topics: {preview}"
        )
    raise TimeoutError(
        "Timed out waiting for PX4 ROS topics to appear. "
        f"Wanted one of {topics!r}; last error: {last_error or 'no /fmu topics seen'}"
    )


def wait_for_ros_topic_message(topic: str, timeout_s: float) -> None:
    probe_cmd = ros_wrapped(
        f"timeout 5 ros2 topic echo --once {topic} >/tmp/start_llm_planner_goto_cpp_stack_probe.log 2>&1"
    )
    deadline = time.monotonic() + float(timeout_s)
    last_exit_code = None
    last_probe_log = ""
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
            last_probe_log = Path("/tmp/start_llm_planner_goto_cpp_stack_probe.log").read_text().strip()
        except FileNotFoundError:
            last_probe_log = ""
        time.sleep(1.0)
    suffix = f"; last probe log: {last_probe_log}" if last_probe_log else ""
    raise TimeoutError(
        f"Timed out waiting for ROS topic {topic!r} to emit a message "
        f"(last probe exit code {last_exit_code}{suffix})"
    )


def main() -> int:
    args = parse_args()
    if not args.master_command:
        raise SystemExit(
            "No master command provided. Pass --master-command '<your launcher>' "
            "or set LLM_MASTER_COMMAND."
        )

    model_spec_raw = str(args.model).strip()
    if not model_spec_raw:
        raise ValueError("--model must not be empty")
    local_model_candidate = Path(model_spec_raw).expanduser()
    model_spec = str(local_model_candidate.resolve()) if local_model_candidate.exists() else model_spec_raw

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
    master = ManagedProcess(
        name="master",
        cmd=workspace_wrapped(args.master_command),
    )

    planner_tokens = [
        "ros2",
        "run",
        "llm_drone",
        "llm_planner_goto_cpp",
        "--ros-args",
        "-p",
        f"llm_backend:={args.llm_backend}",
        "-p",
        f"goal_x:={args.goal_x}",
        "-p",
        f"goal_y:={args.goal_y}",
        "-p",
        f"goal_z:={args.goal_z}",
        "-p",
        f"goal_frame:={args.goal_frame}",
        "-p",
        f"vllm_url:={vllm_url}",
        "-p",
        f"vllm_model:={args.served_model_name}",
        "-p",
        f"vllm_api_key:={args.api_key}",
        "-p",
        f"openai_url:={args.openai_url}",
        "-p",
        f"openai_model:={args.openai_model}",
    ]
    if args.openai_api_key:
        planner_tokens.extend(["-p", f"openai_api_key:={args.openai_api_key}"])
    if planner_prompt_file:
        planner_tokens.extend(["-p", f"prompt_file:={planner_prompt_file}"])

    planner = ManagedProcess(
        name="llm_planner_goto_cpp",
        cmd=ros_wrapped(shlex.join(planner_tokens)),
    )

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

    processes = [planner, master]
    if args.llm_backend == "vllm" and args.start_local_vllm:
        processes.append(vllm)
    try:
        print("[startup] launching PX4/Gazebo master stack", flush=True)
        print(f"[startup] master command: {args.master_command}", flush=True)
        print(f"[startup] llm backend: {args.llm_backend}", flush=True)
        if args.llm_backend == "vllm":
            print(f"[startup] vLLM model spec: {model_spec}", flush=True)
        else:
            print(f"[startup] OpenAI model: {args.openai_model}", flush=True)
        if planner_prompt_file:
            print(f"[startup] planner prompt file: {planner_prompt_file}", flush=True)
        master.start()

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

        print(
            f"[startup] waiting for odometry on {args.odom_topic} before starting vLLM/planner",
            flush=True,
        )
        wait_for_ros_topic_message(args.odom_topic, timeout_s=args.odom_timeout_s)

        if args.llm_backend == "vllm" and args.start_local_vllm:
            print("[startup] launching local vLLM server", flush=True)
            vllm.start()
        elif args.llm_backend == "openai":
            print("[startup] using OpenAI API backend", flush=True)
        else:
            print("[startup] using remote/already-running vLLM server", flush=True)

        if args.llm_backend == "vllm":
            print(f"[startup] waiting for LLM health endpoint: {vllm_health_url}", flush=True)
            wait_for_http_ready(vllm_health_url, timeout_s=args.vllm_timeout_s)

        print("[startup] launching llm_planner_goto_cpp", flush=True)
        planner.start()

        exit_code = planner.process.wait() if planner.process is not None else 1
        print(f"[startup] llm_planner_goto_cpp exited with code {exit_code}", flush=True)
        return exit_code
    except KeyboardInterrupt:
        print("[startup] interrupted by user", flush=True)
        return 130
    finally:
        for process in processes:
            process.stop(timeout_s=args.shutdown_timeout_s)


if __name__ == "__main__":
    sys.exit(main())
