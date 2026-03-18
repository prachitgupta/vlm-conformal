#!/usr/bin/env python3
"""Launch the full LLM planner stack in the same order we use manually.

This script is intentionally verbose and heavily commented because it is meant
to be edited by hand during experiments. The default flow is:

1. Launch the external PX4/Gazebo "master" script that brings up the simulator
   stack (Gazebo, PX4 SITL, MicroDDS/bridge tools, RQT if your script does so).
2. Start the ROS mission executor and wait until PX4 reports the vehicle is
   airborne and in OFFBOARD mode.
3. In parallel, start a vLLM server hosting the merged fine-tuned model.
4. Once both the mission side and the vLLM side are ready, launch llm_planner.

If your lab setup changes, the easiest things to tweak are:
- DEFAULT_MASTER_COMMAND
- DEFAULT_MODEL_SPEC
- DEFAULT_MAX_MODEL_LEN
- the ros2 run command used for llm_planner
"""

from __future__ import annotations

import argparse
import os
import signal
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
DEFAULT_MODEL_SPEC = str(REPO_ROOT / "models" / "drone_planner_checkpoints" / "final_merged")
DEFAULT_SERVED_MODEL_NAME = "qwen25_7b_drone_planner"
DEFAULT_VLLM_PORT = 8000
DEFAULT_VLLM_HOST = "127.0.0.1"
DEFAULT_API_KEY = "token-abc123"
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
    """
    return (
        "set -euo pipefail\n"
        "source /opt/ros/humble/setup.bash\n"
        f"cd {WORKSPACE_ROOT}\n"
        "source install/setup.bash\n"
        "mkdir -p /tmp/roslog\n"
        "export ROS_LOG_DIR=/tmp/roslog\n"
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

    vllm_url = f"http://{args.vllm_host}:{args.vllm_port}/v1/chat/completions"
    vllm_health_url = f"http://{args.vllm_host}:{args.vllm_port}/health"

    # 1) Master process: your existing PX4/Gazebo full-stack launcher.
    #    This is the main integration point with your local sim setup.
    master = ManagedProcess(
        name="master",
        cmd=ros_wrapped(args.master_command),
    )
    # 2) Mission executor: arms, takes off, and holds OFFBOARD until the planner
    #    starts publishing setpoints.
    mission = ManagedProcess(
        name="mission_executor",
        cmd=ros_wrapped("ros2 run llm_drone mission_executor --sim"),
    )
    # 3) Planner process: this stays dormant until both mission and vLLM are
    #    confirmed ready.
    planner = ManagedProcess(
        name="llm_planner",
        cmd=ros_wrapped(
            "ros2 run llm_drone llm --ros-args "
            f"-p llm_provider:=vllm "
            f"-p vllm_url:={vllm_url} "
            f"-p vllm_model:={args.served_model_name} "
            f"-p vllm_api_key:={args.api_key}"
        ),
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
        master.start()
        # Small head start so the simulator stack can begin booting before the
        # mission executor tries to connect to MAVSDK / PX4.
        time.sleep(2.0)

        print("[startup] launching mission_executor and vLLM in parallel", flush=True)
        mission.start()
        vllm.start()

        print(f"[startup] waiting for mission readiness marker: {MISSION_READY_MARKER}", flush=True)
        mission.wait_for_output(MISSION_READY_MARKER, timeout_s=args.mission_timeout_s)

        print(f"[startup] waiting for vLLM health endpoint: {vllm_health_url}", flush=True)
        wait_for_http_ready(vllm_health_url, timeout_s=args.vllm_timeout_s)

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
