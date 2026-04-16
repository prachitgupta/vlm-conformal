#!/usr/bin/env python3
"""Launch vLLM first, then the new C++ goto-based LLM planner node."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_SPEC = str(REPO_ROOT / "models" / "drone_planner_chat")
DEFAULT_SERVED_MODEL_NAME = "drone_planner_chat"
DEFAULT_VLLM_HOST = "127.0.0.1"
DEFAULT_VLLM_PORT = 8000
DEFAULT_API_KEY = "token-abc123"
DEFAULT_PROMPT_FILE = str(REPO_ROOT / "config" / "variant_X.txt")


class ManagedProcess:
    def __init__(self, name: str, cmd: str, env: dict[str, str] | None = None) -> None:
        self.name = name
        self.cmd = cmd
        self.env = dict(env or {})
        self.process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        proc_env = os.environ.copy()
        proc_env.update(self.env)
        self.process = subprocess.Popen(
            ["bash", "-lc", self.cmd],
            cwd=str(REPO_ROOT),
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
                print(f"[{self.name}] {line.rstrip()}", flush=True)

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGINT)
            self.process.wait(timeout=timeout_s)
            return
        except Exception:
            pass
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=timeout_s)
            return
        except Exception:
            pass
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--model", default=DEFAULT_MODEL_SPEC)
    parser.add_argument("--served-model-name", default=DEFAULT_SERVED_MODEL_NAME)
    parser.add_argument("--vllm-host", default=DEFAULT_VLLM_HOST)
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--goal-frame", default="gazebo")
    parser.add_argument("--goal-x", type=float, default=29.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-z", type=float, default=2.5)
    parser.add_argument("--planner-prompt-file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--vllm-timeout-s", type=float, default=300.0)
    return parser.parse_args()


def wait_for_vllm(host: str, port: int, timeout_s: float) -> None:
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if 200 <= response.status < 300:
                    return
        except urllib.error.URLError:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for vLLM health endpoint at {url}")


def main() -> int:
    args = parse_args()

    vllm_cmd = (
        f"source /opt/ros/humble/setup.bash && "
        f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} "
        f"vllm serve {args.model} "
        f"--served-model-name {args.served_model_name} "
        f"--host 0.0.0.0 "
        f"--port {args.vllm_port} "
        f"--dtype {args.dtype} "
        f"--api-key {args.api_key} "
        f"--gpu-memory-utilization {args.gpu_memory_utilization} "
        f"--max-model-len {args.max_model_len}"
    )

    planner_cmd = (
        f"source /opt/ros/humble/setup.bash && "
        f"cd {WORKSPACE_ROOT} && "
        f"source install/setup.bash && "
        f"mkdir -p /tmp/roslog && "
        f"ROS_LOG_DIR=/tmp/roslog "
        f"ros2 run llm_drone llm_planner_goto_cpp --ros-args "
        f"-p vllm_url:=http://{args.vllm_host}:{args.vllm_port}/v1/chat/completions "
        f"-p vllm_model:={args.served_model_name} "
        f"-p vllm_api_key:={args.api_key} "
        f"-p goal_frame:={args.goal_frame} "
        f"-p goal_x:={args.goal_x} "
        f"-p goal_y:={args.goal_y} "
        f"-p goal_z:={args.goal_z}"
    )
    if args.planner_prompt_file.strip():
        planner_cmd += f" -p prompt_file:={args.planner_prompt_file}"

    vllm_process = ManagedProcess("vllm", vllm_cmd)
    planner_process: ManagedProcess | None = None
    try:
        print("[launcher] starting vLLM", flush=True)
        vllm_process.start()
        wait_for_vllm(args.vllm_host, args.vllm_port, args.vllm_timeout_s)
        print("[launcher] vLLM is healthy; starting llm_planner_goto_cpp", flush=True)
        planner_process = ManagedProcess("llm_planner_goto_cpp", planner_cmd)
        planner_process.start()
        assert planner_process.process is not None
        return planner_process.process.wait()
    except KeyboardInterrupt:
        print("[launcher] interrupted, shutting down", flush=True)
        return 130
    finally:
        if planner_process is not None:
            planner_process.stop(timeout_s=5.0)
        vllm_process.stop(timeout_s=5.0)


if __name__ == "__main__":
    sys.exit(main())
