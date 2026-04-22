#!/usr/bin/env python3
"""Print the live generated environment prompt from ROS topics for manual validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import rclpy

from llm_drone.llm.llm_prompt_common import (
    compose_user_prompt,
    load_system_prompt_from_path,
    serialize_prompt_bundle,
    translate_vector_to_nlp,
)
from llm_drone.llm.prompt_generator_samples import (
    DEFAULT_PROMPT_FILE,
    LLM_WAYPOINT_COUNT,
    LivePromptRecorder,
)


class LiveEnvPromptMonitor(LivePromptRecorder):
    """Reuse the live prompt plumbing, but print prompt text instead of writing files."""

    def __init__(
        self,
        system_prompt: str,
        interval_sec: float = 0.5,
        goal_xyz: tuple[float, float, float] = (35.0, 3.0, 2.5),
        goal_frame: str = "ned",
        print_full_bundle: bool = False,
        only_on_change: bool = True,
    ) -> None:
        self._print_full_bundle = bool(print_full_bundle)
        self._only_on_change = bool(only_on_change)
        self._last_printed_text = ""
        super().__init__(
            system_prompt=system_prompt,
            out_dir=Path("/tmp/live_env_prompt_monitor"),
            interval_sec=interval_sec,
            goal_xyz=goal_xyz,
            goal_frame=goal_frame,
        )
        self.get_logger().info(
            "Live env prompt monitor started; printing generated environment prompt to stdout."
        )

    def record_prompt_tick(self) -> None:
        if not self.depth_received:
            self.get_logger().warn("Waiting for /depth_camera...", throttle_duration_sec=5.0)
            return
        if not self.odom_received:
            self.get_logger().warn("Waiting for /fmu/out/vehicle_odometry...", throttle_duration_sec=5.0)
            return

        env_vector = self.build_environment_vector()
        if env_vector is None:
            self.get_logger().warn("Environment vector unavailable", throttle_duration_sec=5.0)
            return

        env_text = translate_vector_to_nlp(env_vector)
        user_prompt = compose_user_prompt(env_text)
        text = env_text.strip()
        if self._print_full_bundle:
            text = serialize_prompt_bundle(self.system_prompt, user_prompt).strip()

        if self._only_on_change and text == self._last_printed_text:
            return

        self._last_printed_text = text
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        separator = "=" * 80
        print(
            f"\n{separator}\n"
            f"timestamp_utc: {now}\n"
            f"waypoint_count_contract: {LLM_WAYPOINT_COUNT}\n"
            f"{separator}\n"
            f"{text}\n",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the live generated environment prompt for manual validation."
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="Path to the system prompt file used when --print-full-bundle is enabled.",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=0.5,
        help="How often to sample and print the environment prompt.",
    )
    parser.add_argument("--goal-x", type=float, default=35.0)
    parser.add_argument("--goal-y", type=float, default=3.0)
    parser.add_argument("--goal-z", type=float, default=2.5)
    parser.add_argument(
        "--goal-frame",
        type=str,
        default="ned",
        help="Goal frame: ned or gazebo/map/enu.",
    )
    parser.add_argument(
        "--print-full-bundle",
        action="store_true",
        help="Print the system+user prompt bundle instead of only the environment prompt.",
    )
    parser.add_argument(
        "--print-every-tick",
        action="store_true",
        help="Print every timer tick even if the prompt text did not change.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_path = Path(args.prompt_file).expanduser().resolve()
    system_prompt = load_system_prompt_from_path(prompt_path)

    rclpy.init()
    node = LiveEnvPromptMonitor(
        system_prompt=system_prompt,
        interval_sec=args.interval_sec,
        goal_xyz=(args.goal_x, args.goal_y, args.goal_z),
        goal_frame=args.goal_frame,
        print_full_bundle=args.print_full_bundle,
        only_on_change=not args.print_every_tick,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
