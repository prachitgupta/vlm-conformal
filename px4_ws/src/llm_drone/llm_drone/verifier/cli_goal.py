#!/usr/bin/env python3
"""Shared CLI helpers for configuring goal coordinates from terminal args."""

import argparse
import sys
from typing import List, Optional, Tuple

import numpy as np


def parse_goal_overrides(args: Optional[List[str]] = None) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Parse optional goal overrides while preserving all other args for rclpy.

    Supported forms:
      --goal X Y Z
      --goal-x X --goal-y Y --goal-z Z
    """
    argv = list(sys.argv[1:] if args is None else args)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--goal', nargs=3, type=float, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--goal-x', type=float)
    parser.add_argument('--goal-y', type=float)
    parser.add_argument('--goal-z', type=float)

    known, remaining = parser.parse_known_args(argv)

    goal = None
    if known.goal is not None:
        goal = np.array(known.goal, dtype=float)
    elif (
        known.goal_x is not None
        and known.goal_y is not None
        and known.goal_z is not None
    ):
        goal = np.array([known.goal_x, known.goal_y, known.goal_z], dtype=float)

    return goal, remaining
