"""Min-jerk waypoint interpolation and gripper open/close profiles.

These operate purely in joint space on the 5-arm-joint (+ gripper) vectors
IK produces -- no MuJoCo objects here, so they're trivially unit-testable
and reusable by both the Phase 3 scripted experts and (later) a policy's
receding-horizon executor.
"""

from __future__ import annotations

import numpy as np

from bimanual.sim import robot_spec as rs


def min_jerk_scaling(t: np.ndarray) -> np.ndarray:
    """Standard 10t^3 - 15t^4 + 6t^5 min-jerk time-scaling, t in [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def min_jerk_trajectory(start: np.ndarray, end: np.ndarray, n_steps: int) -> np.ndarray:
    """Returns an (n_steps, D) array interpolating start->end with a
    min-jerk time profile (zero velocity/acceleration at both ends)."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n_steps)
    s = min_jerk_scaling(t)
    return start[None, :] + s[:, None] * (end[None, :] - start[None, :])


def gripper_profile(n_steps: int, open_at_start: bool, hold_frac: float = 0.3) -> np.ndarray:
    """A (n_steps,) gripper-opening trajectory in [0, 1] (fraction of
    rs.GRIPPER_LIMIT range) that holds its starting state for hold_frac of
    the duration, then min-jerk transitions to the opposite state -- used
    so a grasp/release doesn't happen mid-transport."""
    start_val = 1.0 if open_at_start else 0.0
    end_val = 1.0 - start_val
    hold_steps = int(n_steps * hold_frac)
    move_steps = n_steps - hold_steps
    hold = np.full(hold_steps, start_val)
    move = min_jerk_scaling(np.linspace(0.0, 1.0, max(move_steps, 1)))
    move = start_val + move * (end_val - start_val)
    profile = np.concatenate([hold, move])[:n_steps]
    if len(profile) < n_steps:
        profile = np.concatenate([profile, np.full(n_steps - len(profile), end_val)])
    return profile


def gripper_fraction_to_qpos(fraction: np.ndarray | float) -> np.ndarray | float:
    """Maps a [0, 1] "how open" fraction (1=open, 0=closed) to the real
    gripper joint's qpos, using rs.GRIPPER_OPEN/GRIPPER_CLOSED rather than
    raw lo/hi -- the vendored joint's numeric range order is the OPPOSITE
    of open/closed (verified by rendering both extremes, docs/decisions.md
    Phase 3 notes; an earlier version of this function used lo/hi directly
    and silently commanded the wrong gripper state everywhere)."""
    open_q, closed_q = rs.GRIPPER_OPEN, rs.GRIPPER_CLOSED
    clipped = np.clip(np.asarray(fraction, dtype=np.float64), 0.0, 1.0)
    result = closed_q + clipped * (open_q - closed_q)
    return float(result) if np.isscalar(fraction) or np.ndim(fraction) == 0 else result
