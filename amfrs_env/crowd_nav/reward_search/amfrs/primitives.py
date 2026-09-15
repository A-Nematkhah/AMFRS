"""
Axis 3 — reward primitive registry (pure functions over RewardState).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict

from crowd_nav.reward_search.state import RewardState

Primitive = Callable[[RewardState, Dict[str, Any]], float]


def _finite_or(value: float, fallback: float) -> float:
    v = float(value)
    return v if math.isfinite(v) else float(fallback)


def goal_progress(state: RewardState, memory: Dict[str, Any]) -> float:
    """Negative change in distance-to-goal since last step (positive = progress)."""
    dist = (
        (state.robot.px - state.robot.gx) ** 2 + (state.robot.py - state.robot.gy) ** 2
    ) ** 0.5
    prev = memory.get("prev_dist")
    memory["prev_dist"] = dist
    if prev is None:
        return 0.0
    return _finite_or(float(prev) - float(dist), 0.0)


def discomfort_penalty(state: RewardState, memory: Dict[str, Any]) -> float:
    """0 outside discomfort_dist; linearly increasing as dmin -> 0 inside it."""
    _ = memory
    d = float(state.discomfort_dist)
    dmin = _finite_or(state.dmin, d + 1.0)
    if dmin >= d:
        return 0.0
    return float(d - dmin)


def collision_indicator(state: RewardState, memory: Dict[str, Any]) -> float:
    _ = memory
    return 1.0 if state.collision else 0.0


def time_penalty(state: RewardState, memory: Dict[str, Any]) -> float:
    _ = memory
    return 1.0


def jerk_penalty(state: RewardState, memory: Dict[str, Any]) -> float:
    """Penalize large consecutive action deltas via memory['prev_action']."""
    action = state.action
    if action is None:
        return 0.0
    try:
        ax, ay = float(action[0]), float(action[1])
    except Exception:
        return 0.0
    prev = memory.get("prev_action")
    memory["prev_action"] = (ax, ay)
    if prev is None:
        return 0.0
    return _finite_or(
        ((ax - float(prev[0])) ** 2 + (ay - float(prev[1])) ** 2) ** 0.5, 0.0
    )


def goal_alignment_bonus(state: RewardState, memory: Dict[str, Any]) -> float:
    _ = memory
    dx = state.robot.gx - state.robot.px
    dy = state.robot.gy - state.robot.py
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 1e-8:
        return 1.0
    speed = (state.robot.vx ** 2 + state.robot.vy ** 2) ** 0.5
    if speed < 1e-8:
        return 0.0
    return _finite_or(
        (state.robot.vx * dx + state.robot.vy * dy) / (speed * dist), 0.0
    )


def min_distance_margin(state: RewardState, memory: Dict[str, Any]) -> float:
    """Closest human clearance; never returns ±inf (env may leave dmin=+inf)."""
    _ = memory
    return _finite_or(state.dmin, 15.0)


def success_indicator(state: RewardState, memory: Dict[str, Any]) -> float:
    _ = memory
    return 1.0 if state.reaching_goal else 0.0


def timeout_indicator(state: RewardState, memory: Dict[str, Any]) -> float:
    _ = memory
    return 1.0 if state.timeout else 0.0


def backing_up_penalty(state: RewardState, memory: Dict[str, Any]) -> float:
    _ = memory
    dx = state.robot.gx - state.robot.px
    dy = state.robot.gy - state.robot.py
    # Negative radial speed toward goal
    radial = state.robot.vx * dx + state.robot.vy * dy
    return _finite_or(max(0.0, -radial), 0.0)


def spin_penalty(state: RewardState, memory: Dict[str, Any]) -> float:
    _ = memory
    speed = (state.robot.vx ** 2 + state.robot.vy ** 2) ** 0.5
    # High angular-ish change relative to preferred speed without progress
    return _finite_or(max(0.0, speed - state.robot.v_pref), 0.0)


def energy_penalty(state: RewardState, memory: Dict[str, Any]) -> float:
    _ = memory
    return _finite_or(state.robot.vx ** 2 + state.robot.vy ** 2, 0.0)


def heading_alignment(state: RewardState, memory: Dict[str, Any]) -> float:
    return goal_alignment_bonus(state, memory)


def social_force_alignment(state: RewardState, memory: Dict[str, Any]) -> float:
    """Crude repulsion: prefer velocity away from nearest human if present."""
    _ = memory
    if not state.humans:
        return 0.0
    h = state.humans[0]
    dx = state.robot.px - h.px
    dy = state.robot.py - h.py
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 1e-8:
        return 0.0
    return _finite_or((state.robot.vx * dx + state.robot.vy * dy) / dist, 0.0)


PRIMITIVE_REGISTRY: Dict[str, Primitive] = {
    "goal_progress": goal_progress,
    "discomfort_penalty": discomfort_penalty,
    "collision_indicator": collision_indicator,
    "time_penalty": time_penalty,
    "jerk_penalty": jerk_penalty,
    "goal_alignment_bonus": goal_alignment_bonus,
    "min_distance_margin": min_distance_margin,
    "success_indicator": success_indicator,
    "timeout_indicator": timeout_indicator,
    "backing_up_penalty": backing_up_penalty,
    "spin_penalty": spin_penalty,
    "energy_penalty": energy_penalty,
    "heading_alignment": heading_alignment,
    "social_force_alignment": social_force_alignment,
}


def primitive_signatures_block() -> str:
    lines = ["Available primitives (prefer composing these with numeric weights):"]
    for name in sorted(PRIMITIVE_REGISTRY):
        lines.append(f"  - {name}(state, memory) -> float")
    return "\n".join(lines)
