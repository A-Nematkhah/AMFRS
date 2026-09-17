"""
Axis 5 — static / symbolic reward sanity gate (CPU-only, no torch/gym).

Hard-rejects only non-finite rewards. Monotonicity and human-blindness are
soft diagnostics attached to ``candidate.metadata["static_report"]``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from crowd_nav.reward_search.state import (
    HumanObservable,
    RewardFunction,
    RewardState,
    RobotRewardState,
)

ComputeFn = Callable[[RewardState, Dict[str, Any]], float]

# Source of truth: crowd_nav/reward_search/prompts.py :: D5_SEED_FUNCTION
# (collision_penalty=-20.0, success_reward=10.0). Keep in sync if the seed
# function's terminal magnitudes ever change.
SEED_REWARD_RANGE: Tuple[float, float] = (-20.0, 10.0)


@dataclass(frozen=True)
class StaticGateReport:
    all_finite: bool = True
    min_val: float = 0.0
    max_val: float = 0.0
    offending_state_idx: Optional[int] = None
    monotonicity_ok: bool = True
    ignores_humans: bool = False
    # Observed |range| vs seed terminal magnitudes; >1 means inflation.
    scale_drift_ratio: float = 1.0
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _robot(
    px: float = 0.0,
    py: float = 0.0,
    gx: float = 4.0,
    gy: float = 0.0,
    vx: float = 0.5,
    vy: float = 0.0,
) -> RobotRewardState:
    return RobotRewardState(
        px=px, py=py, vx=vx, vy=vy, radius=0.3, gx=gx, gy=gy, v_pref=1.0
    )


def _state(
    *,
    dmin: float,
    humans: Tuple[HumanObservable, ...] = (),
    collision: bool = False,
    reaching_goal: bool = False,
    timeout: bool = False,
    robot: Optional[RobotRewardState] = None,
    discomfort_dist: float = 0.25,
) -> RewardState:
    return RewardState(
        robot=robot or _robot(),
        humans=humans,
        dmin=float(dmin),
        discomfort_dist=float(discomfort_dist),
        collision=collision,
        reaching_goal=reaching_goal,
        timeout=timeout,
        action=(0.5, 0.0),
        time_step=0.25,
        global_time=1.0,
        time_limit=50.0,
    )


def sample_synthetic_states(n: int = 200, seed: int = 0) -> List[RewardState]:
    """
    Deterministic edge-case battery for static checks.

    Covers: 0 humans, human at discomfort_dist, overlap, dmin sweep,
    timeout/collision/reaching_goal in isolation.
    """
    states: List[RewardState] = []
    # Fixed edge cases first
    states.append(_state(dmin=2.0, humans=()))
    states.append(
        _state(
            dmin=0.25,
            humans=(HumanObservable(0.55, 0.0, 0.0, 0.0, 0.3),),
        )
    )
    states.append(
        _state(
            dmin=-0.05,
            humans=(HumanObservable(0.4, 0.0, 0.0, 0.0, 0.3),),
            collision=True,
        )
    )
    states.append(_state(dmin=1.0, reaching_goal=True, robot=_robot(px=3.95, gx=4.0)))
    states.append(_state(dmin=1.0, timeout=True))

    # dmin sweep inside discomfort zone
    for i in range(max(0, n - len(states))):
        # LCG-ish deterministic floats from seed+i
        t = ((seed * 1103515245 + 12345 + i * 9973) & 0x7FFFFFFF) / 0x7FFFFFFF
        dmin = 0.01 + 0.24 * t
        hx = 0.3 + 0.3 + dmin  # robot at 0 with r=0.3, human r=0.3
        states.append(
            _state(
                dmin=dmin,
                humans=(HumanObservable(hx, 0.0, 0.0, 0.0, 0.3),),
            )
        )
    return states[:n]


def _call(fn: ComputeFn, state: RewardState, memory: Optional[Dict[str, Any]] = None) -> float:
    mem = memory if memory is not None else {}
    return float(fn(state, mem))


def _as_compute_fn(reward: Any) -> ComputeFn:
    if callable(reward) and not isinstance(reward, RewardFunction):
        return reward  # type: ignore[return-value]
    if isinstance(reward, RewardFunction):
        # SandboxedReward.compute only takes state; memory is internal.
        def _fn(state: RewardState, memory: Dict[str, Any]) -> float:
            # Prefer underlying compute if present for fresh memory control
            underlying = getattr(reward, "_compute_fn", None)
            if callable(underlying):
                return float(underlying(state, memory))
            return float(reward.compute(state))

        return _fn
    raise TypeError(f"unsupported reward type: {type(reward)!r}")


def check_bounded(
    compute_fn: Any,
    states: Sequence[RewardState],
) -> StaticGateReport:
    fn = _as_compute_fn(compute_fn)
    values: List[float] = []
    offending: Optional[int] = None
    for i, st in enumerate(states):
        try:
            v = _call(fn, st, {})
        except Exception:
            return StaticGateReport(
                all_finite=False,
                offending_state_idx=i,
                notes=("runtime_error",),
            )
        if not math.isfinite(v):
            return StaticGateReport(
                all_finite=False,
                offending_state_idx=i,
                min_val=v,
                max_val=v,
                notes=("non_finite",),
            )
        values.append(v)
    return StaticGateReport(
        all_finite=True,
        min_val=min(values) if values else 0.0,
        max_val=max(values) if values else 0.0,
    )


def check_monotonicity_in_danger(
    compute_fn: Any,
    *,
    discomfort_dist: float = 0.25,
    n_pairs: int = 16,
    seed: int = 0,
    atol: float = 1e-6,
) -> bool:
    """
    Reward should be non-increasing as dmin decreases inside the discomfort zone
    (getting closer to humans must not increase reward), all else equal.
    """
    fn = _as_compute_fn(compute_fn)
    for i in range(n_pairs):
        t = ((seed + i * 7919) % 1000) / 1000.0
        d_far = discomfort_dist * (0.5 + 0.5 * t)
        d_near = max(1e-4, d_far * 0.5)
        human = (HumanObservable(0.6, 0.0, 0.0, 0.0, 0.3),)
        s_far = _state(dmin=d_far, humans=human, discomfort_dist=discomfort_dist)
        s_near = _state(dmin=d_near, humans=human, discomfort_dist=discomfort_dist)
        try:
            r_far = _call(fn, s_far, {})
            r_near = _call(fn, s_near, {})
        except Exception:
            return False
        if not (math.isfinite(r_far) and math.isfinite(r_near)):
            return False
        # Nearer must not yield strictly higher reward
        if r_near > r_far + atol:
            return False
    return True


def check_scale_drift(
    bounded: StaticGateReport,
    *,
    seed_range: Tuple[float, float] = SEED_REWARD_RANGE,
    warn_threshold: float = 5.0,
) -> float:
    """
    Ratio of this candidate's observed |min|/|max| to the seed reward range.

    ``ratio > 1`` means at least one side of the observed range exceeds the
    corresponding seed terminal magnitude (inflation). Values below 1
    (smaller-scale rewards) are fine and do not warn. ``warn_threshold`` is
    unused for the return value; callers soft-flag when ``ratio > warn_threshold``.
    """
    del warn_threshold  # documented for callers; ratio itself is the return
    seed_min_abs = abs(float(seed_range[0]))
    seed_max_abs = abs(float(seed_range[1]))
    left = (
        abs(float(bounded.min_val)) / seed_min_abs if seed_min_abs > 0.0 else 0.0
    )
    right = (
        abs(float(bounded.max_val)) / seed_max_abs if seed_max_abs > 0.0 else 0.0
    )
    return float(max(left, right))


def check_ignores_humans_and_dmin(
    compute_fn: Any,
    *,
    n_pairs: int = 16,
    seed: int = 0,
) -> bool:
    """
    True if reward is bit-identical across empty vs populated humans AND across
    different dmin values — i.e. truly crowd-blind (ignores humans and dmin).
    """
    fn = _as_compute_fn(compute_fn)
    for i in range(n_pairs):
        t = ((seed + i * 104729) % 1000) / 1000.0
        dmin_a = 0.05 + 0.5 * t
        dmin_b = 1.0 + 0.5 * t
        empty = _state(dmin=dmin_a, humans=())
        populated = _state(
            dmin=dmin_a,
            humans=(HumanObservable(1.0, 0.0, 0.0, 0.0, 0.3),),
        )
        other_dmin = _state(
            dmin=dmin_b,
            humans=(HumanObservable(1.0, 0.0, 0.0, 0.0, 0.3),),
        )
        try:
            r0 = _call(fn, empty, {})
            r1 = _call(fn, populated, {})
            r2 = _call(fn, other_dmin, {})
        except Exception:
            return False
        if r0 != r1 or r0 != r2:
            return False
    return True


@dataclass
class StaticGate:
    n_states: int = 64
    seed: int = 0

    def check(self, reward: Any) -> StaticGateReport:
        states = sample_synthetic_states(n=self.n_states, seed=self.seed)
        bounded = check_bounded(reward, states)
        if not bounded.all_finite:
            return bounded
        mono_ok = check_monotonicity_in_danger(reward, seed=self.seed)
        ignores = check_ignores_humans_and_dmin(reward, seed=self.seed)
        drift_ratio = check_scale_drift(bounded)
        notes: List[str] = []
        if not mono_ok:
            notes.append("monotonicity_violation")
        if ignores:
            notes.append("ignores_humans_and_dmin")
        if drift_ratio > 5.0:
            notes.append("scale_drift")
        return StaticGateReport(
            all_finite=True,
            min_val=bounded.min_val,
            max_val=bounded.max_val,
            monotonicity_ok=mono_ok,
            ignores_humans=ignores,
            scale_drift_ratio=float(drift_ratio),
            notes=tuple(notes),
        )
