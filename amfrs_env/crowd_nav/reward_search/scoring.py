"""
Stage I analytical Score1 over a pre-collected trajectory dataset.

Score1(r) = mean_j mean_f Spearman(rank_rules(j,f), rank_reward(j,f))

Reward ranks use cumulative sum(reward_fn.compute(state)) — never env-logged
reward scalars. ``make_smoke_score_fn`` remains only as an opt-in fast-test
fixture (``--score1 smoke`` / ``fast`` profile).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Union

import numpy as np

from crowd_nav.reward_search.dataset import Stage1Dataset, TrajectoryRecord
from crowd_nav.reward_search.rules import (
    dist_to_goal,
    rule_preference_score,
    spearman_correlation,
)
from crowd_nav.reward_search.state import RewardFunction


class Score1Fn(Protocol):
    def __call__(self, reward_fn: RewardFunction, *, candidate_id: str = "") -> float:
        ...


def _cumulative_reward(
    reward_fn: RewardFunction,
    traj: TrajectoryRecord,
    *,
    max_frame: int,
) -> List[float]:
    """
    Prefix sums of recomputed rewards for frames 0..max_frame inclusive.

    Pads by holding the last state (same as rule ranking). Resets the
    reward function once at the start of the trajectory.
    """
    reward_fn.reset()
    totals: List[float] = []
    running = 0.0
    # Pad short trajs by holding the last RewardState, then
    # sum(compute(state) for state in padded[:f+1]) — same as sister project.
    n_pad = max_frame + 1
    for f in range(n_pad):
        state = traj.state_at(f)
        try:
            running += float(reward_fn.compute(state))
            totals.append(running)
        except Exception:  # noqa: BLE001
            totals.append(float("nan"))
    return totals


def _scenario_frame_correlations(
    trajs: Sequence[TrajectoryRecord],
    reward_fn: RewardFunction,
) -> List[float]:
    """Spearman ρ for each frame f = 0..F_j-1 within one scenario."""
    if len(trajs) < 2:
        return []
    f_max = max(t.length for t in trajs)
    categories = [t.category for t in trajs]
    nav_lengths = [t.nav_length for t in trajs]

    # Precompute cumulative rewards per trajectory (reset between trajs).
    cumuls: List[List[float]] = []
    for traj in trajs:
        cumuls.append(_cumulative_reward(reward_fn, traj, max_frame=f_max - 1))

    corrs: List[float] = []
    for f in range(f_max):
        rule_scores = []
        reward_scores = []
        ok = True
        for i, traj in enumerate(trajs):
            state = traj.state_at(f)
            d_goal = dist_to_goal(
                state.robot.px, state.robot.py, state.robot.gx, state.robot.gy
            )
            rule_scores.append(
                rule_preference_score(
                    categories[i], nav_length=nav_lengths[i], dist_goal=d_goal
                )
            )
            val = cumuls[i][f]
            if not math.isfinite(val):
                ok = False
                break
            reward_scores.append(val)
        if not ok:
            continue
        rho = spearman_correlation(rule_scores, reward_scores)
        if math.isfinite(rho):
            corrs.append(float(rho))
    return corrs


def score1_for_dataset(
    dataset: Union[Stage1Dataset, RewardFunction],
    reward_fn: Optional[RewardFunction] = None,
    *,
    candidate_id: str = "",
) -> float:
    """
    Analytical Score1 over pre-collected scenarios (EvoNav Eq. 1 / Figure 3).

    Call as ``score1_for_dataset(dataset, reward_fn)``. Scenarios with fewer
    than 2 trajectories are skipped. Raises ``ValueError`` only if every
    scenario is unusable.
    """
    # Back-compat: accidental score1_for_dataset(reward_fn) without dataset.
    if reward_fn is None:
        if isinstance(dataset, RewardFunction):
            raise ValueError(
                "score1_for_dataset requires a loaded Stage I dataset as the "
                "first argument: score1_for_dataset(dataset, reward_fn). "
                "Use make_score1_fn(dataset) for StageIEvolver.score_fn."
            )
        raise TypeError("reward_fn is required")

    if not isinstance(dataset, dict) or not dataset:
        raise ValueError("dataset must be a non-empty dict[scenario_id, trajectories]")

    scenario_means: List[float] = []
    for _sid, trajs in dataset.items():
        usable = [t for t in trajs if t.length >= 1]
        if len(usable) < 2:
            continue
        frame_corrs = _scenario_frame_correlations(usable, reward_fn)
        if not frame_corrs:
            continue
        scenario_means.append(float(np.mean(frame_corrs)))

    if not scenario_means:
        raise ValueError(
            "score1_for_dataset: no scoreable scenarios "
            "(need ≥2 trajectories with finite frame correlations)."
        )
    score = float(np.mean(scenario_means))
    # Spearman is in [-1, 1]; keep that contract even with float noise.
    return float(max(-1.0, min(1.0, score)))


def make_score1_fn(dataset: Stage1Dataset) -> Callable[..., float]:
    """Bind a loaded dataset once for StageIEvolver.score_fn."""

    def _fn(reward_fn: RewardFunction, *, candidate_id: str = "") -> float:
        return score1_for_dataset(dataset, reward_fn, candidate_id=candidate_id)

    return _fn


def make_constant_score_fn(scores: dict) -> Callable[..., float]:
    """Test helper: map candidate_id -> score (missing ids get -inf)."""

    def _fn(reward_fn: RewardFunction, *, candidate_id: str = "") -> float:
        return float(scores.get(candidate_id, float("-inf")))

    return _fn


def make_smoke_score_fn() -> Callable[..., float]:
    """
    Opt-in fast-test fixture only (``--score1 smoke`` / ``--fast``).

    Not used by the real Stage I loop. Prefer ``make_score1_fn(dataset)``.
    """
    from crowd_nav.reward_search.sandbox.runtime import default_smoke_states

    states = default_smoke_states()

    def _fn(reward_fn: RewardFunction, *, candidate_id: str = "") -> float:
        vals = []
        for s in states:
            try:
                vals.append(float(reward_fn.compute(s)))
            except Exception:  # noqa: BLE001
                return float("-inf")
        if not vals:
            return float("-inf")
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / max(len(vals), 1)
        polarity = (vals[0] - vals[1]) if len(vals) >= 2 else 0.0
        return float(var + 0.1 * polarity)

    return _fn
