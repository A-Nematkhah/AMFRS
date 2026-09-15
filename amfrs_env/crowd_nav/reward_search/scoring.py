"""
Stage I analytical Score1 over a pre-collected trajectory dataset.

Score1(r) = mean_j mean_f Spearman(rank_rules(j,f), rank_reward(j,f))

Reward ranks use cumulative sum(reward_fn.compute(state)) — never env-logged
reward scalars. ``make_smoke_score_fn`` remains only as an opt-in fast-test
fixture (``--score1 smoke`` / ``fast`` profile).

Phase 1 honesty:
  * pad frames do not keep accruing reward (cumulative freezes after traj end)
  * near-constant / non-finite / explosive streams are rejected (anti-exploit)
  * train vs holdout splits are supported by callers via dataset partitioning
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Union

import numpy as np

from crowd_nav.reward_search.dataset import Stage1Dataset, TrajectoryRecord
from crowd_nav.reward_search.rules import (
    dist_to_goal,
    rule_preference_score,
    spearman_correlation,
)
from crowd_nav.reward_search.sandbox.runtime import default_smoke_states
from crowd_nav.reward_search.state import RewardFunction


class Score1Fn(Protocol):
    def __call__(self, reward_fn: RewardFunction, *, candidate_id: str = "") -> float:
        ...


# Default anti-exploit thresholds (documented; tune under Needs Validation).
DEFAULT_MIN_STEP_STD = 1e-8
DEFAULT_MAX_ABS_STEP = 1.0e4


@dataclass(frozen=True)
class Score1Options:
    """Knobs for trustworthy Score1 (Phase 1)."""

    mask_pads: bool = True
    anti_exploit: bool = True
    min_step_std: float = DEFAULT_MIN_STEP_STD
    max_abs_step: float = DEFAULT_MAX_ABS_STEP


@dataclass
class Score1Report:
    """Structured Score1 result for logging / holdout / mutation diagnostics."""

    score: float
    n_scenarios_scored: int
    n_scenarios_total: int
    anti_exploit_triggered: bool = False
    anti_exploit_reason: str = ""
    mask_pads: bool = True
    options: Dict[str, Any] = field(default_factory=dict)
    # Phase 4 / T14 — per-scenario mean Spearman (empty when anti-exploit fires early).
    per_scenario: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def weak_scenarios(self, n: int = 3) -> List[Dict[str, Any]]:
        """Lowest mean-Spearman scenarios (actionable mutation focus)."""
        ranked = sorted(self.per_scenario.items(), key=lambda kv: kv[1])
        return [
            {"scenario_id": sid, "mean_spearman": float(rho)}
            for sid, rho in ranked[: max(0, int(n))]
        ]


def format_score1_diagnostics(
    report: Optional[Union[Score1Report, Mapping[str, Any]]],
    *,
    n_weak: int = 3,
) -> str:
    """
    Compact diagnostics string for D.2 mutation / reflection prompts (T14).

    Accepts a ``Score1Report`` or its ``to_dict()`` payload from candidate metadata.
    """
    if report is None:
        return ""
    if isinstance(report, Score1Report):
        data = report.to_dict()
    elif isinstance(report, Mapping):
        data = dict(report)
    else:
        return ""

    parts: List[str] = []
    if data.get("anti_exploit_triggered"):
        reason = data.get("anti_exploit_reason") or "anti_exploit"
        parts.append(f"anti-exploit={reason}")
    per = data.get("per_scenario") or {}
    if isinstance(per, Mapping) and per:
        ranked = sorted(per.items(), key=lambda kv: float(kv[1]))
        weak = ranked[: max(0, int(n_weak))]
        if weak:
            bits = ", ".join(f"{sid}:{float(rho):.3f}" for sid, rho in weak)
            parts.append(f"weak scenarios [{bits}]")
            strong = ranked[-1]
            parts.append(f"best scenario {strong[0]}:{float(strong[1]):.3f}")
    n_scored = data.get("n_scenarios_scored")
    n_total = data.get("n_scenarios_total")
    if n_scored is not None and n_total is not None:
        parts.append(f"scored {n_scored}/{n_total} scenarios")
    return "; ".join(parts)


def _cumulative_reward(
    reward_fn: RewardFunction,
    traj: TrajectoryRecord,
    *,
    max_frame: int,
    mask_pads: bool = True,
    step_out: Optional[List[float]] = None,
) -> List[float]:
    """
    Prefix sums of recomputed rewards for frames 0..max_frame inclusive.

    When ``mask_pads`` is True (default), frames ``f >= traj.length`` do **not**
    call ``compute`` again; the cumulative total freezes. This prevents pad
    hold-last-state from inflating returns (S1-3).

    Optional ``step_out`` collects per-step rewards on real frames only
    (for anti-exploit checks).
    """
    reward_fn.reset()
    totals: List[float] = []
    running = 0.0
    n_pad = max_frame + 1
    for f in range(n_pad):
        if mask_pads and f >= traj.length:
            totals.append(running)
            continue
        state = traj.state_at(f)
        try:
            step = float(reward_fn.compute(state))
        except Exception:  # noqa: BLE001
            totals.append(float("nan"))
            running = float("nan")
            continue
        if step_out is not None and f < traj.length:
            step_out.append(step)
        if not math.isfinite(step) or not math.isfinite(running):
            running = float("nan")
            totals.append(float("nan"))
            continue
        running += step
        totals.append(running)
    return totals


def _anti_exploit_reason(
    step_rewards: Sequence[float],
    *,
    min_step_std: float,
    max_abs_step: float,
) -> str:
    """Return a reason string if the stream looks degenerate; else ''."""
    if not step_rewards:
        return "no_steps"
    if any(not math.isfinite(x) for x in step_rewards):
        return "non_finite_step"
    if any(abs(float(x)) > max_abs_step for x in step_rewards):
        return "explosive_magnitude"
    if len(step_rewards) >= 2:
        std = float(np.std(np.asarray(step_rewards, dtype=np.float64)))
        if not math.isfinite(std) or std < min_step_std:
            return "near_constant_reward"
    return ""


def _scenario_frame_correlations(
    trajs: Sequence[TrajectoryRecord],
    reward_fn: RewardFunction,
    *,
    mask_pads: bool = True,
    step_out: Optional[List[float]] = None,
) -> List[float]:
    """
    Spearman ρ for each frame f within one scenario.

    When ``mask_pads`` is True, a trajectory only participates at frame ``f`` if
    ``f < traj.length`` (padded hold-last frames are excluded from the
    correlation, not merely frozen in the cumulative). Frames with fewer than
    2 active trajectories are skipped.
    """
    if len(trajs) < 2:
        return []
    f_max = max(t.length for t in trajs)
    categories = [t.category for t in trajs]
    nav_lengths = [t.nav_length for t in trajs]

    cumuls: List[List[float]] = []
    for traj in trajs:
        cumuls.append(
            _cumulative_reward(
                reward_fn,
                traj,
                max_frame=f_max - 1,
                mask_pads=mask_pads,
                step_out=step_out,
            )
        )

    corrs: List[float] = []
    for f in range(f_max):
        rule_scores = []
        reward_scores = []
        ok = True
        for i, traj in enumerate(trajs):
            if mask_pads and f >= traj.length:
                continue
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
        if not ok or len(reward_scores) < 2:
            continue
        rho = spearman_correlation(rule_scores, reward_scores)
        if math.isfinite(rho):
            corrs.append(float(rho))
    return corrs


def score1_report(
    dataset: Stage1Dataset,
    reward_fn: RewardFunction,
    *,
    candidate_id: str = "",
    options: Optional[Score1Options] = None,
) -> Score1Report:
    """
    Compute Score1 with pad masking and optional anti-exploit gates.

    Anti-exploit failures return ``score=-inf`` with ``anti_exploit_triggered``.
    """
    del candidate_id  # reserved for logging hooks
    opts = options or Score1Options()
    if not isinstance(dataset, dict) or not dataset:
        raise ValueError("dataset must be a non-empty dict[scenario_id, trajectories]")

    all_steps: List[float] = []
    scenario_means: List[float] = []
    per_scenario: Dict[str, float] = {}
    opt_dict = {
        "mask_pads": opts.mask_pads,
        "anti_exploit": opts.anti_exploit,
        "min_step_std": opts.min_step_std,
        "max_abs_step": opts.max_abs_step,
    }
    try:
        for sid, trajs in dataset.items():
            usable = [t for t in trajs if t.length >= 1]
            if len(usable) < 2:
                continue
            frame_corrs = _scenario_frame_correlations(
                usable,
                reward_fn,
                mask_pads=opts.mask_pads,
                step_out=all_steps if opts.anti_exploit else None,
            )
            if not frame_corrs:
                continue
            mean_rho = float(np.mean(frame_corrs))
            scenario_means.append(mean_rho)
            per_scenario[str(sid)] = mean_rho
    except Exception as exc:  # noqa: BLE001 — sandbox non-finite / runtime
        from crowd_nav.reward_search.sandbox.errors import RewardSandboxError

        if isinstance(exc, RewardSandboxError) or "non-finite" in str(exc).lower():
            return Score1Report(
                score=float("-inf"),
                n_scenarios_scored=0,
                n_scenarios_total=len(dataset),
                anti_exploit_triggered=True,
                anti_exploit_reason=f"runtime_non_finite:{type(exc).__name__}",
                mask_pads=opts.mask_pads,
                options=opt_dict,
                per_scenario={},
            )
        raise

    if not scenario_means:
        # Degenerate streams can yield no finite Spearman frames; still apply
        # anti-exploit when we observed real step rewards (Phase 1.5).
        if opts.anti_exploit and all_steps:
            reason = _anti_exploit_reason(
                all_steps,
                min_step_std=opts.min_step_std,
                max_abs_step=opts.max_abs_step,
            )
            if reason:
                return Score1Report(
                    score=float("-inf"),
                    n_scenarios_scored=0,
                    n_scenarios_total=len(dataset),
                    anti_exploit_triggered=True,
                    anti_exploit_reason=reason,
                    mask_pads=opts.mask_pads,
                    options=opt_dict,
                    per_scenario={},
                )
        raise ValueError(
            "score1_for_dataset: no scoreable scenarios "
            "(need ≥2 trajectories with finite frame correlations)."
        )

    if opts.anti_exploit:
        reason = _anti_exploit_reason(
            all_steps,
            min_step_std=opts.min_step_std,
            max_abs_step=opts.max_abs_step,
        )
        if reason:
            return Score1Report(
                score=float("-inf"),
                n_scenarios_scored=0,
                n_scenarios_total=len(dataset),
                anti_exploit_triggered=True,
                anti_exploit_reason=reason,
                mask_pads=opts.mask_pads,
                options=opt_dict,
                per_scenario=dict(per_scenario),
            )

    score = float(np.mean(scenario_means))
    score = float(max(-1.0, min(1.0, score)))
    return Score1Report(
        score=score,
        n_scenarios_scored=len(scenario_means),
        n_scenarios_total=len(dataset),
        anti_exploit_triggered=False,
        anti_exploit_reason="",
        mask_pads=opts.mask_pads,
        options=opt_dict,
        per_scenario=dict(per_scenario),
    )


def score1_for_dataset(
    dataset: Union[Stage1Dataset, RewardFunction],
    reward_fn: Optional[RewardFunction] = None,
    *,
    candidate_id: str = "",
    options: Optional[Score1Options] = None,
) -> float:
    """
    Analytical Score1 over pre-collected scenarios (baseline paper Eq. 1 / Figure 3).

    Call as ``score1_for_dataset(dataset, reward_fn)``. Scenarios with fewer
    than 2 trajectories are skipped. Raises ``ValueError`` only if every
    scenario is unusable (anti-exploit returns ``-inf`` without raising).
    """
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

    return score1_report(
        dataset, reward_fn, candidate_id=candidate_id, options=options
    ).score


def make_score1_fn(
    dataset: Stage1Dataset,
    *,
    options: Optional[Score1Options] = None,
) -> Callable[..., float]:
    """Bind a loaded dataset once for StageIEvolver.score_fn (float only)."""

    opts = options or Score1Options()

    def _fn(reward_fn: RewardFunction, *, candidate_id: str = "") -> float:
        return score1_for_dataset(
            dataset, reward_fn, candidate_id=candidate_id, options=opts
        )

    return _fn


def make_score1_report_fn(
    dataset: Stage1Dataset,
    *,
    options: Optional[Score1Options] = None,
) -> Callable[..., Score1Report]:
    """
    Bind dataset for Stage I scoring that returns ``Score1Report`` (Phase 4 / T14).

    ``StageIEvolver.score_population`` stores the report on candidate metadata and
    ranks on ``report.score``.
    """

    opts = options or Score1Options()

    def _fn(reward_fn: RewardFunction, *, candidate_id: str = "") -> Score1Report:
        return score1_report(
            dataset, reward_fn, candidate_id=candidate_id, options=opts
        )

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


def score1_mode_artifact(mode: str, *, fast: bool = False) -> Dict[str, Any]:
    """
    Hard labeling for manifests / CLI (S1-9).

    ``is_paper_score1`` is True only for real dataset Score1, never smoke/fast.
    """
    normalized = str(mode).strip().lower()
    is_paper = normalized == "dataset" and not fast
    warning = None
    if normalized == "smoke" or fast:
        warning = (
            "score1_mode is smoke/fast fixture — NOT paper Score1; "
            "do not use this run for paper claims or thesis metrics."
        )
    elif normalized != "dataset":
        warning = f"Unknown or non-paper score1_mode={mode!r}"
    return {
        "score1_mode": normalized,
        "is_paper_score1": is_paper,
        "score1_claim_ok": is_paper,
        "warning": warning,
    }
