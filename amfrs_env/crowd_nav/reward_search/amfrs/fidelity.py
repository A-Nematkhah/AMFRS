"""
Axis 1 — fidelity ladder (CPU-safe stubs; real trainers via deferred import).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.selection import navigation_scalar_from_dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FidelityResult:
    metric: float
    cost: float
    raw_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FidelityLevel:
    name: str
    cost_units: float
    evaluate: Callable[[RewardCandidate], FidelityResult]


@dataclass
class TrainerContext:
    """Settings forwarded into Stage2/Stage3 RealPolicyTrainer configs."""

    stage2_train_steps: int = 50_000
    stage2_train_steps_short: int = 12_500
    stage2_eval_episodes: int = 50
    stage3_train_steps: int = 500_000
    stage3_eval_episodes: int = 500
    seed: int = 425
    device: str = "cpu"
    num_processes: Optional[int] = None
    human_num: int = 20
    predict_method: str = "inferred"
    randomization_regime: str = "without_random"
    output_root: str = "trained_models/amfrs2"


class FidelityLadder:
    def __init__(self, levels: Sequence[FidelityLevel]) -> None:
        if not levels:
            raise ValueError("FidelityLadder requires at least one level")
        ordered = sorted(levels, key=lambda lv: float(lv.cost_units))
        for i in range(1, len(ordered)):
            if ordered[i].cost_units <= ordered[i - 1].cost_units:
                raise ValueError("fidelity costs must be strictly increasing")
        self._levels = list(ordered)

    def __iter__(self):
        return iter(self._levels)

    def __len__(self) -> int:
        return len(self._levels)

    @property
    def levels(self) -> List[FidelityLevel]:
        return list(self._levels)

    def get(self, name: str) -> FidelityLevel:
        for lv in self._levels:
            if lv.name == name:
                return lv
        raise KeyError(name)

    def next_level(self, current: FidelityLevel) -> Optional[FidelityLevel]:
        for i, lv in enumerate(self._levels):
            if lv.name == current.name:
                if i + 1 < len(self._levels):
                    return self._levels[i + 1]
                return None
        raise KeyError(current.name)

    def truncate_to(self, max_rung_name: str) -> "FidelityLadder":
        names = [lv.name for lv in self._levels]
        if max_rung_name not in names:
            raise KeyError(max_rung_name)
        idx = names.index(max_rung_name)
        return FidelityLadder(self._levels[: idx + 1])


def _code_hash(code: str) -> int:
    return int(hashlib.md5((code or "").encode("utf-8")).hexdigest()[:8], 16)


def _attach_goal_dist0(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure BD-x has goal_dist0 without modifying ProxyMetrics."""
    out = dict(raw)
    if "goal_dist0" not in out or float(out.get("goal_dist0") or 0.0) <= 0.0:
        pl = float(out.get("PL", out.get("pl", 12.0)))
        sr = float(out.get("SR", out.get("sr", 0.5)))
        # Heuristic: more successful policies → path closer to straight-line.
        out["goal_dist0"] = max(1.0, pl / max(1.05, 1.0 + (1.0 - sr)))
    return out


def _stub_proxy_metrics(candidate: RewardCandidate, *, rung: str) -> Dict[str, Any]:
    m = re.search(
        r"return\s+float\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\)",
        candidate.code or "",
    )
    if m:
        base = min(1.0, max(0.0, float(m.group(1)) / 20.0))
    else:
        base = (_code_hash(candidate.code) % 100) / 100.0
    rung_boost = {
        "F0_score1": 0.0,
        "F1_short_a2c": 0.02,
        "F2_full_a2c": 0.04,
        "F3_full_ppo": 0.06,
    }.get(rung, 0.0)
    sr = min(1.0, max(0.0, 0.4 + 0.5 * base + rung_boost))
    cr = min(1.0, max(0.0, 0.4 * (1.0 - base)))
    tr = max(0.0, 1.0 - sr - cr)
    pl = 12.0 + 8.0 * (1.0 - base)
    goal_dist0 = 8.0 + 2.0 * base
    return {
        "SR": sr,
        "CR": cr,
        "TR": tr,
        "NT": 10.0 + 5.0 * (1.0 - base),
        "PL": pl,
        "ITR": 5.0 + 20.0 * (1.0 - base),
        "SD": 0.2 + 0.3 * base,
        "goal_dist0": goal_dist0,
        "n_seeds": 1.0,
    }


def _evaluate_f0_smoke(candidate: RewardCandidate, cost: float) -> FidelityResult:
    from crowd_nav.reward_search.scoring import make_smoke_score_fn

    if candidate.reward_fn is None:
        return FidelityResult(
            metric=float("-inf"), cost=cost, raw_metrics={"error": "no_reward_fn"}
        )
    score = float(
        make_smoke_score_fn()(candidate.reward_fn, candidate_id=candidate.candidate_id)
    )
    return FidelityResult(
        metric=score, cost=cost, raw_metrics={"score1": score, "mode": "smoke"}
    )


def _evaluate_f0_dataset(
    candidate: RewardCandidate, dataset_path: str, cost: float
) -> FidelityResult:
    from crowd_nav.reward_search.dataset import load_stage1_dataset
    from crowd_nav.reward_search.scoring import score1_for_dataset

    if candidate.reward_fn is None:
        return FidelityResult(
            metric=float("-inf"), cost=cost, raw_metrics={"error": "no_reward_fn"}
        )
    try:
        dataset = load_stage1_dataset(dataset_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Stage I dataset missing at {dataset_path!r}. "
            "Run: python scripts/collect_stage1_dataset.py "
            "or use --fast / --score1 smoke."
        ) from exc
    score = float(
        score1_for_dataset(
            dataset, candidate.reward_fn, candidate_id=candidate.candidate_id
        )
    )
    return FidelityResult(
        metric=score, cost=cost, raw_metrics={"score1": score, "mode": "dataset"}
    )


def _evaluate_stub_rung(
    candidate: RewardCandidate, *, rung: str, cost: float
) -> FidelityResult:
    raw = _stub_proxy_metrics(candidate, rung=rung)
    metric = navigation_scalar_from_dict(raw)
    return FidelityResult(metric=float(metric), cost=cost, raw_metrics=raw)


def _env_name_for_predict(predict_method: str) -> str:
    from crowd_nav.reward_search.regime import (
        EVOLUTION_ENV_NAME_INFERRED,
        EVOLUTION_ENV_NAME_NONE,
    )

    return (
        EVOLUTION_ENV_NAME_INFERRED
        if str(predict_method).lower() == "inferred"
        else EVOLUTION_ENV_NAME_NONE
    )


def _evaluate_real_stage2(
    candidate: RewardCandidate,
    *,
    short: bool,
    cost: float,
    ctx: TrainerContext,
) -> FidelityResult:
    from crowd_nav.reward_search.stage2 import RealPolicyTrainer, Stage2Config

    steps = (
        int(ctx.stage2_train_steps_short) if short else int(ctx.stage2_train_steps)
    )
    cfg = Stage2Config(
        train_env_steps=steps,
        eval_episodes=int(ctx.stage2_eval_episodes),
        seed=int(ctx.seed),
        device=str(ctx.device),
        num_processes=ctx.num_processes,
        human_num=int(ctx.human_num),
        predict_method=str(ctx.predict_method),
        env_name=_env_name_for_predict(ctx.predict_method),
        randomization_regime=str(ctx.randomization_regime),
        output_root=os_path_join(ctx.output_root, "stage2"),
        n_eval_seeds=1,
        accept_reject_refine=False,
    )
    logger.info(
        "Real Stage II %s steps=%s predict=%s device=%s",
        candidate.candidate_id,
        steps,
        cfg.predict_method,
        cfg.device,
    )
    metrics = RealPolicyTrainer().train_and_eval(
        candidate, round_index=0, config=cfg
    )
    raw = _attach_goal_dist0(metrics.as_dict())
    raw["trainer"] = "RealPolicyTrainer"
    raw["rung"] = "F1_short_a2c" if short else "F2_full_a2c"
    return FidelityResult(
        metric=float(navigation_scalar_from_dict(raw)),
        cost=cost,
        raw_metrics=raw,
    )


def _evaluate_real_stage3(
    candidate: RewardCandidate,
    *,
    cost: float,
    ctx: TrainerContext,
) -> FidelityResult:
    from crowd_nav.reward_search.stage3 import RealPolicyTrainer, Stage3Config

    cfg = Stage3Config(
        train_env_steps=int(ctx.stage3_train_steps),
        eval_episodes=int(ctx.stage3_eval_episodes),
        seed=int(ctx.seed),
        device=str(ctx.device),
        num_processes=ctx.num_processes,
        train_human_num=int(ctx.human_num),
        predict_method=str(ctx.predict_method),
        env_name=_env_name_for_predict(ctx.predict_method),
        randomization_regime=str(ctx.randomization_regime),
        output_root=os_path_join(ctx.output_root, "stage3"),
        accept_reject_refine=False,
    )
    logger.info(
        "Real Stage III %s steps=%s predict=%s device=%s",
        candidate.candidate_id,
        cfg.train_env_steps,
        cfg.predict_method,
        cfg.device,
    )
    bundle = RealPolicyTrainer().train_and_eval(
        candidate, round_index=0, config=cfg
    )
    metrics = bundle.metrics
    raw = _attach_goal_dist0(metrics.as_dict())
    raw["trainer"] = "RealPolicyTrainer"
    raw["rung"] = "F3_full_ppo"
    if bundle.checkpoint_path:
        raw["checkpoint_path"] = bundle.checkpoint_path
    return FidelityResult(
        metric=float(navigation_scalar_from_dict(raw)),
        cost=cost,
        raw_metrics=raw,
    )


def os_path_join(*parts: str) -> str:
    import os

    return os.path.join(*parts)


def build_default_ladder(
    *,
    use_stub: bool = True,
    score1_mode: str = "smoke",
    stage1_dataset_path: str = "data/stage1_dataset",
    cost_f0: float = 1.0,
    cost_f1: float = 10.0,
    cost_f2: float = 40.0,
    cost_f3: float = 400.0,
    trainer_ctx: Optional[TrainerContext] = None,
) -> FidelityLadder:
    """
    Build F0→F3 ladder.

    ``use_stub=True`` → deterministic hash metrics (no torch).
    ``use_stub=False`` → ``RealPolicyTrainer`` for F1–F3 (needs GST if inferred).
    """
    ctx = trainer_ctx or TrainerContext()

    def eval_f0(cand: RewardCandidate) -> FidelityResult:
        if score1_mode == "smoke":
            return _evaluate_f0_smoke(cand, cost_f0)
        return _evaluate_f0_dataset(cand, stage1_dataset_path, cost_f0)

    def eval_f1(cand: RewardCandidate) -> FidelityResult:
        if use_stub:
            return _evaluate_stub_rung(cand, rung="F1_short_a2c", cost=cost_f1)
        return _evaluate_real_stage2(cand, short=True, cost=cost_f1, ctx=ctx)

    def eval_f2(cand: RewardCandidate) -> FidelityResult:
        if use_stub:
            return _evaluate_stub_rung(cand, rung="F2_full_a2c", cost=cost_f2)
        return _evaluate_real_stage2(cand, short=False, cost=cost_f2, ctx=ctx)

    def eval_f3(cand: RewardCandidate) -> FidelityResult:
        if use_stub:
            return _evaluate_stub_rung(cand, rung="F3_full_ppo", cost=cost_f3)
        return _evaluate_real_stage3(cand, cost=cost_f3, ctx=ctx)

    return FidelityLadder(
        [
            FidelityLevel("F0_score1", cost_f0, eval_f0),
            FidelityLevel("F1_short_a2c", cost_f1, eval_f1),
            FidelityLevel("F2_full_a2c", cost_f2, eval_f2),
            FidelityLevel("F3_full_ppo", cost_f3, eval_f3),
        ]
    )


def evaluate_at(level: FidelityLevel, candidate: RewardCandidate) -> FidelityResult:
    return level.evaluate(candidate)


def append_fidelity_history(
    candidate: RewardCandidate,
    level_name: str,
    result: FidelityResult,
) -> RewardCandidate:
    from dataclasses import replace

    hist = list((candidate.metadata or {}).get("fidelity_history") or [])
    hist.append(
        {
            "level": level_name,
            "metric": float(result.metric),
            "cost": float(result.cost),
            "raw_metrics": dict(result.raw_metrics),
        }
    )
    md = {
        **(candidate.metadata or {}),
        "fidelity_history": hist,
        "last_fidelity": level_name,
        "last_metric": float(result.metric),
        "last_raw_metrics": dict(result.raw_metrics),
    }
    return replace(candidate, score=float(result.metric), metadata=md)
