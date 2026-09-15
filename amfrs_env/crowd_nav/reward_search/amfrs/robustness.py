"""
Axis 4 — multi-policy robustness (no adversarial YAML in this milestone).

Real trainer failures do not silently fall back to stub metrics used for
finalist ranking unless ``allow_stub_fallback=True``.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.selection import navigation_scalar_from_dict

logger = logging.getLogger(__name__)


def robustness_scalar(by_policy: Mapping[str, Mapping[str, Any]]) -> float:
    """Worst-case navigation scalar across human policies."""
    if not by_policy:
        return float("-inf")
    vals = [navigation_scalar_from_dict(m) for m in by_policy.values()]
    finite = [v for v in vals if v > float("-inf")]
    if not finite:
        return float("-inf")
    return float(min(finite))


def attach_robustness_profile(
    candidate: RewardCandidate,
    by_policy: Mapping[str, Mapping[str, Any]],
) -> RewardCandidate:
    score = robustness_scalar(by_policy)
    return replace(
        candidate,
        metadata={
            **(candidate.metadata or {}),
            "robustness_scalar": float(score),
            "robustness_by_policy": {k: dict(v) for k, v in by_policy.items()},
        },
    )


def pick_best_by_robustness(
    candidates: Sequence[RewardCandidate],
) -> Optional[RewardCandidate]:
    scored = []
    for cand in candidates:
        md = cand.metadata or {}
        if "robustness_scalar" not in md:
            continue
        scored.append((float(md["robustness_scalar"]), cand))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    best = scored[0][1]
    return replace(
        best,
        metadata={**(best.metadata or {}), "robustness_selected": True},
    )


def run_policy_sweep_stub(
    candidate: RewardCandidate,
    policies: Tuple[str, ...] = ("orca", "social_force"),
) -> Dict[str, Dict[str, Any]]:
    """
    Deterministic stub sweep for --fast / unit tests.

    Slightly degrades social_force vs orca so robustness_scalar < orca-only scalar.
    """
    from crowd_nav.reward_search.amfrs.fidelity import _stub_proxy_metrics

    base = _stub_proxy_metrics(candidate, rung="F3_full_ppo")
    out: Dict[str, Dict[str, Any]] = {}
    for i, pol in enumerate(policies):
        m = dict(base)
        if pol != "orca":
            m["SR"] = max(0.0, float(m["SR"]) - 0.05 * (i + 1))
            m["CR"] = min(1.0, float(m["CR"]) + 0.03 * (i + 1))
            m["SD"] = max(0.0, float(m["SD"]) - 0.05)
        m["policy"] = pol
        out[pol] = m
    return out


def _run_policy_sweep_real(
    candidate: RewardCandidate,
    policies: Tuple[str, ...],
    *,
    train_steps: int = 8_000,
    eval_episodes: int = 20,
    device: str = "cpu",
    predict_method: str = "none",
    human_num: int = 5,
    seed: int = 425,
    output_root: str = "trained_models/amfrs2_robustness",
    randomization_regime: str = "without_random",
) -> Dict[str, Dict[str, Any]]:
    """
    Short real A2C train+eval per human policy (expensive).

    Uses ``predict_method`` as given (prefer ``none`` if GST missing).
    """
    from crowd_nav.reward_search.regime import (
        EVOLUTION_ENV_NAME_INFERRED,
        EVOLUTION_ENV_NAME_NONE,
    )
    from crowd_nav.reward_search.stage2 import RealPolicyTrainer, Stage2Config

    if candidate.reward_fn is None:
        raise ValueError("candidate has no reward_fn for real policy sweep")

    env_name = (
        EVOLUTION_ENV_NAME_INFERRED
        if str(predict_method).lower() == "inferred"
        else EVOLUTION_ENV_NAME_NONE
    )
    out: Dict[str, Dict[str, Any]] = {}
    trainer = RealPolicyTrainer()
    for i, pol in enumerate(policies):
        cfg = Stage2Config(
            train_env_steps=int(train_steps),
            eval_episodes=int(eval_episodes),
            seed=int(seed) + i,
            device=str(device),
            human_num=int(human_num),
            predict_method=str(predict_method),
            env_name=env_name,
            randomization_regime=str(randomization_regime),
            output_root=f"{output_root}/{pol}",
            n_eval_seeds=1,
            accept_reject_refine=False,
        )
        # Human policy is applied inside env Config; Stage2Config does not
        # expose it, so we patch via a thin monkeyhook on _make_proxy_env_config.
        import crowd_nav.reward_search.stage2 as stage2_mod

        orig = stage2_mod._make_proxy_env_config

        def _make_cfg(config, _pol=pol, _orig=orig):
            env_cfg = _orig(config)
            env_cfg.humans.policy = _pol
            return env_cfg

        stage2_mod._make_proxy_env_config = _make_cfg  # type: ignore[assignment]
        try:
            logger.info("Robustness real sweep policy=%s candidate=%s", pol, candidate.candidate_id)
            metrics = trainer.train_and_eval(candidate, round_index=0, config=cfg)
            raw = metrics.as_dict()
            raw["policy"] = pol
            raw["trainer"] = "RealPolicyTrainer"
            out[pol] = raw
        finally:
            stage2_mod._make_proxy_env_config = orig  # type: ignore[assignment]
    return out


def run_policy_sweep(
    candidate: RewardCandidate,
    policies: Tuple[str, ...] = ("orca", "social_force"),
    *,
    use_stub: bool = True,
    allow_stub_fallback: bool = False,
    train_steps: int = 8_000,
    eval_episodes: int = 20,
    device: str = "cpu",
    predict_method: str = "none",
    human_num: int = 5,
    seed: int = 425,
    output_root: str = "trained_models/amfrs2_robustness",
    randomization_regime: str = "without_random",
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate candidate under different ``humans.policy`` settings.

    When ``use_stub`` is False, real failures raise unless
    ``allow_stub_fallback`` is True (intentionally opt-in; never used to
    silently rank finalists after a real-trainer crash).
    """
    if use_stub:
        return run_policy_sweep_stub(candidate, policies=policies)
    try:
        return _run_policy_sweep_real(
            candidate,
            policies,
            train_steps=train_steps,
            eval_episodes=eval_episodes,
            device=device,
            predict_method=predict_method,
            human_num=human_num,
            seed=seed,
            output_root=output_root,
            randomization_regime=randomization_regime,
        )
    except Exception as exc:  # noqa: BLE001
        if not allow_stub_fallback:
            raise
        logger.warning(
            "Real policy sweep failed (%s); falling back to stub metrics "
            "(allow_stub_fallback=True)",
            exc,
        )
        out = run_policy_sweep_stub(candidate, policies=policies)
        for v in out.values():
            v["note"] = f"real_policy_sweep_failed:{type(exc).__name__}"
            v["stub_fallback"] = True
        return out
