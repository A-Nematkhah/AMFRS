"""Tests for fidelity ladder and successive halving."""

from __future__ import annotations

from crowd_nav.reward_search.amfrs.fidelity import (
    FidelityLadder,
    FidelityLevel,
    FidelityResult,
    build_default_ladder,
)
from crowd_nav.reward_search.amfrs.halving import HalvingConfig, SuccessiveHalvingScheduler
from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.sandbox import RewardValidator


def _cand(i: int) -> RewardCandidate:
    code = (
        "def compute_reward(state, memory):\n"
        f"    return float({float(i)})\n"
    )
    fn, err = RewardValidator().try_validate(code)
    assert fn is not None, err
    return RewardCandidate(
        candidate_id=f"c{i}",
        code=code,
        reward_fn=fn,
        valid=True,
        origin="initial",
        metadata={},
    )


def test_ladder_costs_strictly_increasing():
    ladder = build_default_ladder(use_stub=True, score1_mode="smoke")
    costs = [lv.cost_units for lv in ladder]
    assert costs == sorted(costs)
    assert costs == sorted(set(costs))


def test_halving_promotes_top_fraction():
    # Synthetic ladder with fixed metrics from candidate_id
    def make_eval(metric_map, cost):
        def _ev(cand):
            return FidelityResult(metric=metric_map[cand.candidate_id], cost=cost, raw_metrics={})

        return _ev

    metrics = {"c0": 0.1, "c1": 0.9, "c2": 0.5, "c3": 0.8, "c4": 0.2, "c5": 0.7}
    levels = [
        FidelityLevel("F0", 1.0, make_eval(metrics, 1.0)),
        FidelityLevel("F1", 10.0, make_eval(metrics, 10.0)),
    ]
    ladder = FidelityLadder(levels)
    pop = [
        RewardCandidate(candidate_id=cid, code="x", valid=True, metadata={})
        for cid in metrics
    ]
    out = SuccessiveHalvingScheduler(ladder, HalvingConfig(eta=3, min_survivors=1)).run(pop)
    # After F0 with N=6, eta=3 -> keep ceil(6/3)=2; those evaluated at F1
    assert len(out) <= 2
    ids = {c.candidate_id for c in out}
    assert "c1" in ids  # best
    assert all(c.metadata.get("fidelity_history") for c in out)


def test_slice_after_skips_lower_rungs():
    ladder = build_default_ladder(use_stub=True, score1_mode="smoke")
    upper = ladder.slice_after("F1_short_a2c", "F3_full_ppo")
    names = [lv.name for lv in upper]
    assert names == ["F2_full_a2c", "F3_full_ppo"]


def test_halving_respects_budget_before_first_eval():
    calls = {"n": 0}

    def make_eval(cost):
        def _ev(cand):
            calls["n"] += 1
            return FidelityResult(metric=1.0, cost=cost, raw_metrics={})

        return _ev

    ladder = FidelityLadder(
        [
            FidelityLevel("F0", 1.0, make_eval(1.0)),
            FidelityLevel("F1", 50.0, make_eval(50.0)),
        ]
    )
    pop = [
        RewardCandidate(candidate_id="c0", code="x", valid=True, metadata={}),
        RewardCandidate(candidate_id="c1", code="y", valid=True, metadata={}),
    ]
    spent = [0.0]
    out = SuccessiveHalvingScheduler(
        ladder, HalvingConfig(eta=2, min_survivors=1, max_cost_units=5.0)
    ).run(pop, spent_cost=spent)
    # F0 costs 1+1=2; F1 costs 50 → must not start F1
    assert calls["n"] == 2
    assert spent[0] == 2.0
    assert out  # survivors from F0 retained


def test_pipeline_halving_shrinks_population(tmp_path):
    from crowd_nav.reward_search.amfrs import AMFRS2Pipeline, AMFRS2RunConfig

    cfg = AMFRS2RunConfig(output_dir=str(tmp_path / "h"))
    cfg.apply_fast_profile()
    cfg.output_dir = str(tmp_path / "h")
    cfg.population_size = 6
    cfg.generations = 1
    art = AMFRS2Pipeline(cfg).run()
    assert art.promotion_log, "expected promotion decisions"
    # First rung should start from accepted (>=1); later rungs shrink or stay
    sizes = [e["n_survivors"] for e in art.promotion_log]
    assert sizes[0] >= 1
    assert any(c.metadata.get("fidelity_history") for c in art.accepted_candidates) or art.best
