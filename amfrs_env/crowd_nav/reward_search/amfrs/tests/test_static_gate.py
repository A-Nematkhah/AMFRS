"""Unit tests for Axis 5 static gate (no gym/torch)."""

from __future__ import annotations

from crowd_nav.reward_search.amfrs.static_gate import (
    StaticGate,
    check_bounded,
    check_ignores_humans_and_dmin,
    check_monotonicity_in_danger,
    sample_synthetic_states,
)
from crowd_nav.reward_search.sandbox import RewardValidator
from crowd_nav.reward_search.state import RewardState


def test_sampler_produces_typed_states():
    states = sample_synthetic_states(n=40, seed=1)
    assert len(states) == 40
    assert any(len(s.humans) == 0 for s in states)
    assert any(s.collision for s in states)
    assert all(isinstance(s, RewardState) for s in states)


def test_nan_candidate_hard_rejected():
    def bad(state, memory):
        return float("nan")

    report = check_bounded(bad, sample_synthetic_states(n=8, seed=0))
    assert report.all_finite is False


def test_constant_reward_flagged_human_blind():
    def const(state, memory):
        return 1.0

    assert check_ignores_humans_and_dmin(const) is True
    assert check_monotonicity_in_danger(const) is True  # flat is non-increasing


def test_bad_monotonicity_flagged():
    def closer_is_better(state, memory):
        # Rewards getting closer to humans (bad)
        return float(-state.dmin)

    assert check_monotonicity_in_danger(closer_is_better) is False


def test_good_seed_passes_finite():
    from crowd_nav.reward_search.prompts import D5_SEED_FUNCTION

    fn = RewardValidator().validate_code(D5_SEED_FUNCTION)
    gate = StaticGate(n_states=32, seed=0)
    report = gate.check(fn)
    assert report.all_finite is True
    # D5 seed uses goal potential + collision/goal flags, not dmin/humans —
    # soft-flagging crowd-blindness is expected and non-fatal.


def test_dmin_aware_reward_not_human_blind():
    code = (
        "def compute_reward(state, memory):\n"
        "    return float(-state.dmin - len(state.humans))\n"
    )
    fn = RewardValidator().validate_code(code)
    gate = StaticGate(n_states=32, seed=0)
    report = gate.check(fn)
    assert report.all_finite is True
    assert report.ignores_humans is False


def test_pipeline_hard_rejects_nan(tmp_path):
    from crowd_nav.reward_search.amfrs import AMFRSPipeline, AMFRSRunConfig
    from crowd_nav.reward_search.amfrs.static_gate import StaticGate
    from crowd_nav.reward_search.evolver import RewardCandidate

    cfg = AMFRSRunConfig(output_dir=str(tmp_path / "g"))
    cfg.apply_fast_profile()
    cfg.output_dir = str(tmp_path / "g")
    pipe = AMFRSPipeline(cfg)

    # Directly exercise gate path used by pipeline
    def nan_fn(state, memory):
        return float("nan")

    report = pipe.static_gate.check(nan_fn)
    assert report.all_finite is False
