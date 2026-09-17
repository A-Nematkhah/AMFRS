"""Tests for static-gate reward scale-drift soft flag."""

from __future__ import annotations

from crowd_nav.reward_search.amfrs.static_gate import (
    SEED_REWARD_RANGE,
    StaticGate,
    StaticGateReport,
    check_scale_drift,
)
from crowd_nav.reward_search.sandbox import RewardValidator


def test_seed_scale_not_flagged():
    # Magnitudes matching D5_SEED_FUNCTION / SEED_REWARD_RANGE.
    code = (
        "def compute_reward(state, memory):\n"
        "    if state.collision:\n"
        "        return float(-20.0)\n"
        "    if state.reaching_goal:\n"
        "        return float(10.0)\n"
        "    return float(0.0)\n"
    )
    fn = RewardValidator().validate_code(code)
    report = StaticGate(n_states=32, seed=0).check(fn)
    assert "scale_drift" not in report.notes
    assert report.scale_drift_ratio <= 5.0


def test_inflated_scale_flagged_like_g4_m2():
    # Magnitudes matching the real-run g4_m2-style inflation (~300 / -200).
    code = (
        "def compute_reward(state, memory):\n"
        "    if state.collision:\n"
        "        return float(-200.0)\n"
        "    if state.reaching_goal:\n"
        "        return float(300.0)\n"
        "    return float(0.0)\n"
    )
    fn = RewardValidator().validate_code(code)
    report = StaticGate(n_states=32, seed=0).check(fn)
    assert "scale_drift" in report.notes
    assert report.scale_drift_ratio > 5.0


def test_smaller_than_seed_not_flagged():
    code = (
        "def compute_reward(state, memory):\n"
        "    if state.collision:\n"
        "        return float(-2.0)\n"
        "    if state.reaching_goal:\n"
        "        return float(1.0)\n"
        "    return float(0.0)\n"
    )
    fn = RewardValidator().validate_code(code)
    report = StaticGate(n_states=32, seed=0).check(fn)
    assert "scale_drift" not in report.notes
    assert report.scale_drift_ratio < 1.0


def test_check_scale_drift_formula_and_to_dict():
    bounded = StaticGateReport(all_finite=True, min_val=-200.0, max_val=300.0)
    ratio = check_scale_drift(bounded, seed_range=SEED_REWARD_RANGE)
    # max(200/20, 300/10) = 30
    assert abs(ratio - 30.0) < 1e-9
    report = StaticGateReport(
        all_finite=True,
        min_val=-200.0,
        max_val=300.0,
        scale_drift_ratio=ratio,
        notes=("scale_drift",),
    )
    d = report.to_dict()
    assert d["scale_drift_ratio"] == ratio
    assert "scale_drift" in d["notes"]
