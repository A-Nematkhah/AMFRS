"""Tests for Axis 4 robustness helpers."""

from __future__ import annotations

from crowd_nav.reward_search.amfrs.robustness import (
    attach_robustness_profile,
    pick_best_by_robustness,
    robustness_scalar,
    run_policy_sweep,
)
from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.selection import navigation_scalar_from_dict


def test_robustness_scalar_is_worst_case():
    by = {
        "orca": {"SR": 0.9, "CR": 0.05, "TR": 0.05, "ITR": 0.0, "SD": 0.5},
        "social_force": {"SR": 0.4, "CR": 0.3, "TR": 0.3, "ITR": 0.0, "SD": 0.1},
    }
    rs = robustness_scalar(by)
    assert rs == navigation_scalar_from_dict(by["social_force"])


def test_policy_sweep_stub_keys(tmp_path):
    cand = RewardCandidate(
        candidate_id="c0",
        code="def compute_reward(state, memory):\n    return float(5.0)\n",
        valid=True,
        metadata={},
    )
    out = run_policy_sweep(cand, use_stub=True)
    assert "orca" in out and "social_force" in out
    attached = attach_robustness_profile(cand, out)
    assert "robustness_scalar" in attached.metadata
    best = pick_best_by_robustness([attached])
    assert best is not None


def test_real_sweep_does_not_silent_stub_fallback(monkeypatch):
    cand = RewardCandidate(
        candidate_id="c0",
        code="def compute_reward(state, memory):\n    return float(5.0)\n",
        valid=True,
        metadata={},
    )

    def _boom(*_a, **_k):
        raise RuntimeError("trainer exploded")

    monkeypatch.setattr(
        "crowd_nav.reward_search.amfrs.robustness._run_policy_sweep_real",
        _boom,
    )
    try:
        run_policy_sweep(cand, use_stub=False, allow_stub_fallback=False)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "trainer exploded" in str(exc)


def test_opt_in_stub_fallback_marks_metrics(monkeypatch):
    cand = RewardCandidate(
        candidate_id="c0",
        code="def compute_reward(state, memory):\n    return float(5.0)\n",
        valid=True,
        metadata={},
    )

    def _boom(*_a, **_k):
        raise RuntimeError("trainer exploded")

    monkeypatch.setattr(
        "crowd_nav.reward_search.amfrs.robustness._run_policy_sweep_real",
        _boom,
    )
    out = run_policy_sweep(cand, use_stub=False, allow_stub_fallback=True)
    assert out["orca"].get("stub_fallback") is True
