"""Unit tests for best-ever navigation scalar selection + Phase 3 helpers."""

from __future__ import annotations

from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.selection import (
    attach_h_profile,
    best_by_ever_metrics,
    candidate_nav_scalar,
    h_profile_scalar,
    navigation_scalar,
    pick_best_by_h_profile,
    pick_best_trained,
    select_top_k_finalists,
)
from crowd_nav.reward_search.stage2 import ProxyMetrics, Stage2RoundRecord
from crowd_nav.reward_search.stage3 import HumanSweepReport


def test_navigation_scalar():
    # Legacy SR/CR/TR-only call matches historical base under default weights.
    assert navigation_scalar(0.68, 0.267, 0.053) == 0.68 - 0.267 - 0.5 * 0.053
    # T7: ITR penalized, SD rewarded.
    with_social = navigation_scalar(0.68, 0.267, 0.053, itr=10.0, sd=0.4)
    assert with_social == 0.68 - 0.267 - 0.5 * 0.053 - 0.01 * 10.0 + 0.2 * 0.4


def test_navigation_scalar_prefers_lower_itr_when_sr_tied():
    from crowd_nav.reward_search.selection import navigation_scalar_from_dict

    a = navigation_scalar_from_dict(
        {"SR": 0.5, "CR": 0.1, "TR": 0.1, "ITR": 20.0, "SD": 0.2}
    )
    b = navigation_scalar_from_dict(
        {"SR": 0.5, "CR": 0.1, "TR": 0.1, "ITR": 2.0, "SD": 0.2}
    )
    assert b > a


def test_pick_best_trained_prefers_higher_scalar():
    weak = RewardCandidate(
        candidate_id="weak",
        code="def compute_reward(state, memory):\n    return 0.0\n",
        metadata={
            "last_metrics": {"SR": 0.1, "CR": 0.5, "TR": 0.4},
            "trained_snapshot": True,
        },
    )
    strong = RewardCandidate(
        candidate_id="strong",
        code="def compute_reward(state, memory):\n    return 1.0\n",
        metadata={
            "last_metrics": {"SR": 0.68, "CR": 0.267, "TR": 0.053},
            "trained_snapshot": True,
            "checkpoint_path": "ckpt.pt",
        },
    )
    best = pick_best_trained([weak, strong])
    assert best is not None
    assert best.candidate_id == "strong"
    assert candidate_nav_scalar(best) > candidate_nav_scalar(weak)


def test_select_top_k_finalists_prefers_and_dedupes():
    a = RewardCandidate(
        candidate_id="a",
        code="def compute_reward(state, memory):\n    return 1.0\n",
        metadata={"last_metrics": {"SR": 0.9, "CR": 0.0, "TR": 0.1}},
    )
    b = RewardCandidate(
        candidate_id="b",
        code="def compute_reward(state, memory):\n    return 2.0\n",
        metadata={"last_metrics": {"SR": 0.5, "CR": 0.1, "TR": 0.1}},
    )
    a_clone = RewardCandidate(
        candidate_id="a2",
        code=a.code,
        metadata={"last_metrics": {"SR": 0.95, "CR": 0.0, "TR": 0.0}},
    )
    c = RewardCandidate(
        candidate_id="c",
        code="def compute_reward(state, memory):\n    return 3.0\n",
        metadata={"last_metrics": {"SR": 0.2, "CR": 0.2, "TR": 0.2}},
    )
    selected = select_top_k_finalists([a, b, a_clone, c], k=2, prefer=b)
    assert len(selected) == 2
    assert selected[0].candidate_id == "b"
    assert selected[1].candidate_id in {"a", "a2"}


def test_h_profile_scalar_blends_mean_and_worst():
    by_h = {
        5: {"SR": 0.8, "CR": 0.0, "TR": 0.0},
        20: {"SR": 0.4, "CR": 0.0, "TR": 0.0},
    }
    score = h_profile_scalar(by_h, mean_weight=0.5)
    # mean=0.6, worst=0.4 → 0.5*0.6 + 0.5*0.4 = 0.5
    assert abs(score - 0.5) < 1e-9


def test_pick_best_by_h_profile():
    weak = RewardCandidate(
        candidate_id="w",
        code="def compute_reward(state, memory):\n    return 0.0\n",
        metadata={"h_profile_scalar": 0.1},
    )
    strong = RewardCandidate(
        candidate_id="s",
        code="def compute_reward(state, memory):\n    return 1.0\n",
        metadata={"h_profile_scalar": 0.7},
    )
    best = pick_best_by_h_profile([weak, strong])
    assert best is not None
    assert best.candidate_id == "s"
    assert (best.metadata or {}).get("h_aware_selected") is True


def test_attach_h_profile():
    cand = RewardCandidate(
        candidate_id="c0",
        code="def compute_reward(state, memory):\n    return 1.0\n",
    )
    report = HumanSweepReport(
        candidate_id="c0",
        by_human_count={
            5: ProxyMetrics(sr=0.9, cr=0.0, tr=0.0),
            20: ProxyMetrics(sr=0.5, cr=0.1, tr=0.0),
        },
    )
    out = attach_h_profile(cand, report, mean_weight=0.5)
    assert "h_profile_scalar" in (out.metadata or {})
    assert "h_sweep" in (out.metadata or {})


def test_pipeline_best_by_ever_uses_snapshots_not_last_round_only():
    """Regression: R0 peak must beat a worse final-round candidate."""
    r0 = RewardCandidate(
        candidate_id="mut_0060_v2",
        code="def compute_reward(state, memory):\n    return 0.5\n",
        metadata={
            "last_metrics": {"SR": 0.68, "CR": 0.267, "TR": 0.053},
            "checkpoint_path": "r00.pt",
            "trained_snapshot": True,
        },
    )
    r1 = RewardCandidate(
        candidate_id="mut_0060_v3",
        code="def compute_reward(state, memory):\n    return 0.1\n",
        metadata={
            "last_metrics": {"SR": 0.467, "CR": 0.453, "TR": 0.08},
            "checkpoint_path": "r01.pt",
            "trained_snapshot": True,
        },
    )
    history = [
        Stage2RoundRecord(
            round_index=0,
            candidate_id="mut_0060_v2",
            metrics=ProxyMetrics(sr=0.68, cr=0.267, tr=0.053),
            refined=False,
            kept_previous=True,
        ),
        Stage2RoundRecord(
            round_index=1,
            candidate_id="mut_0060_v2",
            metrics=ProxyMetrics(sr=0.467, cr=0.453, tr=0.08),
            refined=True,
            kept_previous=False,
        ),
    ]
    best = best_by_ever_metrics(
        [r1], history, trained_snapshots=[r0, r1]
    )
    assert best.candidate_id == "mut_0060_v2"
    assert abs(candidate_nav_scalar(best) - navigation_scalar(0.68, 0.267, 0.053)) < 1e-9
