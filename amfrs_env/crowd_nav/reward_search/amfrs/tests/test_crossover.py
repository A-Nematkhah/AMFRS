"""Tests for Axis 3 crossover / diagnostics helpers."""

from __future__ import annotations

from crowd_nav.reward_search.amfrs.crossover import (
    build_mutation_prompt,
    build_semantic_crossover_prompt,
    format_candidate_diagnostics,
)
from crowd_nav.reward_search.evolver import RewardCandidate


def test_format_candidate_diagnostics_from_raw_metrics():
    cand = RewardCandidate(
        candidate_id="p0",
        code="def compute_reward(state, memory):\n    return 1.0\n",
        valid=True,
        metadata={
            "last_metric": 0.42,
            "last_fidelity_level": "F1_short_a2c",
            "last_raw_metrics": {"SR": 0.62, "CR": 0.18, "SD": 0.31},
        },
    )
    diag = format_candidate_diagnostics(cand)
    assert "SR=0.620" in diag
    assert "CR=0.180" in diag
    assert "last_rung=F1_short_a2c" in diag


def test_mutation_prompt_includes_diagnostics():
    cand = RewardCandidate(
        candidate_id="p0",
        code="def compute_reward(state, memory):\n    return 1.0\n",
        valid=True,
        metadata={"last_raw_metrics": {"SR": 0.5}},
    )
    prompt = build_mutation_prompt(cand, diagnostics=format_candidate_diagnostics(cand))
    assert "Diagnostics:" in prompt
    assert "SR=0.500" in prompt


def test_crossover_prompt_includes_both_diagnostics():
    a = RewardCandidate(
        candidate_id="a",
        code="def compute_reward(state, memory):\n    return 1.0\n",
        valid=True,
        metadata={"last_raw_metrics": {"SR": 0.9}},
    )
    b = RewardCandidate(
        candidate_id="b",
        code="def compute_reward(state, memory):\n    return 2.0\n",
        valid=True,
        metadata={"last_raw_metrics": {"SR": 0.4}},
    )
    prompt = build_semantic_crossover_prompt(
        a,
        b,
        diagnostics_a=format_candidate_diagnostics(a),
        diagnostics_b=format_candidate_diagnostics(b),
    )
    assert "Diagnostics A:" in prompt
    assert "SR=0.900" in prompt
    assert "SR=0.400" in prompt
