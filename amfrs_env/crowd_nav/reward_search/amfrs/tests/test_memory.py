"""Tests for Axis 3 retrieval memory."""

from __future__ import annotations

from crowd_nav.reward_search.amfrs.memory import RewardMemory


def test_memory_similarity_and_persistence(tmp_path):
    db = tmp_path / "mem.sqlite"
    mem = RewardMemory(str(db))
    code_a = (
        "def compute_reward(state, memory):\n"
        "    return goal_progress(state, memory) - collision_indicator(state, memory)\n"
    )
    code_b = (
        "def compute_reward(state, memory):\n"
        "    return goal_progress(state, memory) - discomfort_penalty(state, memory)\n"
    )
    mem.add(code_a, run_id="r1", outcome_metrics={"SR": 0.8})
    mem.add(code_b, run_id="r2", outcome_metrics={"SR": 0.5})
    hits = mem.query_similar(code_a, k=2)
    mem.close()
    assert hits
    assert hits[0].similarity >= hits[-1].similarity
    assert hits[0].code == code_a
