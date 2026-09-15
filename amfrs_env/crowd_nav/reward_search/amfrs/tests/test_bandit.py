"""Tests for UCB1 allocator."""

from __future__ import annotations

from crowd_nav.reward_search.amfrs.bandit import UCB1Allocator


def test_ucb1_explores_unpulled_then_exploits():
    alloc = UCB1Allocator(exploration_c=2.0, cost_aware=False)
    ids = ["a", "b", "c"]
    # Unpulled arms sort first (inf UCB)
    first = alloc.select_next(ids, k=1)[0]
    assert first in ids
    # Pull a mediocre arm many times, leave best under-explored
    for _ in range(20):
        alloc.record("a", 0.2, 1.0)
    alloc.record("b", 0.9, 1.0)
    # After some pulls, b should outrank a when selecting among {a,b}
    ranked = alloc.select_next(["a", "b"], k=2)
    assert ranked[0] == "b"
