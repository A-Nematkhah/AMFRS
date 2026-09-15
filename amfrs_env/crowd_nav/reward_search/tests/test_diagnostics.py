"""Tests for T6 failure-mode diagnostics and D.3 extra context wiring."""

from __future__ import annotations

from crowd_nav.reward_search.diagnostics import (
    failure_mode_bullets,
    failure_mode_summary,
)
from crowd_nav.reward_search.prompts import format_d3_refinement
from crowd_nav.reward_search.stage2 import ProxyMetrics


def test_timeout_dominant_failure_mode():
    m = ProxyMetrics(sr=0.0, cr=0.1, tr=0.9, itr=2.0, sd=0.4)
    bullets = failure_mode_bullets(m)
    assert any("TIMEOUT" in b for b in bullets)
    text = failure_mode_summary(m)
    assert "Failure-mode analysis" in text
    assert "TIMEOUT" in text


def test_collision_and_intrusion_failure_modes():
    m = ProxyMetrics(sr=0.2, cr=0.4, tr=0.4, itr=12.0, sd=0.2)
    bullets = failure_mode_bullets(m)
    joined = " ".join(bullets)
    assert "collision" in joined.lower() or "COLLISION" in joined
    assert "intrusion" in joined.lower() or "ITR" in joined


def test_d3_prompt_includes_failure_mode_extra_context():
    m = ProxyMetrics(sr=0.0, cr=0.2, tr=0.8, itr=5.0, sd=0.3)
    prompt = format_d3_refinement(
        "def compute_reward(state, memory):\n    return 0.0\n",
        last_score=m.scalar_score(),
        feedback=m.feedback_text(),
        extra_context_if_any=failure_mode_summary(m),
    )
    assert "Failure-mode analysis" in prompt
    assert "SR=" in prompt
