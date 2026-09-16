"""Tests for ensemble critique agreement."""

from __future__ import annotations

from crowd_nav.reward_search.amfrs.ensemble_critique import critique_agrees
from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.llm import LLMClient


class _FixedLLM(LLMClient):
    def __init__(self, text: str) -> None:
        self._text = text

    def complete(self, prompt: str) -> str:
        return self._text


def test_critique_agreement_ignores_code_identifiers():
    code = (
        "def compute_reward(state, memory):\n"
        "    return discomfort_penalty(state, memory) - collision_indicator(state, memory)\n"
    )
    cand = RewardCandidate(candidate_id="c0", code=code, valid=True, metadata={})
    # Both mention identifiers from the code — should NOT count as agreement.
    a = _FixedLLM("discomfort_penalty and collision_indicator dominate the signal")
    b = _FixedLLM("collision_indicator and discomfort_penalty are over-weighted")
    report = critique_agrees(cand, "", a, b, min_shared=2)
    assert not report.agreed
    assert not report.shared_terms


def test_critique_agreement_counts_added_terms():
    code = "def compute_reward(state, memory):\n    return 1.0\n"
    cand = RewardCandidate(candidate_id="c0", code=code, valid=True, metadata={})
    a = _FixedLLM("too aggressive near crowds causes unsafe shortcuts")
    b = _FixedLLM("unsafe shortcuts near crowds from aggressive weighting")
    report = critique_agrees(cand, "", a, b, min_shared=2)
    assert report.agreed
    assert len(report.shared_terms) >= 2
