"""
Axis 3 — ensemble critique agreement gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.llm import LLMClient


@dataclass(frozen=True)
class AgreementReport:
    agreed: bool
    shared_terms: List[str]
    critique_a: str
    critique_b: str


_TERM_RE = re.compile(r"[a-z_][a-z0-9_]{2,}", re.I)


def _extract_terms(text: str) -> set:
    stop = {
        "the",
        "and",
        "for",
        "this",
        "that",
        "with",
        "from",
        "reward",
        "function",
        "should",
        "would",
        "could",
        "state",
        "memory",
    }
    return {t.lower() for t in _TERM_RE.findall(text or "") if t.lower() not in stop}


def critique_agrees(
    candidate: RewardCandidate,
    diagnostics: str,
    client_a: LLMClient,
    client_b: LLMClient,
    *,
    min_shared: int = 2,
) -> AgreementReport:
    """
    Query two LLM clients independently. Agree when they share enough critique terms.
    """
    prompt = (
        "Critique this robot navigation reward in under 80 words. "
        "List concrete flaws as short keywords.\n"
        f"Diagnostics:\n{diagnostics}\n"
        f"Code:\n```python\n{candidate.code}\n```\n"
    )
    critique_a = client_a.complete(prompt)
    critique_b = client_b.complete(prompt)
    terms_a = _extract_terms(critique_a)
    terms_b = _extract_terms(critique_b)
    shared = sorted(terms_a & terms_b)
    agreed = len(shared) >= int(min_shared)
    return AgreementReport(
        agreed=agreed,
        shared_terms=shared,
        critique_a=critique_a,
        critique_b=critique_b,
    )
