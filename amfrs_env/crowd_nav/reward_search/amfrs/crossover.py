"""
Axis 3 — primitive-aware / semantic crossover prompt builders (AMFRS-only).

Does not modify baseline ``prompts.py``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from crowd_nav.reward_search.amfrs.primitives import primitive_signatures_block
from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.scoring import format_score1_diagnostics

def format_candidate_diagnostics(candidate: RewardCandidate) -> str:
    """Short metrics summary for mutation/crossover prompts (from fidelity history)."""
    md = candidate.metadata or {}
    parts: list[str] = []
    last_metric = md.get("last_metric")
    if last_metric is not None:
        parts.append(f"last_metric={last_metric}")
    raw = md.get("last_raw_metrics") or {}
    if isinstance(raw, Mapping):
        for key in ("SR", "CR", "TR", "SD", "PL", "ITR", "score1"):
            if key in raw:
                try:
                    parts.append(f"{key}={float(raw[key]):.3f}")
                except (TypeError, ValueError):
                    parts.append(f"{key}={raw[key]}")
    score1_diag = format_score1_diagnostics(md.get("score1_report"))
    if score1_diag:
        parts.append(f"Score1: {score1_diag}")
    level = md.get("last_fidelity_level")
    if level:
        parts.append(f"last_rung={level}")
    return " — ".join(parts)


AMFRS_SYSTEM_PROMPT = (
    "You are an expert in reinforcement learning and robot crowd navigation. "
    "Design compute_reward(state, memory) functions. Prefer composing named "
    "primitives with explicit numeric weights. Return only a Python code block."
)


def build_initial_prompt(*, reflection: str = "", memory_block: str = "") -> str:
    parts = [
        "Write def compute_reward(state, memory): returning a finite float.",
        "No imports, no classes. Use memory dict for episode state.",
        primitive_signatures_block(),
        "Example shape:",
        "```python",
        "def compute_reward(state, memory):",
        "    w_goal = 2.0",
        "    w_coll = 20.0",
        "    r = w_goal * goal_progress(state, memory)",
        "    r -= w_coll * collision_indicator(state, memory)",
        "    r -= 1.0 * discomfort_penalty(state, memory)",
        "    return float(r)",
        "```",
    ]
    if reflection.strip():
        parts.append("Reflection:\n" + reflection.strip())
    if memory_block.strip():
        parts.append(memory_block.strip())
    return "\n".join(parts)


def build_mutation_prompt(
    parent: RewardCandidate,
    *,
    diagnostics: str = "",
    memory_block: str = "",
) -> str:
    parts = [
        "Mutate the following reward. Keep the compute_reward(state, memory) signature.",
        primitive_signatures_block(),
        "Parent code:",
        "```python",
        parent.code.strip(),
        "```",
    ]
    if diagnostics.strip():
        parts.append("Diagnostics:\n" + diagnostics.strip())
    if memory_block.strip():
        parts.append(memory_block.strip())
    return "\n".join(parts)


def build_semantic_crossover_prompt(
    parent_a: RewardCandidate,
    parent_b: RewardCandidate,
    diagnostics_a: str = "",
    diagnostics_b: str = "",
    *,
    memory_block: str = "",
) -> str:
    """
    Ask the LLM to name which primitive/term from each parent to keep.
    """
    parts = [
        "Semantically crossover two reward functions.",
        "Explicitly name which primitive/term from each parent to keep,",
        "then emit a single compute_reward(state, memory).",
        primitive_signatures_block(),
        "Parent A:",
        "```python",
        parent_a.code.strip(),
        "```",
        f"Diagnostics A:\n{diagnostics_a or '(none)'}",
        "Parent B:",
        "```python",
        parent_b.code.strip(),
        "```",
        f"Diagnostics B:\n{diagnostics_b or '(none)'}",
    ]
    if memory_block.strip():
        parts.append(memory_block.strip())
    return "\n".join(parts)


def build_memory_block(entries: list) -> str:
    if not entries:
        return ""
    lines = ["Similar past rewards and outcomes:"]
    for e in entries:
        code = getattr(e, "code", "") or ""
        metrics = getattr(e, "outcome_metrics", {}) or {}
        sim = getattr(e, "similarity", 0.0)
        lines.append(f"- similarity={sim:.3f} metrics={metrics}")
        lines.append("```python")
        lines.append(code.strip()[:800])
        lines.append("```")
    return "\n".join(lines)
