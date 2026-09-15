"""
Shared Stage II / III D.3 accept-reject helpers (Phase 4 / T12).

Train loops stay stage-specific (A2C+K2 vs PPO+K3). This module owns the
fair-budget compare + metadata / failure-record shapes so both runners innovate
along one path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from crowd_nav.reward_search.llm import (
    LLMClient,
    extract_python_code,
    normalize_to_compute_reward,
)
from crowd_nav.reward_search.prompts import D3_SYSTEM_PROMPT, format_d3_repair


def resolve_verify_train_steps(
    train_env_steps: int,
    accept_reject_steps: Optional[int],
) -> int:
    """Reduced verify budget: explicit steps, else ``train_env_steps // 4``."""
    if accept_reject_steps is None:
        return max(1, int(train_env_steps) // 4)
    return max(1, int(accept_reject_steps))


@dataclass(frozen=True)
class AcceptRejectDecision:
    """Outcome of comparing parent vs proposed under the same verify budget."""

    accepted: bool
    verify_scalar: float
    parent_verify_scalar: float
    parent_full_scalar: float
    tolerance: float

    @property
    def kept_previous(self) -> bool:
        return not self.accepted


def decide_accept_reject(
    *,
    parent_verify_scalar: float,
    proposed_verify_scalar: float,
    parent_full_scalar: float,
    tolerance: float = 0.0,
) -> AcceptRejectDecision:
    """
    Accept proposed when ``proposed >= parent_verify - tol`` (fair verify).

    A tiny epsilon avoids float jitter rejecting equal scores.
    """
    tol = float(tolerance or 0.0)
    accepted = float(proposed_verify_scalar) + 1e-12 >= float(
        parent_verify_scalar
    ) - tol
    return AcceptRejectDecision(
        accepted=accepted,
        verify_scalar=float(proposed_verify_scalar),
        parent_verify_scalar=float(parent_verify_scalar),
        parent_full_scalar=float(parent_full_scalar),
        tolerance=tol,
    )


def build_accept_metadata(
    proposed_metadata: Optional[Mapping[str, Any]],
    *,
    parent_id: str,
    parent_code: str,
    proposed_code: str,
    decision: AcceptRejectDecision,
    proposed_verify_metrics: Mapping[str, Any],
    parent_verify_metrics: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Metadata stamped on an accepted refined genome."""
    md: Dict[str, Any] = {
        **dict(proposed_metadata or {}),
        "refine_accepted": True,
        "refine_rejected_metric": False,
        "refine_verify_metrics": dict(proposed_verify_metrics),
        "refine_verify_scalar": decision.verify_scalar,
        "parent_verify_metrics": dict(parent_verify_metrics),
        "parent_verify_scalar": decision.parent_verify_scalar,
        "parent_train_scalar": decision.parent_full_scalar,
        "accept_reject_fair_budget": True,
        "proposed_refined_genome_code": proposed_code,
        "evaluated_genome_id": parent_id,
        "evaluated_genome_code": parent_code,
    }
    if extra:
        md.update(dict(extra))
    return md


def build_reject_metadata(
    parent_metadata: Optional[Mapping[str, Any]],
    *,
    parent_id: str,
    parent_code: str,
    parent_metrics: Mapping[str, Any],
    proposed_id: str,
    proposed_code: str,
    decision: AcceptRejectDecision,
    proposed_verify_metrics: Mapping[str, Any],
    parent_verify_metrics: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Metadata stamped when metric-reject keeps the parent genome."""
    md: Dict[str, Any] = {
        **dict(parent_metadata or {}),
        "last_metrics": dict(parent_metrics),
        "refine_kept_previous": True,
        "refine_accepted": False,
        "refine_rejected_metric": True,
        "refine_proposed_id": proposed_id,
        "proposed_refined_genome_code": proposed_code,
        "refine_verify_metrics": dict(proposed_verify_metrics),
        "refine_verify_scalar": decision.verify_scalar,
        "parent_verify_metrics": dict(parent_verify_metrics),
        "parent_verify_scalar": decision.parent_verify_scalar,
        "parent_train_scalar": decision.parent_full_scalar,
        "accept_reject_fair_budget": True,
        "evaluated_genome_id": parent_id,
        "evaluated_genome_code": parent_code,
    }
    if extra:
        md.update(dict(extra))
    return md


def build_metric_reject_failure(
    *,
    parent_id: str,
    proposed_id: str,
    decision: AcceptRejectDecision,
) -> Dict[str, Any]:
    """Row appended to ``validation_failures`` on metric reject."""
    return {
        "candidate_id": parent_id,
        "reason": "metric_reject",
        "kept_previous": True,
        "proposed_id": proposed_id,
        "verify_scalar": decision.verify_scalar,
        "parent_verify_scalar": decision.parent_verify_scalar,
        "parent_scalar": decision.parent_full_scalar,
    }


def repair_invalid_code(
    llm: LLMClient,
    *,
    bad_code: str,
    validation_error: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    One D.3 repair attempt.

    Returns ``(repaired_code, None)`` on LLM success, else ``(None, error)``.
    Caller still must re-validate the repaired source.
    """
    repair_prompt = format_d3_repair(
        bad_code=bad_code,
        validation_error=validation_error,
    )
    full_prompt = f"{D3_SYSTEM_PROMPT}\n\n{repair_prompt}"
    try:
        raw = llm.complete(full_prompt)
        repaired_code = normalize_to_compute_reward(extract_python_code(raw))
        return repaired_code, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
