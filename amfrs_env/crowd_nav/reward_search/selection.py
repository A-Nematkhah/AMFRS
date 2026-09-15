"""
Navigation fitness helpers for Stage II / III selection.

Default scalar (Phase 2 / T7, Decision D-3 option B v0):
    w_sr·SR - w_cr·CR - w_tr·TR - w_itr·ITR + w_sd·SD

with defaults matching the historical base ``SR - CR - 0.5·TR`` when
ITR=SD=0, plus small social terms (ITR penalized, SD rewarded).

Phase 3 also adds top-k finalist selection and an H-aware scalar over
generalization sweeps (mean and worst-H blend).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from crowd_nav.reward_search.evolver import RewardCandidate

MetricsLike = Union[Mapping[str, Any], Any]


@dataclass(frozen=True)
class SelectionWeights:
    """
    Shared II/III selection weights (D-3 / T7).

    ``w_itr`` applies to ITR in **percent** (same units as ``rl.evaluation`` /
    ``ProxyMetrics.itr``). ``w_sd`` applies to mean min-distance (meters) during
    intrusion frames — higher SD is better.
    """

    w_sr: float = 1.0
    w_cr: float = 1.0
    w_tr: float = 0.5
    w_itr: float = 0.01
    w_sd: float = 0.2

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


# Documented v0 defaults — Needs Validation against paper / human preference.
DEFAULT_SELECTION_WEIGHTS = SelectionWeights()


def navigation_scalar(
    sr: float,
    cr: float,
    tr: float,
    itr: float = 0.0,
    sd: float = 0.0,
    *,
    weights: Optional[SelectionWeights] = None,
) -> float:
    """
    Weighted navigation fitness (higher better).

    Legacy call ``navigation_scalar(sr, cr, tr)`` matches
    ``SR - CR - 0.5·TR`` under default weights with ITR=SD=0.
    """
    w = weights or DEFAULT_SELECTION_WEIGHTS
    return (
        float(w.w_sr) * float(sr)
        - float(w.w_cr) * float(cr)
        - float(w.w_tr) * float(tr)
        - float(w.w_itr) * float(itr)
        + float(w.w_sd) * float(sd)
    )


def navigation_scalar_from_dict(
    metrics: Optional[Mapping[str, Any]],
    *,
    weights: Optional[SelectionWeights] = None,
) -> float:
    if not metrics:
        return float("-inf")
    sr = float(metrics.get("SR", metrics.get("sr", 0.0)))
    cr = float(metrics.get("CR", metrics.get("cr", 0.0)))
    tr = float(metrics.get("TR", metrics.get("tr", 0.0)))
    itr = float(metrics.get("ITR", metrics.get("itr", 0.0)))
    sd = float(metrics.get("SD", metrics.get("sd", 0.0)))
    return navigation_scalar(sr, cr, tr, itr, sd, weights=weights)


def _metrics_as_mapping(metrics: MetricsLike) -> Mapping[str, Any]:
    if hasattr(metrics, "as_dict"):
        return metrics.as_dict()
    if isinstance(metrics, Mapping):
        return metrics
    return {
        "SR": float(getattr(metrics, "sr", 0.0)),
        "CR": float(getattr(metrics, "cr", 0.0)),
        "TR": float(getattr(metrics, "tr", 0.0)),
        "ITR": float(getattr(metrics, "itr", 0.0)),
        "SD": float(getattr(metrics, "sd", 0.0)),
    }


def candidate_nav_scalar(
    candidate: RewardCandidate,
    *,
    weights: Optional[SelectionWeights] = None,
) -> float:
    md = candidate.metadata or {}
    return navigation_scalar_from_dict(md.get("last_metrics"), weights=weights)


def pick_best_trained(
    snapshots: Sequence[RewardCandidate],
) -> Optional[RewardCandidate]:
    """Return the trained snapshot with highest navigation scalar, if any."""
    scored = [c for c in snapshots if candidate_nav_scalar(c) > float("-inf")]
    if not scored:
        return None
    return max(scored, key=candidate_nav_scalar)


def is_same_genome(a: RewardCandidate, b: RewardCandidate) -> bool:
    return a.code.strip() == b.code.strip()


def select_top_k_finalists(
    population: Sequence[RewardCandidate],
    k: int,
    *,
    prefer: Optional[RewardCandidate] = None,
) -> List[RewardCandidate]:
    """
    Stage III admission: keep top-``k`` genomes by navigation scalar (Phase 3 / T8).

    Deduplicates by code. If ``prefer`` (e.g. Stage II best_trained) is set, it
    is always included when present in / compatible with the pool.
    """
    k = max(1, int(k))
    ranked = sorted(
        population,
        key=candidate_nav_scalar,
        reverse=True,
    )
    selected: List[RewardCandidate] = []
    seen_codes = set()

    def _try_add(cand: RewardCandidate) -> None:
        code = cand.code.strip()
        if code in seen_codes:
            return
        seen_codes.add(code)
        selected.append(cand)

    if prefer is not None:
        _try_add(prefer)
    for cand in ranked:
        if len(selected) >= k:
            break
        _try_add(cand)
    return selected


def h_profile_scalar(
    by_human_count: Mapping[int, MetricsLike],
    *,
    mean_weight: float = 0.5,
    weights: Optional[SelectionWeights] = None,
) -> float:
    """
    H-aware fitness: blend of mean and worst-H navigation scalars (Phase 3 / T11).

    ``final = mean_weight * mean_H(scalar) + (1 - mean_weight) * min_H(scalar)``
    """
    if not by_human_count:
        return float("-inf")
    scalars = [
        navigation_scalar_from_dict(_metrics_as_mapping(m), weights=weights)
        for m in by_human_count.values()
    ]
    finite = [s for s in scalars if s > float("-inf")]
    if not finite:
        return float("-inf")
    mean_s = sum(finite) / len(finite)
    worst_s = min(finite)
    w = min(1.0, max(0.0, float(mean_weight)))
    return float(w * mean_s + (1.0 - w) * worst_s)


def attach_h_profile(
    cand: RewardCandidate,
    report: Any,
    *,
    mean_weight: float = 0.5,
) -> RewardCandidate:
    """Attach H-sweep metrics + profile scalar onto a candidate."""
    score = h_profile_scalar(report.by_human_count, mean_weight=mean_weight)
    return replace(
        cand,
        metadata={
            **(cand.metadata or {}),
            "h_profile_scalar": float(score),
            "h_sweep": {
                str(h): _metrics_as_mapping(m)
                for h, m in report.by_human_count.items()
            },
        },
    )


def pick_best_by_h_profile(
    candidates: Sequence[RewardCandidate],
) -> Optional[RewardCandidate]:
    """Pick candidate with highest ``metadata['h_profile_scalar']``."""
    scored = []
    for cand in candidates:
        md = cand.metadata or {}
        if "h_profile_scalar" not in md:
            continue
        scored.append((float(md["h_profile_scalar"]), cand))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    best = replace(
        scored[0][1],
        metadata={**(scored[0][1].metadata or {}), "h_aware_selected": True},
    )
    return best
