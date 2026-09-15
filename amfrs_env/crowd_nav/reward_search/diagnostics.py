"""
Failure-mode summaries for Stage II / III D.3 refinement (Phase 2 / T6).

Turns aggregate ProxyMetrics into short, actionable critique for the LLM —
beyond raw SR/CR/TR dumps — without requiring per-episode logs.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Union


def _f(metrics: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                continue
    return float(default)


def metrics_mapping(metrics: Any) -> Mapping[str, Any]:
    if metrics is None:
        return {}
    if hasattr(metrics, "as_dict"):
        return metrics.as_dict()
    if isinstance(metrics, Mapping):
        return metrics
    return {
        "SR": float(getattr(metrics, "sr", 0.0)),
        "CR": float(getattr(metrics, "cr", 0.0)),
        "TR": float(getattr(metrics, "tr", 0.0)),
        "NT": float(getattr(metrics, "nt", 0.0)),
        "PL": float(getattr(metrics, "pl", 0.0)),
        "ITR": float(getattr(metrics, "itr", 0.0)),
        "SD": float(getattr(metrics, "sd", 0.0)),
    }


def failure_mode_bullets(metrics: Any) -> List[str]:
    """
    Interpret aggregate navigation metrics into failure-mode bullets.

    Thresholds are practical v0 defaults (Needs Validation); they are for LLM
    guidance only and do not change selection math.
    """
    m = metrics_mapping(metrics)
    sr = _f(m, "SR", "sr")
    cr = _f(m, "CR", "cr")
    tr = _f(m, "TR", "tr")
    nt = _f(m, "NT", "nt")
    pl = _f(m, "PL", "pl")
    itr = _f(m, "ITR", "itr")
    sd = _f(m, "SD", "sd")

    bullets: List[str] = []

    if sr < 0.05 and tr >= 0.45:
        bullets.append(
            "TIMEOUT-dominant (low SR, high TR): strengthen dense goal-progress / "
            "potential shaping; avoid rewards that freeze the robot or over-penalize motion."
        )
    elif sr < 0.15:
        bullets.append(
            "Low success rate: increase terminal success reward and mid-episode progress "
            "toward (gx, gy); keep collision costs strong."
        )

    if cr >= 0.25:
        bullets.append(
            "High collision rate: enlarge collision / close-proximity penalties and "
            "discomfort terms when dmin < discomfort_dist."
        )
    elif cr >= 0.12 and sr < 0.4:
        bullets.append(
            "Moderate collisions with weak success: rebalance safety vs progress "
            "(do not only amplify collision cost)."
        )

    if itr >= 8.0:
        bullets.append(
            f"High intrusion (ITR={itr:.1f}%): tighten personal-space shaping so the "
            "robot spends less time inside discomfort zones."
        )
    elif itr >= 3.0:
        bullets.append(
            f"Non-trivial intrusion (ITR={itr:.1f}%): mild discomfort penalty refinement."
        )

    if sd > 0.0 and sd < 0.28 and (itr >= 1.0 or cr >= 0.1):
        bullets.append(
            f"Close approaches during intrusion (SD={sd:.2f}m): prefer larger clearance "
            "when humans are near."
        )

    if sr >= 0.5 and cr <= 0.1 and tr <= 0.2:
        bullets.append(
            f"Mostly successful (NT≈{nt:.1f}, PL≈{pl:.1f}): refine efficiency / smoothness "
            "without weakening collision or progress terms."
        )

    if not bullets:
        bullets.append(
            "Mixed outcomes: keep dense progress shaping, explicit collision/timeout "
            "terminals, and discomfort when dmin is small; avoid near-constant rewards."
        )

    return bullets


def failure_mode_summary(metrics: Any) -> str:
    """Single block for D.3 ``extra_context_if_any`` / logging."""
    bullets = failure_mode_bullets(metrics)
    numbered = " ".join(f"({i}) {b}" for i, b in enumerate(bullets, start=1))
    return f"Failure-mode analysis: {numbered}"
