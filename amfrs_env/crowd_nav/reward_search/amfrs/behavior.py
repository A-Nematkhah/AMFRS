"""
Axis 2 — behavior descriptors for MAP-Elites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple


@dataclass(frozen=True)
class DescriptorSpaceConfig:
    path_efficiency_edges: Tuple[float, ...] = (1.0, 1.2, 1.5, 2.0, 3.0)
    social_margin_edges: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)


def _bin_index(value: float, edges: Sequence[float]) -> int:
    """Clip to nearest edge bin; returns index in [0, len(edges)-2]."""
    if len(edges) < 2:
        return 0
    if value <= edges[0]:
        return 0
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2


def compute_behavior_descriptor(
    metrics: Mapping[str, float],
    *,
    config: DescriptorSpaceConfig = DescriptorSpaceConfig(),
    eps: float = 1e-6,
) -> Tuple[int, int]:
    """
    BD-x: path efficiency = PL / max(goal_dist0, eps)
    BD-y: social margin = SD
    """
    pl = float(metrics.get("PL", metrics.get("pl", 0.0)))
    goal = float(metrics.get("goal_dist0", 0.0))
    if goal <= 0.0:
        # Fallback if adapter forgot goal_dist0: treat PL as already-ratio-ish
        efficiency = max(1.0, pl / max(eps, 8.0))
    else:
        efficiency = pl / max(eps, goal)
    sd = float(metrics.get("SD", metrics.get("sd", 0.0)))
    bx = _bin_index(efficiency, config.path_efficiency_edges)
    by = _bin_index(sd, config.social_margin_edges)
    return (bx, by)


def grid_shape_from_config(config: DescriptorSpaceConfig = DescriptorSpaceConfig()) -> Tuple[int, int]:
    return (
        max(1, len(config.path_efficiency_edges) - 1),
        max(1, len(config.social_margin_edges) - 1),
    )
