"""
Axis 1 — UCB1 budget allocator (cost-aware by default).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ArmStats:
    pulls: int = 0
    total_reward: float = 0.0
    total_cost: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.pulls <= 0:
            return 0.0
        return self.total_reward / float(self.pulls)

    @property
    def mean_reward_per_cost(self) -> float:
        if self.total_cost <= 0.0:
            return self.mean_reward
        return self.total_reward / float(self.total_cost)


class UCB1Allocator:
    """
    Standard UCB1 over candidate_id arms.

    When ``cost_aware=True``, uses mean_reward_per_cost as the exploitation term.
    Callers should pass rewards already normalized to roughly [0,1] per rung
    (see ``normalize_last_rung``).
    """

    def __init__(self, exploration_c: float = 2.0, cost_aware: bool = True) -> None:
        self.exploration_c = float(exploration_c)
        self.cost_aware = bool(cost_aware)
        self._arms: Dict[str, ArmStats] = {}
        self._total_pulls: int = 0
        self._pending: Dict[str, float] = {}

    def record(self, candidate_id: str, reward: float, cost: float) -> None:
        arm = self._arms.setdefault(candidate_id, ArmStats())
        arm.pulls += 1
        arm.total_reward += float(reward)
        arm.total_cost += max(0.0, float(cost))
        self._total_pulls += 1
        self._pending[candidate_id] = float(reward)

    def normalize_last_rung(self, candidate_ids: List[str], metrics: List[float]) -> None:
        """Re-scale the most recent rung's rewards into [0,1] for fair UCB."""
        if not metrics:
            return
        lo = min(metrics)
        hi = max(metrics)
        span = hi - lo if hi > lo else 1.0
        for cid, raw in zip(candidate_ids, metrics):
            arm = self._arms.get(cid)
            if arm is None or arm.pulls <= 0:
                continue
            # Undo last raw add and replace with normalized
            prev = self._pending.get(cid, raw)
            arm.total_reward -= prev
            norm = (raw - lo) / span
            arm.total_reward += norm
            self._pending[cid] = norm

    def _ucb(self, cid: str) -> float:
        arm = self._arms.setdefault(cid, ArmStats())
        if arm.pulls == 0:
            return float("inf")
        exploit = arm.mean_reward_per_cost if self.cost_aware else arm.mean_reward
        explore = self.exploration_c * math.sqrt(
            math.log(max(1, self._total_pulls)) / float(arm.pulls)
        )
        return exploit + explore

    def select_next(self, candidate_ids: List[str], k: int) -> List[str]:
        if k <= 0:
            return []
        scored = sorted(
            candidate_ids,
            key=lambda cid: (-self._ucb(cid), str(cid)),
        )
        return scored[: min(k, len(scored))]
