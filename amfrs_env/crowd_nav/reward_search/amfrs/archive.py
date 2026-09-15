"""
Axis 2 — MAP-Elites elite grid.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from crowd_nav.reward_search.evolver import RewardCandidate


class EliteGrid:
    def __init__(self, shape: Tuple[int, int]) -> None:
        if shape[0] < 1 or shape[1] < 1:
            raise ValueError(f"invalid archive shape: {shape}")
        self.shape = (int(shape[0]), int(shape[1]))
        self._cells: Dict[Tuple[int, int], Tuple[RewardCandidate, float]] = {}

    def try_insert(
        self, candidate: RewardCandidate, cell: Tuple[int, int], fitness: float
    ) -> bool:
        bx = min(self.shape[0] - 1, max(0, int(cell[0])))
        by = min(self.shape[1] - 1, max(0, int(cell[1])))
        key = (bx, by)
        cur = self._cells.get(key)
        if cur is None or float(fitness) > cur[1]:
            self._cells[key] = (candidate, float(fitness))
            return True
        return False

    def get(self, cell: Tuple[int, int]) -> Optional[RewardCandidate]:
        item = self._cells.get((int(cell[0]), int(cell[1])))
        return item[0] if item else None

    def all_elites(self) -> List[RewardCandidate]:
        return [c for c, _ in self._cells.values()]

    def coverage(self) -> float:
        total = self.shape[0] * self.shape[1]
        if total <= 0:
            return 0.0
        return float(len(self._cells)) / float(total)

    def qd_score(self) -> float:
        return float(sum(fit for _, fit in self._cells.values()))

    def to_json(self, path: str) -> None:
        payload = {
            "shape": list(self.shape),
            "cells": [
                {
                    "cell": [bx, by],
                    "fitness": fit,
                    "candidate_id": cand.candidate_id,
                    "code": cand.code,
                    "score": cand.score,
                    "metadata": cand.metadata or {},
                }
                for (bx, by), (cand, fit) in self._cells.items()
            ],
            "coverage": self.coverage(),
            "qd_score": self.qd_score(),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

    @classmethod
    def from_json(cls, path: str) -> "EliteGrid":
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        grid = cls(tuple(payload["shape"]))
        for item in payload.get("cells") or []:
            cell = tuple(item["cell"])
            cand = RewardCandidate(
                candidate_id=str(item["candidate_id"]),
                code=str(item.get("code") or ""),
                score=item.get("score"),
                valid=True,
                origin="archive",
                metadata=dict(item.get("metadata") or {}),
            )
            grid.try_insert(cand, cell, float(item["fitness"]))
        return grid
