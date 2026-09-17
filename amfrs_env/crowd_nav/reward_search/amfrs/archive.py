"""
Axis 2 — MAP-Elites elite grid.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from crowd_nav.reward_search.evolver import RewardCandidate


class EliteGrid:
    def __init__(self, shape: Tuple[int, int]) -> None:
        if shape[0] < 1 or shape[1] < 1:
            raise ValueError(f"invalid archive shape: {shape}")
        self.shape = (int(shape[0]), int(shape[1]))
        self._cells: Dict[Tuple[int, int], Tuple[RewardCandidate, float]] = {}
        # Non-competitive: most recent evaluation per candidate_id (any fidelity).
        self.latest_by_candidate_id: Dict[str, RewardCandidate] = {}

    def try_insert(
        self, candidate: RewardCandidate, cell: Tuple[int, int], fitness: float
    ) -> bool:
        bx = min(self.shape[0] - 1, max(0, int(cell[0])))
        by = min(self.shape[1] - 1, max(0, int(cell[1])))
        key = (bx, by)
        # Always record the latest evaluation for this id, even if the cell
        # keeps an older/higher-fitness snapshot (cross-candidate competition
        # is unchanged below).
        self.latest_by_candidate_id[str(candidate.candidate_id)] = candidate
        cur = self._cells.get(key)
        if cur is None or float(fitness) > cur[1]:
            self._cells[key] = (candidate, float(fitness))
            return True
        return False

    def get(self, cell: Tuple[int, int]) -> Optional[RewardCandidate]:
        item = self._cells.get((int(cell[0]), int(cell[1])))
        return item[0] if item else None

    def get_latest(self, candidate_id: str) -> Optional[RewardCandidate]:
        return self.latest_by_candidate_id.get(str(candidate_id))

    def all_elites(self) -> List[RewardCandidate]:
        return [c for c, _ in self._cells.values()]

    def unique_elites_by_id(self) -> List[RewardCandidate]:
        """
        One elite per ``candidate_id`` (best cell fitness wins).

        MAP-Elites may place the same genome in several cells; expensive
        final-rung / robustness climbs should not retrain duplicates.
        """
        best: Dict[str, Tuple[RewardCandidate, float]] = {}
        for cand, fit in self._cells.values():
            cid = str(cand.candidate_id)
            cur = best.get(cid)
            if cur is None or float(fit) > cur[1]:
                best[cid] = (cand, float(fit))
        # Stable order: higher fitness first, then id.
        ranked = sorted(
            best.values(),
            key=lambda t: (-t[1], str(t[0].candidate_id)),
        )
        return [c for c, _ in ranked]

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
            "latest_by_candidate_id": {
                cid: {
                    "candidate_id": cand.candidate_id,
                    "code": cand.code,
                    "score": cand.score,
                    "metadata": cand.metadata or {},
                }
                for cid, cand in self.latest_by_candidate_id.items()
            },
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
        # Restore latest evaluations (may differ from cell-winning snapshots).
        latest_payload = payload.get("latest_by_candidate_id") or {}
        restored: Dict[str, RewardCandidate] = {}
        for cid, item in latest_payload.items():
            restored[str(cid)] = RewardCandidate(
                candidate_id=str(item.get("candidate_id") or cid),
                code=str(item.get("code") or ""),
                score=item.get("score"),
                valid=True,
                origin="archive",
                metadata=dict(item.get("metadata") or {}),
            )
        if restored:
            grid.latest_by_candidate_id = restored
        return grid
