"""
Axis 1 — successive halving scheduler (ASHA-style promotion).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from crowd_nav.reward_search.amfrs.fidelity import (
    FidelityLadder,
    FidelityLevel,
    append_fidelity_history,
    evaluate_at,
)
from crowd_nav.reward_search.evolver import RewardCandidate


@dataclass(frozen=True)
class HalvingConfig:
    eta: int = 3
    min_survivors: int = 1
    max_cost_units: Optional[float] = None


class SuccessiveHalvingScheduler:
    def __init__(
        self,
        ladder: FidelityLadder,
        config: HalvingConfig = HalvingConfig(),
        *,
        bandit: Optional[object] = None,
    ) -> None:
        self.ladder = ladder
        self.config = config
        self.bandit = bandit  # optional UCB1Allocator

    def run(
        self,
        population: Sequence[RewardCandidate],
        *,
        on_rung_complete: Optional[
            Callable[[str, List[RewardCandidate]], None]
        ] = None,
        spent_cost: Optional[List[float]] = None,
    ) -> List[RewardCandidate]:
        """
        Evaluate at each rung, keep top ceil(N/eta), continue until ladder ends
        or min_survivors reached. Ties broken by lower candidate_id.
        """
        alive: List[RewardCandidate] = list(population)
        if not alive:
            return []
        cost_box = spent_cost if spent_cost is not None else [0.0]
        eta = max(2, int(self.config.eta))

        for level in self.ladder:
            if len(alive) < self.config.min_survivors and level != self.ladder.levels[0]:
                break
            if self.config.max_cost_units is not None and cost_box[0] >= self.config.max_cost_units:
                break
            # Skip entire rung if we cannot afford even one evaluation.
            if (
                self.config.max_cost_units is not None
                and cost_box[0] + float(level.cost_units) > self.config.max_cost_units
            ):
                break

            scored: List[RewardCandidate] = []
            unevaluated: List[RewardCandidate] = []
            order = list(alive)
            if self.bandit is not None and hasattr(self.bandit, "select_next"):
                # Optional non-uniform funding order within the rung
                ids = [c.candidate_id for c in order]
                chosen = self.bandit.select_next(ids, k=len(ids))
                id_to_c = {c.candidate_id: c for c in order}
                order = [id_to_c[i] for i in chosen if i in id_to_c]
                # append any missing
                seen = set(chosen)
                order.extend([c for c in alive if c.candidate_id not in seen])

            budget_exhausted = False
            for cand in order:
                if (
                    self.config.max_cost_units is not None
                    and cost_box[0] + float(level.cost_units) > self.config.max_cost_units
                ):
                    budget_exhausted = True
                    unevaluated.append(cand)
                    continue
                if budget_exhausted:
                    unevaluated.append(cand)
                    continue
                result = evaluate_at(level, cand)
                cost_box[0] += float(result.cost)
                updated = append_fidelity_history(cand, level.name, result)
                if self.bandit is not None and hasattr(self.bandit, "record"):
                    # Normalize reward within rung later; record raw for now
                    self.bandit.record(cand.candidate_id, float(result.metric), float(result.cost))
                scored.append(updated)

            if not scored and not unevaluated:
                break

            # Unevaluated arms keep prior-rung scores so they are not dropped silently.
            pool = scored + unevaluated
            if not pool:
                break

            # Normalize bandit rewards within rung if supported
            if self.bandit is not None and hasattr(self.bandit, "normalize_last_rung") and scored:
                metrics = [float(c.score or 0.0) for c in scored]
                self.bandit.normalize_last_rung(
                    [c.candidate_id for c in scored], metrics
                )

            # Drop arms already dead at prior rung — do not spend F1+/budget on -inf.
            # Keep score=None (unevaluated under budget) so they are not silently lost.
            finite_pool = [
                c
                for c in pool
                if c.score is None or math.isfinite(float(c.score))
            ]
            if not finite_pool:
                if on_rung_complete is not None:
                    on_rung_complete(level.name, [])
                return []
            pool = finite_pool

            pool.sort(
                key=lambda c: (
                    -(float(c.score) if c.score is not None else float("-inf")),
                    str(c.candidate_id),
                )
            )
            n_keep = max(
                int(self.config.min_survivors),
                int(math.ceil(len(pool) / float(eta))),
            )
            n_keep = min(n_keep, len(pool))
            # Exact-tie at cutoff: already sorted by candidate_id ascending as secondary key
            alive = pool[:n_keep]
            if on_rung_complete is not None:
                on_rung_complete(level.name, list(alive))

        return alive
