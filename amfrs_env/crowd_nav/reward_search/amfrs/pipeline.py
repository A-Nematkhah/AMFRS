"""
AMFRS pipeline orchestrator.

Wires axes 1–5: static gate → successive halving → MAP-Elites illumination →
optional retrieval memory / ensemble critique → robustness on final elites.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from crowd_nav.reward_search.amfrs.archive import EliteGrid
from crowd_nav.reward_search.amfrs.bandit import UCB1Allocator
from crowd_nav.reward_search.amfrs.behavior import (
    DescriptorSpaceConfig,
    compute_behavior_descriptor,
    grid_shape_from_config,
)
from crowd_nav.reward_search.amfrs.config import AMFRSRunConfig
from crowd_nav.reward_search.amfrs.crossover import (
    AMFRS_SYSTEM_PROMPT,
    build_initial_prompt,
    build_memory_block,
    build_mutation_prompt,
    build_semantic_crossover_prompt,
    format_candidate_diagnostics,
)
from crowd_nav.reward_search.amfrs.ensemble_critique import critique_agrees
from crowd_nav.reward_search.amfrs.assets import require_amfrs_assets
from crowd_nav.reward_search.amfrs.fidelity import TrainerContext, build_default_ladder
from crowd_nav.reward_search.amfrs.halving import HalvingConfig, SuccessiveHalvingScheduler
from crowd_nav.reward_search.amfrs.memory import RewardMemory
from crowd_nav.reward_search.amfrs.primitives import PRIMITIVE_REGISTRY
from crowd_nav.reward_search.amfrs.robustness import (
    attach_robustness_profile,
    pick_best_by_robustness,
    run_policy_sweep,
)
from crowd_nav.reward_search.amfrs.static_gate import StaticGate
from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.llm import (
    LLMClient,
    extract_python_code,
    make_llm_client,
    normalize_to_compute_reward,
)
from crowd_nav.reward_search.prompts import D5_SEED_FUNCTION
from crowd_nav.reward_search.sandbox import RewardValidator
from crowd_nav.reward_search.sandbox.config import SandboxConfig
from crowd_nav.reward_search.selection import navigation_scalar_from_dict

logger = logging.getLogger(__name__)


@dataclass
class AMFRSArtifacts:
    """Serializable run outputs for compare / thesis figures."""

    output_dir: str
    seed_candidates: List[RewardCandidate] = field(default_factory=list)
    accepted_candidates: List[RewardCandidate] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    archive: Optional[EliteGrid] = None
    best: Optional[RewardCandidate] = None
    promotion_log: List[Dict[str, Any]] = field(default_factory=list)
    cost_trace: List[Dict[str, Any]] = field(default_factory=list)
    manifest: Dict[str, Any] = field(default_factory=dict)

    def write(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "amfrs_manifest.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.manifest, fh, indent=2, default=str)
        if self.archive is not None:
            self.archive.to_json(os.path.join(self.output_dir, "map_elites_archive.json"))
        cost_path = os.path.join(self.output_dir, "cost_trace.json")
        with open(cost_path, "w", encoding="utf-8") as fh:
            json.dump(self.cost_trace, fh, indent=2)
        logger.info("Wrote %s", path)


class AMFRSPipeline:
    """Additive AMFRS v2 search pipeline."""

    def __init__(self, config: AMFRSRunConfig) -> None:
        self.config = config
        sandbox_cfg = SandboxConfig(extra_namespace=dict(PRIMITIVE_REGISTRY))
        self.validator = RewardValidator(config=sandbox_cfg)
        self.static_gate = StaticGate(
            n_states=int(config.static_gate_n_states),
            seed=int(config.static_gate_seed),
        )
        self.descriptor_cfg = DescriptorSpaceConfig()
        shape = config.archive_shape
        if shape[0] <= 0 or shape[1] <= 0:
            shape = grid_shape_from_config(self.descriptor_cfg)
        self.archive = EliteGrid(shape)
        self.memory: Optional[RewardMemory] = None
        if config.use_retrieval_memory:
            db = config.memory_db_path
            if not os.path.isabs(db):
                db = os.path.join(os.getcwd(), db)
            self.memory = RewardMemory(db)
        self.llm_a = self._make_llm(config.llm_provider, config.llm_model)
        self.llm_b = self._make_llm(config.llm_provider_b, config.llm_model)
        self.promotion_log: List[Dict[str, Any]] = []
        self.cost_trace: List[Dict[str, Any]] = []
        self._spent = [0.0]
        self._best_metric = float("-inf")

    @staticmethod
    def _make_llm(provider: str, model: Optional[str]) -> LLMClient:
        try:
            return make_llm_client(provider, model=model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM provider %s failed (%s); falling back to seed", provider, exc)
            return make_llm_client("seed", model=model)

    def _validate_and_gate(self, code: str, cid: str, origin: str) -> Optional[RewardCandidate]:
        reward_fn, err = self.validator.try_validate(code)
        if reward_fn is None:
            return RewardCandidate(
                candidate_id=cid,
                code=code,
                valid=False,
                origin=origin,
                validation_error=err,
                metadata={},
            )
        report = self.static_gate.check(reward_fn)
        cand = RewardCandidate(
            candidate_id=cid,
            code=code,
            reward_fn=reward_fn,
            valid=True,
            origin=origin,
            metadata={"static_report": report.to_dict()},
        )
        if not report.all_finite:
            cand = replace(
                cand,
                valid=False,
                reward_fn=None,
                validation_error="static_gate_non_finite",
            )
        return cand

    def _memory_block_for(self, code: str) -> str:
        if self.memory is None:
            return ""
        entries = self.memory.query_similar(code, k=3)
        return build_memory_block(entries)

    def _append_reject(
        self,
        rejected: List[Dict[str, Any]],
        cand: Optional[RewardCandidate],
        *,
        reason: str,
        attempt: int,
    ) -> None:
        rejected.append(
            {
                "candidate_id": cand.candidate_id if cand is not None else f"attempt_{attempt}",
                "reason": reason
                if cand is None
                else (cand.validation_error or reason),
                "static_report": (cand.metadata or {}).get("static_report")
                if cand is not None
                else None,
                "attempt": int(attempt),
            }
        )

    def _is_executable(self, cand: Optional[RewardCandidate]) -> bool:
        return cand is not None and bool(cand.valid) and cand.reward_fn is not None

    def _propose_initial_code(self, slot: int, attempt: int) -> str:
        """slot is the target index in [0, N); attempt counts retries for that slot."""
        cfg = self.config
        if slot == 0 and attempt == 0:
            return D5_SEED_FUNCTION.strip() + "\n"
        if cfg.fast or cfg.llm_provider == "seed":
            # Vary coefficients so retries are distinct genomes.
            i = slot + attempt * max(1, cfg.population_size)
            return (
                "def compute_reward(state, memory):\n"
                f"    g = goal_progress(state, memory)\n"
                f"    c = collision_indicator(state, memory)\n"
                f"    d = discomfort_penalty(state, memory)\n"
                f"    return float({1.0 + 0.1 * i} * g - 20.0 * c - {0.5 + 0.1 * i} * d)\n"
            )
        seed_code = D5_SEED_FUNCTION.strip() + "\n"
        prompt = build_initial_prompt(memory_block=self._memory_block_for(seed_code))
        raw = self.llm_a.complete(AMFRS_SYSTEM_PROMPT + "\n" + prompt)
        return normalize_to_compute_reward(extract_python_code(raw) or raw)

    def _generate_initial_population(
        self, rejected: List[Dict[str, Any]]
    ) -> List[RewardCandidate]:
        """
        Fill exactly ``population_size`` *valid* candidates.

        Invalid sandbox/static-gate outputs are logged and replaced until the
        budget ``population_size * max_invalid_replacements`` is exhausted.
        """
        cfg = self.config
        n = max(1, int(cfg.population_size))
        max_attempts = n * max(1, int(cfg.max_invalid_replacements))
        out: List[RewardCandidate] = []
        attempt = 0
        while len(out) < n and attempt < max_attempts:
            slot = len(out)
            retry_for_slot = 0
            while len(out) == slot and attempt < max_attempts:
                cid = (
                    f"seed_{slot}"
                    if retry_for_slot == 0
                    else f"seed_{slot}_r{retry_for_slot}"
                )
                code = self._propose_initial_code(slot, retry_for_slot)
                cand = self._validate_and_gate(code, cid, "initial")
                attempt += 1
                if self._is_executable(cand):
                    assert cand is not None
                    out.append(cand)
                    break
                self._append_reject(
                    rejected,
                    cand,
                    reason="invalid_initial",
                    attempt=attempt,
                )
                retry_for_slot += 1
                logger.info(
                    "Initial slot %s rejected (%s); have %s/%s valid",
                    slot,
                    cand.validation_error if cand else "None",
                    len(out),
                    n,
                )
        if len(out) < n:
            logger.warning(
                "Could only fill %s/%s valid initial candidates after %s attempts",
                len(out),
                n,
                attempt,
            )
        return out

    def _propose_mutation_code(self, parent: RewardCandidate, slot: int, retry: int) -> str:
        cfg = self.config
        if cfg.fast or cfg.llm_provider == "seed":
            i = slot + retry * 17
            return (
                "def compute_reward(state, memory):\n"
                f"    g = goal_progress(state, memory)\n"
                f"    c = collision_indicator(state, memory)\n"
                f"    d = discomfort_penalty(state, memory)\n"
                f"    return float({1.5 + 0.05 * i} * g - {18.0 + (i % 7)} * c - d)\n"
            )
        mem = self._memory_block_for(parent.code)
        prompt = build_mutation_prompt(
            parent,
            diagnostics=format_candidate_diagnostics(parent),
            memory_block=mem,
        )
        raw = self.llm_a.complete(AMFRS_SYSTEM_PROMPT + "\n" + prompt)
        return normalize_to_compute_reward(extract_python_code(raw) or raw)

    def _propose_children(
        self,
        parents: Sequence[RewardCandidate],
        gen: int,
        rejected: List[Dict[str, Any]],
    ) -> List[RewardCandidate]:
        """Propose exactly ``population_size`` valid children (refill on reject)."""
        cfg = self.config
        if not parents:
            return []
        n = max(1, int(cfg.population_size))
        max_attempts = n * max(1, int(cfg.max_invalid_replacements))
        parent_list = list(parents)
        rng = random.Random(int(cfg.seed) + int(gen) * 1009 + 17)
        children: List[RewardCandidate] = []
        attempt = 0
        while len(children) < n and attempt < max_attempts:
            slot = len(children)
            parent = rng.choice(parent_list) if parent_list else parent_list[0]
            # Last slot: prefer crossover when >=2 parents (once), else mutate
            use_xover = (
                slot == n - 1
                and len(parent_list) >= 2
                and not any(c.origin == "crossover" for c in children)
            )
            retry = 0
            filled = False
            while not filled and attempt < max_attempts:
                if use_xover:
                    cid = f"g{gen}_x{slot}" if retry == 0 else f"g{gen}_x{slot}_r{retry}"
                    if cfg.fast or cfg.llm_provider == "seed":
                        code = (
                            "def compute_reward(state, memory):\n"
                            "    return float("
                            f"{2.0 + 0.05 * retry} * goal_progress(state, memory) "
                            "- 20.0 * collision_indicator(state, memory) "
                            "- discomfort_penalty(state, memory))\n"
                        )
                    else:
                        if len(parent_list) >= 2:
                            a, b = rng.sample(parent_list, 2)
                        else:
                            a = b = parent_list[0]
                        prompt = build_semantic_crossover_prompt(
                            a,
                            b,
                            diagnostics_a=format_candidate_diagnostics(a),
                            diagnostics_b=format_candidate_diagnostics(b),
                            memory_block=self._memory_block_for(a.code),
                        )
                        raw = self.llm_a.complete(AMFRS_SYSTEM_PROMPT + "\n" + prompt)
                        code = normalize_to_compute_reward(extract_python_code(raw) or raw)
                    origin = "crossover"
                else:
                    cid = f"g{gen}_m{slot}" if retry == 0 else f"g{gen}_m{slot}_r{retry}"
                    code = self._propose_mutation_code(parent, slot, retry)
                    origin = "mutation"
                cand = self._validate_and_gate(code, cid, origin)
                attempt += 1
                if not self._is_executable(cand):
                    self._append_reject(
                        rejected, cand, reason=f"invalid_{origin}", attempt=attempt
                    )
                    retry += 1
                    continue
                assert cand is not None
                if cfg.use_ensemble_critique and not cfg.fast:
                    diag = format_candidate_diagnostics(parent)
                    report = critique_agrees(cand, diag, self.llm_a, self.llm_b)
                    cand = replace(
                        cand,
                        metadata={
                            **(cand.metadata or {}),
                            "ensemble_critique": {
                                "agreed": report.agreed,
                                "shared_terms": report.shared_terms,
                            },
                        },
                    )
                    if not report.agreed:
                        self._append_reject(
                            rejected,
                            cand,
                            reason="ensemble_critique_disagree",
                            attempt=attempt,
                        )
                        retry += 1
                        continue
                children.append(cand)
                filled = True
        if len(children) < n:
            logger.warning(
                "Could only fill %s/%s valid children for gen=%s after %s attempts",
                len(children),
                n,
                gen,
                attempt,
            )
        return children

    def _on_rung_complete(self, level_name: str, survivors: List[RewardCandidate]) -> None:
        spent_now = float(self._spent[0])
        self.promotion_log.append(
            {
                "level": level_name,
                "n_survivors": len(survivors),
                "ids": [c.candidate_id for c in survivors],
                "spent_cost": spent_now,
            }
        )
        for cand in survivors:
            metric = float(cand.score) if cand.score is not None else float("-inf")
            # F0 Score1 is not on the same scale as navigation_scalar (F1+).
            if not str(level_name).startswith("F0") and metric > self._best_metric:
                self._best_metric = metric
            self.cost_trace.append(
                {
                    "spent_cost": spent_now,
                    "best_metric": float(self._best_metric),
                    "level": level_name,
                    "candidate_id": cand.candidate_id,
                    "rung_metric": metric,
                }
            )
            raw = (cand.metadata or {}).get("last_raw_metrics") or {}
            if level_name.startswith("F0"):
                continue  # no reliable PL/SD for archive
            if "PL" not in raw and "pl" not in raw:
                continue
            cell = compute_behavior_descriptor(raw, config=self.descriptor_cfg)
            fitness = navigation_scalar_from_dict(raw)
            updated = replace(
                cand,
                metadata={
                    **(cand.metadata or {}),
                    "behavior_descriptor": list(cell),
                },
            )
            if self.config.selection_mode == "map_elites":
                self.archive.try_insert(updated, cell, fitness)
            if self.memory is not None:
                self.memory.add(cand.code, run_id=cand.candidate_id, outcome_metrics=dict(raw))

    def _trainer_ctx(self) -> TrainerContext:
        cfg = self.config
        return TrainerContext(
            stage2_train_steps=int(cfg.stage2_train_steps),
            stage2_train_steps_short=int(cfg.stage2_train_steps_short),
            stage2_eval_episodes=int(cfg.stage2_eval_episodes),
            stage3_train_steps=int(cfg.stage3_train_steps),
            stage3_eval_episodes=int(cfg.stage3_eval_episodes),
            seed=int(cfg.seed),
            device=str(cfg.device),
            num_processes=cfg.num_processes,
            human_num=int(cfg.human_num),
            predict_method=str(cfg.predict_method),
            randomization_regime=str(cfg.randomization_regime),
            output_root=os.path.join(cfg.output_dir, "trained_models"),
        )

    def _build_scheduler(
        self,
        max_rung: str,
        *,
        after_rung: Optional[str] = None,
        max_cost_units: Optional[float] = None,
        use_generation_budget: bool = True,
        cost_offset: float = 0.0,
    ) -> SuccessiveHalvingScheduler:
        use_stub = bool(self.config.use_stub_trainers or self.config.fast)
        ladder = build_default_ladder(
            use_stub=use_stub,
            score1_mode=self.config.score1_mode,
            stage1_dataset_path=self.config.stage1_dataset_path,
            trainer_ctx=None if use_stub else self._trainer_ctx(),
        )
        try:
            if after_rung is not None and after_rung != max_rung:
                ladder = ladder.slice_after(after_rung, max_rung)
            else:
                ladder = ladder.truncate_to(max_rung)
        except (KeyError, ValueError) as exc:
            logger.warning(
                "Could not build ladder max=%s after=%s (%s); using truncate_to(%s)",
                max_rung,
                after_rung,
                exc,
                max_rung,
            )
            ladder = build_default_ladder(
                use_stub=use_stub,
                score1_mode=self.config.score1_mode,
                stage1_dataset_path=self.config.stage1_dataset_path,
                trainer_ctx=None if use_stub else self._trainer_ctx(),
            ).truncate_to(max_rung)
        bandit = UCB1Allocator(cost_aware=True) if self.config.use_bandit else None
        if use_generation_budget:
            cost_cap: Optional[float] = float(self.config.max_cost_units_per_generation)
        else:
            cost_cap = max_cost_units  # may be None → uncapped
        return SuccessiveHalvingScheduler(
            ladder,
            HalvingConfig(
                eta=int(self.config.halving_eta),
                min_survivors=int(self.config.halving_min_survivors),
                max_cost_units=cost_cap,
                cost_offset=float(cost_offset),
            ),
            bandit=bandit,
        )

    def run(self) -> AMFRSArtifacts:
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        use_stub = bool(cfg.use_stub_trainers or cfg.fast)
        asset_report = require_amfrs_assets(
            regime=cfg.randomization_regime,
            predict_method=cfg.predict_method,
            score1_mode=cfg.score1_mode,
            stage1_dataset_path=cfg.stage1_dataset_path,
            use_stub=use_stub,
        )
        logger.info("Assets:\n%s", asset_report.format_text())

        rejected: List[Dict[str, Any]] = []
        accepted = self._generate_initial_population(rejected)
        seed_candidates = list(accepted)
        logger.info(
            "Initial population: %s valid (target=%s), %s rejected/replaced",
            len(accepted),
            cfg.population_size,
            len(rejected),
        )

        # Initial halving climb (up to illumination_max_rung, then optionally final)
        sched = self._build_scheduler(cfg.illumination_max_rung, cost_offset=0.0)
        survivors = sched.run(
            accepted,
            on_rung_complete=self._on_rung_complete,
            spent_cost=self._spent,
        )

        # Illumination generations from archive (or scalar survivors)
        coverages: List[float] = [self.archive.coverage()]
        for gen in range(int(cfg.generations)):
            if cfg.selection_mode == "map_elites" and self.archive.all_elites():
                parents = self.archive.all_elites()
            else:
                parents = survivors or accepted
            children = self._propose_children(parents, gen=gen, rejected=rejected)
            logger.info(
                "Gen %s children: %s valid (target=%s)",
                gen,
                len(children),
                cfg.population_size,
            )
            if not children:
                coverages.append(self.archive.coverage())
                continue
            gen_start = float(self._spent[0])
            gen_sched = self._build_scheduler(
                cfg.illumination_max_rung,
                cost_offset=gen_start,
            )
            survivors = gen_sched.run(
                children,
                on_rung_complete=self._on_rung_complete,
                spent_cost=self._spent,
            )
            coverages.append(self.archive.coverage())

        # Finalists: archive elites or scalar top survivors
        if cfg.selection_mode == "map_elites" and self.archive.all_elites():
            finalists = self.archive.all_elites()
        else:
            finalists = list(survivors)

        # Optional final rung bump: only rungs ABOVE illumination (no F0 restart).
        if finalists and cfg.final_rung and cfg.final_rung != cfg.illumination_max_rung:
            final_start = float(self._spent[0])
            final_sched = self._build_scheduler(
                cfg.final_rung,
                after_rung=cfg.illumination_max_rung,
                max_cost_units=cfg.final_rung_max_cost_units,
                use_generation_budget=False,
                cost_offset=final_start,
            )
            finalists = final_sched.run(
                finalists,
                on_rung_complete=self._on_rung_complete,
                spent_cost=self._spent,
            )

        # Axis 4 robustness on finalists
        robustness_sweep_degraded = False
        if cfg.run_robustness_sweep and finalists:
            robustified = []
            for cand in finalists:
                try:
                    by_pol = run_policy_sweep(
                        cand,
                        policies=tuple(cfg.robustness_policies),
                        use_stub=use_stub,
                        allow_stub_fallback=False,
                        train_steps=min(8_000, int(cfg.stage2_train_steps_short)),
                        eval_episodes=min(20, int(cfg.stage2_eval_episodes)),
                        device=str(cfg.device),
                        predict_method=str(cfg.predict_method),
                        human_num=int(cfg.human_num),
                        seed=int(cfg.seed),
                        output_root=os.path.join(cfg.output_dir, "robustness"),
                        randomization_regime=str(cfg.randomization_regime),
                    )
                    if any(
                        isinstance(v, dict) and v.get("stub_fallback")
                        for v in by_pol.values()
                    ):
                        robustness_sweep_degraded = True
                    robustified.append(attach_robustness_profile(cand, by_pol))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Robustness sweep failed for %s (%s); "
                        "candidate kept without robustness profile",
                        cand.candidate_id,
                        exc,
                    )
                    robustified.append(
                        replace(
                            cand,
                            metadata={
                                **(cand.metadata or {}),
                                "robustness_error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                    )
            finalists = robustified
            best = pick_best_by_robustness(finalists)
        else:
            best = None

        if best is None and finalists:
            ranked = sorted(
                finalists,
                key=lambda c: (
                    -(float(c.score) if c.score is not None else float("-inf")),
                    str(c.candidate_id),
                ),
            )
            best = ranked[0]

        manifest: Dict[str, Any] = {
            "pipeline": "AMFRS",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": cfg.to_dict(),
            "n_seed": len(seed_candidates),
            "n_accepted": len(accepted),
            "n_rejected": len(rejected),
            "promotion_log": self.promotion_log,
            "archive_coverage": self.archive.coverage(),
            "archive_qd_score": self.archive.qd_score(),
            "coverage_over_generations": coverages,
            "spent_cost_units": float(self._spent[0]),
            "best_id": best.candidate_id if best else None,
            "best_robustness": (best.metadata or {}).get("robustness_scalar") if best else None,
            "assets": asset_report.to_dict(),
            "use_stub_trainers": use_stub,
            "robustness_sweep_degraded": bool(robustness_sweep_degraded),
            "notes": (
                "AMFRS multi-fidelity search. "
                "Axes: static gate, successive halving, MAP-Elites, "
                "primitives/memory, multi-policy robustness."
            ),
        }
        artifacts = AMFRSArtifacts(
            output_dir=cfg.output_dir,
            seed_candidates=seed_candidates,
            accepted_candidates=accepted,
            rejected=rejected,
            archive=self.archive,
            best=best,
            promotion_log=self.promotion_log,
            cost_trace=self.cost_trace,
            manifest=manifest,
        )
        artifacts.write()
        if self.memory is not None:
            self.memory.close()
        return artifacts
