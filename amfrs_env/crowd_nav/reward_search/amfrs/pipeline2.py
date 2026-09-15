"""
AMFRS2 pipeline orchestrator (innovation track).

Wires axes 1–5: static gate → successive halving → MAP-Elites illumination →
optional retrieval memory / ensemble critique → robustness on final elites.

Does not call or modify ``AMFRSPipeline``.
"""

from __future__ import annotations

import json
import logging
import os
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
from crowd_nav.reward_search.amfrs.config import AMFRS2RunConfig
from crowd_nav.reward_search.amfrs.crossover import (
    AMFRS2_SYSTEM_PROMPT,
    build_initial_prompt,
    build_memory_block,
    build_mutation_prompt,
    build_semantic_crossover_prompt,
)
from crowd_nav.reward_search.amfrs.ensemble_critique import critique_agrees
from crowd_nav.reward_search.amfrs.assets import require_amfrs2_assets
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
class AMFRS2Artifacts:
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
        path = os.path.join(self.output_dir, "amfrs2_manifest.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.manifest, fh, indent=2, default=str)
        if self.archive is not None:
            self.archive.to_json(os.path.join(self.output_dir, "map_elites_archive.json"))
        cost_path = os.path.join(self.output_dir, "cost_trace.json")
        with open(cost_path, "w", encoding="utf-8") as fh:
            json.dump(self.cost_trace, fh, indent=2)
        logger.info("Wrote %s", path)


class AMFRS2Pipeline:
    """Additive AMFRS v2 search pipeline."""

    def __init__(self, config: AMFRS2RunConfig) -> None:
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
        # Active cost box for the current scheduler.run (may be a per-gen box).
        self._active_spent: List[float] = self._spent
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

    def _generate_initial_population(self) -> List[RewardCandidate]:
        cfg = self.config
        out: List[RewardCandidate] = []
        seed_code = D5_SEED_FUNCTION.strip() + "\n"
        # Always include D5 seed
        c0 = self._validate_and_gate(seed_code, "seed_0", "initial")
        if c0 is not None:
            out.append(c0)

        # Fill via LLM or deterministic numeric variants in fast/seed mode
        for i in range(1, cfg.population_size):
            cid = f"seed_{i}"
            if cfg.fast or cfg.llm_provider == "seed":
                code = (
                    "def compute_reward(state, memory):\n"
                    f"    g = goal_progress(state, memory)\n"
                    f"    c = collision_indicator(state, memory)\n"
                    f"    d = discomfort_penalty(state, memory)\n"
                    f"    return float({1.0 + 0.1 * i} * g - 20.0 * c - {0.5 + 0.1 * i} * d)\n"
                )
            else:
                prompt = build_initial_prompt(memory_block=self._memory_block_for(seed_code))
                raw = self.llm_a.complete(AMFRS2_SYSTEM_PROMPT + "\n" + prompt)
                code = normalize_to_compute_reward(extract_python_code(raw) or raw)
            cand = self._validate_and_gate(code, cid, "initial")
            if cand is not None:
                out.append(cand)
        return out

    def _on_rung_complete(self, level_name: str, survivors: List[RewardCandidate]) -> None:
        active = self._active_spent if self._active_spent is not None else self._spent
        self.promotion_log.append(
            {
                "level": level_name,
                "n_survivors": len(survivors),
                "ids": [c.candidate_id for c in survivors],
                "spent_cost": float(active[0]),
            }
        )
        for cand in survivors:
            metric = float(cand.score) if cand.score is not None else float("-inf")
            # F0 Score1 is not on the same scale as navigation_scalar (F1+).
            if not str(level_name).startswith("F0") and metric > self._best_metric:
                self._best_metric = metric
            self.cost_trace.append(
                {
                    "spent_cost": float(active[0]),
                    "global_spent_cost": float(self._spent[0]),
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
            ),
            bandit=bandit,
        )

    def _propose_children(self, parents: Sequence[RewardCandidate], gen: int) -> List[RewardCandidate]:
        cfg = self.config
        if not parents:
            return []
        children: List[RewardCandidate] = []
        # Mutate each elite (or scalar top parents)
        for i, parent in enumerate(list(parents)[: cfg.population_size]):
            cid = f"g{gen}_m{i}"
            mem = self._memory_block_for(parent.code)
            if cfg.fast or cfg.llm_provider == "seed":
                code = (
                    "def compute_reward(state, memory):\n"
                    f"    g = goal_progress(state, memory)\n"
                    f"    c = collision_indicator(state, memory)\n"
                    f"    d = discomfort_penalty(state, memory)\n"
                    f"    return float({1.5 + 0.05 * i} * g - {18.0 + i} * c - d)\n"
                )
            else:
                prompt = build_mutation_prompt(parent, memory_block=mem)
                raw = self.llm_a.complete(AMFRS2_SYSTEM_PROMPT + "\n" + prompt)
                code = normalize_to_compute_reward(extract_python_code(raw) or raw)
            cand = self._validate_and_gate(code, cid, "mutation")
            if cand is not None and cand.valid:
                if cfg.use_ensemble_critique and not cfg.fast:
                    diag = str((parent.metadata or {}).get("last_metric", ""))
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
                        # Keep parent unchanged (skip child)
                        continue
                children.append(cand)

        # One semantic crossover if >=2 parents
        if len(parents) >= 2:
            a, b = parents[0], parents[1]
            cid = f"g{gen}_x0"
            if cfg.fast or cfg.llm_provider == "seed":
                code = (
                    "def compute_reward(state, memory):\n"
                    "    return float("
                    "2.0 * goal_progress(state, memory) "
                    "- 20.0 * collision_indicator(state, memory) "
                    "- discomfort_penalty(state, memory))\n"
                )
            else:
                prompt = build_semantic_crossover_prompt(
                    a,
                    b,
                    memory_block=self._memory_block_for(a.code),
                )
                raw = self.llm_a.complete(AMFRS2_SYSTEM_PROMPT + "\n" + prompt)
                code = normalize_to_compute_reward(extract_python_code(raw) or raw)
            xc = self._validate_and_gate(code, cid, "crossover")
            if xc is not None and xc.valid:
                children.append(xc)
        return children

    def run(self) -> AMFRS2Artifacts:
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        use_stub = bool(cfg.use_stub_trainers or cfg.fast)
        asset_report = require_amfrs2_assets(
            regime=cfg.randomization_regime,
            predict_method=cfg.predict_method,
            score1_mode=cfg.score1_mode,
            stage1_dataset_path=cfg.stage1_dataset_path,
            use_stub=use_stub,
        )
        logger.info("Assets:\n%s", asset_report.format_text())

        seed_candidates = self._generate_initial_population()
        rejected: List[Dict[str, Any]] = []
        accepted: List[RewardCandidate] = []
        for cand in seed_candidates:
            if not cand.valid or cand.reward_fn is None:
                rejected.append(
                    {
                        "candidate_id": cand.candidate_id,
                        "reason": cand.validation_error or "invalid",
                        "static_report": (cand.metadata or {}).get("static_report"),
                    }
                )
            else:
                accepted.append(cand)

        # Initial halving climb (up to illumination_max_rung, then optionally final)
        self._active_spent = self._spent
        sched = self._build_scheduler(cfg.illumination_max_rung)
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
            children = self._propose_children(parents, gen=gen)
            if not children:
                coverages.append(self.archive.coverage())
                continue
            # Reset per-generation spent counter but keep global trace
            gen_spent = [0.0]
            self._active_spent = gen_spent
            gen_sched = self._build_scheduler(cfg.illumination_max_rung)
            survivors = gen_sched.run(
                children,
                on_rung_complete=self._on_rung_complete,
                spent_cost=gen_spent,
            )
            self._spent[0] += gen_spent[0]
            self._active_spent = self._spent
            coverages.append(self.archive.coverage())

        # Finalists: archive elites or scalar top survivors
        if cfg.selection_mode == "map_elites" and self.archive.all_elites():
            finalists = self.archive.all_elites()
        else:
            finalists = list(survivors)

        # Optional final rung bump: only rungs ABOVE illumination (no F0 restart).
        if finalists and cfg.final_rung and cfg.final_rung != cfg.illumination_max_rung:
            final_spent = [0.0]
            self._active_spent = final_spent
            final_sched = self._build_scheduler(
                cfg.final_rung,
                after_rung=cfg.illumination_max_rung,
                max_cost_units=cfg.final_rung_max_cost_units,
                use_generation_budget=False,
            )
            finalists = final_sched.run(
                finalists,
                on_rung_complete=self._on_rung_complete,
                spent_cost=final_spent,
            )
            self._spent[0] += final_spent[0]
            self._active_spent = self._spent

        # Axis 4 robustness on finalists
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
            "pipeline": "AMFRS2",
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
            "notes": (
                "Innovation track. Baseline AMFRSPipeline untouched. "
                "Axes: static gate, successive halving, MAP-Elites, "
                "primitives/memory, multi-policy robustness."
            ),
        }
        artifacts = AMFRS2Artifacts(
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
