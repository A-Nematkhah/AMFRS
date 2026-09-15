"""
AMFRS Algorithm 1 orchestrator (faithful replication baseline).

seed → Stage I → Stage II → Stage III. No AMFRS mechanisms.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from crowd_nav.reward_search.evolver import (
    RewardCandidate,
    StageIConfig,
    StageIEvolver,
)
from crowd_nav.reward_search import console
from crowd_nav.reward_search.llm import LLMClient, make_llm_client
from crowd_nav.reward_search.prompts import D5_SEED_FUNCTION
from crowd_nav.reward_search.reporting import (
    candidate_to_dict,
    stage2_stage3_calibration,
    write_json,
)
from crowd_nav.reward_search.sandbox import RewardValidator
from crowd_nav.reward_search.scoring import (
    Score1Options,
    make_score1_report_fn,
    make_smoke_score_fn,
    score1_mode_artifact,
    score1_report,
)
from crowd_nav.reward_search.dataset import load_stage1_dataset, split_stage1_dataset
from crowd_nav.reward_search.selection import (
    candidate_nav_scalar,
    navigation_scalar_from_dict,
    pick_best_trained,
    select_top_k_finalists,
)
from crowd_nav.reward_search.stage2 import (
    Stage2Config,
    Stage2Runner,
    StubPolicyTrainer as Stage2StubTrainer,
)
from crowd_nav.reward_search.stage3 import (
    STAGE3_PAPER_STEPS,
    STAGE3_STEPS,
    Stage3Config,
    Stage3Runner,
    StubPolicyTrainer as Stage3StubTrainer,
)

logger = logging.getLogger(__name__)


@dataclass
class AMFRSRunConfig:
    """End-to-end Algorithm 1 settings (paper defaults + practical overrides)."""

    output_dir: str = "results/amfrs_run"
    seed: int = 425
    llm_provider: str = "seed"  # seed | groq | vllm | ollama | scripted
    llm_model: Optional[str] = None
    # Stage I Score1: "dataset" (default, paper) | "smoke" (opt-in fast tests only)
    score1_mode: str = "dataset"
    stage1_dataset_path: str = "data/stage1_dataset"
    # Fraction of scenarios held out from Stage I evolution scoring (Phase 1).
    stage1_holdout_fraction: float = 0.2

    # Stage I (Table 5 / §5.1)
    stage1_population: int = 8
    stage1_generations: int = 10

    # Stage II — paper Table 5 uses K2=8000; that is too short to rank rewards
    # reliably in practice. Default 5e4 for local/scaled runs; paper_scale.py
    # still forces PAPER_K2=8000 when reproducing the paper budget.
    stage2_rounds: int = 16
    stage2_train_steps: int = 50_000
    stage2_eval_episodes: int = 50
    stage2_horizon: int = 100
    stage2_use_stub: bool = False
    # Phase 2 Stage II gate
    stage2_n_eval_seeds: int = 2
    stage2_accept_reject_refine: bool = True
    stage2_accept_reject_tolerance: float = 0.0
    stage2_accept_reject_steps: Optional[int] = None

    # Stage III
    stage3_rounds: int = 3
    stage3_train_steps: int = STAGE3_STEPS  # paper: 1e7 — see STAGE3_PAPER_STEPS
    stage3_eval_episodes: int = 500
    stage3_use_stub: bool = False
    stage3_run_h_sweep: bool = True
    # Phase 3: admit only top-k Stage II genomes into Stage III.
    stage3_max_finalists: int = 3
    stage3_accept_reject_refine: bool = True
    stage3_accept_reject_tolerance: float = 0.0
    stage3_accept_reject_steps: Optional[int] = None
    stage3_skip_refine_last_round: bool = True
    stage3_h_sweep_max_finalists: int = 2
    stage3_h_profile_mean_weight: float = 0.5
    # None → paper-ish defaults clipped to human_num: {5,10,15,20} ∩ [1,H].
    # Explicit list (e.g. (3,5,7)) for easier local validation sweeps.
    stage3_h_counts: Optional[Tuple[int, ...]] = None

    device: str = "cuda"
    # None → auto (min(16, cpu-1)); set low on 4GB GPUs to avoid OOM.
    num_processes: Optional[int] = None
    # AUDIT.md §8.1 — first validation pass defaults to without_random.
    randomization_regime: str = "without_random"
    # AUDIT.md §8.2 choice (a): Stage II/III use GST-inferred obs.
    predict_method: str = "inferred"
    # Crowd size for Stage II/III train+eval (paper=20). Lower for easier debugging.
    human_num: int = 20
    # Fast dry-run profile (tests / laptop)
    fast: bool = False

    def apply_fast_profile(self) -> None:
        """Seconds-scale Algorithm 1 walk with stub trainers + tiny budgets."""
        self.fast = True
        self.score1_mode = "smoke"
        self.stage1_population = 2
        self.stage1_generations = 1
        self.stage2_rounds = 1
        self.stage2_train_steps = 8
        self.stage2_eval_episodes = 2
        self.stage2_horizon = 5
        self.stage2_use_stub = True
        self.stage2_n_eval_seeds = 1
        self.stage2_accept_reject_refine = True
        self.stage3_rounds = 1
        self.stage3_train_steps = 8
        self.stage3_eval_episodes = 2
        self.stage3_use_stub = True
        self.stage3_run_h_sweep = True
        self.stage3_max_finalists = 2
        self.stage3_h_sweep_max_finalists = 2
        self.llm_provider = "seed"
        # Stubs never load GST; keep flags consistent for config builders.
        self.predict_method = "none"

    def apply_easy_profile(self) -> None:
        """
        Easier simulator for pipeline result-getting (not paper claims).

        - No GST prediction (faster, simpler obs)
        - Fewer humans (5 vs 20)
        - Longer Stage II horizon (~50s episodes)
        """
        self.predict_method = "none"
        self.human_num = 5
        self.stage2_horizon = 200  # 200 * 0.25s = 50s
        self.stage3_run_h_sweep = False


@dataclass
class AMFRSArtifacts:
    """Paths / populations produced by one Algorithm 1 run."""

    output_dir: str
    seed_code: str
    stage1_population: List[RewardCandidate] = field(default_factory=list)
    stage2_population: List[RewardCandidate] = field(default_factory=list)
    stage3_population: List[RewardCandidate] = field(default_factory=list)
    best_stage1: Optional[RewardCandidate] = None
    best_stage2: Optional[RewardCandidate] = None
    best_stage3: Optional[RewardCandidate] = None
    manifest: Dict[str, Any] = field(default_factory=dict)


class AMFRSPipeline:
    """Reproduce Algorithm 1 end-to-end and persist JSON artifacts."""

    def __init__(
        self,
        config: Optional[AMFRSRunConfig] = None,
        *,
        llm: Optional[LLMClient] = None,
        checkpoint_store: Optional[Any] = None,
    ) -> None:
        self.config = config or AMFRSRunConfig()
        self.llm = llm
        self.validator = RewardValidator()
        # Optional paper-scale resume store (seed/stage/round/candidate).
        self.checkpoint_store = checkpoint_store
        self._score1_train: Optional[Dict[str, Any]] = None
        self._score1_holdout: Optional[Dict[str, Any]] = None
        self._score1_options: Optional[Score1Options] = None

    @staticmethod
    def resolve_h_sweep_counts(
        human_num: int,
        *,
        run_h_sweep: bool = True,
        requested: Optional[Sequence[int]] = None,
    ) -> Tuple[int, ...]:
        """
        Build Stage III H-sweep set.

        Counts must be ``1 .. human_num`` (Policy / obs width = train H).
        If ``requested`` is set, use that list (clipped). Otherwise use the
        paper Table-6 set ``{5,10,15,20}`` clipped to ``human_num``, always
        including ``human_num`` itself.
        """
        h_train = max(1, int(human_num))
        if not run_h_sweep:
            return (h_train,)
        if requested:
            swept = [int(h) for h in requested if 1 <= int(h) <= h_train]
            if not swept:
                swept = [h_train]
            return tuple(sorted(set(swept)))
        swept = [h for h in (5, 10, 15, 20) if h <= h_train]
        if h_train not in swept:
            swept.append(h_train)
        return tuple(sorted(set(swept)))

    def _build_llm(self) -> LLMClient:
        if self.llm is not None:
            return self.llm
        if self.config.llm_model is None:
            return make_llm_client(self.config.llm_provider)
        return make_llm_client(self.config.llm_provider, model=self.config.llm_model)

    def _score_fn(self):
        """
        Build Stage I score_fn and Score1 artifact metadata.

        Returns ``(score_fn, score1_info)`` where ``score1_info`` is written
        into the run manifest (mode labeling, holdout split, options).
        """
        mode = str(self.config.score1_mode).strip().lower()
        mode_meta = score1_mode_artifact(mode, fast=bool(self.config.fast))
        score1_info: Dict[str, Any] = {
            **mode_meta,
            "mask_pads": True,
            "anti_exploit": True,
            "holdout": None,
        }

        if mode == "smoke":
            logger.warning(
                "Using make_smoke_score_fn (opt-in fast fixture) — not paper Score1"
            )
            console.status(
                "WARNING: score1_mode=smoke — NOT paper Score1; do not claim paper results",
                stage="pipeline",
            )
            return make_smoke_score_fn(), score1_info

        if mode != "dataset":
            raise ValueError(f"Unknown score1_mode: {self.config.score1_mode!r}")

        path = self.config.stage1_dataset_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Stage I dataset not found at {path}. "
                f"Run: python scripts/collect_stage1_dataset.py --out {path}"
            )
        dataset = load_stage1_dataset(path)
        logger.info(
            "Loaded Stage I dataset from %s (%d scenarios)", path, len(dataset)
        )

        train, holdout, split_meta = split_stage1_dataset(
            dataset,
            holdout_fraction=float(self.config.stage1_holdout_fraction),
            seed=int(self.config.seed),
        )
        score1_info["holdout"] = split_meta
        score1_info["n_scenarios_total"] = len(dataset)
        opts = Score1Options(mask_pads=True, anti_exploit=True)
        score1_info["options"] = {
            "mask_pads": opts.mask_pads,
            "anti_exploit": opts.anti_exploit,
            "min_step_std": opts.min_step_std,
            "max_abs_step": opts.max_abs_step,
        }
        # Stash holdout for post-Stage-I reporting (same pipeline instance).
        self._score1_train = train
        self._score1_holdout = holdout
        self._score1_options = opts

        console.status(
            f"Score1 dataset split: train={split_meta['n_train_scenarios']} "
            f"holdout={split_meta['n_holdout_scenarios']} "
            f"(fraction={split_meta['holdout_fraction']})",
            stage="pipeline",
        )
        return make_score1_report_fn(train, options=opts), score1_info

    def _holdout_scores_for_population(
        self, population: List[RewardCandidate]
    ) -> Dict[str, Any]:
        """Evaluate train-scored candidates on holdout scenarios (Phase 1)."""
        holdout = getattr(self, "_score1_holdout", None) or {}
        opts = getattr(self, "_score1_options", None) or Score1Options()
        if not holdout:
            return {"available": False, "scores": {}}

        scores: Dict[str, Any] = {}
        for cand in population:
            if not cand.is_executable:
                scores[cand.candidate_id] = None
                continue
            try:
                report = score1_report(
                    holdout,
                    cand.as_reward_function(),
                    candidate_id=cand.candidate_id,
                    options=opts,
                )
                scores[cand.candidate_id] = {
                    "holdout_score": report.score,
                    "anti_exploit_triggered": report.anti_exploit_triggered,
                    "anti_exploit_reason": report.anti_exploit_reason,
                    "n_scenarios_scored": report.n_scenarios_scored,
                }
            except Exception as exc:  # noqa: BLE001
                scores[cand.candidate_id] = {
                    "holdout_score": None,
                    "error": str(exc),
                }
        return {"available": True, "scores": scores}

    @staticmethod
    def _include_global_best(
        population: List[RewardCandidate], global_best: RewardCandidate
    ) -> List[RewardCandidate]:
        """Keep the best Stage-I candidate in the population sent to Stage II."""
        if any(candidate.candidate_id == global_best.candidate_id for candidate in population):
            return population
        worst_index = min(
            range(len(population)),
            key=lambda index: population[index].score if population[index].score is not None else float("-inf"),
        )
        population[worst_index] = global_best
        return population

    def _best(self, population: List[RewardCandidate]) -> RewardCandidate:
        ranked = sorted(
            population,
            key=lambda c: (
                float(c.score) if c.score is not None else float("-inf"),
                float((c.metadata or {}).get("last_metrics", {}).get("SR", 0.0)),
            ),
            reverse=True,
        )
        return ranked[0]

    def run(self) -> AMFRSArtifacts:
        import time

        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)
        wall_t0 = time.perf_counter()
        from crowd_nav.reward_search.regime import (
            EVOLUTION_PREDICT_METHOD,
            assert_gst_matches_regime,
            env_name_for_predict_method,
            gst_model_dir_for_regime,
            parse_regime,
            randomization_flags,
        )

        regime = parse_regime(cfg.randomization_regime)
        cfg.randomization_regime = regime
        predict_method = (cfg.predict_method or EVOLUTION_PREDICT_METHOD).strip().lower()
        cfg.predict_method = predict_method
        attrs, goals = randomization_flags(regime)
        console.banner("AMFRS Algorithm 1")
        console.status(
            f"output={cfg.output_dir} seed={cfg.seed} llm={cfg.llm_provider} "
            f"fast={cfg.fast} device={cfg.device}"
        )
        console.status(
            f"regime={regime} (randomize_attributes={attrs}, "
            f"random_goal_changing={goals}); predict_method={predict_method}"
        )
        if predict_method == "inferred":
            assert_gst_matches_regime(
                gst_model_dir_for_regime(regime),
                regime,
                predict_method=predict_method,
                entry_point="pipeline_startup",
            )
        else:
            assert_gst_matches_regime(
                "",
                regime,
                predict_method=predict_method,
                entry_point="pipeline_startup",
            )

        llm = self._build_llm()
        seed_code = D5_SEED_FUNCTION.strip() + "\n"

        manifest: Dict[str, Any] = {
            "algorithm": "AMFRS Algorithm 1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(cfg),
            "stage3_paper_steps": STAGE3_PAPER_STEPS,
            "evolution_randomization_regime": regime,
            "predict_method": predict_method,
            "notes": (
                "Faithful replication baseline. No AMFRS novelty / archive / "
                "Pareto / adaptive controller. Obs-space choice (a): "
                "Stage II/III use predict_method=inferred when not --fast "
                "(AUDIT.md §8)."
            ),
        }
        write_json(os.path.join(cfg.output_dir, "config.json"), asdict(cfg))
        with open(os.path.join(cfg.output_dir, "seed_reward.py"), "w", encoding="utf-8") as f:
            f.write(seed_code)

        # ----- Stage I -----
        n = cfg.stage1_population
        n_crossover = min(2, n)
        n_mutation = min(4, max(0, n - n_crossover))
        n_random = n - n_crossover - n_mutation
        s1_cfg = StageIConfig(
            population_size=n,
            generations=cfg.stage1_generations,
            n_crossover=n_crossover,
            n_mutation=n_mutation,
            n_random=n_random,
        )
        score_fn, score1_info = self._score_fn()
        manifest["score1"] = score1_info
        if score1_info.get("warning"):
            manifest["score1_warning"] = score1_info["warning"]
        evolver = StageIEvolver(
            llm,
            score_fn=score_fn,
            validator=self.validator,
            config=s1_cfg,
            rejection_log_path=os.path.join(cfg.output_dir, "stage1_rejections.jsonl"),
        )
        stage1_pop = evolver.run()
        if evolver.global_best is None:
            raise RuntimeError("Stage I did not produce a global best candidate.")
        best_s1 = evolver.global_best
        stage1_pop = self._include_global_best(stage1_pop, best_s1)
        holdout_block = self._holdout_scores_for_population(stage1_pop)
        best_holdout = None
        if holdout_block.get("available"):
            best_holdout = (holdout_block.get("scores") or {}).get(best_s1.candidate_id)
        manifest["score1"]["best_train_score"] = best_s1.score
        manifest["score1"]["best_holdout"] = best_holdout
        write_json(
            os.path.join(cfg.output_dir, "stage1_population.json"),
            {
                "ranking": [c.candidate_id for c in stage1_pop],
                "population": [candidate_to_dict(c) for c in stage1_pop],
                "holdout": holdout_block,
                "score1": score1_info,
                "history": [
                    {
                        "generation": h.generation,
                        "ranking": h.ranking,
                        "best_id": h.best_id,
                        "best_score": h.best_score,
                        "reflection": h.reflection,
                    }
                    for h in evolver.history
                ],
            },
        )
        write_json(
            os.path.join(cfg.output_dir, "best_stage1.json"),
            {
                **candidate_to_dict(best_s1),
                "holdout": best_holdout,
                "score1_mode": score1_info.get("score1_mode"),
                "is_paper_score1": score1_info.get("is_paper_score1"),
            },
        )
        console.status(
            f"Stage I complete - best={best_s1.candidate_id} score={best_s1.score}",
            stage="pipeline",
        )
        if best_holdout and best_holdout.get("holdout_score") is not None:
            console.status(
                f"Stage I holdout score for best={best_holdout['holdout_score']}",
                stage="pipeline",
            )

        # ----- Stage II -----
        s2_cfg = Stage2Config(
            population_size=len(stage1_pop),
            rounds=cfg.stage2_rounds,
            train_env_steps=cfg.stage2_train_steps,
            eval_episodes=cfg.stage2_eval_episodes,
            horizon_steps=cfg.stage2_horizon,
            seed=cfg.seed,
            device=cfg.device,
            num_processes=cfg.num_processes,
            human_num=int(cfg.human_num),
            output_root=os.path.join(cfg.output_dir, "stage2_train"),
            randomization_regime=regime,
            predict_method=predict_method,
            env_name=env_name_for_predict_method(predict_method),
            n_eval_seeds=int(cfg.stage2_n_eval_seeds),
            accept_reject_refine=bool(cfg.stage2_accept_reject_refine),
            accept_reject_tolerance=float(cfg.stage2_accept_reject_tolerance),
            accept_reject_steps=cfg.stage2_accept_reject_steps,
        )
        if cfg.stage2_use_stub:
            s2_trainer = Stage2StubTrainer()
            console.status("Stage II using StubPolicyTrainer", stage="pipeline")
        else:
            from crowd_nav.reward_search.stage2 import RealPolicyTrainer as S2Real

            s2_trainer = S2Real()
        s2_runner = Stage2Runner(
            llm, s2_trainer, validator=self.validator, config=s2_cfg
        )
        if self.checkpoint_store is not None:
            s2_runner.checkpoint_store = self.checkpoint_store
            s2_runner.checkpoint_seed = int(cfg.seed)
        stage2_pop = s2_runner.run(stage1_pop)
        best_s2 = s2_runner.best_trained or self._best_by_ever_metrics(
            stage2_pop, s2_runner.history, s2_runner.trained_snapshots
        )
        write_json(
            os.path.join(cfg.output_dir, "stage2_population.json"),
            {
                "population": [candidate_to_dict(c) for c in stage2_pop],
                "best_trained_id": (
                    best_s2.candidate_id if best_s2 is not None else None
                ),
                "history": [
                    {
                        "round_index": r.round_index,
                        "candidate_id": r.candidate_id,
                        "metrics": r.metrics.as_dict(),
                        "refined": r.refined,
                        "kept_previous": r.kept_previous,
                        "refine_accepted": getattr(r, "refine_accepted", None),
                        "verify_scalar": getattr(r, "verify_scalar", None),
                    }
                    for r in s2_runner.history
                ],
            },
        )
        write_json(
            os.path.join(cfg.output_dir, "best_stage2.json"),
            candidate_to_dict(best_s2),
        )
        console.status(
            f"Stage II complete - best={best_s2.candidate_id} "
            f"scalar={candidate_nav_scalar(best_s2):.3f}",
            stage="pipeline",
        )

        # ----- Stage III -----
        # H-sweep only at H <= training crowd size (obs / Policy width).
        h_train = max(1, int(cfg.human_num))
        human_counts = self.resolve_h_sweep_counts(
            h_train,
            run_h_sweep=bool(cfg.stage3_run_h_sweep),
            requested=cfg.stage3_h_counts,
        )
        if cfg.stage3_run_h_sweep and len(human_counts) <= 1:
            console.status(
                f"H-sweep enabled but only H={human_counts} "
                f"(train human_num={h_train}); pass --h-sweep-counts "
                f"or raise --human-num for multi-H generalization",
                stage="pipeline",
            )
        elif cfg.stage3_run_h_sweep:
            console.status(
                f"H-sweep counts={human_counts} (train H={h_train})",
                stage="pipeline",
            )

        # Phase 3 / T8 — Stage III is a finalist tournament, not full Stage II pop.
        pool_by_code: Dict[str, RewardCandidate] = {}
        for cand in list(stage2_pop) + list(s2_runner.trained_snapshots):
            code = cand.code.strip()
            prev = pool_by_code.get(code)
            if prev is None or candidate_nav_scalar(cand) > candidate_nav_scalar(prev):
                pool_by_code[code] = cand
        finalists = select_top_k_finalists(
            list(pool_by_code.values()),
            int(cfg.stage3_max_finalists),
            prefer=best_s2,
        )
        console.status(
            f"Stage III admission: {len(finalists)}/{len(stage2_pop)} finalists "
            f"(max_k={cfg.stage3_max_finalists})",
            stage="pipeline",
        )
        write_json(
            os.path.join(cfg.output_dir, "stage3_finalists.json"),
            {
                "max_finalists": int(cfg.stage3_max_finalists),
                "finalists": [candidate_to_dict(c) for c in finalists],
            },
        )

        s3_cfg = Stage3Config(
            population_size=len(finalists),
            rounds=cfg.stage3_rounds,
            train_env_steps=cfg.stage3_train_steps,
            eval_episodes=cfg.stage3_eval_episodes,
            seed=cfg.seed,
            device=cfg.device,
            num_processes=cfg.num_processes,
            train_human_num=h_train,
            output_root=os.path.join(cfg.output_dir, "stage3_train"),
            human_counts=human_counts,
            randomization_regime=regime,
            predict_method=predict_method,
            env_name=env_name_for_predict_method(predict_method),
            accept_reject_refine=bool(cfg.stage3_accept_reject_refine),
            accept_reject_tolerance=float(cfg.stage3_accept_reject_tolerance),
            accept_reject_steps=cfg.stage3_accept_reject_steps,
            skip_refine_last_round=bool(cfg.stage3_skip_refine_last_round),
            h_sweep_max_finalists=int(cfg.stage3_h_sweep_max_finalists),
            h_profile_mean_weight=float(cfg.stage3_h_profile_mean_weight),
        )
        if cfg.stage3_use_stub:
            s3_trainer = Stage3StubTrainer()
            console.status("Stage III using StubPolicyTrainer", stage="pipeline")
        else:
            from crowd_nav.reward_search.stage3 import RealPolicyTrainer as S3Real

            s3_trainer = S3Real()
        s3_runner = Stage3Runner(
            llm, s3_trainer, validator=self.validator, config=s3_cfg
        )
        if self.checkpoint_store is not None:
            s3_runner.checkpoint_store = self.checkpoint_store
            s3_runner.checkpoint_seed = int(cfg.seed)
        stage3_pop = s3_runner.run(finalists, run_h_sweep=cfg.stage3_run_h_sweep)
        best_s3 = (
            s3_runner.best_h_aware
            or s3_runner.best_trained
            or self._best_by_ever_metrics(
                stage3_pop, s3_runner.history, s3_runner.trained_snapshots
            )
        )
        write_json(
            os.path.join(cfg.output_dir, "stage3_population.json"),
            {
                "population": [candidate_to_dict(c) for c in stage3_pop],
                "finalist_ids": [c.candidate_id for c in finalists],
                "best_trained_id": (
                    best_s3.candidate_id if best_s3 is not None else None
                ),
                "h_aware_selected": bool(
                    (best_s3.metadata or {}).get("h_aware_selected")
                )
                if best_s3 is not None
                else False,
                "history": [
                    {
                        "round_index": r.round_index,
                        "candidate_id": r.candidate_id,
                        "metrics": r.metrics.as_dict(),
                        "refined": r.refined,
                        "kept_previous": r.kept_previous,
                        "checkpoint_path": r.checkpoint_path,
                    }
                    for r in s3_runner.history
                ],
                "h_sweep": [
                    {
                        "candidate_id": rep.candidate_id,
                        "by_human_count": {
                            str(h): m.as_dict()
                            for h, m in rep.by_human_count.items()
                        },
                        "summary_table": rep.summary_table(),
                    }
                    for rep in s3_runner.sweep_reports
                ],
                "h_profiled_finalists": [
                    candidate_to_dict(c) for c in s3_runner.h_profiled_finalists
                ],
            },
        )
        write_json(
            os.path.join(cfg.output_dir, "best_stage3.json"),
            candidate_to_dict(best_s3),
        )
        write_json(
            os.path.join(cfg.output_dir, "final_candidate.json"),
            candidate_to_dict(best_s3),
        )
        console.status(
            f"Stage III complete - best={best_s3.candidate_id} "
            f"scalar={candidate_nav_scalar(best_s3):.3f}",
            stage="pipeline",
        )

        # Phase 4 / T13 — II↔III rank calibration (additive; does not change selection).
        stage2_scores: Dict[str, float] = {}
        for cand in list(pool_by_code.values()):
            stage2_scores[cand.code.strip()] = candidate_nav_scalar(cand)
        # Prefer stable genome keys; also index by admission finalist ids for readability.
        stage2_by_id: Dict[str, float] = {
            c.candidate_id: candidate_nav_scalar(c) for c in finalists
        }
        stage3_by_id: Dict[str, float] = {}
        for cand in list(s3_runner.trained_snapshots) + list(stage3_pop):
            # Prefer H-aware scalar when present.
            md = cand.metadata or {}
            if md.get("h_profile_scalar") is not None:
                stage3_by_id[cand.candidate_id] = float(md["h_profile_scalar"])
            else:
                stage3_by_id[cand.candidate_id] = candidate_nav_scalar(cand)
        # Align on shared IDs when possible; else map finalists' Stage II scalars
        # to Stage III snapshots with matching code.
        code_to_s2_id = {c.code.strip(): c.candidate_id for c in finalists}
        s2_for_corr: Dict[str, float] = dict(stage2_by_id)
        s3_for_corr: Dict[str, float] = {}
        for cand in list(s3_runner.h_profiled_finalists) or list(
            s3_runner.trained_snapshots
        ):
            cid = code_to_s2_id.get(cand.code.strip(), cand.candidate_id)
            md = cand.metadata or {}
            if md.get("h_profile_scalar") is not None:
                s3_for_corr[cid] = float(md["h_profile_scalar"])
            else:
                s3_for_corr[cid] = candidate_nav_scalar(cand)
            if cid not in s2_for_corr:
                s2_for_corr[cid] = stage2_scores.get(
                    cand.code.strip(), candidate_nav_scalar(cand)
                )
        if not s3_for_corr:
            s3_for_corr = dict(stage3_by_id)
        calibration = stage2_stage3_calibration(
            s2_for_corr,
            s3_for_corr,
            stage2_best_id=best_s2.candidate_id if best_s2 is not None else None,
            stage3_best_id=best_s3.candidate_id if best_s3 is not None else None,
        )
        calib_path = os.path.join(cfg.output_dir, "stage2_stage3_calibration.json")
        write_json(calib_path, calibration)
        manifest["calibration"] = {
            "stage2_stage3": "stage2_stage3_calibration.json",
            "spearman": (calibration.get("correlation") or {}).get("spearman"),
            "winner_match": calibration.get("winner_match"),
        }
        console.status(
            f"II↔III calibration spearman="
            f"{(calibration.get('correlation') or {}).get('spearman')} "
            f"winner_match={calibration.get('winner_match')}",
            stage="pipeline",
        )

        manifest["best_stage1_id"] = best_s1.candidate_id
        manifest["best_stage2_id"] = best_s2.candidate_id
        manifest["best_stage3_id"] = best_s3.candidate_id
        write_json(os.path.join(cfg.output_dir, "manifest.json"), manifest)

        wall = time.perf_counter() - wall_t0
        console.final_run_summary(
            output_dir=cfg.output_dir,
            wall_seconds=wall,
            best_stage1=best_s1,
            best_stage2=best_s2,
            best_stage3=best_s3,
        )

        return AMFRSArtifacts(
            output_dir=cfg.output_dir,
            seed_code=seed_code,
            stage1_population=stage1_pop,
            stage2_population=stage2_pop,
            stage3_population=stage3_pop,
            best_stage1=best_s1,
            best_stage2=best_s2,
            best_stage3=best_s3,
            manifest=manifest,
        )

    @staticmethod
    def _best_by_ever_metrics(
        population,
        history,
        trained_snapshots=None,
    ) -> RewardCandidate:
        """
        Pick the best-ever trained genome by SR - CR - 0.5*TR across all rounds.

        Prefers explicit trained snapshots (correct code+metrics+checkpoint).
        Falls back to history scan + final population mapping.
        """
        if trained_snapshots:
            best = pick_best_trained(trained_snapshots)
            if best is not None:
                return best

        best_hist = None
        best_score = float("-inf")
        if history:
            for r in history:
                score = float(r.metrics.scalar_score())
                if score > best_score:
                    best_score = score
                    best_hist = r

        def _key(c: RewardCandidate):
            m = (c.metadata or {}).get("last_metrics")
            if m:
                return navigation_scalar_from_dict(m)
            if best_hist is not None and (
                c.candidate_id == best_hist.candidate_id
                or best_hist.candidate_id in (c.parent_ids or ())
            ):
                return best_score
            return float(c.score or float("-inf"))

        return max(population, key=_key)

    # Backward-compatible alias (older call sites / notebooks).
    _best_by_last_metrics = _best_by_ever_metrics
