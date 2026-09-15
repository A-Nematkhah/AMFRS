"""
EvoNav Algorithm 1 orchestrator (faithful replication baseline).

seed → Stage I → Stage II → Stage III. No AMFRS mechanisms.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from crowd_nav.reward_search.evolver import (
    RewardCandidate,
    StageIConfig,
    StageIEvolver,
)
from crowd_nav.reward_search import console
from crowd_nav.reward_search.llm import LLMClient, make_llm_client
from crowd_nav.reward_search.prompts import D5_SEED_FUNCTION
from crowd_nav.reward_search.reporting import candidate_to_dict, write_json
from crowd_nav.reward_search.sandbox import RewardValidator
from crowd_nav.reward_search.scoring import make_score1_fn, make_smoke_score_fn
from crowd_nav.reward_search.dataset import load_stage1_dataset
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
class EvoNavRunConfig:
    """End-to-end Algorithm 1 settings (paper defaults + practical overrides)."""

    output_dir: str = "results/evonav_run"
    seed: int = 425
    llm_provider: str = "seed"  # seed | groq | vllm | ollama | scripted
    llm_model: Optional[str] = None
    # Stage I Score1: "dataset" (default, paper) | "smoke" (opt-in fast tests only)
    score1_mode: str = "dataset"
    stage1_dataset_path: str = "data/stage1_dataset"

    # Stage I (Table 5 / §5.1)
    stage1_population: int = 8
    stage1_generations: int = 10

    # Stage II
    stage2_rounds: int = 16
    stage2_train_steps: int = 8000
    stage2_eval_episodes: int = 50
    stage2_horizon: int = 100
    stage2_use_stub: bool = False

    # Stage III
    stage3_rounds: int = 3
    stage3_train_steps: int = STAGE3_STEPS  # paper: 1e7 — see STAGE3_PAPER_STEPS
    stage3_eval_episodes: int = 500
    stage3_use_stub: bool = False
    stage3_run_h_sweep: bool = True

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
        self.stage3_rounds = 1
        self.stage3_train_steps = 8
        self.stage3_eval_episodes = 2
        self.stage3_use_stub = True
        self.stage3_run_h_sweep = True
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
class EvoNavArtifacts:
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


class EvoNavPipeline:
    """Reproduce Algorithm 1 end-to-end and persist JSON artifacts."""

    def __init__(
        self,
        config: Optional[EvoNavRunConfig] = None,
        *,
        llm: Optional[LLMClient] = None,
        checkpoint_store: Optional[Any] = None,
    ) -> None:
        self.config = config or EvoNavRunConfig()
        self.llm = llm
        self.validator = RewardValidator()
        # Optional paper-scale resume store (seed/stage/round/candidate).
        self.checkpoint_store = checkpoint_store

    def _build_llm(self) -> LLMClient:
        if self.llm is not None:
            return self.llm
        if self.config.llm_model is None:
            return make_llm_client(self.config.llm_provider)
        return make_llm_client(self.config.llm_provider, model=self.config.llm_model)

    def _score_fn(self):
        mode = str(self.config.score1_mode).strip().lower()
        if mode == "smoke":
            logger.warning(
                "Using make_smoke_score_fn (opt-in fast fixture) — not paper Score1"
            )
            return make_smoke_score_fn()
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
        return make_score1_fn(dataset)

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

    def run(self) -> EvoNavArtifacts:
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
        console.banner("EvoNav Algorithm 1")
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
            "algorithm": "EvoNav Algorithm 1",
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
        evolver = StageIEvolver(
            llm,
            score_fn=self._score_fn(),
            validator=self.validator,
            config=s1_cfg,
            rejection_log_path=os.path.join(cfg.output_dir, "stage1_rejections.jsonl"),
        )
        stage1_pop = evolver.run()
        if evolver.global_best is None:
            raise RuntimeError("Stage I did not produce a global best candidate.")
        best_s1 = evolver.global_best
        stage1_pop = self._include_global_best(stage1_pop, best_s1)
        write_json(
            os.path.join(cfg.output_dir, "stage1_population.json"),
            {
                "ranking": [c.candidate_id for c in stage1_pop],
                "population": [candidate_to_dict(c) for c in stage1_pop],
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
            candidate_to_dict(best_s1),
        )
        console.status(
            f"Stage I complete - best={best_s1.candidate_id} score={best_s1.score}",
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
        best_s2 = self._best_by_last_metrics(stage2_pop, s2_runner.history)
        write_json(
            os.path.join(cfg.output_dir, "stage2_population.json"),
            {
                "population": [candidate_to_dict(c) for c in stage2_pop],
                "history": [
                    {
                        "round_index": r.round_index,
                        "candidate_id": r.candidate_id,
                        "metrics": r.metrics.as_dict(),
                        "refined": r.refined,
                        "kept_previous": r.kept_previous,
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
            f"Stage II complete - best={best_s2.candidate_id}",
            stage="pipeline",
        )

        # ----- Stage III -----
        s3_cfg = Stage3Config(
            population_size=len(stage2_pop),
            rounds=cfg.stage3_rounds,
            train_env_steps=cfg.stage3_train_steps,
            eval_episodes=cfg.stage3_eval_episodes,
            seed=cfg.seed,
            device=cfg.device,
            num_processes=cfg.num_processes,
            train_human_num=int(cfg.human_num),
            output_root=os.path.join(cfg.output_dir, "stage3_train"),
            human_counts=(5, 10, 15, 20)
            if cfg.stage3_run_h_sweep
            else (int(cfg.human_num),),
            randomization_regime=regime,
            predict_method=predict_method,
            env_name=env_name_for_predict_method(predict_method),
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
        stage3_pop = s3_runner.run(stage2_pop, run_h_sweep=cfg.stage3_run_h_sweep)
        best_s3 = self._best_by_last_metrics(stage3_pop, s3_runner.history)
        write_json(
            os.path.join(cfg.output_dir, "stage3_population.json"),
            {
                "population": [candidate_to_dict(c) for c in stage3_pop],
                "history": [
                    {
                        "round_index": r.round_index,
                        "candidate_id": r.candidate_id,
                        "metrics": r.metrics.as_dict(),
                        "refined": r.refined,
                        "kept_previous": r.kept_previous,
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
            f"Stage III complete - best={best_s3.candidate_id}",
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

        return EvoNavArtifacts(
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
    def _best_by_last_metrics(population, history) -> RewardCandidate:
        """Pick candidate with highest last-round SR - CR - 0.5*TR."""
        last_metrics: Dict[str, Any] = {}
        if history:
            max_round = max(r.round_index for r in history)
            for r in history:
                if r.round_index == max_round:
                    last_metrics[r.candidate_id] = r.metrics

        def _key(c: RewardCandidate):
            m = (c.metadata or {}).get("last_metrics")
            if m:
                return float(m.get("SR", 0) - m.get("CR", 0) - 0.5 * m.get("TR", 0))
            # Map refined ids back to parent history keys.
            for pid in list(c.parent_ids) + [c.candidate_id]:
                if pid in last_metrics:
                    pm = last_metrics[pid]
                    return float(pm.sr - pm.cr - 0.5 * pm.tr)
            return float(c.score or float("-inf"))

        return max(population, key=_key)
