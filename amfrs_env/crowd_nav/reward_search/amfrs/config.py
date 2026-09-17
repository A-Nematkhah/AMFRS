"""
AMFRS run configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple


@dataclass
class AMFRSRunConfig:
    """End-to-end AMFRS settings (axes 1–5)."""

    output_dir: str = "results/amfrs_run"
    seed: int = 425

    # Population / illumination
    population_size: int = 8
    generations: int = 4
    selection_mode: Literal["map_elites", "scalar"] = "map_elites"

    # Axis 1 — fidelity / budget
    # Cost units are relative (F0=1, F1=10, F2=40, F3=400); see fidelity.build_default_ladder.
    # Default must cover illumination through F1 (and leave headroom). Final F3
    # climb uses ``final_rung_max_cost_units`` (None = uncapped for that phase).
    max_cost_units_per_generation: float = 500.0
    illumination_max_rung: str = "F1_short_a2c"
    final_rung: str = "F3_full_ppo"
    # None → no cost cap on the post-illumination final rung climb.
    final_rung_max_cost_units: Optional[float] = None
    use_bandit: bool = False
    halving_eta: int = 3
    halving_min_survivors: int = 1

    # Fidelity stub / smoke (fast path)
    fast: bool = False
    use_stub_trainers: bool = False
    score1_mode: Literal["dataset", "smoke"] = "dataset"
    stage1_dataset_path: str = "data/stage1_dataset"

    # Stage II/III budgets reused by fidelity adapters (when not stubbing)
    stage2_train_steps: int = 50_000
    stage2_train_steps_short: int = 12_500
    stage2_eval_episodes: int = 50
    stage3_train_steps: int = 500_000
    stage3_eval_episodes: int = 500

    # Axis 2 — MAP-Elites grid (bins = len(edges) - 1 along each axis)
    archive_shape: Tuple[int, int] = (4, 4)

    # Axis 3 — LLM / memory
    llm_provider: str = "seed"  # seed | groq | vllm | ollama | scripted
    llm_provider_b: str = "seed"  # second critic for ensemble (often ollama)
    llm_model: Optional[str] = None
    allow_seed_llm: bool = False
    require_primitives: bool = False
    memory_db_path: str = "data/reward_memory.sqlite"
    use_retrieval_memory: bool = True
    use_ensemble_critique: bool = False
    # Retries when sandbox/static-gate rejects a proposal (fill to population_size).
    max_invalid_replacements: int = 24

    # Axis 4 — robustness (finalists only)
    run_robustness_sweep: bool = True
    robustness_policies: Tuple[str, ...] = ("orca", "social_force")

    # Axis 5 — static gate
    static_gate_n_states: int = 64
    static_gate_seed: int = 0

    # Env / device (used when real trainers run)
    device: str = "cuda"
    num_processes: Optional[int] = None
    randomization_regime: str = "without_random"
    predict_method: str = "inferred"
    human_num: int = 20

    # Free-form bookkeeping
    extra: Dict[str, Any] = field(default_factory=dict)

    def apply_fast_profile(self) -> None:
        """Seconds-scale dry run: stubs, smoke Score1, tiny population."""
        self.fast = True
        self.use_stub_trainers = True
        self.score1_mode = "smoke"
        self.population_size = 4
        self.generations = 2
        self.stage2_train_steps = 8
        self.stage2_train_steps_short = 4
        self.stage2_eval_episodes = 2
        self.stage3_train_steps = 8
        self.stage3_eval_episodes = 2
        self.max_cost_units_per_generation = 50.0
        self.illumination_max_rung = "F1_short_a2c"
        self.final_rung = "F2_full_a2c"
        self.predict_method = "none"
        self.device = "cpu"
        self.use_bandit = False
        self.use_ensemble_critique = False
        self.run_robustness_sweep = True  # stub path still exercises A4
        self.allow_seed_llm = True
        self.llm_provider = "seed"
        self.llm_provider_b = "seed"
        self.static_gate_n_states = 32

    def apply_short_gpu_profile(self) -> None:
        """
        Real trainers, aimed at ~15–30 min on one GPU (order-of-magnitude).

        Skips full PPO (stops at F2 A2C) and robustness sweep; fewer humans
        and smaller pop/gen. Timing varies with GPU and GST on/off.
        """
        self.fast = False
        self.use_stub_trainers = False
        self.population_size = 4
        self.generations = 1
        self.score1_mode = "dataset"
        self.stage2_train_steps = 8_000
        self.stage2_train_steps_short = 2_000
        self.stage2_eval_episodes = 20
        self.stage3_train_steps = 15_000
        self.stage3_eval_episodes = 30
        self.max_cost_units_per_generation = 200.0
        self.illumination_max_rung = "F1_short_a2c"
        self.final_rung = "F2_full_a2c"
        self.run_robustness_sweep = False
        self.predict_method = "none"
        self.human_num = 5
        self.num_processes = 4
        self.device = "cuda"
        self.allow_seed_llm = True
        self.llm_provider = "seed"
        self.llm_provider_b = "seed"

    def apply_6h_gpu_profile(self) -> None:
        """
        Real trainers aimed at ~4–8 h on one GPU (order-of-magnitude).

        Includes GST (inferred), illumination through F2, final F3 PPO, and
        multi-policy robustness. Tuned from short-run timings (~5 min for
        pop=4/gen=1/F2/no-GST); wall-clock varies with GPU and LLM latency.
        """
        self.fast = False
        self.use_stub_trainers = False
        self.population_size = 8
        self.generations = 3
        self.score1_mode = "dataset"
        # F1 / F2 / F3 budgets (env steps)
        self.stage2_train_steps_short = 12_500
        self.stage2_train_steps = 40_000
        self.stage2_eval_episodes = 40
        self.stage3_train_steps = 120_000
        self.stage3_eval_episodes = 80
        self.max_cost_units_per_generation = 400.0
        self.illumination_max_rung = "F2_full_a2c"
        self.final_rung = "F3_full_ppo"
        self.final_rung_max_cost_units = None
        self.run_robustness_sweep = True
        self.predict_method = "inferred"
        self.human_num = 10
        self.num_processes = 4
        self.device = "cuda"
        self.use_bandit = False
        self.use_ensemble_critique = False
        self.allow_seed_llm = True
        self.llm_provider = "seed"
        self.llm_provider_b = "seed"
        self.static_gate_n_states = 64

    def apply_12h_gpu_profile(self) -> None:
        """
        Real trainers aimed at ~10–14 h on one GPU (order-of-magnitude).

        Same shape as 6h (GST + F2 illumination + F3 PPO + robustness) with
        more population/generations and higher train/eval budgets. Prefer
        ``--llm groq`` (or ollama). Wall-clock varies with GPU and LLM latency.
        """
        self.fast = False
        self.use_stub_trainers = False
        self.population_size = 12
        self.generations = 5
        self.score1_mode = "dataset"
        # F1 / F2 / F3 budgets (env steps)
        self.stage2_train_steps_short = 20_000
        self.stage2_train_steps = 60_000
        self.stage2_eval_episodes = 50
        self.stage3_train_steps = 250_000
        self.stage3_eval_episodes = 100
        self.max_cost_units_per_generation = 600.0
        self.illumination_max_rung = "F2_full_a2c"
        self.final_rung = "F3_full_ppo"
        self.final_rung_max_cost_units = None
        self.run_robustness_sweep = True
        self.predict_method = "inferred"
        self.human_num = 15
        self.num_processes = 4
        self.device = "cuda"
        self.use_bandit = False
        self.use_ensemble_critique = False
        self.allow_seed_llm = True
        self.llm_provider = "seed"
        self.llm_provider_b = "seed"
        self.static_gate_n_states = 64

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
