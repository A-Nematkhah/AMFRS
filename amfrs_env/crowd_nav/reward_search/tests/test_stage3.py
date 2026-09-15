"""
Fast Stage III tests using StubPolicyTrainer (no real RL).

Run: pytest crowd_nav/reward_search/tests/test_stage3.py -q
"""

from __future__ import annotations

import logging

from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.llm import ScriptedLLMClient
from crowd_nav.reward_search.sandbox import RewardValidator
from crowd_nav.reward_search.stage3 import (
    STAGE3_HUMAN_COUNTS,
    STAGE3_PAPER_STEPS,
    STAGE3_STEPS,
    Stage3Config,
    Stage3Runner,
    StubPolicyTrainer,
    _v3_candidate_id,
)


def _valid_code(value: float) -> str:
    return (
        "```python\n"
        "def compute_reward(state, memory):\n"
        f"    return float({value})\n"
        "```\n"
    )


def _invalid_import_code() -> str:
    return (
        "```python\n"
        "import os\n"
        "def compute_reward(state, memory):\n"
        "    return float(0.0)\n"
        "```\n"
    )


def _make_population(n: int = 2) -> list:
    validator = RewardValidator()
    pop = []
    for i in range(n):
        code = f"def compute_reward(state, memory):\n    return float({i}.0)\n"
        reward_fn, err = validator.try_validate(code)
        assert reward_fn is not None, err
        pop.append(
            RewardCandidate(
                candidate_id=f"c{i}",
                code=code,
                reward_fn=reward_fn,
                valid=True,
                origin="initial",
            )
        )
    return pop


def test_table6_defaults_and_k3_constant():
    cfg = Stage3Config()
    assert cfg.population_size == 8
    assert cfg.rounds == 3
    assert cfg.eval_episodes == 500
    assert cfg.algo == "ppo"
    assert cfg.train_env_steps == STAGE3_STEPS
    assert STAGE3_STEPS < STAGE3_PAPER_STEPS
    assert STAGE3_PAPER_STEPS == int(1e7)
    assert cfg.human_counts == STAGE3_HUMAN_COUNTS == (5, 10, 15, 20)
    assert cfg.num_processes is None  # resolved at train time


def test_stage3_argv_k3_independent_of_num_processes():
    """Total env-step budget K3 is unchanged; only wall-clock should change."""
    from crowd_nav.reward_search.stage3 import _stage3_train_argv

    cfg_a = Stage3Config(num_processes=1, train_env_steps=100_000)
    cfg_b = Stage3Config(num_processes=4, train_env_steps=100_000)
    argv_a = _stage3_train_argv(cfg_a, "c0", 0)
    argv_b = _stage3_train_argv(cfg_b, "c0", 0)
    assert argv_a[argv_a.index("--num-env-steps") + 1] == "100000"
    assert argv_b[argv_b.index("--num-env-steps") + 1] == "100000"
    assert int(argv_a[argv_a.index("--num-processes") + 1]) == 1
    assert int(argv_b[argv_b.index("--num-processes") + 1]) == 4
    # Same K3 string; update count floors but sample budget stays K3.
    steps = 100_000
    n_steps = cfg_a.num_steps
    updates_a = steps // n_steps // 1
    updates_b = steps // n_steps // 4
    assert updates_a > updates_b
    collected_a = updates_a * n_steps * 1
    collected_b = updates_b * n_steps * 4
    # Integer division can drop < one full update of env steps.
    assert abs(collected_a - collected_b) < n_steps * 4
    assert collected_a <= steps and collected_b <= steps



def test_stage3_run_refines_to_v3_and_h_sweep():
    n = 2
    pop = _make_population(n)
    # 1 round × 2 candidates
    client = ScriptedLLMClient([_valid_code(10.0), _valid_code(11.0)])
    runner = Stage3Runner(
        client,
        StubPolicyTrainer(),
        config=Stage3Config(
            population_size=n,
            rounds=1,
            train_env_steps=100,
            eval_episodes=2,
            human_counts=(5, 10, 20),
            protect_elite_refine=False,
            accept_reject_refine=True,
            h_sweep_max_finalists=2,
        ),
    )
    out = runner.run(pop, run_h_sweep=True)
    assert len(out) == n
    assert runner.best_trained is not None
    assert len(runner.history) == n
    assert any(r.refined and not r.kept_previous for r in runner.history)
    # H-sweep runs on up to h_sweep_max_finalists unique genomes.
    assert 1 <= len(runner.sweep_reports) <= 2
    assert runner.best_h_aware is not None
    assert (runner.best_h_aware.metadata or {}).get("h_aware_selected") is True
    for report in runner.sweep_reports:
        assert set(report.by_human_count) == {5, 10, 20}
        table = report.summary_table()
        assert "SR" in table and "CR" in table and "TR" in table
        for h, m in report.by_human_count.items():
            assert 0.0 <= m.sr <= 1.0
            assert 0.0 <= m.cr <= 1.0
            assert 0.0 <= m.tr <= 1.0


def test_failed_refinement_keeps_previous(caplog):
    pop = _make_population(1)
    client = ScriptedLLMClient([_invalid_import_code()])
    runner = Stage3Runner(
        client,
        StubPolicyTrainer(),
        config=Stage3Config(
            population_size=1,
            rounds=1,
            human_counts=(5,),
            protect_elite_refine=False,
        ),
    )
    with caplog.at_level(logging.WARNING):
        out = runner.run(pop, run_h_sweep=True)
    assert out[0].candidate_id == "c0"
    assert out[0].metadata.get("refine_kept_previous") is True
    assert runner.history[0].kept_previous
    assert len(runner.sweep_reports) == 1


def test_stage3_skip_refine_on_last_round_when_g3_gt_1():
    pop = _make_population(2)
    # Round 0: two refines; round 1 (last): no LLM calls.
    client = ScriptedLLMClient([_valid_code(12.0), _valid_code(13.0)])
    runner = Stage3Runner(
        client,
        StubPolicyTrainer(),
        config=Stage3Config(
            population_size=2,
            rounds=2,
            train_env_steps=50,
            human_counts=(5,),
            protect_elite_refine=False,
            skip_refine_last_round=True,
            accept_reject_refine=True,
        ),
    )
    out = runner.run(pop, run_h_sweep=False)
    last_round = [r for r in runner.history if r.round_index == 1]
    assert len(last_round) == 2
    assert all(r.kept_previous for r in last_round)
    assert client.remaining == 0  # both scripted replies consumed in round 0 only
    assert any(
        (c.metadata or {}).get("refine_skipped_last_round") for c in out
    )


def test_stage3_accept_reject_rejects_worse_refine():
    validator = RewardValidator()
    strong_code = "def compute_reward(state, memory):\n    return float(18.0)\n"
    weak_code = "def compute_reward(state, memory):\n    return float(1.0)\n"
    fn, err = validator.try_validate(strong_code)
    assert fn is not None, err
    pop = [
        RewardCandidate(
            candidate_id="strong",
            code=strong_code,
            reward_fn=fn,
            valid=True,
            origin="initial",
        ),
        RewardCandidate(
            candidate_id="other",
            code=weak_code,
            reward_fn=validator.try_validate(weak_code)[0],
            valid=True,
            origin="initial",
        ),
    ]
    # First candidate gets a worse refine; second gets a mild refine.
    client = ScriptedLLMClient([_valid_code(1.0), _valid_code(2.0)])
    runner = Stage3Runner(
        client,
        StubPolicyTrainer(),
        config=Stage3Config(
            population_size=2,
            rounds=1,
            human_counts=(5,),
            protect_elite_refine=False,
            accept_reject_refine=True,
        ),
    )
    runner.run(pop, run_h_sweep=False)
    assert any(f.get("reason") == "metric_reject" for f in runner.validation_failures)


def test_stage3_accept_reject_keeps_better_refine():
    pop = _make_population(2)  # c0→0.0, c1→1.0
    client = ScriptedLLMClient([_valid_code(18.0), _valid_code(19.0)])
    runner = Stage3Runner(
        client,
        StubPolicyTrainer(),
        config=Stage3Config(
            population_size=2,
            rounds=1,
            human_counts=(5,),
            protect_elite_refine=False,
            accept_reject_refine=True,
        ),
    )
    out = runner.run(pop, run_h_sweep=False)
    assert any(c.candidate_id.endswith("_v3") for c in out)
    assert any((c.metadata or {}).get("refine_accepted") for c in out)


def test_candidate_to_dict_genome_clarity():
    from crowd_nav.reward_search.reporting import candidate_to_dict

    cand = RewardCandidate(
        candidate_id="parent_v3",
        code="def compute_reward(state, memory):\n    return float(9.0)\n",
        origin="refinement",
        metadata={
            "trained_snapshot": True,
            "evaluated_genome_id": "parent",
            "evaluated_genome_code": (
                "def compute_reward(state, memory):\n    return float(1.0)\n"
            ),
            "last_metrics": {"SR": 0.5, "CR": 0.1, "TR": 0.1},
            "refine_accepted": True,
            "refine_verify_scalar": 0.4,
            "parent_verify_scalar": 0.3,
        },
    )
    d = candidate_to_dict(cand)
    assert d["evaluated_genome"]["candidate_id"] == "parent"
    assert "float(1.0)" in d["evaluated_genome"]["code"]
    assert d["proposed_refined_genome"]["accepted"] is True
    assert d["proposed_refined_genome"]["candidate_id"] == "parent_v3"


def test_v3_id_helper():
    assert _v3_candidate_id("c3") == "c3_v3"
    assert _v3_candidate_id("c3_v2") == "c3_v3"


def test_stub_h_sweep_deterministic():
    pop = _make_population(1)
    trainer = StubPolicyTrainer()
    cfg = Stage3Config(population_size=1, human_counts=(5, 15, 20))
    bundle = trainer.train_and_eval(pop[0], round_index=0, config=cfg)
    a = trainer.evaluate_at_human_counts(pop[0], bundle, config=cfg)
    b = trainer.evaluate_at_human_counts(pop[0], bundle, config=cfg)
    assert a.summary_table() == b.summary_table()
    assert a.by_human_count[20].sr <= a.by_human_count[5].sr
