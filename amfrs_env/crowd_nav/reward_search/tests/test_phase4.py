"""Phase 4 unit tests: refine harness, II↔III calibration, Score1 diagnostics."""

from __future__ import annotations

from dataclasses import replace

import json

from crowd_nav.reward_search.evolver import RewardCandidate, StageIConfig, StageIEvolver
from crowd_nav.reward_search.llm import ScriptedLLMClient
from crowd_nav.reward_search.pipeline import AMFRSPipeline, AMFRSRunConfig
from crowd_nav.reward_search.refine_harness import (
    AcceptRejectDecision,
    build_accept_metadata,
    build_reject_metadata,
    decide_accept_reject,
    resolve_verify_train_steps,
)
from crowd_nav.reward_search.reporting import (
    rank_table,
    spearman_rank_correlation,
    stage2_stage3_calibration,
)
from crowd_nav.reward_search.sandbox import RewardValidator
from crowd_nav.reward_search.scoring import (
    Score1Report,
    format_score1_diagnostics,
)


def test_resolve_verify_train_steps_default_quarter():
    assert resolve_verify_train_steps(8000, None) == 2000
    assert resolve_verify_train_steps(10, None) == 2
    assert resolve_verify_train_steps(100, 7) == 7


def test_decide_accept_reject_tolerance():
    ok = decide_accept_reject(
        parent_verify_scalar=0.5,
        proposed_verify_scalar=0.49,
        parent_full_scalar=0.8,
        tolerance=0.02,
    )
    assert ok.accepted is True
    bad = decide_accept_reject(
        parent_verify_scalar=0.5,
        proposed_verify_scalar=0.40,
        parent_full_scalar=0.8,
        tolerance=0.0,
    )
    assert bad.accepted is False
    assert bad.kept_previous is True


def test_accept_reject_metadata_shapes():
    decision = AcceptRejectDecision(
        accepted=True,
        verify_scalar=0.6,
        parent_verify_scalar=0.5,
        parent_full_scalar=0.7,
        tolerance=0.0,
    )
    acc = build_accept_metadata(
        {},
        parent_id="p",
        parent_code="def compute_reward(state, memory):\n    return 1.0\n",
        proposed_code="def compute_reward(state, memory):\n    return 2.0\n",
        decision=decision,
        proposed_verify_metrics={"SR": 0.6},
        parent_verify_metrics={"SR": 0.5},
    )
    assert acc["refine_accepted"] is True
    assert acc["accept_reject_fair_budget"] is True
    rej_decision = AcceptRejectDecision(
        accepted=False,
        verify_scalar=0.1,
        parent_verify_scalar=0.5,
        parent_full_scalar=0.7,
        tolerance=0.0,
    )
    rej = build_reject_metadata(
        {},
        parent_id="p",
        parent_code="parent",
        parent_metrics={"SR": 0.0},
        proposed_id="p_v2",
        proposed_code="proposed",
        decision=rej_decision,
        proposed_verify_metrics={"SR": 0.1},
        parent_verify_metrics={"SR": 0.5},
    )
    assert rej["refine_rejected_metric"] is True
    assert rej["refine_kept_previous"] is True


def test_rank_table_and_spearman():
    table = rank_table({"a": 0.9, "b": 0.1, "c": 0.5})
    assert table[0]["candidate_id"] == "a" and table[0]["rank"] == 1
    ranks_a = {"a": 1, "b": 2, "c": 3}
    ranks_b = {"a": 1, "b": 2, "c": 3}
    corr = spearman_rank_correlation(ranks_a, ranks_b)
    assert corr["n"] == 3
    assert abs(corr["spearman"] - 1.0) < 1e-9
    inverted = spearman_rank_correlation(ranks_a, {"a": 3, "b": 2, "c": 1})
    assert abs(inverted["spearman"] - (-1.0)) < 1e-9


def test_stage2_stage3_calibration_payload():
    payload = stage2_stage3_calibration(
        {"x": 0.5, "y": -0.2},
        {"x": -0.1, "y": 0.4},
        stage2_best_id="x",
        stage3_best_id="y",
    )
    assert payload["winner_match"] is False
    assert payload["correlation"]["n"] == 2
    assert len(payload["stage2_ranking"]) == 2


def test_format_score1_diagnostics_weak_scenarios():
    report = Score1Report(
        score=0.4,
        n_scenarios_scored=2,
        n_scenarios_total=2,
        per_scenario={"scenario_bad": -0.2, "scenario_good": 0.9},
    )
    text = format_score1_diagnostics(report)
    assert "scenario_bad" in text
    assert "weak scenarios" in text
    assert format_score1_diagnostics(None) == ""


def test_mutation_prompt_includes_score1_diagnostics():
    parent_code = (
        "def compute_reward(state, memory):\n"
        "    return float(0.0)\n"
    )
    child_code = (
        "```python\n"
        "def compute_reward(state, memory):\n"
        "    return float(1.0)\n"
        "```\n"
    )
    client = ScriptedLLMClient([child_code])

    def _score_with_report(reward_fn, *, candidate_id: str = ""):
        del reward_fn, candidate_id
        return Score1Report(
            score=0.2,
            n_scenarios_scored=1,
            n_scenarios_total=1,
            per_scenario={"scenario_hard": -0.5},
        )

    parent = RewardCandidate(
        candidate_id="mut_parent",
        code=parent_code,
        reward_fn=None,
        valid=True,
        score=0.2,
        metadata={
            "score1_report": Score1Report(
                score=0.2,
                n_scenarios_scored=1,
                n_scenarios_total=1,
                per_scenario={"scenario_hard": -0.5},
            ).to_dict()
        },
    )
    validator = RewardValidator()
    fn, err = validator.try_validate(parent_code)
    assert fn is not None, err
    parent = replace(parent, reward_fn=fn, valid=True)

    evolver = StageIEvolver(
        client,
        score_fn=_score_with_report,
        validator=validator,
        config=StageIConfig(
            population_size=2,
            generations=1,
            n_crossover=0,
            n_mutation=1,
            n_random=1,
        ),
    )
    child = evolver._try_mutate(parent, "global note", phase="mutation", attempt=1)
    assert child.valid
    assert "scenario_hard" in (child.metadata or {}).get("weakness", "")


def test_pipeline_fast_writes_calibration(tmp_path):
    out = tmp_path / "run"
    cfg = AMFRSRunConfig(output_dir=str(out))
    cfg.apply_fast_profile()
    AMFRSPipeline(cfg).run()
    calib = out / "stage2_stage3_calibration.json"
    assert calib.is_file()
    data = json.loads(calib.read_text(encoding="utf-8"))
    assert "correlation" in data
    assert "stage2_ranking" in data
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert "calibration" in manifest
