"""Tests for Axis 3 primitives, memory, crossover prompts, ensemble critique."""

from __future__ import annotations

import math

from crowd_nav.reward_search.amfrs.crossover import (
    build_semantic_crossover_prompt,
    build_initial_prompt,
)
from crowd_nav.reward_search.amfrs.ensemble_critique import critique_agrees
from crowd_nav.reward_search.amfrs.memory import RewardMemory
from crowd_nav.reward_search.amfrs.primitives import PRIMITIVE_REGISTRY
from crowd_nav.reward_search.amfrs.static_gate import sample_synthetic_states
from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.llm import ScriptedLLMClient
from crowd_nav.reward_search.sandbox import RewardValidator
from crowd_nav.reward_search.sandbox.config import SandboxConfig


def test_primitives_finite_on_synthetic_states():
    states = sample_synthetic_states(n=20, seed=2)
    for name, fn in PRIMITIVE_REGISTRY.items():
        for st in states:
            v = fn(st, {})
            assert v == v and abs(v) != float("inf"), f"{name} non-finite: {v}"


def test_min_distance_margin_handles_inf_dmin():
    from crowd_nav.reward_search.amfrs.primitives import min_distance_margin
    from crowd_nav.reward_search.state import RewardState, RobotRewardState

    st = RewardState(
        robot=RobotRewardState(
            px=0.0, py=0.0, vx=0.0, vy=0.0, radius=0.3, gx=1.0, gy=0.0, v_pref=1.0
        ),
        humans=(),
        dmin=float("inf"),
        discomfort_dist=0.25,
        collision=True,
        reaching_goal=False,
        timeout=False,
        action=None,
        time_step=0.25,
        global_time=0.0,
        time_limit=25.0,
    )
    v = min_distance_margin(st, {})
    assert math.isfinite(v)


def test_primitives_injected_into_sandbox():
    code = (
        "def compute_reward(state, memory):\n"
        "    return float(goal_progress(state, memory) "
        "- collision_indicator(state, memory))\n"
    )
    cfg = SandboxConfig(extra_namespace=dict(PRIMITIVE_REGISTRY))
    fn = RewardValidator(config=cfg).validate_code(code)
    st = sample_synthetic_states(n=1, seed=0)[0]
    fn.reset()
    val = fn.compute(st)
    assert val == val


def test_baseline_sandbox_unchanged_without_extra_namespace():
    code = "def compute_reward(state, memory):\n    return float(1.0)\n"
    RewardValidator().validate_code(code)  # default config, no primitives needed


def test_semantic_crossover_prompt_contains_parents_and_primitives():
    a = RewardCandidate(candidate_id="a", code="def compute_reward(state, memory):\n return 1.0\n")
    b = RewardCandidate(candidate_id="b", code="def compute_reward(state, memory):\n return 2.0\n")
    prompt = build_semantic_crossover_prompt(a, b, "diagA", "diagB")
    assert "return 1.0" in prompt and "return 2.0" in prompt
    assert "diagA" in prompt and "diagB" in prompt
    assert "goal_progress" in prompt
    init = build_initial_prompt()
    for name in PRIMITIVE_REGISTRY:
        assert name in init


def test_memory_retrieves_near_duplicate(tmp_path):
    mem = RewardMemory(str(tmp_path / "m.sqlite"))
    code_a = "def compute_reward(state, memory):\n    return float(state.dmin)\n"
    code_b = "def compute_reward(state, memory):\n    return float(state.dmin + 0.0)\n"
    code_c = "def compute_reward(state, memory):\n    x = 1\n    y = 2\n    return float(x*y)\n"
    mem.add(code_a, "r1", {"SR": 0.9})
    mem.add(code_c, "r2", {"SR": 0.1})
    hits = mem.query_similar(code_b, k=2)
    assert hits
    assert hits[0].code == code_a
    mem.close()


def test_ensemble_critique_agree_and_disagree():
    cand = RewardCandidate(
        candidate_id="c",
        code="def compute_reward(state, memory):\n return 0.0\n",
    )
    agree_a = ScriptedLLMClient(["aggressive near humans discomfort collision"])
    agree_b = ScriptedLLMClient(["collision discomfort humans aggressive"])
    report = critique_agrees(cand, "sr=0.1", agree_a, agree_b, min_shared=2)
    assert report.agreed is True
    disagree_a = ScriptedLLMClient(["too slow timeout"])
    disagree_b = ScriptedLLMClient(["collision only"])
    report2 = critique_agrees(cand, "sr=0.1", disagree_a, disagree_b, min_shared=2)
    assert report2.agreed is False
