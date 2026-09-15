"""Initial / child population always fills to N valid candidates."""

from __future__ import annotations

from crowd_nav.reward_search.amfrs import AMFRSPipeline, AMFRSRunConfig
from crowd_nav.reward_search.llm import ScriptedLLMClient
from crowd_nav.reward_search.prompts import D5_SEED_FUNCTION


def _valid_fence(i: int) -> str:
    return (
        "```python\n"
        "def compute_reward(state, memory):\n"
        f"    return float({1.0 + 0.1 * i} * goal_progress(state, memory) "
        f"- 20.0 * collision_indicator(state, memory))\n"
        "```\n"
    )


def _invalid_fence() -> str:
    return (
        "```python\n"
        "def compute_reward(state, memory):\n"
        "    import os\n"
        "    return float(1.0)\n"
        "```\n"
    )


def test_initial_population_refills_to_n(tmp_path, monkeypatch):
    cfg = AMFRSRunConfig(output_dir=str(tmp_path / "fill"))
    cfg.apply_fast_profile()
    cfg.output_dir = str(tmp_path / "fill")
    cfg.population_size = 4
    cfg.fast = False  # take LLM propose path (not seed templates)
    cfg.use_stub_trainers = True
    cfg.llm_provider = "scripted"
    cfg.max_invalid_replacements = 8

    # seed_0 uses D5 (always valid, no LLM). Then for slots 1..3:
    # two invalids then three valids.
    scripted = ScriptedLLMClient(
        [
            _invalid_fence(),
            _invalid_fence(),
            _valid_fence(2),
            _valid_fence(3),
            _valid_fence(4),
        ]
    )
    pipe = AMFRSPipeline(cfg)
    monkeypatch.setattr(pipe, "llm_a", scripted)
    monkeypatch.setattr(pipe, "llm_b", scripted)

    rejected: list = []
    accepted = pipe._generate_initial_population(rejected)
    assert len(accepted) == 4
    assert all(c.valid and c.reward_fn is not None for c in accepted)
    assert len(rejected) >= 2
