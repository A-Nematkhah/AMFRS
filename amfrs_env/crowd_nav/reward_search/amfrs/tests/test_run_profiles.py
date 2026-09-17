"""Named GPU profiles: full AMFRS stack, scaled budgets."""

from __future__ import annotations

from crowd_nav.reward_search.amfrs.config import AMFRSRunConfig


def _assert_full_stack(cfg: AMFRSRunConfig) -> None:
    assert cfg.fast is False
    assert cfg.use_stub_trainers is False
    assert cfg.predict_method == "inferred"
    assert cfg.illumination_max_rung == "F2_full_a2c"
    assert cfg.final_rung == "F3_full_ppo"
    assert cfg.run_robustness_sweep is True
    assert cfg.use_bandit is True
    assert cfg.use_ensemble_critique is True
    assert cfg.use_retrieval_memory is True
    assert cfg.selection_mode == "map_elites"


def test_3h_profile_is_full_stack_h2():
    cfg = AMFRSRunConfig()
    cfg.apply_3h_gpu_profile()
    _assert_full_stack(cfg)
    assert cfg.human_num == 2
    assert cfg.population_size == 8
    assert cfg.generations == 3
    assert cfg.stage2_train_steps_short >= 20_000
    assert cfg.stage3_train_steps >= 120_000


def test_18h_profile_is_full_stack_h10():
    cfg = AMFRSRunConfig()
    cfg.apply_18h_gpu_profile()
    _assert_full_stack(cfg)
    assert cfg.human_num == 10
    assert cfg.generations >= cfg.population_size // 4


def test_full_profile_is_full_stack_h20():
    cfg = AMFRSRunConfig()
    cfg.apply_full_gpu_profile()
    _assert_full_stack(cfg)
    assert cfg.human_num == 20
    assert cfg.stage3_train_steps >= 500_000


def test_legacy_aliases_map_to_new_profiles():
    a, b = AMFRSRunConfig(), AMFRSRunConfig()
    a.apply_2h_gpu_profile()
    b.apply_3h_gpu_profile()
    assert a.human_num == b.human_num == 2
    assert a.final_rung == b.final_rung

    c, d = AMFRSRunConfig(), AMFRSRunConfig()
    c.apply_12h_gpu_profile()
    d.apply_18h_gpu_profile()
    assert c.human_num == d.human_num == 10
    assert c.final_rung == d.final_rung
