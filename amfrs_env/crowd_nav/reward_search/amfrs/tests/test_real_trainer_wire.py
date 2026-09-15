"""Ensure non-stub ladder wires RealPolicyTrainer (mocked, no GPU)."""

from __future__ import annotations

from crowd_nav.reward_search.amfrs.fidelity import TrainerContext, build_default_ladder
from crowd_nav.reward_search.evolver import RewardCandidate
from crowd_nav.reward_search.sandbox import RewardValidator
from crowd_nav.reward_search.stage2 import ProxyMetrics


def test_real_ladder_calls_stage2_real_trainer(monkeypatch):
    calls = {"n": 0}

    class FakeTrainer:
        def train_and_eval(self, candidate, *, round_index, config):
            calls["n"] += 1
            assert config.predict_method == "none"
            assert config.train_env_steps == 123
            return ProxyMetrics(sr=0.8, cr=0.1, tr=0.1, nt=10.0, pl=12.0, itr=5.0, sd=0.4)

    monkeypatch.setattr(
        "crowd_nav.reward_search.stage2.RealPolicyTrainer",
        FakeTrainer,
    )

    code = "def compute_reward(state, memory):\n    return float(1.0)\n"
    fn, err = RewardValidator().try_validate(code)
    assert fn is not None, err
    cand = RewardCandidate(
        candidate_id="t0", code=code, reward_fn=fn, valid=True, metadata={}
    )
    ctx = TrainerContext(
        stage2_train_steps_short=123,
        stage2_eval_episodes=2,
        predict_method="none",
        device="cpu",
    )
    ladder = build_default_ladder(
        use_stub=False, score1_mode="smoke", trainer_ctx=ctx
    )
    f1 = ladder.get("F1_short_a2c")
    result = f1.evaluate(cand)
    assert calls["n"] == 1
    assert result.raw_metrics.get("trainer") == "RealPolicyTrainer"
    assert "goal_dist0" in result.raw_metrics
    assert result.metric > float("-inf")


def test_real_ladder_survives_trainer_crash(monkeypatch):
    class BoomTrainer:
        def train_and_eval(self, candidate, *, round_index, config):
            raise RuntimeError("compute_reward returned a non-finite value: -inf")

    monkeypatch.setattr(
        "crowd_nav.reward_search.stage2.RealPolicyTrainer",
        BoomTrainer,
    )
    code = "def compute_reward(state, memory):\n    return float('-inf')\n"
    # Bypass validator smoke (would reject -inf); attach a dummy reward_fn.
    cand = RewardCandidate(
        candidate_id="bad",
        code=code,
        reward_fn=object(),  # unused — trainer raises first
        valid=True,
        metadata={},
    )
    ctx = TrainerContext(
        stage2_train_steps_short=8,
        stage2_eval_episodes=2,
        predict_method="none",
        device="cpu",
    )
    ladder = build_default_ladder(
        use_stub=False, score1_mode="smoke", trainer_ctx=ctx
    )
    result = ladder.get("F1_short_a2c").evaluate(cand)
    assert result.metric == float("-inf")
    assert result.raw_metrics.get("trainer") == "failed"
    assert "non-finite" in str(result.raw_metrics.get("error", ""))


def test_real_ladder_skips_when_prior_metric_non_finite(monkeypatch):
    calls = {"n": 0}

    class FakeTrainer:
        def train_and_eval(self, candidate, *, round_index, config):
            calls["n"] += 1
            return ProxyMetrics(sr=0.8, cr=0.1, tr=0.1, nt=10.0, pl=12.0, itr=5.0, sd=0.4)

    monkeypatch.setattr(
        "crowd_nav.reward_search.stage2.RealPolicyTrainer",
        FakeTrainer,
    )
    cand = RewardCandidate(
        candidate_id="dead_f0",
        code="def compute_reward(state, memory):\n    return 1.0\n",
        reward_fn=object(),
        valid=True,
        score=float("-inf"),
        metadata={"last_metric": float("-inf"), "last_fidelity": "F0_score1"},
    )
    ctx = TrainerContext(
        stage2_train_steps_short=8,
        stage2_eval_episodes=2,
        predict_method="none",
        device="cpu",
    )
    ladder = build_default_ladder(
        use_stub=False, score1_mode="smoke", trainer_ctx=ctx
    )
    result = ladder.get("F1_short_a2c").evaluate(cand)
    assert calls["n"] == 0
    assert result.metric == float("-inf")
    assert result.raw_metrics.get("trainer") == "skipped"
    assert result.raw_metrics.get("skipped") == "prior_rung_non_finite"
    assert result.cost == 0.0
