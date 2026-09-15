"""Phase 1 Score1 honesty: pad mask, anti-exploit, holdout, mode labeling."""

from __future__ import annotations

import math
from typing import List, Tuple

import pytest

from crowd_nav.reward_search.dataset import (
    TrajectoryRecord,
    split_stage1_dataset,
)
from crowd_nav.reward_search.scoring import (
    Score1Options,
    _cumulative_reward,
    score1_for_dataset,
    score1_mode_artifact,
    score1_report,
)
from crowd_nav.reward_search.state import (
    HumanObservable,
    RewardFunction,
    RewardState,
    RobotRewardState,
)


def _robot(*, px: float, py: float) -> RobotRewardState:
    return RobotRewardState(
        px=px, py=py, vx=0.0, vy=0.0, radius=0.3, gx=0.0, gy=0.0, v_pref=1.0
    )


def _state(*, px: float, py: float, global_time: float, **flags) -> RewardState:
    return RewardState(
        robot=_robot(px=px, py=py),
        humans=(HumanObservable(10.0, 10.0, 0.0, 0.0, 0.3),),
        dmin=5.0,
        discomfort_dist=0.25,
        collision=bool(flags.get("collision", False)),
        reaching_goal=bool(flags.get("reaching_goal", False)),
        timeout=bool(flags.get("timeout", False)),
        action=None,
        time_step=0.25,
        global_time=global_time,
        time_limit=50.0,
    )


def _traj(
    scenario_id: str,
    tid: str,
    label: str,
    frames: List[Tuple[float, float, float]],
) -> TrajectoryRecord:
    states = tuple(
        _state(
            px=px,
            py=py,
            global_time=gt,
            reaching_goal=(label == "success" and i == len(frames) - 1),
            collision=(label == "collision" and i == len(frames) - 1),
            timeout=(label == "timeout" and i == len(frames) - 1),
        )
        for i, (px, py, gt) in enumerate(frames)
    )
    return TrajectoryRecord(
        trajectory_id=tid,
        scenario_id=scenario_id,
        seed=0,
        states=states,
        label=label,
        behavior="synthetic",
    )


class _ConstantReward(RewardFunction):
    def reset(self) -> None:
        return None

    def compute(self, state: RewardState) -> float:
        return 1.0


class _ExplosiveReward(RewardFunction):
    def reset(self) -> None:
        return None

    def compute(self, state: RewardState) -> float:
        return 1.0e9


class _CountingReward(RewardFunction):
    """Returns +1 every compute call (detects pad inflation)."""

    def __init__(self) -> None:
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def compute(self, state: RewardState) -> float:
        self.calls += 1
        return 1.0


class _AlignedReward(RewardFunction):
    def reset(self) -> None:
        return None

    def compute(self, state: RewardState) -> float:
        if state.collision:
            return -20.0
        if state.reaching_goal:
            return 10.0
        dist = (
            (state.robot.px - state.robot.gx) ** 2
            + (state.robot.py - state.robot.gy) ** 2
        ) ** 0.5
        return float(-dist)


def _mini_dataset():
    sid = "scenario_000"
    return {
        sid: [
            _traj(sid, "ok", "success", [(4.0, 0.0, 0.25), (2.0, 0.0, 0.5), (0.2, 0.0, 0.75)]),
            _traj(
                sid,
                "ok_l",
                "success",
                [(5.0, 0.0, 0.25), (3.0, 0.0, 1.0), (1.0, 0.0, 2.0), (0.2, 0.0, 3.0)],
            ),
            _traj(sid, "to", "timeout", [(2.0, 0.0, 0.25), (1.5, 0.0, 50.0)]),
            _traj(sid, "col", "collision", [(3.0, 0.0, 0.25), (2.5, 0.0, 0.5)]),
        ]
    }


def test_pad_mask_freezes_cumulative_and_skips_compute():
    sid = "s"
    short = _traj(sid, "short", "success", [(1.0, 0.0, 0.25), (0.2, 0.0, 0.5)])
    # max_frame forces padding beyond length=2
    reward = _CountingReward()
    masked = _cumulative_reward(reward, short, max_frame=4, mask_pads=True)
    assert reward.calls == short.length
    assert masked == [1.0, 2.0, 2.0, 2.0, 2.0]

    reward2 = _CountingReward()
    unmasked = _cumulative_reward(reward2, short, max_frame=4, mask_pads=False)
    assert reward2.calls == 5
    assert unmasked == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_score1_invariant_to_pad_length_under_mask():
    """With mask_pads, only shared real frames matter (timeout uses dist_goal)."""
    sid = "scenario_pad"
    frames_a = [(2.0, 0.0, 0.25), (0.2, 0.0, 0.5)]
    frames_b = [
        (3.0, 0.0, 0.25),
        (2.0, 0.0, 0.5),
        (1.0, 0.0, 0.75),
        (0.5, 0.0, 1.0),
        (0.2, 0.0, 1.25),
    ]
    frames_c = [(4.0, 0.0, 0.25), (3.5, 0.0, 0.5)]
    full = {
        sid: [
            _traj(sid, "a", "timeout", frames_a),
            _traj(sid, "b", "timeout", frames_b),
            _traj(sid, "c", "collision", frames_c),
        ]
    }
    truncated = {
        sid: [
            _traj(sid, "a", "timeout", frames_a),
            _traj(sid, "b", "timeout", frames_b[:2]),
            _traj(sid, "c", "collision", frames_c),
        ]
    }
    opts = Score1Options(mask_pads=True, anti_exploit=False)
    score_full = score1_for_dataset(full, _AlignedReward(), options=opts)
    score_trunc = score1_for_dataset(truncated, _AlignedReward(), options=opts)
    assert score_full == pytest.approx(score_trunc)


def test_pad_frames_excluded_from_spearman_length():
    from crowd_nav.reward_search.scoring import _scenario_frame_correlations

    sid = "s"
    # lengths 2, 5, 2 → only f=0,1 have ≥2 real trajs under mask_pads
    trajs = [
        _traj(sid, "a", "success", [(2.0, 0.0, 0.25), (0.2, 0.0, 0.5)]),
        _traj(
            sid,
            "b",
            "success",
            [(3.0, 0.0, 0.25), (2.0, 0.0, 0.5), (1.0, 0.0, 0.75), (0.5, 0.0, 1.0), (0.2, 0.0, 1.25)],
        ),
        _traj(sid, "c", "collision", [(4.0, 0.0, 0.25), (3.5, 0.0, 0.5)]),
    ]
    corrs = _scenario_frame_correlations(trajs, _AlignedReward(), mask_pads=True)
    assert len(corrs) == 2


def test_anti_exploit_rejects_constant_reward():
    ds = _mini_dataset()
    report = score1_report(ds, _ConstantReward(), options=Score1Options(anti_exploit=True))
    assert report.anti_exploit_triggered
    assert report.anti_exploit_reason == "near_constant_reward"
    assert report.score == float("-inf")


def test_anti_exploit_rejects_explosive_reward():
    ds = _mini_dataset()
    report = score1_report(ds, _ExplosiveReward(), options=Score1Options(anti_exploit=True))
    assert report.anti_exploit_triggered
    assert report.anti_exploit_reason == "explosive_magnitude"
    assert report.score == float("-inf")


def test_anti_exploit_off_allows_scoring_when_ranks_defined():
    """With anti-exploit off, non-constant shaped rewards still score."""
    ds = _mini_dataset()
    score = score1_for_dataset(
        ds, _AlignedReward(), options=Score1Options(anti_exploit=False)
    )
    assert math.isfinite(score)
    assert -1.0 <= score <= 1.0


def test_equal_length_constant_without_anti_exploit_unscoreable():
    """Identical constant streams → Spearman undefined → ValueError."""
    sid = "eq"
    frames = [(3.0, 0.0, 0.25), (2.0, 0.0, 0.5), (1.0, 0.0, 0.75)]
    ds = {
        sid: [
            _traj(sid, "a", "success", frames),
            _traj(sid, "b", "timeout", frames),
            _traj(sid, "c", "collision", frames),
        ]
    }
    with pytest.raises(ValueError, match="no scoreable"):
        score1_for_dataset(
            ds, _ConstantReward(), options=Score1Options(anti_exploit=False)
        )


def test_split_stage1_dataset_deterministic():
    ds = {f"scenario_{i:03d}": _mini_dataset()["scenario_000"] for i in range(10)}
    train_a, hold_a, meta_a = split_stage1_dataset(ds, holdout_fraction=0.2, seed=7)
    train_b, hold_b, meta_b = split_stage1_dataset(ds, holdout_fraction=0.2, seed=7)
    assert meta_a == meta_b
    assert set(train_a) == set(train_b)
    assert set(hold_a) == set(hold_b)
    assert meta_a["n_holdout_scenarios"] == 2
    assert meta_a["n_train_scenarios"] == 8
    assert set(train_a).isdisjoint(set(hold_a))
    assert set(train_a) | set(hold_a) == set(ds)


def test_split_single_scenario_holdout_empty():
    ds = _mini_dataset()
    train, hold, meta = split_stage1_dataset(ds, holdout_fraction=0.2, seed=0)
    assert len(train) == 1
    assert hold == {}
    assert meta["n_holdout_scenarios"] == 0


def test_score1_mode_artifact_smoke_not_paper():
    art = score1_mode_artifact("smoke", fast=False)
    assert art["score1_mode"] == "smoke"
    assert art["is_paper_score1"] is False
    assert art["score1_claim_ok"] is False
    assert art["warning"]

    art2 = score1_mode_artifact("dataset", fast=True)
    assert art2["is_paper_score1"] is False
    assert art2["score1_claim_ok"] is False

    art3 = score1_mode_artifact("dataset", fast=False)
    assert art3["is_paper_score1"] is True
    assert art3["score1_claim_ok"] is True
    assert art3["warning"] is None


def test_pipeline_fast_manifest_labels_smoke(tmp_path):
    from crowd_nav.reward_search.amfrs import AMFRSPipeline, AMFRSRunConfig

    cfg = AMFRSRunConfig(output_dir=str(tmp_path / "run"))
    cfg.apply_fast_profile()
    cfg.output_dir = str(tmp_path / "run")
    arts = AMFRSPipeline(cfg).run()
    assert arts.manifest["pipeline"] == "AMFRS"
    assert arts.manifest["config"]["score1_mode"] == "smoke"
    assert arts.manifest["config"]["fast"] is True
    assert arts.best is not None
