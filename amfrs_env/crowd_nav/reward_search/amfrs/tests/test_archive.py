"""Tests for MAP-Elites archive and behavior descriptors."""

from __future__ import annotations

from crowd_nav.reward_search.amfrs.archive import EliteGrid
from crowd_nav.reward_search.amfrs.behavior import (
    DescriptorSpaceConfig,
    compute_behavior_descriptor,
)
from crowd_nav.reward_search.evolver import RewardCandidate


def test_behavior_bins_stable_and_clip():
    cfg = DescriptorSpaceConfig()
    bx, by = compute_behavior_descriptor({"PL": 12.0, "goal_dist0": 8.0, "SD": 0.3}, config=cfg)
    assert isinstance(bx, int) and isinstance(by, int)
    # Extreme values clip, do not raise
    bx2, by2 = compute_behavior_descriptor(
        {"PL": 1e6, "goal_dist0": 1.0, "SD": -10.0}, config=cfg
    )
    assert bx2 == len(cfg.path_efficiency_edges) - 2
    assert by2 == 0


def test_elite_grid_insert_replace():
    grid = EliteGrid((3, 3))
    a = RewardCandidate(candidate_id="a", code="a", valid=True, metadata={})
    b = RewardCandidate(candidate_id="b", code="b", valid=True, metadata={})
    assert grid.try_insert(a, (0, 0), 0.5) is True
    assert grid.try_insert(b, (0, 0), 0.4) is False
    assert grid.get((0, 0)).candidate_id == "a"
    assert grid.try_insert(b, (0, 0), 0.9) is True
    assert grid.get((0, 0)).candidate_id == "b"
    assert grid.coverage() > 0
    assert grid.qd_score() == 0.9


def test_archive_json_roundtrip(tmp_path):
    grid = EliteGrid((2, 2))
    c = RewardCandidate(candidate_id="x", code="def f():\n pass", valid=True, metadata={"k": 1})
    grid.try_insert(c, (1, 0), 1.2)
    path = tmp_path / "arch.json"
    grid.to_json(str(path))
    loaded = EliteGrid.from_json(str(path))
    assert loaded.get((1, 0)).candidate_id == "x"


def test_pipeline_coverage_nondecreasing(tmp_path):
    from crowd_nav.reward_search.amfrs import AMFRS2Pipeline, AMFRS2RunConfig

    cfg = AMFRS2RunConfig(output_dir=str(tmp_path / "a"))
    cfg.apply_fast_profile()
    cfg.output_dir = str(tmp_path / "a")
    cfg.generations = 3
    cfg.population_size = 4
    art = AMFRS2Pipeline(cfg).run()
    cov = art.manifest.get("coverage_over_generations") or []
    assert cov, "expected coverage trace"
    # Non-decreasing: each step >= previous
    for i in range(1, len(cov)):
        assert cov[i] + 1e-9 >= cov[i - 1]
