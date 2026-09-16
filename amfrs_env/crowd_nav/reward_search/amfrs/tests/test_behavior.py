"""Tests for Axis 2 behavior descriptors."""

from __future__ import annotations

from crowd_nav.reward_search.amfrs.behavior import (
    DescriptorSpaceConfig,
    compute_behavior_descriptor,
    grid_shape_from_config,
)


def test_compute_behavior_descriptor_bins():
    cfg = DescriptorSpaceConfig()
    cell = compute_behavior_descriptor(
        {"PL": 10.0, "goal_dist0": 8.0, "SD": 0.5},
        config=cfg,
    )
    assert isinstance(cell, tuple)
    assert len(cell) == 2
    assert all(isinstance(i, int) for i in cell)


def test_grid_shape_matches_edges():
    cfg = DescriptorSpaceConfig()
    shape = grid_shape_from_config(cfg)
    assert shape[0] == len(cfg.path_efficiency_edges) - 1
    assert shape[1] == len(cfg.social_margin_edges) - 1
