"""Shared fixtures for AMFRS tests (no torch/gym required for pure modules)."""

from __future__ import annotations

import pytest

from crowd_nav.reward_search.amfrs.config import AMFRSRunConfig


@pytest.fixture
def fast_config(tmp_path) -> AMFRSRunConfig:
    cfg = AMFRSRunConfig(output_dir=str(tmp_path / "amfrs_fast"))
    cfg.apply_fast_profile()
    cfg.output_dir = str(tmp_path / "amfrs_fast")
    return cfg
