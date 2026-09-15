"""Shared fixtures for AMFRS2 tests (no torch/gym required for pure modules)."""

from __future__ import annotations

import pytest

from crowd_nav.reward_search.amfrs.config import AMFRS2RunConfig


@pytest.fixture
def fast_config(tmp_path) -> AMFRS2RunConfig:
    cfg = AMFRS2RunConfig(output_dir=str(tmp_path / "amfrs2_fast"))
    cfg.apply_fast_profile()
    cfg.output_dir = str(tmp_path / "amfrs2_fast")
    return cfg
