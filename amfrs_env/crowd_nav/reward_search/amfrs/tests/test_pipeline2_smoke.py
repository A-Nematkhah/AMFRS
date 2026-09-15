"""M0 smoke: AMFRS2Pipeline validates seeds and writes a manifest."""

from __future__ import annotations

import json
import os

from crowd_nav.reward_search.amfrs import AMFRS2Pipeline, AMFRS2RunConfig


def test_amfrs2_fast_run_writes_manifest(tmp_path):
    cfg = AMFRS2RunConfig(output_dir=str(tmp_path / "out"))
    cfg.apply_fast_profile()
    cfg.output_dir = str(tmp_path / "out")

    artifacts = AMFRS2Pipeline(cfg).run()

    assert artifacts.accepted_candidates, "expected at least the D5 seed to validate"
    assert artifacts.best is not None
    manifest_path = os.path.join(cfg.output_dir, "amfrs2_manifest.json")
    assert os.path.isfile(manifest_path)
    with open(manifest_path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["pipeline"] == "AMFRS2"
    assert data["n_accepted"] >= 1
    assert data["config"]["fast"] is True
    # Final climb must evaluate rungs above illumination (F2 in fast profile).
    levels = [e["level"] for e in data.get("promotion_log") or []]
    assert any(str(lv).startswith("F2") for lv in levels), levels
    # Per-gen cost_trace spent_cost must move (not stick at the global counter).
    costs = [float(e["spent_cost"]) for e in artifacts.cost_trace]
    assert costs, "expected cost_trace entries"
    assert max(costs) > 0.0


def test_amfrs2_config_does_not_subclass_baseline():
    from crowd_nav.reward_search.pipeline import AMFRSRunConfig

    assert not issubclass(AMFRS2RunConfig, AMFRSRunConfig)


def test_pure_amfrs_modules_import_without_torch():
    """Guardrail: config/pipeline2 must not pull torch at import time."""
    import sys

    # Allow torch if already loaded by the test runner; only check our modules
    # do not *require* it by inspecting their globals for torch symbols.
    import crowd_nav.reward_search.amfrs.config as cfg_mod
    import crowd_nav.reward_search.amfrs.pipeline2 as pipe_mod

    for mod in (cfg_mod, pipe_mod):
        assert "torch" not in getattr(mod, "__dict__", {})
        # Module source should not top-level-import torch/gym
        src_path = getattr(mod, "__file__", None)
        assert src_path and os.path.isfile(src_path)
        with open(src_path, encoding="utf-8") as fh:
            text = fh.read()
        for bad in ("import torch", "import gym", "from torch", "from gym"):
            assert bad not in text, f"{src_path} must not contain {bad!r}"
