"""Smoke: AMFRSPipeline validates seeds and writes a manifest."""

from __future__ import annotations

import json
import os

from crowd_nav.reward_search.amfrs import AMFRSPipeline, AMFRSRunConfig


def test_amfrs_fast_run_writes_manifest(tmp_path):
    cfg = AMFRSRunConfig(output_dir=str(tmp_path / "out"))
    cfg.apply_fast_profile()
    cfg.output_dir = str(tmp_path / "out")

    artifacts = AMFRSPipeline(cfg).run()

    assert artifacts.accepted_candidates, "expected at least the D5 seed to validate"
    assert artifacts.best is not None
    manifest_path = os.path.join(cfg.output_dir, "amfrs_manifest.json")
    assert os.path.isfile(manifest_path)
    with open(manifest_path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["pipeline"] == "AMFRS"
    assert data["n_accepted"] >= 1
    assert data["n_accepted"] == data["config"]["population_size"]
    assert data["config"]["fast"] is True
    # Final climb must evaluate rungs above illumination (F2 in fast profile).
    levels = [e["level"] for e in data.get("promotion_log") or []]
    assert any(str(lv).startswith("F2") for lv in levels), levels
    costs = [float(e["spent_cost"]) for e in artifacts.cost_trace]
    assert costs, "expected cost_trace entries"
    assert max(costs) > 0.0


def test_pipeline_reexport_matches_package():
    from crowd_nav.reward_search import pipeline as pipe_mod
    from crowd_nav.reward_search.amfrs import pipeline as amfrs_pipe

    assert pipe_mod.AMFRSPipeline is amfrs_pipe.AMFRSPipeline
    assert pipe_mod.AMFRSRunConfig is amfrs_pipe.AMFRSRunConfig


def test_pure_amfrs_modules_import_without_torch():
    """Guardrail: config/pipeline must not pull torch at import time."""
    import crowd_nav.reward_search.amfrs.config as cfg_mod
    import crowd_nav.reward_search.amfrs.pipeline as pipe_mod

    for mod in (cfg_mod, pipe_mod):
        assert "torch" not in getattr(mod, "__dict__", {})
        src_path = getattr(mod, "__file__", None)
        assert src_path and os.path.isfile(src_path)
        with open(src_path, encoding="utf-8") as fh:
            text = fh.read()
        for bad in ("import torch", "import gym", "from torch", "from gym"):
            assert bad not in text, f"{src_path} must not contain {bad!r}"
