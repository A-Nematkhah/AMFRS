"""Asset checker unit tests (no torch/gym)."""

from __future__ import annotations

import os

from crowd_nav.reward_search.amfrs.assets import (
    check_amfrs2_assets,
    check_gst_for_regime,
    check_stage1_dataset,
    require_amfrs2_assets,
)


def test_stub_mode_assets_ok():
    report = check_amfrs2_assets(use_stub=True)
    assert report.ok


def test_inferred_without_gst_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = check_amfrs2_assets(
        regime="without_random",
        predict_method="inferred",
        score1_mode="smoke",
        use_stub=False,
        root=str(tmp_path),
    )
    assert not report.ok
    assert any(i.name.startswith("gst") for i in report.missing())


def test_none_predict_skips_gst(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = check_amfrs2_assets(
        predict_method="none",
        score1_mode="smoke",
        use_stub=False,
        root=str(tmp_path),
    )
    assert report.ok


def test_require_raises_with_remediation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    try:
        require_amfrs2_assets(
            predict_method="inferred",
            score1_mode="smoke",
            use_stub=False,
            root=str(tmp_path),
        )
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "fetch_gst_weights" in str(exc)


def test_stage1_detects_npz(tmp_path):
    d = tmp_path / "data" / "stage1_dataset"
    d.mkdir(parents=True)
    (d / "stage1_dataset.npz").write_bytes(b"x")
    st = check_stage1_dataset(str(d), root=str(tmp_path))
    assert st.ok
