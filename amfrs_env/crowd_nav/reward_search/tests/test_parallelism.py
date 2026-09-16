"""Tests for Stage II/III parallelism helpers."""

from __future__ import annotations

import os

from crowd_nav.reward_search.parallelism import (
    check_run_disk_space,
    configure_run_temp,
    configure_worker_thread_env,
    default_num_processes,
    resolve_num_processes,
)


def test_configure_worker_thread_env_sets_blas_caps(monkeypatch):
    for key in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.delenv(key, raising=False)
    configure_worker_thread_env()
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_resolve_explicit_num_processes():
    assert resolve_num_processes(4) == 4
    assert resolve_num_processes(1) == 1


def test_windows_auto_cap(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert default_num_processes() == 2
    assert resolve_num_processes(None) == 2


def test_configure_run_temp_honors_amfrs_temp(monkeypatch, tmp_path):
    preferred = tmp_path / "preferred_temp"
    monkeypatch.setenv("AMFRS_TEMP", str(preferred))
    monkeypatch.delenv("TEMP", raising=False)
    monkeypatch.delenv("TMP", raising=False)
    got = configure_run_temp(output_dir=str(tmp_path / "out"))
    assert got == str(preferred)
    assert preferred.is_dir()
    assert os.environ["TEMP"] == str(preferred)
    assert os.environ["TMP"] == str(preferred)


def test_configure_run_temp_redirects_when_default_full(monkeypatch, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    cramped = tmp_path / "cramped_temp"
    cramped.mkdir()
    monkeypatch.delenv("AMFRS_TEMP", raising=False)
    monkeypatch.setenv("TEMP", str(cramped))
    monkeypatch.setenv("TMP", str(cramped))

    def fake_free(path):
        class Usage:
            free = 10 * 1024 * 1024 if str(path).startswith(str(cramped)) else 5 * 1024 * 1024 * 1024

        return Usage()

    monkeypatch.setattr(
        "crowd_nav.reward_search.parallelism.shutil.disk_usage",
        fake_free,
    )
    got = configure_run_temp(output_dir=str(out))
    assert got.endswith(".amfrs_tmp")
    assert os.path.isdir(got)


def test_check_run_disk_space_rejects_full_output(monkeypatch, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setenv("AMFRS_TEMP", str(temp))

    def fake_free(path):
        class Usage:
            free = 10 * 1024 * 1024 if str(path).startswith(str(out)) else 5 * 1024 * 1024 * 1024

        return Usage()

    monkeypatch.setattr(
        "crowd_nav.reward_search.parallelism.shutil.disk_usage",
        fake_free,
    )
    ok, msg = check_run_disk_space(str(out))
    assert not ok
    assert "only" in msg and "MB free" in msg
