"""Tests for Stage II/III parallelism helpers."""

from __future__ import annotations

import os

from crowd_nav.reward_search.parallelism import (
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
