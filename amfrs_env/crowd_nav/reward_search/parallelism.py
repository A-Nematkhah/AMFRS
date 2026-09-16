"""
CPU parallelism defaults for Stage II/III RealPolicyTrainer.

Kept separate from Config/argparse so Stage2Config / Stage3Config can share
one policy: prefer true multi-process rollouts up to a soft cap, leave one
core free for the learner / OS.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Windows ``spawn`` duplicates the full interpreter + GST/torch/OpenBLAS into
# every worker. Auto ``cpu_count-1`` (often 8–15) OOMs with MemoryError /
# OpenBLAS allocation failures (see smoke logs). Keep the auto cap low.
_WINDOWS_AUTO_CAP = 2
_DEFAULT_AUTO_CAP = 16


def configure_worker_thread_env() -> None:
    """
    Cap BLAS/OpenMP threads so N env workers do not each try to use all cores.

    Safe to call multiple times; only sets keys that are not already present.
    """
    for key, value in (
        ("OPENBLAS_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OMP_NUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
    ):
        os.environ.setdefault(key, value)


def default_num_processes(*, cap: Optional[int] = None) -> int:
    """
    Reasonable parallel env count for this machine.

    ``min(cap, max(1, cpu_count - 1))`` — never returns 0.
    On Windows the effective auto cap is at most ``_WINDOWS_AUTO_CAP``.
    """
    configure_worker_thread_env()
    if cap is None:
        cap = _DEFAULT_AUTO_CAP
    effective_cap = int(cap)
    if os.name == "nt":
        effective_cap = min(effective_cap, _WINDOWS_AUTO_CAP)
    n = os.cpu_count() or 1
    if n <= 1:
        return 1
    return max(1, min(int(effective_cap), int(n) - 1))


def default_num_mini_batch(num_processes: int) -> int:
    """PPO/A2C mini-batches must be <= num_processes (storage assert)."""
    n = max(1, int(num_processes))
    return max(1, min(2, n))


def resolve_num_processes(value: Optional[int], *, cap: Optional[int] = None) -> int:
    """``None`` or ``<= 0`` → auto; otherwise clamp to at least 1."""
    configure_worker_thread_env()
    if value is None or int(value) <= 0:
        return default_num_processes(cap=cap if cap is not None else _DEFAULT_AUTO_CAP)
    return max(1, int(value))
