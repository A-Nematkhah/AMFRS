"""
CPU parallelism defaults for Stage II/III RealPolicyTrainer.

Kept separate from Config/argparse so Stage2Config / Stage3Config can share
one policy: prefer true multi-process rollouts up to a soft cap, leave one
core free for the learner / OS.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Windows ``spawn`` duplicates the full interpreter + GST/torch/OpenBLAS into
# every worker. Auto ``cpu_count-1`` (often 8–15) OOMs with MemoryError /
# OpenBLAS allocation failures (see smoke logs). Keep the auto cap low.
_WINDOWS_AUTO_CAP = 2
_DEFAULT_AUTO_CAP = 16
_MIN_TEMP_FREE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB
_MIN_OUTPUT_FREE_BYTES = 500 * 1024 * 1024  # 500 MiB


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


def _free_bytes(path: str) -> int:
    return int(shutil.disk_usage(path).free)


def configure_run_temp(*, output_dir: Optional[str] = None) -> str:
    """
    Ensure TEMP/TMP point at a volume with enough free space.

    Order:
    1. ``AMFRS_TEMP`` if set
    2. Keep current TEMP if it already has ``>= 1 GiB`` free
    3. Else create ``<output_dir>/.amfrs_tmp`` when output volume is roomy
    4. Else leave TEMP unchanged (caller should refuse via ``check_run_disk_space``)
    """
    preferred = (os.environ.get("AMFRS_TEMP") or "").strip()
    if preferred:
        os.makedirs(preferred, exist_ok=True)
        os.environ["TEMP"] = preferred
        os.environ["TMP"] = preferred
        tempfile.tempdir = preferred
        return preferred

    current = os.environ.get("TEMP") or os.environ.get("TMP") or tempfile.gettempdir()
    try:
        if _free_bytes(current) >= _MIN_TEMP_FREE_BYTES:
            return current
    except OSError:
        pass

    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            if _free_bytes(output_dir) >= _MIN_TEMP_FREE_BYTES:
                fallback = os.path.join(os.path.abspath(output_dir), ".amfrs_tmp")
                os.makedirs(fallback, exist_ok=True)
                os.environ["TEMP"] = fallback
                os.environ["TMP"] = fallback
                tempfile.tempdir = fallback
                logger.warning(
                    "Default TEMP has low free space; redirecting TEMP/TMP to %s",
                    fallback,
                )
                return fallback
        except OSError as exc:
            logger.warning("Could not redirect TEMP under %s: %s", output_dir, exc)

    return current


def check_run_disk_space(
    output_dir: str,
    *,
    min_output_free: int = _MIN_OUTPUT_FREE_BYTES,
    min_temp_free: int = _MIN_TEMP_FREE_BYTES,
) -> Tuple[bool, str]:
    """
    Return ``(ok, message)``. Message is empty when ok.

    Checks both the output volume and the effective TEMP volume — Stage II
    writes baselines logs + torch checkpoints under TEMP; a full TEMP drive
    surfaces as torch zip ``unexpected pos`` / ``Errno 28``.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        out_free = _free_bytes(output_dir)
    except OSError as exc:
        return False, f"Could not check disk space for {output_dir}: {exc}"

    if out_free < min_output_free:
        return (
            False,
            f"Refusing real run: only {out_free / 1e6:.0f} MB free under "
            f"{output_dir}. Use a larger drive for --output-dir "
            f"(e.g. J:\\amfrs_runs\\...) or free disk space.",
        )

    temp_dir = configure_run_temp(output_dir=output_dir)
    try:
        temp_free = _free_bytes(temp_dir)
    except OSError as exc:
        return False, f"Could not check TEMP disk space for {temp_dir}: {exc}"

    if temp_free < min_temp_free:
        return (
            False,
            f"Refusing real run: only {temp_free / 1e6:.0f} MB free under TEMP "
            f"({temp_dir}). Set AMFRS_TEMP to a roomy path "
            f"(e.g. J:\\amfrs_temp) or free disk space.",
        )
    return True, ""


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
