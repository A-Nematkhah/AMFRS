"""Fail-closed guards for --llm seed on non-fast entry points."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_AMFRS_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _AMFRS_ROOT / "scripts"


def _run_script(script_name: str, argv: list[str], monkeypatch) -> int:
    """Execute a scripts/*.py main with controlled argv; return exit code."""
    script = _SCRIPTS / script_name
    assert script.is_file(), script
    monkeypatch.chdir(_AMFRS_ROOT)
    root_str = str(_AMFRS_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    monkeypatch.setattr(sys, "argv", [str(script), *argv])
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_run_amfrs_refuses_seed_without_fast(monkeypatch, capsys):
    """Non-fast + default seed LLM must exit before GST/dataset/simulator."""
    out_dir = _AMFRS_ROOT / "results" / "_seed_refuse_test"
    code = _run_script(
        "run_amfrs.py",
        ["--output-dir", str(out_dir.relative_to(_AMFRS_ROOT))],
        monkeypatch,
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "Refusing non-fast AMFRS" in err
    assert "--allow-seed-llm" in err
    assert not (out_dir / "amfrs_manifest.json").exists()
    assert not (out_dir / "manifest.json").exists()


def test_run_amfrs_fast_still_allows_seed(monkeypatch):
    code = _run_script(
        "run_amfrs.py",
        ["--fast", "--output-dir", "results/_seed_fast_ok", "--device", "cpu"],
        monkeypatch,
    )
    assert code == 0
    assert (_AMFRS_ROOT / "results" / "_seed_fast_ok" / "amfrs_manifest.json").is_file()
