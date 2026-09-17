"""Tests for AMFRS terminal logging setup."""

from __future__ import annotations

import logging
import sys

from crowd_nav.reward_search.console import amfrs_finished, configure_amfrs_logging


def test_configure_amfrs_logging_quiets_httpx():
    configure_amfrs_logging(verbose=False)
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("groq").level == logging.WARNING
    assert logging.getLogger("crowd_sim").level == logging.WARNING
    root = logging.getLogger()
    streams = [getattr(h, "stream", None) for h in root.handlers]
    assert sys.stdout in streams


def test_amfrs_finished_emits_review_lines(caplog):
    caplog.set_level(logging.INFO)
    amfrs_finished(
        output_dir="results/amfrs_12h",
        best_id="g4_m2",
        best_metrics={"SR": 0.12, "CR": 0.4, "TR": 0.48, "SD": 0.39},
        coverage=0.125,
        spent_cost=1280.0,
        n_accepted=12,
        n_rejected=2,
        n_scale_drift=7,
    )
    text = "\n".join(r.message for r in caplog.records)
    assert "AMFRS finished" in text
    assert "best=g4_m2" in text
    assert "scale_drift_flags=7" in text
    assert "SR=0.120" in text
    assert "amfrs_manifest.json" in text
