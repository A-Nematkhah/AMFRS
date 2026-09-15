#!/usr/bin/env python
"""
Compare baseline AMFRS vs AMFRS2 run manifests (cost / best metric).

Reads:
  results/.../manifest.json or amfrs2_manifest.json + optional cost_trace.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare AMFRS vs AMFRS2 run artifacts")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--amfrs2-dir", required=True)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON summary path")
    args = parser.parse_args()

    def find_manifest(d: str) -> str:
        for name in ("amfrs2_manifest.json", "manifest.json", "run_manifest.json"):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"No manifest in {d}")

    base = _load(find_manifest(args.baseline_dir))
    v2 = _load(find_manifest(args.amfrs2_dir))

    cost_trace_path = os.path.join(args.amfrs2_dir, "cost_trace.json")
    cost_trace = []
    if os.path.isfile(cost_trace_path):
        cost_trace = _load(cost_trace_path)

    summary = {
        "baseline_dir": args.baseline_dir,
        "amfrs2_dir": args.amfrs2_dir,
        "baseline_best_id": base.get("best_id") or base.get("best_candidate_id"),
        "amfrs2_best_id": v2.get("best_id"),
        "amfrs2_spent_cost_units": v2.get("spent_cost_units"),
        "amfrs2_archive_coverage": v2.get("archive_coverage"),
        "amfrs2_archive_qd_score": v2.get("archive_qd_score"),
        "amfrs2_best_robustness": v2.get("best_robustness"),
        "amfrs2_cost_trace_len": len(cost_trace),
        "amfrs2_final_best_metric": (
            cost_trace[-1].get("best_metric") if cost_trace else None
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
