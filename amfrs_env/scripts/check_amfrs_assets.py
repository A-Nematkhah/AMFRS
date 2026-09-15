#!/usr/bin/env python
"""Print GST / Stage I dataset readiness for AMFRS real runs."""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)


def main() -> int:
    from crowd_nav.reward_search.amfrs.assets import check_amfrs_assets, check_all_gst

    parser = argparse.ArgumentParser(description="Check AMFRS runtime assets")
    parser.add_argument("--regime", default="without_random")
    parser.add_argument("--predict-method", default="inferred")
    parser.add_argument("--score1", default="dataset", choices=["dataset", "smoke"])
    parser.add_argument("--stage1-dataset", default="data/stage1_dataset")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--all-gst", action="store_true", help="Check both GST trees")
    args = parser.parse_args()

    if args.all_gst:
        items = check_all_gst(root=_ROOT)
        payload = {"items": [i.to_dict() for i in items], "ok": all(i.ok for i in items)}
        print(json.dumps(payload, indent=2) if args.json else "\n".join(
            f"[{'OK' if i.ok else 'MISS'}] {i.name}: {i.path} — {i.detail}" for i in items
        ))
        return 0 if payload["ok"] else 1

    report = check_amfrs_assets(
        regime=args.regime,
        predict_method=args.predict_method,
        score1_mode=args.score1,
        stage1_dataset_path=args.stage1_dataset,
        use_stub=False,
        root=_ROOT,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.format_text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
