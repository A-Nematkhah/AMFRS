#!/usr/bin/env python
"""
Download CrowdNav++ GST checkpoint trees needed for ``predict_method=inferred``.

Fetches minimal files (checkpoint/args.pickle + epoch_*.pt + train_hist.pickle)
from the upstream GitHub raw URLs into ``gst_updated/results/...``.

Examples (from ``amfrs_env/``)::

    python scripts/fetch_gst_weights.py
    python scripts/fetch_gst_weights.py --regime without_random
    python scripts/check_amfrs_assets.py
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

RAW_BASE = (
    "https://raw.githubusercontent.com/Shuijing725/"
    "CrowdNav_Prediction_AttnGraph/main/"
)

TREES = {
    "without_random": (
        "gst_updated/results/"
        "100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-"
        "edge_head_0-ebd_64-snl_1-snh_8-seed_1000/sj"
    ),
    "with_random": (
        "gst_updated/results/"
        "100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-"
        "edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj"
    ),
}

FILES = (
    "checkpoint/args.pickle",
    "checkpoint/epoch_100.pt",
    "checkpoint/train_hist.pickle",
)


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print(f"  skip (exists): {dest}")
        return
    print(f"  GET {url}")
    tmp = dest + ".partial"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    except Exception:
        if os.path.isfile(tmp):
            os.remove(tmp)
        raise
    print(f"  wrote {dest} ({os.path.getsize(dest)} bytes)")


def fetch_tree(rel_sj: str) -> None:
    print(f"Fetching {rel_sj}")
    for rel in FILES:
        url = RAW_BASE + rel_sj.replace("\\", "/") + "/" + rel
        dest = os.path.join(_ROOT, rel_sj.replace("/", os.sep), *rel.split("/"))
        _download(url, dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch GST weights for AMFRS")
    parser.add_argument(
        "--regime",
        choices=["without_random", "with_random", "both"],
        default="both",
    )
    args = parser.parse_args()
    regimes = (
        ["without_random", "with_random"]
        if args.regime == "both"
        else [args.regime]
    )
    for r in regimes:
        fetch_tree(TREES[r])

    from crowd_nav.reward_search.amfrs.assets import check_all_gst

    report_items = check_all_gst(root=_ROOT)
    for item in report_items:
        print(f"[{'OK' if item.ok else 'MISS'}] {item.name}: {item.detail}")
    return 0 if all(i.ok for i in report_items) else 1


if __name__ == "__main__":
    raise SystemExit(main())
