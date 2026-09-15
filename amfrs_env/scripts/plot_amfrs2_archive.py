#!/usr/bin/env python
"""
Plot AMFRS2 MAP-Elites archive heatmap and optional cost-vs-quality curve.

Usage (from amfrs_env/)::

    python scripts/plot_amfrs2_archive.py --run-dir results/amfrs2_fast
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot AMFRS2 archive / cost trace")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.run_dir
    os.makedirs(out_dir, exist_ok=True)

    arch_path = os.path.join(args.run_dir, "map_elites_archive.json")
    cost_path = os.path.join(args.run_dir, "cost_trace.json")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required for plotting", file=sys.stderr)
        return 2

    if os.path.isfile(arch_path):
        with open(arch_path, encoding="utf-8") as fh:
            arch = json.load(fh)
        shape = arch.get("shape") or [4, 4]
        grid = np.full((int(shape[1]), int(shape[0])), np.nan)
        annotations = {}
        for cell in arch.get("cells") or []:
            bx, by = cell["cell"]
            grid[int(by), int(bx)] = float(cell["fitness"])
            annotations[(int(by), int(bx))] = cell.get("candidate_id", "")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(grid, origin="lower", aspect="auto")
        ax.set_xlabel("path_efficiency bin")
        ax.set_ylabel("social_margin bin")
        ax.set_title("AMFRS2 MAP-Elites archive")
        fig.colorbar(im, ax=ax, label="fitness")
        for (by, bx), cid in annotations.items():
            ax.text(bx, by, str(cid)[:8], ha="center", va="center", fontsize=7, color="w")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "map_elites_heatmap.png"), dpi=140)
        plt.close(fig)
        print(f"Wrote {out_dir}/map_elites_heatmap.png")

    if os.path.isfile(cost_path):
        with open(cost_path, encoding="utf-8") as fh:
            trace = json.load(fh)
        xs = [float(p.get("spent_cost", 0.0)) for p in trace]
        ys = [float(p.get("best_metric", 0.0)) for p in trace]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, marker="o", linewidth=1.5)
        ax.set_xlabel("cumulative cost units")
        ax.set_ylabel("best metric so far")
        ax.set_title("AMFRS2 cost vs quality")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "cost_vs_quality.png"), dpi=140)
        plt.close(fig)
        print(f"Wrote {out_dir}/cost_vs_quality.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
