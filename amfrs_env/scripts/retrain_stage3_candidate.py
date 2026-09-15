#!/usr/bin/env python
"""
Retrain one reward candidate with Stage III PPO (diagnosis / longer K3).

Loads code from a prior AMFRS run JSON (final_candidate / best_stage3 / path)
and trains from scratch — no LLM refine. Useful to test whether higher K3
lifts SR without re-running Stage I/II.

Examples (from ``amfrs_env/``)::

    # Overnight winner, ~K3=5e5 (~1–2h on same GPU as overnight_h7)
    python scripts/retrain_stage3_candidate.py \\
        --candidate-json results/amfrs_overnight_h7/final_candidate.json \\
        --human-num 7 --h-sweep 3,5,7 \\
        --train-env-steps 500000 --eval-episodes 150 \\
        --device cuda --num-processes 4 \\
        --output-dir results/retrain_cro0016_k3_5e5

    # Stronger probe (~K3=1e6)
    python scripts/retrain_stage3_candidate.py \\
        --candidate-json results/amfrs_overnight_h7/final_candidate.json \\
        --human-num 7 --h-sweep 3,5,7 \\
        --train-env-steps 1000000 --eval-episodes 200 \\
        --device cuda --num-processes 4 \\
        --output-dir results/retrain_cro0016_k3_1e6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_BASELINES_ROOT = os.path.abspath(os.path.join(_ROOT, "..", "baselines_openai"))
if _BASELINES_ROOT not in sys.path:
    sys.path.insert(0, _BASELINES_ROOT)
os.chdir(_ROOT)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pick_candidate(data: Dict[str, Any], candidate_id: Optional[str]) -> Dict[str, Any]:
    if "population" in data:
        items = list(data["population"])
        if candidate_id is None:
            raise SystemExit(
                "JSON has a population; pass --candidate-id to select one entry"
            )
        for cand in items:
            if str(cand.get("candidate_id")) == candidate_id:
                return cand
        raise SystemExit(f"candidate_id={candidate_id!r} not found in population")
    if candidate_id is not None and str(data.get("candidate_id")) != candidate_id:
        raise SystemExit(
            f"JSON candidate_id={data.get('candidate_id')!r} "
            f"!= requested {candidate_id!r}"
        )
    return data


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrain one AMFRS candidate with Stage III PPO (no refine)"
    )
    parser.add_argument(
        "--candidate-json",
        required=True,
        help="Path to final_candidate.json / best_stage3.json / population entry",
    )
    parser.add_argument(
        "--candidate-id",
        default=None,
        help="Required when --candidate-json is a *population*.json",
    )
    parser.add_argument("--train-env-steps", type=int, default=500_000)
    parser.add_argument("--eval-episodes", type=int, default=150)
    parser.add_argument("--human-num", type=int, default=7)
    parser.add_argument(
        "--h-sweep",
        type=str,
        default="3,5,7",
        help="Comma-separated H counts (<= human-num). Empty to skip.",
    )
    parser.add_argument("--no-h-sweep", action="store_true")
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument(
        "--predict-method",
        choices=("inferred", "none", "const_vel", "truth"),
        default="inferred",
    )
    parser.add_argument(
        "--regime",
        default="without_random",
        choices=("without_random", "with_random"),
    )
    parser.add_argument(
        "--output-dir",
        default="results/retrain_stage3_candidate",
        help="Directory for metrics JSON + stage3_train checkpoints",
    )
    args = parser.parse_args()

    # Isolate Config.get_args() from our CLI.
    sys.argv = [
        sys.argv[0],
        "--no-cuda" if args.device == "cpu" else "--seed",
        str(args.seed),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    import crowd_sim  # noqa: F401
    from crowd_nav.reward_search.evolver import RewardCandidate
    from crowd_nav.reward_search.regime import env_name_for_predict_method
    from crowd_nav.reward_search.sandbox import RewardValidator
    from crowd_nav.reward_search.stage3 import (
        STAGE3_PAPER_STEPS,
        RealPolicyTrainer,
        Stage3Config,
    )

    raw = _load_json(os.path.abspath(args.candidate_json))
    payload = _pick_candidate(raw, args.candidate_id)
    code = str(payload.get("code") or "")
    cid = str(payload.get("candidate_id") or "retrain")
    if not code.strip():
        logging.error("Candidate JSON has empty code")
        return 1

    validator = RewardValidator()
    reward_fn, err = validator.try_validate(code)
    if reward_fn is None:
        logging.error("Sandbox rejected candidate code: %s", err)
        return 1

    human_num = max(1, int(args.human_num))
    if args.no_h_sweep or not str(args.h_sweep).strip():
        human_counts: Tuple[int, ...] = (human_num,)
        run_h_sweep = False
    else:
        human_counts = tuple(
            int(x) for x in str(args.h_sweep).split(",") if x.strip()
        )
        human_counts = tuple(h for h in human_counts if 1 <= h <= human_num)
        if not human_counts:
            human_counts = (human_num,)
        run_h_sweep = len(human_counts) >= 1

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    train_root = os.path.join(out_dir, "stage3_train")

    cfg = Stage3Config(
        population_size=1,
        rounds=1,
        train_env_steps=int(args.train_env_steps),
        eval_episodes=int(args.eval_episodes),
        horizon_steps=None,
        algo="ppo",
        seed=int(args.seed),
        device=str(args.device),
        num_processes=int(args.num_processes),
        train_human_num=human_num,
        human_counts=human_counts,
        output_root=train_root,
        randomization_regime=str(args.regime),
        predict_method=str(args.predict_method),
        env_name=env_name_for_predict_method(str(args.predict_method)),
        protect_elite_refine=True,
        accept_reject_refine=False,
        skip_refine_last_round=True,
    )

    cand = RewardCandidate(
        candidate_id=cid,
        code=code,
        reward_fn=reward_fn,
        valid=True,
        origin=str(payload.get("origin") or "retrain"),
        parent_ids=tuple(payload.get("parent_ids") or ()),
        metadata={"source_json": os.path.abspath(args.candidate_json)},
    )

    logging.info(
        "Retrain %s K3=%d (paper=%d) H_train=%d predict=%s nproc=%d → %s",
        cid,
        cfg.train_env_steps,
        STAGE3_PAPER_STEPS,
        human_num,
        cfg.predict_method,
        cfg.num_processes,
        out_dir,
    )

    trainer = RealPolicyTrainer()
    t0 = time.perf_counter()
    bundle = trainer.train_and_eval(cand, round_index=0, config=cfg)
    train_wall = time.perf_counter() - t0
    metrics = bundle.metrics.as_dict()
    logging.info("Train+eval metrics: %s", bundle.metrics.feedback_text())
    logging.info(
        "Train wall=%.1fs checkpoint=%s",
        train_wall,
        bundle.checkpoint_path,
    )

    h_report = None
    if run_h_sweep:
        logging.info("H-sweep counts=%s", human_counts)
        report = trainer.evaluate_at_human_counts(
            cand, bundle, config=cfg, human_counts=human_counts
        )
        logging.info("H-sweep:\n%s", report.summary_table())
        h_report = {
            "candidate_id": report.candidate_id,
            "by_human_count": {
                str(h): m.as_dict() for h, m in report.by_human_count.items()
            },
            "summary_table": report.summary_table(),
        }

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": cid,
        "source_json": os.path.abspath(args.candidate_json),
        "config": {
            "train_env_steps": cfg.train_env_steps,
            "eval_episodes": cfg.eval_episodes,
            "human_num": human_num,
            "human_counts": list(human_counts),
            "predict_method": cfg.predict_method,
            "regime": cfg.randomization_regime,
            "device": cfg.device,
            "num_processes": cfg.num_processes,
            "seed": cfg.seed,
        },
        "metrics": metrics,
        "scalar": float(
            metrics.get("SR", 0) - metrics.get("CR", 0) - 0.5 * metrics.get("TR", 0)
        ),
        "checkpoint_path": bundle.checkpoint_path,
        "train_wall_seconds": train_wall,
        "h_sweep": h_report,
        "code": code,
    }
    out_path = os.path.join(out_dir, "retrain_result.json")
    _write_json(out_path, result)
    logging.info("Wrote %s", out_path)
    logging.info(
        "Done. SR=%.3f CR=%.3f TR=%.3f scalar=%.3f",
        float(metrics.get("SR", 0)),
        float(metrics.get("CR", 0)),
        float(metrics.get("TR", 0)),
        result["scalar"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
