#!/usr/bin/env python
"""
Single entry point for AMFRS (multi-fidelity reward search).

Examples (from ``amfrs_env/``)::

    python scripts/run_amfrs.py --fast --output-dir results/amfrs_fast

    # Real trainers without GST (obs = none):
    python scripts/run_amfrs.py --allow-seed-llm --predict-method none \\
        --score1 smoke --device cpu --output-dir results/amfrs_real_none

    # Real trainers with GST (after scripts/fetch_gst_weights.py):
    python scripts/run_amfrs.py --allow-seed-llm --predict-method inferred \\
        --score1 smoke --device cuda --output-dir results/amfrs_real_gst
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_BASELINES_ROOT = os.path.abspath(os.path.join(_ROOT, "..", "baselines_openai"))
if _BASELINES_ROOT not in sys.path:
    sys.path.insert(0, _BASELINES_ROOT)
os.chdir(_ROOT)


def main() -> int:
    from crowd_nav.reward_search.amfrs import AMFRSPipeline, AMFRSRunConfig
    from crowd_nav.reward_search.amfrs.assets import check_amfrs_assets
    from crowd_nav.reward_search.parallelism import configure_worker_thread_env

    configure_worker_thread_env()

    parser = argparse.ArgumentParser(description="AMFRS end-to-end")
    parser.add_argument("--output-dir", type=str, default="results/amfrs_run")
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument("--fast", action="store_true", help="Stub / smoke dry-run profile")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["short"],
        help=(
            "Named budget profile. 'short' ≈ 15–30 min real A2C on one GPU "
            "(no stub, no full PPO, no robustness sweep)."
        ),
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="seed",
        choices=["seed", "groq", "vllm", "ollama", "scripted"],
    )
    parser.add_argument("--allow-seed-llm", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument(
        "--predict-method",
        type=str,
        default=None,
        choices=["inferred", "none"],
        help=(
            "Default inferred; --fast forces none. Use none if GST missing. "
            "Only inferred|none are wired through AMFRS trainers/regime."
        ),
    )
    parser.add_argument(
        "--score1",
        type=str,
        default=None,
        choices=["dataset", "smoke"],
        help="F0 scorer (default dataset; smoke if --fast)",
    )
    parser.add_argument("--stage1-dataset", type=str, default="data/stage1_dataset")
    parser.add_argument("--regime", type=str, default="without_random")
    parser.add_argument("--human-num", type=int, default=None)
    parser.add_argument(
        "--use-stub",
        action="store_true",
        help="Force stub trainers even without --fast",
    )
    parser.add_argument(
        "--check-assets-only",
        action="store_true",
        help="Print asset status and exit (0=ok, 1=missing)",
    )
    parser.add_argument("--stage2-train-steps", type=int, default=None)
    parser.add_argument("--stage2-train-steps-short", type=int, default=None)
    parser.add_argument("--stage3-train-steps", type=int, default=None)
    parser.add_argument("--num-processes", type=int, default=None)
    args = parser.parse_args()

    if args.check_assets_only:
        report = check_amfrs_assets(
            regime=args.regime,
            predict_method=args.predict_method or "inferred",
            score1_mode=args.score1 or "dataset",
            stage1_dataset_path=args.stage1_dataset,
            use_stub=bool(args.fast or args.use_stub),
            root=_ROOT,
        )
        print(report.format_text())
        return 0 if report.ok else 1

    if args.fast and args.profile:
        print("Use either --fast or --profile, not both.", file=sys.stderr)
        return 2

    # --profile short enables seed LLM by design (local short GPU smoke).
    allow_seed = bool(args.allow_seed_llm or args.profile == "short")
    if (
        not args.fast
        and str(args.llm).strip().lower() == "seed"
        and not allow_seed
    ):
        print(
            "Refusing non-fast AMFRS with --llm seed. "
            "Pass a real provider or --allow-seed-llm / --fast / --profile short.",
            file=sys.stderr,
        )
        return 2

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Protect Config.get_args() from our CLI when real trainers parse argv.
    sanitized = [sys.argv[0], "--seed", str(args.seed)]
    if args.device == "cpu":
        sanitized.append("--no-cuda")
    sys.argv = sanitized

    cfg = AMFRSRunConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        llm_provider=args.llm,
        device=args.device,
        allow_seed_llm=allow_seed,
        stage1_dataset_path=args.stage1_dataset,
        randomization_regime=args.regime,
        num_processes=args.num_processes,
    )
    if args.fast:
        cfg.apply_fast_profile()
        cfg.output_dir = args.output_dir
        cfg.seed = args.seed
    if args.profile == "short":
        cfg.apply_short_gpu_profile()
        cfg.output_dir = args.output_dir
        cfg.seed = args.seed
        cfg.device = args.device
        cfg.llm_provider = args.llm
        cfg.allow_seed_llm = True
    if args.use_stub:
        cfg.use_stub_trainers = True
        if args.score1 is None and not args.fast:
            cfg.score1_mode = "smoke"
    if args.predict_method is not None and not args.fast:
        cfg.predict_method = args.predict_method
    if args.score1 is not None and not args.fast:
        cfg.score1_mode = args.score1
    if args.population_size is not None:
        cfg.population_size = max(1, int(args.population_size))
    if args.generations is not None:
        cfg.generations = max(0, int(args.generations))
    if args.human_num is not None:
        cfg.human_num = max(1, int(args.human_num))
    if args.stage2_train_steps is not None:
        cfg.stage2_train_steps = int(args.stage2_train_steps)
    if args.stage2_train_steps_short is not None:
        cfg.stage2_train_steps_short = int(args.stage2_train_steps_short)
    if args.stage3_train_steps is not None:
        cfg.stage3_train_steps = int(args.stage3_train_steps)
    if args.num_processes is not None:
        cfg.num_processes = int(args.num_processes)

    # Fail early when the output volume is nearly full (common on tiny/mapped drives).
    try:
        import shutil

        os.makedirs(cfg.output_dir, exist_ok=True)
        free = shutil.disk_usage(cfg.output_dir).free
        if free < 500 * 1024 * 1024 and not (cfg.fast or cfg.use_stub_trainers):
            print(
                f"Refusing real run: only {free / 1e6:.0f} MB free under "
                f"{cfg.output_dir}. Use a larger drive for --output-dir "
                f"(e.g. J:\\amfrs_runs\\...) or free disk space.",
                file=sys.stderr,
            )
            return 3
    except OSError as exc:
        print(f"Could not check disk space for {cfg.output_dir}: {exc}", file=sys.stderr)

    try:
        artifacts = AMFRSPipeline(cfg).run()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"[amfrs] done | accepted={len(artifacts.accepted_candidates)} "
        f"rejected={len(artifacts.rejected)} "
        f"best={artifacts.best.candidate_id if artifacts.best else None} "
        f"stub={cfg.use_stub_trainers or cfg.fast} "
        f"profile={args.profile or ('fast' if args.fast else 'default')} "
        f"-> {cfg.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
