#!/usr/bin/env python
"""
Single entry point for AMFRS (multi-fidelity reward search).

Examples (from ``amfrs_env/``)::

    python scripts/run_amfrs.py --fast --output-dir results/amfrs_fast

    python scripts/run_amfrs.py --profile 3h --llm groq --device cuda \\
        --output-dir results/amfrs_3h

    python scripts/run_amfrs.py --profile 18h --llm groq --device cuda \\
        --output-dir results/amfrs_18h

    python scripts/run_amfrs.py --profile full --llm groq --device cuda \\
        --output-dir results/amfrs_full
"""

from __future__ import annotations

import argparse
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
    from crowd_nav.reward_search.parallelism import (
        check_run_disk_space,
        configure_run_temp,
        configure_worker_thread_env,
    )
    from crowd_nav.reward_search.console import configure_amfrs_logging

    configure_worker_thread_env()

    parser = argparse.ArgumentParser(description="AMFRS end-to-end")
    parser.add_argument("--output-dir", type=str, default="results/amfrs_run")
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument("--fast", action="store_true", help="Stub / smoke dry-run profile")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["3h", "18h", "full", "2h", "short", "6h", "12h"],
        help=(
            "Named budget profile (all run F0→F3 + GST + robustness + bandit "
            "+ ensemble). '3h' ≈ 2–3 h, H=2. '18h' ≈ 18 h, H=10. "
            "'full' long paper-scale, H=20. Aliases: 2h/short→3h, 6h/12h→18h."
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

    named_profiles = {
        "3h": "apply_3h_gpu_profile",
        "2h": "apply_3h_gpu_profile",
        "short": "apply_3h_gpu_profile",
        "18h": "apply_18h_gpu_profile",
        "6h": "apply_18h_gpu_profile",
        "12h": "apply_18h_gpu_profile",
        "full": "apply_full_gpu_profile",
    }

    # Named profiles may use --llm seed for local wiring; real runs should pass --llm groq.
    allow_seed = bool(args.allow_seed_llm or args.profile in named_profiles)
    if (
        not args.fast
        and str(args.llm).strip().lower() == "seed"
        and not allow_seed
    ):
        print(
            "Refusing non-fast AMFRS with --llm seed. "
            "Pass a real provider or --allow-seed-llm / --fast / "
            "--profile 3h|18h|full.",
            file=sys.stderr,
        )
        return 2

    configure_amfrs_logging(verbose=bool(args.verbose))

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
    if args.profile in named_profiles:
        getattr(cfg, named_profiles[args.profile])()
        cfg.output_dir = args.output_dir
        cfg.seed = args.seed
        cfg.device = args.device
        cfg.llm_provider = args.llm
        cfg.allow_seed_llm = True
        # Second critic: same provider unless the user already set a distinct B.
        if cfg.use_ensemble_critique and str(cfg.llm_provider_b).strip().lower() == "seed":
            cfg.llm_provider_b = args.llm
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

    # Fail early when output/TEMP volumes are nearly full (Errno 28 / torch zip
    # corruption on tiny mapped drives like a full I: or cramped C:\\Temp).
    if not (cfg.fast or cfg.use_stub_trainers):
        configure_run_temp(output_dir=cfg.output_dir)
        ok_disk, disk_msg = check_run_disk_space(cfg.output_dir)
        if not ok_disk:
            print(disk_msg, file=sys.stderr)
            return 3

    try:
        AMFRSPipeline(cfg).run()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
