#!/usr/bin/env python
"""
Visualize an AMFRS Stage II/III checkpoint with its reward candidate.

Loads ``final_candidate.json`` (or ``--candidate``) + checkpoint, runs a few
episodes with matplotlib rendering, optionally saves slide PNGs and a GIF.

Examples::

    # Static archive heatmap (no GPU)
    python scripts/plot_amfrs_archive.py --run-dir results/amfrs_12h

    # Headless: save slides + GIF for 2 episodes
    python scripts/visualize_amfrs.py --run-dir results/amfrs_12h \\
        --stage stage2 --episodes 2 --gif --human-num 15

    # Interactive window
    python scripts/visualize_amfrs.py --run-dir results/amfrs_12h \\
        --stage stage2 --episodes 1 --human-num 15
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _resolve_path(run_dir: str, path: str) -> str:
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    # Normalize Windows-style separators stored in JSON.
    norm = path.replace("\\", os.sep)
    candidates = [
        norm,
        os.path.join(run_dir, norm),
        os.path.join(_ROOT, norm),
        os.path.normpath(os.path.join(run_dir, os.path.basename(norm))),
    ]
    # Also try relative to cwd / repo root as stored.
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    # Glob last .pt under stage*_train matching candidate id folder.
    return os.path.abspath(norm)


def _find_checkpoint(run_dir: str, cand: Dict[str, Any], stage: str) -> str:
    md = cand.get("metadata") or {}
    ckpt = md.get("checkpoint_path")
    if not ckpt:
        raw = md.get("last_raw_metrics") or {}
        if isinstance(raw, dict):
            ckpt = raw.get("checkpoint_path")
    if ckpt:
        resolved = _resolve_path(run_dir, str(ckpt))
        if os.path.isfile(resolved):
            return resolved
    cid = str(cand.get("candidate_id", ""))
    # Prefer latest .pt (more env-steps) within a candidate folder.
    # Alg1: stage2_train / stage3_train — AMFRS: trained_models/stage2|stage3
    patterns = [
        os.path.join(run_dir, f"{stage}_train", f"*_{cid}", "checkpoints", "*.pt"),
        os.path.join(run_dir, "trained_models", stage, f"*_{cid}", "checkpoints", "*.pt"),
        os.path.join(run_dir, "trained_models", stage, f"*_{cid}_*", "checkpoints", "*.pt"),
        os.path.join(run_dir, "robustness", "*", f"*_{cid}*", "checkpoints", "*.pt"),
    ]
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    folders: List[str] = []
    for root in (
        os.path.join(run_dir, f"{stage}_train"),
        os.path.join(run_dir, "trained_models", stage),
    ):
        folders.extend(sorted(glob.glob(os.path.join(root, f"*{cid}*"))))
    for folder in folders:
        pts = sorted(glob.glob(os.path.join(folder, "checkpoints", "*.pt")))
        if pts:
            return pts[-1]
    raise FileNotFoundError(
        f"No checkpoint for candidate={cid} under {run_dir} "
        f"(tried {stage}_train and trained_models/{stage})"
    )


def _load_run_config(run_dir: str) -> Dict[str, Any]:
    """Alg1 writes config.json; AMFRS nests config inside amfrs_manifest.json."""
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(cfg_path):
        return _load_json(cfg_path)
    man_path = os.path.join(run_dir, "amfrs_manifest.json")
    if os.path.isfile(man_path):
        man = _load_json(man_path)
        cfg = man.get("config")
        if isinstance(cfg, dict):
            return cfg
    return {}


def _resolve_candidate_path(run_dir: str, candidate: Optional[str]) -> str:
    """
    Prefer explicit --candidate, then Alg1 finals, then AMFRS archive best cell.
    Writes a temp JSON under ``visuals/_candidate.json`` when resolving from archive.
    """
    if candidate:
        path = candidate if os.path.isabs(candidate) else os.path.join(run_dir, candidate)
        if os.path.isfile(path):
            return os.path.abspath(path)
        raise FileNotFoundError(f"candidate JSON not found: {candidate}")

    for name in ("final_candidate.json", "best_stage3.json", "best_stage2.json"):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            return p

    man_path = os.path.join(run_dir, "amfrs_manifest.json")
    arch_path = os.path.join(run_dir, "map_elites_archive.json")
    best_id = None
    if os.path.isfile(man_path):
        best_id = _load_json(man_path).get("best_id")
    if os.path.isfile(arch_path):
        arch = _load_json(arch_path)
        cells = list(arch.get("cells") or [])
        chosen = None
        if best_id is not None:
            for cell in cells:
                if str(cell.get("candidate_id")) == str(best_id):
                    chosen = cell
                    break
        if chosen is None and cells:
            chosen = max(cells, key=lambda c: float(c.get("fitness", float("-inf"))))
        if chosen is not None:
            payload = {
                "candidate_id": chosen.get("candidate_id"),
                "code": chosen.get("code"),
                "score": chosen.get("score"),
                "valid": True,
                "origin": "archive",
                "parent_ids": [],
                "validation_error": None,
                "metadata": chosen.get("metadata") or {},
            }
            out = os.path.join(run_dir, "visuals", "_candidate.json")
            _ensure_parent(out)
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            return out

    raise FileNotFoundError(
        "No candidate JSON found (final_candidate.json / archive best missing)"
    )


def _frames_to_gif(frame_paths: Sequence[str], gif_path: str, duration_ms: int = 100) -> None:
    from PIL import Image

    if not frame_paths:
        raise ValueError("No frames to encode as GIF")
    images = [Image.open(p).convert("RGB") for p in frame_paths]
    _ensure_parent(gif_path)
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    for im in images:
        im.close()


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _collect_sorted_pngs(folder: str) -> List[str]:
    files = glob.glob(os.path.join(folder, "*.png"))
    def _key(p: str) -> Tuple[int, str]:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            return (int(stem), stem)
        except ValueError:
            return (10**9, stem)

    return sorted(files, key=_key)


def visualize(
    *,
    run_dir: str,
    candidate_path: str,
    stage: str,
    episodes: int,
    test_case: int,
    device_pref: str,
    save_slides: bool,
    gif: bool,
    no_display: bool,
    human_num: Optional[int],
    output_dir: Optional[str],
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    import matplotlib

    if no_display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn

    from crowd_nav.reward_search.regime import env_name_for_predict_method
    from crowd_nav.reward_search.reporting import load_candidate_dict
    from crowd_nav.reward_search.stage3 import Stage3Config, _make_full_env_config, _parse_stage3_algo_args
    from rl.evaluation import evaluate
    from rl.networks.envs import make_vec_envs
    from rl.networks.model import Policy

    run_cfg = _load_run_config(run_dir)
    cand_payload = _load_json(candidate_path)
    candidate = load_candidate_dict(cand_payload)
    if candidate.reward_fn is None:
        raise RuntimeError(
            f"Candidate reward failed sandbox: {candidate.validation_error}"
        )

    ckpt_path = _find_checkpoint(run_dir, cand_payload, stage)
    predict_method = str(run_cfg.get("predict_method") or "inferred")
    regime = str(run_cfg.get("randomization_regime") or "without_random")
    h_num = int(human_num if human_num is not None else run_cfg.get("human_num", 20))
    if seed is not None:
        seed = int(seed)
    elif int(test_case) < 0:
        # Random scenario each invocation (avoid always replaying seed=425 case 0).
        seed = int(random.randint(0, 2**31 - 1))
    else:
        seed = int(run_cfg.get("seed", 425))
    env_name = env_name_for_predict_method(predict_method)

    s3_cfg = Stage3Config(
        train_env_steps=1,
        eval_episodes=int(episodes),
        seed=seed,
        env_name=env_name,
        predict_method=predict_method,
        randomization_regime=regime,
        output_root=os.path.join(run_dir, f"{stage}_viz"),
        device=device_pref,
        train_human_num=h_num,
        num_processes=1,
    )
    algo_args = _parse_stage3_algo_args(s3_cfg, candidate.candidate_id, 0)
    # Force single-process eval argv fields already set; ensure no_cuda if needed.
    if device_pref != "cuda" or not torch.cuda.is_available():
        algo_args.cuda = False

    env_config = _make_full_env_config(s3_cfg)
    env_config.env.test_size = int(episodes)

    out_root = output_dir or os.path.join(run_dir, "visuals", candidate.candidate_id)
    os.makedirs(out_root, exist_ok=True)
    slides_root = os.path.join(out_root, "slides")
    if save_slides or gif:
        env_config.save_slides = True
        env_config.save_path = slides_root
        os.makedirs(slides_root, exist_ok=True)

    ax = None
    if not no_display:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-6.5, 6.5)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.ion()
        plt.show()
    else:
        # Headless slides: still need an axes for artists; skip interactive window.
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-6.5, 6.5)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        # Speed up save_slides path (env always calls plt.pause).
        plt.pause = lambda _interval: None  # type: ignore[assignment]

    device = torch.device("cuda" if algo_args.cuda and torch.cuda.is_available() else "cpu")
    print(f"[viz] candidate={candidate.candidate_id} ckpt={ckpt_path}")
    print(
        f"[viz] env={env_name} predict={predict_method} humans={h_num} "
        f"episodes={episodes} test_case={test_case} seed={seed} device={device}"
    )

    envs = make_vec_envs(
        env_name,
        seed,
        1,
        algo_args.gamma,
        os.path.join(out_root, "eval"),
        device,
        allow_early_resets=True,
        config=env_config,
        ax=ax,
        test_case=test_case,
        pretext_wrapper=env_config.env.use_wrapper,
        reward_fn=candidate.reward_fn,
    )

    actor_critic = Policy(
        envs.observation_space.spaces,
        envs.action_space,
        base_kwargs=algo_args,
        base=env_config.robot.policy,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_critic.load_state_dict(state)
    actor_critic.base.nenv = 1
    nn.DataParallel(actor_critic).to(device)

    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Always render when saving slides or showing UI.
    do_viz = True
    evaluate(
        actor_critic,
        envs,
        1,
        device,
        int(episodes),
        logging,
        env_config,
        algo_args,
        do_viz,
    )

    gif_path = None
    if gif:
        # Prefer first episode folder under slides_root.
        episode_dirs = sorted(
            d for d in glob.glob(os.path.join(slides_root, "*")) if os.path.isdir(d)
        )
        if not episode_dirs:
            print("[viz] warning: save_slides produced no frame folders; GIF skipped")
        else:
            frames = _collect_sorted_pngs(episode_dirs[0])
            gif_path = os.path.join(out_root, f"{candidate.candidate_id}_ep0.gif")
            _frames_to_gif(frames, gif_path, duration_ms=80)
            print(f"[viz] GIF ({len(frames)} frames): {gif_path}")

    if no_display:
        plt.close(fig)

    return {
        "candidate_id": candidate.candidate_id,
        "checkpoint": ckpt_path,
        "output_dir": out_root,
        "gif": gif_path,
        "slides": slides_root if (save_slides or gif) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize AMFRS checkpoint episodes")
    parser.add_argument("--run-dir", required=True, help="results/<run> directory")
    parser.add_argument(
        "--candidate",
        default=None,
        help="Candidate JSON (default: final_candidate.json, else best_stage3.json)",
    )
    parser.add_argument(
        "--stage",
        choices=("stage2", "stage3"),
        default="stage3",
        help="Which train/ folder to search for checkpoints",
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument(
        "--test-case",
        type=int,
        default=-1,
        help="Scenario id (>=0 fixed/reproducible); -1 = random each run (default)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Env RNG seed (default: run config seed). Change this with --test-case -1 "
        "for a different random stream.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--human-num", type=int, default=None, help="Must match training obs width")
    parser.add_argument("--save-slides", action="store_true", help="Write per-step PNGs")
    parser.add_argument("--gif", action="store_true", help="Also stitch first episode to GIF")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Headless (Agg); implied when --gif/--save-slides without a GUI need",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"error: run dir not found: {run_dir}", file=sys.stderr)
        return 1

    try:
        cand = _resolve_candidate_path(run_dir, args.candidate)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Prefer stage3 when asked; fall back to stage2 if no F3 checkpoint.
    stage = args.stage
    if stage == "stage3":
        probe = _load_json(cand)
        try:
            _find_checkpoint(run_dir, probe, "stage3")
        except FileNotFoundError:
            print("[viz] no stage3 checkpoint; falling back to stage2", flush=True)
            stage = "stage2"

    no_display = bool(args.no_display or args.gif or args.save_slides)
    # If user only wants live view, keep display.
    if not args.no_display and not args.gif and not args.save_slides:
        no_display = False

    try:
        info = visualize(
            run_dir=run_dir,
            candidate_path=os.path.abspath(cand),
            stage=stage,
            episodes=int(args.episodes),
            test_case=int(args.test_case),
            device_pref=args.device,
            save_slides=bool(args.save_slides or args.gif),
            gif=bool(args.gif),
            no_display=no_display,
            human_num=args.human_num,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise

    print(f"[viz] done -> {info['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
