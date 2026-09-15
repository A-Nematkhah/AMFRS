"""
Asset presence checks for AMFRS real (non-stub) runs.

Pure stdlib — no torch/gym at import time.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from crowd_nav.reward_search.regime import (
    GST_MODEL_DIR_WITH_RANDOM,
    GST_MODEL_DIR_WITHOUT_RANDOM,
    gst_model_dir_for_regime,
    parse_regime,
)


@dataclass(frozen=True)
class AssetStatus:
    name: str
    path: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssetReport:
    items: List[AssetStatus] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(i.ok for i in self.items)

    def missing(self) -> List[AssetStatus]:
        return [i for i in self.items if not i.ok]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "items": [i.to_dict() for i in self.items],
            "missing": [i.to_dict() for i in self.missing()],
        }

    def format_text(self) -> str:
        lines = ["AMFRS asset check:"]
        for item in self.items:
            mark = "OK  " if item.ok else "MISS"
            lines.append(f"  [{mark}] {item.name}: {item.path}")
            if item.detail:
                lines.append(f"         {item.detail}")
        return "\n".join(lines)


def _abs(path: str, *, root: Optional[str] = None) -> str:
    if os.path.isabs(path):
        return path
    base = root or os.getcwd()
    return os.path.normpath(os.path.join(base, path))


def gst_checkpoint_ready(model_dir: str, *, root: Optional[str] = None) -> Tuple[bool, str]:
    """
    GST load path must contain ``checkpoint/args.pickle`` (see VecPretextNormalize).
    """
    path = _abs(model_dir, root=root)
    if not os.path.isdir(path):
        return False, "directory not found"
    ckpt = os.path.join(path, "checkpoint")
    if not os.path.isdir(ckpt):
        return False, "missing checkpoint/ subdirectory"
    args_pickle = os.path.join(ckpt, "args.pickle")
    if not os.path.isfile(args_pickle):
        # Also accept any .pt / model files as a weak signal
        files = os.listdir(ckpt)
        if not files:
            return False, "checkpoint/ is empty"
        return False, f"missing args.pickle (found: {files[:5]})"
    return True, "checkpoint/args.pickle present"


def stage1_dataset_ready(dataset_path: str, *, root: Optional[str] = None) -> Tuple[bool, str]:
    path = _abs(dataset_path, root=root)
    if not os.path.isdir(path):
        # Allow a single archive file path
        if os.path.isfile(path) and path.endswith((".npz", ".pkl", ".jsonl")):
            return True, "dataset file present"
        return False, "directory not found"
    names = os.listdir(path)
    if not names:
        return False, "directory empty"
    interesting = [
        n
        for n in names
        if n.endswith((".npz", ".pkl", ".jsonl"))
        or n.startswith("scenario_")
        or n == "manifest.json"
    ]
    if not interesting:
        return False, f"no scenario/archive files (contents: {names[:8]})"
    return True, f"found {len(interesting)} dataset artifact(s)"


def check_gst_for_regime(
    regime: str = "without_random",
    *,
    root: Optional[str] = None,
) -> AssetStatus:
    r = parse_regime(regime)
    if r == "both":
        # Report without_random as primary; caller should check both separately.
        model_dir = GST_MODEL_DIR_WITHOUT_RANDOM
        name = "gst_without_random (regime=both → check each pass)"
    else:
        model_dir = gst_model_dir_for_regime(r)
        name = f"gst_{r}"
    ok, detail = gst_checkpoint_ready(model_dir, root=root)
    return AssetStatus(name=name, path=_abs(model_dir, root=root), ok=ok, detail=detail)


def check_all_gst(*, root: Optional[str] = None) -> List[AssetStatus]:
    out = []
    for label, path in (
        ("gst_without_random", GST_MODEL_DIR_WITHOUT_RANDOM),
        ("gst_with_random", GST_MODEL_DIR_WITH_RANDOM),
    ):
        ok, detail = gst_checkpoint_ready(path, root=root)
        out.append(AssetStatus(label, _abs(path, root=root), ok, detail))
    return out


def check_stage1_dataset(
    dataset_path: str = "data/stage1_dataset",
    *,
    root: Optional[str] = None,
) -> AssetStatus:
    ok, detail = stage1_dataset_ready(dataset_path, root=root)
    return AssetStatus(
        name="stage1_dataset",
        path=_abs(dataset_path, root=root),
        ok=ok,
        detail=detail,
    )


def check_amfrs_assets(
    *,
    regime: str = "without_random",
    predict_method: str = "inferred",
    score1_mode: str = "dataset",
    stage1_dataset_path: str = "data/stage1_dataset",
    use_stub: bool = False,
    root: Optional[str] = None,
) -> AssetReport:
    """
    What a non-stub AMFRS run needs given predict/score settings.

    ``use_stub=True`` / ``--fast`` → empty report (ok).
    """
    report = AssetReport()
    if use_stub:
        report.items.append(
            AssetStatus("stub_mode", "(n/a)", True, "stubs active; assets not required")
        )
        return report

    if str(predict_method).strip().lower() == "inferred":
        report.items.append(check_gst_for_regime(regime, root=root))
        if parse_regime(regime) == "both":
            report.items.extend(check_all_gst(root=root))

    if str(score1_mode).strip().lower() == "dataset":
        report.items.append(
            check_stage1_dataset(stage1_dataset_path, root=root)
        )

    if not report.items:
        report.items.append(
            AssetStatus(
                "none_required",
                "(n/a)",
                True,
                "predict_method!=inferred and score1_mode!=dataset",
            )
        )
    return report


def require_amfrs_assets(**kwargs: Any) -> AssetReport:
    """Raise ``FileNotFoundError`` with a clear remediation message if missing."""
    report = check_amfrs_assets(**kwargs)
    if report.ok:
        return report
    tips = [
        report.format_text(),
        "",
        "Remediation:",
        "  1) GST weights: python scripts/fetch_gst_weights.py",
        "     (or copy upstream CrowdNav++ gst_updated/results/... trees)",
        "  2) Stage I dataset: python scripts/collect_stage1_dataset.py",
        "  3) Or run with --fast / --predict-method none / --score1 smoke for wiring only",
    ]
    raise FileNotFoundError("\n".join(tips))


UPSTREAM_GST_REPO = "https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph.git"
GST_SPARSE_PATHS = (
    "gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000",
    "gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand",
)
