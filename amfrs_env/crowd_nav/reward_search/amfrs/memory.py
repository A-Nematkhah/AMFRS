"""
Axis 3 — cross-run retrieval memory (sqlite + AST bag-of-features).
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List


@dataclass(frozen=True)
class MemoryEntry:
    code: str
    run_id: str
    outcome_metrics: dict
    similarity: float = 0.0


def _ast_feature_vector(code: str) -> Dict[str, float]:
    features: Dict[str, float] = {}
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return {"parse_error": 1.0}
    for node in ast.walk(tree):
        name = type(node).__name__
        features[name] = features.get(name, 0.0) + 1.0
    features["n_constants"] = float(
        sum(1 for n in ast.walk(tree) if isinstance(n, ast.Constant))
    )
    features["n_compares"] = float(
        sum(1 for n in ast.walk(tree) if isinstance(n, ast.Compare))
    )
    src = code or ""
    features["mentions_discomfort"] = 1.0 if "discomfort_dist" in src else 0.0
    features["mentions_humans"] = 1.0 if "humans" in src else 0.0
    features["mentions_dmin"] = 1.0 if "dmin" in src else 0.0
    return features


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(float(a.get(k, 0.0)) * float(b.get(k, 0.0)) for k in keys)
    na = math.sqrt(sum(float(a.get(k, 0.0)) ** 2 for k in keys))
    nb = math.sqrt(sum(float(b.get(k, 0.0)) ** 2 for k in keys))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(dot / (na * nb))


class RewardMemory:
    """SQLite-backed store. Default path: amfrs_env/data/reward_memory.sqlite"""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                code_hash TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                run_id TEXT,
                features_json TEXT,
                outcome_json TEXT,
                created_at TEXT
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def add(self, code: str, run_id: str, outcome_metrics: dict) -> None:
        features = _ast_feature_vector(code)
        code_hash = hashlib.sha256((code or "").encode("utf-8")).hexdigest()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO candidates
            (code_hash, code, run_id, features_json, outcome_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                code_hash,
                code,
                run_id,
                json.dumps(features),
                json.dumps(outcome_metrics or {}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def query_similar(self, code: str, k: int = 3) -> List[MemoryEntry]:
        q = _ast_feature_vector(code)
        rows = self._conn.execute(
            "SELECT code, run_id, features_json, outcome_json FROM candidates"
        ).fetchall()
        scored: List[MemoryEntry] = []
        for code_i, run_id, feat_json, out_json in rows:
            feats = json.loads(feat_json or "{}")
            sim = _cosine(q, feats)
            scored.append(
                MemoryEntry(
                    code=code_i,
                    run_id=run_id or "",
                    outcome_metrics=json.loads(out_json or "{}"),
                    similarity=sim,
                )
            )
        scored.sort(key=lambda e: e.similarity, reverse=True)
        return scored[: max(0, int(k))]
