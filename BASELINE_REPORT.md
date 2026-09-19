# BASELINE LOCK REPORT — `baseline-pre-amfrs`

**Date:** 2026-09-19  
**Purpose:** Freeze the paper-faithful EvoNav Algorithm 1 replication (CrowdNav++ fork) before any AMFRS work. All future AMFRS diffs should be measured against tag `baseline-pre-amfrs`.

**Scope of this pass:** Stage I Score1 correctness fixes, repo cleanup, fidelity audit, tests, short live smoke, commit + annotated tag. **No AMFRS code.**

---

## Verdict (three categories — do not blur)

### Byte-faithful to the paper’s stated Algorithm 1 design

| Area | Evidence |
|------|----------|
| Stage I population / gens | `N=8`, `G1=10` — `StageIConfig` (`evolver.py` 74–79), `EvoNavRunConfig` (`pipeline.py` 62–63), `configs/paper_scale.yaml` 19–20 |
| Stage I operators | 2 crossover / 4 mutation / 2 random — `StageIConfig` (`evolver.py` 77–79) |
| Stage I Score1 | Spearman(rules, cumulative recomputed reward) over dataset — `scoring.py` 1–14, 171–186 |
| Stage II algo / rounds / eval | A2C, `G2=16`, `E2=50`, `T_short=100` — `Stage2Config` (`stage2.py` 64–71) |
| Stage III algo / rounds / eval / H set | PPO, `G3=3`, `E3=500`, H∈{5,10,15,20} — `Stage3Config` (`stage3.py` 70–101) |
| Paper-scale budgets | YAML forces `K2=8000`, `K3=1e7` — `configs/paper_scale.yaml` 26–36 |
| Elitism / global_best | Stage I `global_best` + pipeline handoff; Stage II/III `_inject_elite` — see §Elitism |
| Fail-closed `--llm seed` | `run_evonav.py` 161–174; `run_evonav_paper_scale.py` 130–135 |
| Sandbox bans `__import__` | AST + restricted builtins — see §Sandbox |
| GST regime asserts | Wired via `apply_regime_to_config` / direct asserts — see §GST |
| Dynamic `num_processes` | `None` → `min(16, cpu_count-1)` — `parallelism.py` 15–37; Stage2/3 configs |

### Deliberate documented deviations

| Deviation | Paper | This baseline | Reasoning (recorded in code) |
|-----------|-------|---------------|------------------------------|
| **K3 default** | Table 6: `1e7` | `STAGE3_STEPS = 5e5` (`stage3.py` 67–69) | Hardware / local iteration; paper value in `STAGE3_PAPER_STEPS` and `paper_scale.yaml` |
| **K2 default (non-paper entry)** | Table 5: `8000` | `50_000` (`pipeline.py` 65–69, `stage2.py` 69) | Short K2 ranks rewards poorly in practice; paper_scale still forces 8000 |
| **Obs / GST for Stage II–III** | CrowdNav++ predictive | Default `predict_method=inferred` + GST checkpoint per regime (`AUDIT.md` §8.2 choice (a)) | Required for Table-1 architecture parity; Stage I collector may use `predict_method=none` |
| **First validation regime** | Paper may report with/without rand | Default `without_random` (`pipeline.py` 84–85, AUDIT §8.1) | Lower variance for first honest pass |
| **Score1 pad / nav-length** | Underspecified for padding | Freeze cumulative after real length; `nav_length = f+1` (`scoring.py` 10–13, 98–111, 148–149) | Correctness vs naive pad / final-length leak (Part A of this lock) |
| **`--easy` / `--fast`** | N/A | Reduced humans / stubs / smoke Score1 | Debugging only — not paper claims |

### Open bugs / known gaps (not fixed in this lock)

| Item | Status |
|------|--------|
| Primary science run `run_scaled_h5_gst` still far below GST/ORCA SR (see `PROJECT_TECHNICAL_REPORT.md`) | **Empirical gap**, not a newly discovered Score1 pad bug |
| AMFRS / surrogate / active learning | **Not implemented** (by design for this baseline) |
| Standalone `train.py --algo a2c` not used by Algorithm 1 | Documented out-of-scope (`AUDIT.md`) |

Part A Score1 bugs (pad inflation, final-length leak, silent NaN skip) are **closed** in this baseline.

---

## Part A — Score1 fixes (re-verified)

### 1. Cumulative reward freeze on padding

`_cumulative_reward` only calls `reward_fn.compute()` while `f < traj.length`; afterwards repeats the last real cumulative (`scoring.py` 80–112).

### 2. Per-frame `nav_length = f + 1`

`_scenario_frame_correlations` passes `nav_so_far = float(f + 1)` into `rule_preference_score` (`scoring.py` 148–154) — no leak of final episode length into early frames.

### 3. `degenerate_fraction` on Score1 output

`Score1Result` carries `score`, `degenerate_fraction`, `n_pairs`, `n_degenerate` (`scoring.py` 40–77). NaN Spearman frames increment `n_degenerate` (`scoring.py` 162–167). `StageIEvolver.score_population` writes metadata and warns at ≥50% degeneracy (`evolver.py` ~460–470).

### 4. Regression tests

`crowd_nav/reward_search/tests/test_score1_baseline_lock.py` — padding terminal reward, frame-varying Success rule ranks, degeneracy fraction.

---

## Part B — Cleanup summary

### Caches

Removed `__pycache__`, `.pytest_cache`, `*.pyc` across the repo (gitignore already excludes them).

### Results archive (moved, not deleted)

| Path | Role |
|------|------|
| `evonav_env/results/archive/run_5h_easy/` | Non-GST easy profile complete run |
| `evonav_env/results/archive/run_1to2h/` | Severely reduced-budget GST-ish run |
| `evonav_env/results/archive/run_1to1p5h_easy/` | Incomplete |
| `evonav_env/results/archive/paper_scale/` | Stub / dry timings |
| `evonav_env/results/archive/_paper_seed_dry/` | Seed dry-run |
| `evonav_env/results/archive/_seed_fast_ok/` | Fast wiring smoke |
| `evonav_env/results/archive/_prompt_fix_smoke/` | Prompt/sandbox smoke |

**Kept live:** `evonav_env/results/run_scaled_h5_gst/` (primary documented science run).  
**Gitignore choice:** entire `evonav_env/results/` remains gitignored (including archive) — local paper trail only; see `results/archive/README.md`.

### Dead code

| Item | Decision |
|------|----------|
| `rl/vec_env/envs.py` | **Absent** — active path `rl/networks/envs.py` (comment updated ~115–117) |
| `run_evonav_system.ps1` | **Absent** — no leftover references |
| `LLMClient.generate(n=)` | **Kept** — still called from `tests/test_evolver.py` |

### `.gitignore` cross-check

Excludes `evonav_env/results/`, `evonav_env/data/stage1_dataset/`, `*.pt` / `*.pth`.  
`git ls-files` shows **no** tracked `.pt`, results trees, or stage1 dataset blobs.

### Credential re-scan

Tracked tree: only placeholders / examples (`gsk_REPLACE…`, docs, unit-test fakes). Real `groq_keys.json` is gitignored. No live `gsk_` / `sk-` / `AKIA` secrets in tracked files.

---

## Part C — Fidelity detail (Algorithm 1)

### Stage I (LLM evolution + Score1)

| Knob | Paper (Tables 3–5) | Code default | Location |
|------|--------------------|--------------|----------|
| N | 8 | 8 | `evolver.py` 75; `pipeline.py` 62 |
| G1 | 10 | 10 | `evolver.py` 76; `pipeline.py` 63 |
| Score1 | Spearman vs rules | dataset mode | `scoring.py`; `score1_mode="dataset"` |
| Elitism | keep best | `global_best` + `_include_global_best` | `evolver.py` 125, 719–755; `pipeline.py` 187–197, 296–299 |

### Stage II (A2C proxy)

| Knob | Paper Table 5 | Code default | Notes |
|------|---------------|--------------|-------|
| Algo | A2C | `"a2c"` | `stage2.py` 72 |
| G2 | 16 | 16 | `stage2.py` 68 |
| E2 | 50 | 50 | `stage2.py` 70 |
| K2 | 8000 | **50_000** (default) / **8000** (paper_scale) | deliberate deviation |
| T_short | 100 | 100 | `stage2.py` 71 |
| Elite protect | — | `protect_elite_refine=True`, `_inject_elite` | `stage2.py` 87, 975+ |

### Stage III (PPO + H-sweep)

| Knob | Paper Table 6 | Code default | Notes |
|------|---------------|--------------|-------|
| Algo | PPO | `"ppo"` | `stage3.py` 84 |
| G3 | 3 | 3 | `stage3.py` 78 |
| E3 | 500 | 500 | `stage3.py` 81 |
| K3 | 1e7 | **5e5** default / **1e7** paper_scale | deliberate hardware scale-down |
| H-sweep | {5,10,15,20} | `STAGE3_HUMAN_COUNTS` | each H = full E3 eval — documented in `README_EVONAV.md` |

### Elitism / `global_best` (re-verified)

- Stage I tracks `self.global_best` and updates when a generation improves (`evolver.py` 719–755).
- Test: `test_global_best_survives_a_regressive_generation` (`test_evolver.py` 129+).
- Pipeline injects global best into Stage II population (`pipeline.py` 187–197, 296–299).
- Stage II/III: `_inject_elite` + `protect_elite_refine` (`stage2.py` 975+, `stage3.py` 653+); tests in `test_stage2.py`.

### `num_processes` (re-verified)

`resolve_num_processes(None)` → `default_num_processes` = `min(16, max(1, cpu_count-1))` (`parallelism.py` 15–37). Stage2/3 configs default `num_processes: Optional[int] = None` — **not** hardcoded to 1.

### Fail-closed `--llm seed` (re-verified)

- `scripts/run_evonav.py` 161–174  
- `scripts/run_evonav_paper_scale.py` (~130–135)  
- Tests: `test_seed_llm_gate.py`, `test_paper_scale.py`

### Sandbox (re-verified)

- `__import__` forbidden in AST policy messages (`sandbox/ast_policy.py` 26, 40) and stripped from allowed builtins (`sandbox/config.py` 49).
- Stateful contract: `compute_reward(state, memory)` with runner-owned dict cleared on reset — documented in `prompts.py` (header + D1 body) and `sandbox/runtime.py` 6–7, 243–265. **No drift** between prompt contract and runtime.

### GST regime assertions (four entry points)

| Entry | Mechanism | Location |
|-------|-----------|----------|
| Collect Stage I dataset | `apply_regime_to_config` → `assert_gst_matches_regime` | `collect_stage1_dataset.py` 64–72; `regime.py` 174–208 |
| Stage II train config | `apply_regime_to_config` | `stage2.py` 281–286 |
| Stage III train config | `apply_regime_to_config` | `stage3.py` 307–312 |
| Final / proxy eval | direct `assert_gst_matches_regime` | `reporting.py` 278+, 410+ |

Also: pipeline startup assert (`pipeline.py` 240–253).

### H-sweep cost

Implemented in `RealPolicyTrainer.evaluate_at_human_counts` (`stage3.py` 533–564): **one full `evaluate_proxy_policy` pass per H** at `E3` episodes. Documented in `evonav_env/README_EVONAV.md` §“Stage III H-sweep”.

---

## Part D — Verification

### Fast test suite

```text
pytest crowd_nav/reward_search/tests -m "not slow"
→ 114 passed, 1 deselected (21.14s)  # 2026-09-19 baseline lock
```

### Live smoke

Short non-`--fast` pipeline (2026-09-19):

```text
python scripts/run_evonav.py --llm ollama --llm-model qwen3.5:4b --device cpu \
  --stage1-population 2 --stage1-generations 1 --stage2-rounds 1 --stage3-rounds 1 \
  --stage2-stub --stage3-stub --no-h-sweep --score1 dataset \
  --stage1-dataset data/stage1_dataset --predict-method none \
  --output-dir results/_baseline_lock_smoke
→ Done. Final candidate: ini_0000
  Stage I Gen0 accepted 2/2 via real Ollama + sandbox; Score1 on 100 scenarios
  (best Score1 ≈ 0.631). Groq returned 403 Forbidden; Ollama used instead.
```

Artifacts under `results/_baseline_lock_smoke/` (gitignored).

### Tag

```bash
git tag -a baseline-pre-amfrs -m "Paper-faithful EvoNav baseline before AMFRS (Score1 pad/nav/degeneracy lock)"
```

Diff future work with: `git diff baseline-pre-amfrs`.
