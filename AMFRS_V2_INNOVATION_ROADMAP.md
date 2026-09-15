# AMFRS v2 — Innovation Roadmap (Axes 1–5)

**Status:** Draft roadmap for the *actual thesis contribution*, independent of EvoNav fidelity.
**Supersedes:** `AMFRS_STAGE1_STAGE2_STAGE3_MASTER_PLAN.md` §12 Phase 5 (T15–T19). Phases 0–4 of that
document (trustworthy Stage I, stable Stage II accept/reject gate, Stage III finalist tournament,
shared harness) are **kept as-is** and treated as a frozen, working baseline — see "Non-negotiables"
below.
**Audience:** This file is written to be handed to Cursor as an implementation brief. Every task
names the exact file to create/touch, the exact class/function signature expected, its
dependencies, and how to validate it. Task IDs are `A<axis>.<n>` and do not reuse the old `T#`
numbering, to make it unambiguous that this is a new track of work.
**Ground truth this roadmap was built against (verified by direct code inspection, 2026-09-15):**
`amfrs_env/crowd_nav/reward_search/{pipeline,evolver,stage2,stage3,selection,state,sandbox/*,
llm,dataset,scoring,rules,reporting,refine_harness}.py`. Class/function names quoted below are the
real ones in the repo, not paraphrases — check them against the file before writing code that
assumes a different name.

---

## 0. Why this document exists

The existing pipeline (`AMFRSPipeline` in `pipeline.py`) is, by the project's own master plan
(§ Executive Summary), "a faithful EvoNav Algorithm 1 replication renamed to the AMFRS namespace."
That is a legitimate and necessary artifact — **it stays exactly as it is, untouched**, because it
is the baseline your thesis compares against. Everything in this document is *new, additive* code
that produces a second, genuinely different pipeline (`AMFRS2Pipeline`) built from the same
low-level primitives (`RewardCandidate`, `RewardValidator`, `ProxyMetrics`, `RewardState`) but with
a different search algorithm, a different selection philosophy, a different LLM-driving strategy,
a different notion of "good," and an extra safety gate that EvoNav never had.

Concretely, this document expands the five brainstorm axes into buildable specs:

| Axis | One-line pitch | Core novelty vs EvoNav/AMFRS-v1 |
|---|---|---|
| 1 | Real adaptive multi-fidelity search (successive halving + bandit budget allocation) | Fixed N/G1/G2/G3 budgets → dynamic, cost-aware promotion |
| 2 | Quality-Diversity archive (MAP-Elites) instead of a single scalar/Pareto winner | One "best" reward → an illuminated archive of diverse good rewards |
| 3 | Smarter LLM search: primitive DSL, semantic crossover, ensemble critique, retrieval memory | LLM writes raw Python each time from a short reflection → LLM composes from a primitive library and remembers every past run |
| 4 | Robustness/generalization as a first-class fitness axis | H-sweep only varies crowd *count* → also varies crowd *policy* + adversarial scenarios |
| 5 | Static/symbolic safety gate before any simulation is run | Sandbox only checks *code safety* → also checks *reward sanity* (boundedness, monotonicity, human-blindness) for free, before spending any GPU time |

---

## 1. Non-negotiables (do not break these while building v2)

1. **Do not modify** `pipeline.py`, `AMFRSPipeline`, `AMFRSRunConfig`, `run_amfrs.py`,
   `run_amfrs_paper_scale.py`. This is the frozen EvoNav-faithful baseline used for thesis
   comparison tables. If a shared helper needs a behavior change, fork it into the new `amfrs/`
   package instead of editing the original.
2. **Reuse, don't reimplement**, these stable interfaces:
   - `RewardCandidate` (`evolver.py`) — the unit of evolution stays the same dataclass; new axes
     attach data via its existing `metadata: dict` field (e.g. `metadata["behavior_descriptor"]`,
     `metadata["fidelity_level"]`, `metadata["static_report"]`). Do not add new positional fields
     to the dataclass itself — every existing call site constructs it positionally/by-kwarg and a
     new required field breaks all of them.
   - `RewardValidator` / `SandboxedReward` (`sandbox/`) — all new candidate code still goes through
     this before anything else touches it.
   - `ProxyMetrics` (`stage2.py`) — already a full vector (`sr, cr, tr, nt, pl, itr, sd`
     + `n_seeds` + `metric_std`). Axis 2's behavior descriptors and axis 4's robustness score are
     *derived from* this, not a competing metrics object.
   - `RewardState` / `RewardFunction` (`state.py`) — the `(state, memory)` reward signature is
     already stateful-capable (`memory` dict persists across steps within an episode, cleared on
     `reset()`). Axis 3's primitive DSL builds on top of this, not around it.
3. **Keep everything CPU-testable without torch/gym wherever the logic doesn't inherently need a
   trained policy.** Axis 1's scheduler, axis 2's archive, axis 3's memory store and primitive
   library, and axis 5's static gate are pure Python/NumPy — they must import cleanly with no
   `torch`/`gym` at module scope (this project has already been bitten once by non-deferred torch
   imports in `stage2.py`/`stage3.py`/`reporting.py` making 8 test files uncollectable without a
   full environment — do not repeat that mistake in `amfrs/`).
4. Every new module gets tests in `crowd_nav/reward_search/amfrs/tests/` mirroring the existing
   test style (plain `pytest`, no fixtures beyond `tmp_path`/`monkeypatch`, scripted/stub LLM
   clients instead of real network calls — see `ScriptedLLMClient` in `llm.py`).

---

## 2. New package layout

```
amfrs_env/crowd_nav/reward_search/amfrs/          # ALL new v2 code lives here — nothing above touched
    __init__.py
    fidelity.py            # A1 — FidelityLevel / FidelityLadder
    halving.py              # A1 — SuccessiveHalvingScheduler (ASHA-style)
    bandit.py                # A1 — UCB1 / Thompson budget allocator
    behavior.py              # A2 — behavior_descriptor(), descriptor space config
    archive.py                # A2 — EliteGrid (MAP-Elites)
    primitives.py             # A3 — reward primitive function registry
    crossover.py               # A3 — semantic crossover prompt builder
    ensemble_critique.py        # A3 — multi-LLM critique agreement gate
    memory.py                    # A3 — cross-run retrieval-augmented candidate store
    robustness.py                 # A4 — multi-policy sweep + adversarial scenario runner
    scenarios/                     # A4 — YAML scenario definitions
        narrow_corridor.yaml
        dense_crossing.yaml
        bottleneck_swap.yaml
    static_gate.py                  # A5 — symbolic/static reward sanity checker
    config.py                        # AMFRS2RunConfig (new, does NOT extend AMFRSRunConfig)
    pipeline2.py                      # AMFRS2Pipeline — orchestrates all axes
    tests/
        __init__.py
        conftest.py
        test_fidelity.py
        test_halving.py
        test_bandit.py
        test_behavior.py
        test_archive.py
        test_primitives.py
        test_crossover.py
        test_ensemble_critique.py
        test_memory.py
        test_robustness.py
        test_static_gate.py
        test_pipeline2_smoke.py

amfrs_env/scripts/
    run_amfrs2.py             # new entry point, mirrors run_amfrs.py's CLI conventions
    plot_amfrs2_archive.py    # A2 — render the MAP-Elites grid as a figure for the thesis
```

Rationale for a fully separate package instead of editing `pipeline.py` in place: it lets you run
**both** pipelines back-to-back on the same seed/dataset for a clean baseline-vs-contribution
comparison table, it makes `git diff` / thesis appendix ("here is exactly the code I wrote") trivial,
and it means a bug in new code can never regress the already-working, already-tested baseline.

---

## 3. Cross-cutting architecture decisions

State these explicitly now so Cursor doesn't have to guess mid-implementation.

**D-A — New pipeline vs. extend existing.**
Recommended: **new** `AMFRS2Pipeline` (Decision, see §1.1 above). Rejected alternative: adding
`if self.config.use_amfrs2:` branches inside `AMFRSPipeline.run()` — rejected because it would mix
two philosophies in one 800-line method and put the frozen baseline at risk.

**D-B — Fidelity ladder rungs (Axis 1).**
Recommended 4 rungs, reusing existing evaluators, just re-budgeted and re-scheduled instead of
fixed-count:
| Rung | Evaluator | Existing code reused | Typical cost |
|---|---|---|---|
| F0 | Score1 analytical scoring | `scoring.score1_for_dataset` / `Score1Report` | ~ms, CPU |
| F1 | Short proxy A2C | `stage2.Stage2Runner` with `train_env_steps` ≈ K2/4 | seconds–low minutes, CPU/GPU |
| F2 | Full proxy A2C | `stage2.Stage2Runner` at default `K2=8000`+ | minutes |
| F3 | Full PPO | `stage3.Stage3Runner` | hours (GPU) |
Rejected alternative: inventing a brand-new "fidelity 0" non-learning controller reward-blind to
compute a metric — rejected because it doesn't actually evaluate a *reward function* (ORCA/SF don't
consume `compute_reward` at all), so it wouldn't rank candidates on the thing you're searching over.

**D-C — Behavior descriptor space (Axis 2).**
Recommended: start with a **2D grid** so it's visualizable and cheap to keep well-populated with a
small population (8–16 candidates/generation on a 4GB GPU cannot fill a high-dimensional grid).
- BD-x: **path efficiency** = `PL / (straight_line_dist)` from `ProxyMetrics.pl`, clipped/binned.
- BD-y: **social margin** = `ProxyMetrics.sd` (mean min-distance during Danger frames), binned.
Extend to 3D (add `ITR`-derived "caution" axis) only after F2-rung evaluations are cheap/plentiful
enough (measure `n_seeds` throughput first — see §5's validation criteria).

**D-D — Embedding for retrieval memory (Axis 3).**
Recommended: a **local, dependency-free, deterministic** feature vector — bag of AST node-type
counts + a handful of hand-picked scalar features (number of `if` branches, whether `state.humans`
is referenced, coefficient magnitudes found via `ast.Constant` scan) — *not* a call to Groq/Ollama's
embedding endpoint. Reason: embeddings-via-API add a network dependency and nondeterminism to
something that should be reproducible offline and cheap to run thousands of times across a
multi-day paper-scale run. Revisit only if the AST-bag approach is empirically shown to retrieve
poor neighbors (Task A3.4 has an explicit validation step for this).

**D-E — Where the static gate sits (Axis 5).**
Recommended: immediately after `RewardValidator.validate_code` succeeds and *before* F0 Score1
scoring. It is a cheap additional filter on top of the existing AST/sandbox check, not a
replacement for it.

**D-F — Baseline comparison protocol.**
Recommended: same `stage1_dataset_path`, same `seed`, same `randomization_regime`, same LLM
provider/model for both `run_amfrs.py` (baseline) and `run_amfrs2.py` (v2) runs, output to sibling
directories (`results/baseline_seedN/`, `results/amfrs2_seedN/`), compared via a new
`scripts/compare_amfrs_runs.py` (§8) that both pipelines' `AMFRSArtifacts`/`AMFRS2Artifacts`
manifests can feed into.

---

## 4. Axis 1 — Adaptive Multi-Fidelity Search

### 4.1 Problem with the current pipeline

`AMFRSPipeline.run()` uses **fixed** budgets at every stage regardless of how a candidate is
performing: `stage1_population=8` always runs `stage1_generations=10` full generations;
**every** surviving candidate gets the **same** `stage2_rounds=16` × `stage2_train_steps` proxy
training; **every** finalist gets the **same** `stage3_train_steps`. There is no mechanism that
says "this candidate is clearly bad after 2 rounds, stop spending budget on it" or "this candidate
is a clear winner, give it more budget than its peers." That is what "adaptive" in AMFRS is
supposed to mean, and it currently does not exist anywhere in the codebase (confirmed by reading
`stage2.py`/`stage3.py` — the round loops are plain `for round_idx in range(config.stage2_rounds)`
with no early-stop or budget-reallocation logic).

### 4.2 Task table

| ID | Task | New file | Depends on | Expected result | Validation |
|---|---|---|---|---|---|
| A1.1 | Define `FidelityLevel` / `FidelityLadder` | `amfrs/fidelity.py` | — | Typed description of each rung (name, evaluator callable, nominal cost) | `test_fidelity.py`: ladder iterates in order, cost strictly increasing |
| A1.2 | Implement `SuccessiveHalvingScheduler` | `amfrs/halving.py` | A1.1 | Given a population and a ladder, returns which candidates get promoted each rung | `test_halving.py`: synthetic scores, assert top `1/eta` fraction promoted, exact-tie handling deterministic |
| A1.3 | Implement `UCB1Allocator` | `amfrs/bandit.py` | — | Given running per-candidate (mean reward, pull count, cost), returns next candidate(s) to fund | `test_bandit.py`: classic UCB1 unit tests (arm with fewer pulls gets explored, best-mean arm dominates after many pulls) |
| A1.4 | Wire F0→F1→F2→F3 evaluators as thin adapters | `amfrs/fidelity.py` | A1.1, existing `scoring.py`/`stage2.py`/`stage3.py` | `evaluate_at(level, candidate) -> FidelityResult(metric, cost, raw_metrics)` | Integration test with `stage2_use_stub=True` / `--fast`-style stub trainers, no GPU required |
| A1.5 | `AMFRS2Pipeline` uses halving+bandit instead of fixed loops | `amfrs/pipeline2.py` | A1.1–A1.4 | End-to-end run promotes a shrinking population up the ladder, logs promotion decisions to manifest | `test_pipeline2_smoke.py` fast profile (see §7) |
| A1.6 | Cost/quality report vs. baseline | `scripts/plot_amfrs2_archive.py` (shared plotting script, see A2) + `compare_amfrs_runs.py` | A1.5, one baseline run, one v2 run | A plot: cumulative GPU-seconds spent vs. best-found `navigation_scalar` over time, for both pipelines | Manual inspection; this *is* one of your thesis figures |

### 4.3 Concrete specs

```python
# amfrs/fidelity.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Dict

@dataclass(frozen=True)
class FidelityResult:
    metric: float          # navigation_scalar-comparable score at this rung
    cost: float             # nominal cost units spent (see cost model below)
    raw_metrics: Dict[str, Any]  # e.g. ProxyMetrics.as_dict() or Score1Report summary

@dataclass(frozen=True)
class FidelityLevel:
    name: str                                   # "F0_score1" | "F1_short_a2c" | ...
    cost_units: float                           # nominal relative cost (see below)
    evaluate: Callable[["RewardCandidate"], FidelityResult]

# Cost units are NOT wall-clock seconds (too hardware-dependent to hardcode). Use a simple
# proportional model seeded from the run config once at pipeline construction time, e.g.:
#   F0 = 1  (Score1 on the holdout dataset)
#   F1 = stage2_train_steps_short / stage2_train_steps_short   -> normalize to 10
#   F2 = full stage2_train_steps / stage2_train_steps_short    -> e.g. 40
#   F3 = stage3_train_steps / stage2_train_steps               -> e.g. 400
# Exact ratios should be measured once (log wall-clock per rung on your RTX 3050) and hardcoded
# as defaults in AMFRS2RunConfig, not recomputed live — the scheduler needs a stable cost model.

class FidelityLadder:
    def __init__(self, levels: list[FidelityLevel]): ...
    def __iter__(self): ...          # yields levels in ascending cost order
    def next_level(self, current: FidelityLevel) -> FidelityLevel | None: ...
```

```python
# amfrs/halving.py
from dataclasses import dataclass
from typing import Sequence
from crowd_nav.reward_search.evolver import RewardCandidate

@dataclass(frozen=True)
class HalvingConfig:
    eta: int = 3                 # promote top 1/eta fraction each rung
    min_survivors: int = 1        # never promote fewer than this (keeps at least one finalist)

class SuccessiveHalvingScheduler:
    def __init__(self, ladder: "FidelityLadder", config: HalvingConfig = HalvingConfig()): ...

    def run(
        self,
        population: Sequence[RewardCandidate],
        *,
        on_rung_complete: Callable[[str, list[RewardCandidate]], None] | None = None,
    ) -> list[RewardCandidate]:
        """
        Evaluate `population` at the ladder's first rung, keep top ceil(N/eta), evaluate those at
        the next rung, repeat until the ladder is exhausted or min_survivors reached. Each
        candidate's metadata["fidelity_history"] accumulates a list of
        {"level": name, "metric": float, "cost": float} entries — this is what feeds axis 1's
        cost-vs-quality plot (A1.6) and gives axis 3's retrieval memory (A3.4) real provenance.
        Ties at the promotion cutoff: keep the candidate with the lower candidate_id (deterministic,
        matches existing tie-break style in selection.py's is_same_genome dedup).
        """
```

```python
# amfrs/bandit.py
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ArmStats:
    pulls: int = 0
    total_reward: float = 0.0
    total_cost: float = 0.0

    @property
    def mean_reward(self) -> float: ...
    @property
    def mean_reward_per_cost(self) -> float: ...

class UCB1Allocator:
    """
    Standard UCB1 over candidate_id arms, reward = observed fidelity-rung metric normalized to
    [0,1] per rung (normalize using the population's min/max at that rung — do NOT compare raw
    metrics across rungs, they are not on the same scale). Cost-aware variant: score arms by
    mean_reward_per_cost instead of mean_reward when `cost_aware=True` (default True — you are
    explicitly optimizing under a GPU-hour budget).
    """
    def __init__(self, exploration_c: float = 2.0, cost_aware: bool = True): ...
    def record(self, candidate_id: str, reward: float, cost: float) -> None: ...
    def select_next(self, candidate_ids: list[str], k: int) -> list[str]: ...
```

**Integration point in `AMFRS2Pipeline`:** replace the place where the old pipeline would do
"take all Stage-I survivors, run every one through 16 fixed Stage-II rounds" with:
`SuccessiveHalvingScheduler(ladder).run(stage1_population)`. The bandit allocator is a second,
optional layer *within* a single rung when you want to allocate a shared round-budget across
still-alive candidates non-uniformly (e.g. within F2, don't split K2 evenly across all survivors —
let UCB1 decide who gets the next chunk of steps). Ship halving first (A1.1–A1.2, simpler, bigger
win), add the bandit layer (A1.3) as a refinement once halving is validated.

## 5. Axis 2 — Quality-Diversity Archive (MAP-Elites)

### 5.1 Problem with the current pipeline

`selection.py` today reduces every candidate to a single scalar
(`navigation_scalar = w_sr·SR - w_cr·CR - w_tr·TR - w_itr·ITR + w_sd·SD`) and keeps only the
top-k by that scalar (`select_top_k_finalists`). Two reward functions that produce very different
*styles* of navigation (fast-but-close-to-humans vs. slow-but-cautious) but land on similar scalar
values get treated as redundant, and the one that's marginally lower just gets discarded —
there's no notion of "this is a different, still-valuable point in the design space." A
Pareto front (the T17 idea from the old master plan) is a step up but still only tracks the
non-dominated *frontier*; it doesn't actively search for a spread of qualitatively different good
solutions the way MAP-Elites does. MAP-Elites is the stronger, more clearly "novel algorithm"
choice for a thesis contribution.

### 5.2 Task table

| ID | Task | New file | Depends on | Expected result | Validation |
|---|---|---|---|---|---|
| A2.1 | Define behavior descriptor function + binning | `amfrs/behavior.py` | `ProxyMetrics` (exists) | `compute_behavior_descriptor(metrics) -> tuple[int,int]` cell coords | `test_behavior.py`: known metric values map to expected/stable bins; out-of-range values clip instead of erroring |
| A2.2 | Implement `EliteGrid` | `amfrs/archive.py` | A2.1 | Insert/replace-if-better-in-cell, `coverage()`, `qd_score()` (sum of elite fitnesses), `all_elites()`, JSON (de)serialization | `test_archive.py`: insert lower-fitness into occupied cell is a no-op; insert higher-fitness replaces; empty-cell insert always accepted |
| A2.3 | AMFRS2 evolutionary loop samples parents from the archive | `amfrs/pipeline2.py` (or a small `amfrs/illumination.py` if it grows) | A2.2, `evolver.StageIEvolver` internals as reference (do not subclass/monkeypatch it — reimplement the loop shape in the new package) | Each generation: sample K elites (weighted toward less-crowded regions or uniformly), mutate/crossover, insert children back into the grid | `test_pipeline2_smoke.py`: after N generations, `archive.coverage()` is non-decreasing generation over generation |
| A2.4 | Persist + visualize the grid | `amfrs/archive.py` (`to_json`/`from_json`) + `scripts/plot_amfrs2_archive.py` | A2.2 | Heatmap PNG: x=path-efficiency bin, y=social-margin bin, color=fitness, annotated with which `candidate_id` occupies each cell | Manual inspection — this is a thesis figure |
| A2.5 | Attach descriptor to `RewardCandidate.metadata` at the fidelity rung where it's first measurable | `amfrs/pipeline2.py` | A2.1, A1.4 | `candidate.metadata["behavior_descriptor"] = (bx, by)` set right after any F1+ evaluation (F0/Score1 alone doesn't give you `PL`/`SD` reliably — check `Score1Report` fields before deciding whether F0-level descriptors are meaningful; if not, only insert into the archive from F1 onward) | Unit test asserts descriptor is `None` after F0-only evaluation and populated after F1 |

### 5.3 Concrete specs

```python
# amfrs/behavior.py
from dataclasses import dataclass

@dataclass(frozen=True)
class DescriptorSpaceConfig:
    # Bin edges, not counts — explicit edges make the grid reproducible across runs/config
    # changes, and make the plotted axis labels exact.
    path_efficiency_edges: tuple[float, ...] = (1.0, 1.2, 1.5, 2.0, 3.0)   # PL / straight_line_dist
    social_margin_edges: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)    # SD, meters

def compute_behavior_descriptor(
    metrics: dict,  # ProxyMetrics.as_dict() shape: {"PL":.., "SD":.., ...}
    straight_line_dist: float,
    *,
    config: DescriptorSpaceConfig = DescriptorSpaceConfig(),
) -> tuple[int, int]:
    """
    Returns (bx, by) grid cell indices. Values outside the configured edges clip to the nearest
    edge bin (do not raise) — a candidate with an extreme style is still a valid archive citizen,
    it just lands in a boundary cell.
    """
```

```python
# amfrs/archive.py
from dataclasses import dataclass, field
from typing import Optional
from crowd_nav.reward_search.evolver import RewardCandidate

class EliteGrid:
    def __init__(self, shape: tuple[int, int]): ...

    def try_insert(self, candidate: RewardCandidate, cell: tuple[int, int], fitness: float) -> bool:
        """Returns True if the candidate became (or replaced) the elite of `cell`."""

    def get(self, cell: tuple[int, int]) -> Optional[RewardCandidate]: ...
    def all_elites(self) -> list[RewardCandidate]: ...
    def coverage(self) -> float: """Fraction of cells occupied, in [0,1]."""
    def qd_score(self) -> float: """Sum of fitness over all occupied cells — the standard QD metric."""
    def to_json(self, path: str) -> None: ...
    @classmethod
    def from_json(cls, path: str) -> "EliteGrid": ...
```

**Selection-mode flag:** expose `AMFRS2RunConfig.selection_mode: Literal["map_elites","scalar"]` so
you can run an ablation (archive search vs. plain scalar search, everything else identical) — this
ablation is exactly the kind of controlled comparison a thesis committee will ask for.

**Naming note:** call the fitness used *inside a cell* the same `navigation_scalar` from
`selection.py` (don't invent a second fitness function) — the only thing MAP-Elites changes is
*which candidates compete against which* (same-cell only), not *how a single candidate's quality is
scored*.

## 6. Axis 3 — Smarter LLM-Driven Generation

### 6.1 Problem with the current pipeline

`evolver.py`'s `_try_mutate`/`_try_crossover` ask the LLM to emit an entire `compute_reward`
function from scratch every time, guided only by a short textual "reflection"
(`_build_reflection`) about the previous generation's ranking. There is no structured vocabulary
of reward *components*, no cross-run memory (every `run_amfrs.py` invocation starts from zero
knowledge of every previous run you've ever done), and no mechanism to catch a case where two
independent critiques agree on the same flaw vs. one LLM call hallucinating a plausible-sounding
but wrong critique.

### 6.2 Task table

| ID | Task | New file | Depends on | Expected result | Validation |
|---|---|---|---|---|---|
| A3.1 | Reward primitive registry | `amfrs/primitives.py` | `RewardState` (exists) | ~10–15 named pure functions `f(state: RewardState) -> float`, e.g. `goal_progress(state)`, `discomfort_penalty(state)`, `collision_indicator(state)`, `time_penalty(state)`, `jerk_penalty(state, memory)` | `test_primitives.py`: each primitive is finite for a battery of synthetic `RewardState` fixtures, including edge cases (zero humans, human exactly at `discomfort_dist`) |
| A3.2 | Inject primitives into the sandbox namespace | `sandbox/config.py` + `sandbox/runtime.py` (extend, don't fork the sandbox) | A3.1 | New `ALLOWED_MODULES`-style constant `INJECTED_PRIMITIVES: dict[str, Callable]`; `compile_compute_reward` injects them into `namespace` alongside `math`, same pattern already used for `math` | Existing sandbox tests still pass unmodified; new test confirms `primitives.goal_progress(state)` is callable from candidate code and still blocked from calling anything outside the registry |
| A3.3 | Primitive-aware prompt templates | `amfrs/crossover.py` (+ new prompt strings, kept in `amfrs/`, not appended to `prompts.py` to avoid touching the baseline's prompt file) | A3.1 | D.1/D.2-equivalent prompts that list available primitives by name+signature and instruct the LLM to prefer composing them with explicit numeric weights over inventing new low-level math | Manual read-through + a scripted-LLM test asserting the prompt string contains every primitive name currently registered (prevents silent drift when a primitive is added but the prompt isn't regenerated) |
| A3.4 | Semantic crossover | `amfrs/crossover.py` | A3.1–A3.3, `evolver.RewardCandidate` | `build_semantic_crossover_prompt(parent_a, parent_b, diagnostics_a, diagnostics_b) -> str` that explicitly asks the LLM to name which primitive/term from each parent to keep | Golden-file test: prompt contains both parents' code and both diagnostic summaries |
| A3.5 | Ensemble critique agreement gate | `amfrs/ensemble_critique.py` | `llm.LLMClient` ABC (exists — works with any of `GroqLLMClient`/`OllamaLLMClient`/`ScriptedLLMClient`) | `critique_agrees(candidate, diagnostics, client_a, client_b) -> AgreementReport(agreed: bool, shared_terms: list[str])`; only apply a refinement automatically when `agreed=True`, otherwise log and keep the parent unchanged (mirrors existing accept/reject philosophy in `refine_harness.py`, just gating on *critique* agreement instead of *post-hoc metric* comparison) | `test_ensemble_critique.py` using two `ScriptedLLMClient`s: identical canned critiques → agree; contradictory canned critiques → disagree, refinement skipped |
| A3.6 | Cross-run retrieval memory | `amfrs/memory.py` | none (pure Python + `sqlite3`, stdlib only — no new dependency) | `RewardMemory` class: `add(candidate, run_id, outcome_metrics)`, `query_similar(code, k) -> list[MemoryEntry]` using the AST-bag-of-features vector from D-D (§3) | `test_memory.py`: inserted near-duplicate code retrieves as top match; two syntactically very different reward functions retrieve each other with low similarity |
| A3.7 | Wire retrieval memory into prompts | `amfrs/pipeline2.py` | A3.6, A3.3 | Every mutation/crossover/initial-generation prompt gets a "here are 2–3 similar reward functions tried in past runs and what happened to them" block appended | Integration test: with a pre-seeded memory DB, the generated prompt string contains the expected past-run snippet |

### 6.3 Concrete specs

```python
# amfrs/primitives.py
from crowd_nav.reward_search.state import RewardState
from typing import Dict, Any

def goal_progress(state: RewardState, memory: Dict[str, Any]) -> float:
    """Negative change in distance-to-goal since last step (positive = progress)."""

def discomfort_penalty(state: RewardState, memory: Dict[str, Any]) -> float:
    """0 outside discomfort_dist; linearly increasing penalty as dmin -> 0 inside it."""

def collision_indicator(state: RewardState, memory: Dict[str, Any]) -> float:
    """1.0 if state.collision else 0.0 — a primitive, not a full terminal reward; the LLM
    composes this with a learned/prompted coefficient rather than the sandbox hardcoding -20."""

def time_penalty(state: RewardState, memory: Dict[str, Any]) -> float:
    """Small constant per-step penalty, encourages efficient paths."""

def jerk_penalty(state: RewardState, memory: Dict[str, Any]) -> float:
    """Uses memory['prev_action'] (candidate must be told this key exists) to penalize large
    consecutive action deltas. This is the primitive that most needs the (state, memory) contract
    from state.py — a good example to lead the prompt documentation with."""

PRIMITIVE_REGISTRY: dict[str, callable] = {
    "goal_progress": goal_progress,
    "discomfort_penalty": discomfort_penalty,
    "collision_indicator": collision_indicator,
    "time_penalty": time_penalty,
    "jerk_penalty": jerk_penalty,
    # ... 5-10 more: social_force_alignment, heading_alignment, energy_penalty,
    # min_distance_margin, goal_alignment_bonus, backing_up_penalty, spin_penalty
}
```

```python
# amfrs/memory.py
import sqlite3, ast, hashlib
from dataclasses import dataclass

@dataclass(frozen=True)
class MemoryEntry:
    code: str
    run_id: str
    outcome_metrics: dict
    similarity: float  # filled in only by query_similar, not stored

class RewardMemory:
    """
    SQLite-backed, single file (default: amfrs_env/data/reward_memory.sqlite, gitignored).
    Schema: candidates(code TEXT, code_hash TEXT PRIMARY KEY, run_id TEXT, features_json TEXT,
    outcome_json TEXT, created_at TEXT).
    Feature vector = AST node-type histogram (ast.walk + type(node).__name__ counts) concatenated
    with a few hand-picked scalars (count of Constant nodes, count of Compare nodes referencing
    'discomfort_dist', count of Attribute accesses to 'humans'). Similarity = cosine on this vector.
    Deliberately NOT a call to any LLM API for embeddings (see Decision D-D in §3).
    """
    def __init__(self, db_path: str): ...
    def add(self, code: str, run_id: str, outcome_metrics: dict) -> None: ...
    def query_similar(self, code: str, k: int = 3) -> list[MemoryEntry]: ...
```

**Why ensemble critique matters here specifically:** the project's own audit already found one real
failure mode (Gen0 81% rejection rate from a sandbox bug) that a single automated check missed for
a while. An LLM-generated *critique* of a rollout ("this reward is too aggressive near humans") is
exactly the kind of claim that's easy for one model to state confidently and be wrong about your
specific run. Requiring two independently-queried clients (Groq + Ollama, which you already have
both wired up in `llm.py`) to agree before auto-applying a refinement is a cheap, principled
consistency check that directly reuses infrastructure you already have — no new API integration
needed, just two calls instead of one at the refinement decision point.

## 7. Axis 4 — Robustness / Generalization as a Fitness Axis

### 7.1 Problem with the current pipeline

The existing H-sweep (`stage3.py` + `selection.h_profile_scalar`) only varies **crowd size**
(`human_num`), always with the same human policy, `humans.policy = "orca"`
(`crowd_nav/configs/config.py`). A reward function can look robust across H=5..20 while being
completely tuned to ORCA's specific avoidance style, and would visibly fail against a
`social_force`-driven crowd (already implemented and registered as `policy_factory['social_force']`
in `crowd_nav/policy/policy_factory.py` — confirmed present, just never swept over) or against
hand-built adversarial layouts. This axis makes "does it generalize" a first-class thing you
measure and select on, not an afterthought.

### 7.2 Task table

| ID | Task | New file | Depends on | Expected result | Validation |
|---|---|---|---|---|---|
| A4.1 | Multi-policy sweep runner | `amfrs/robustness.py` | Existing `crowd_nav.configs.config` (`humans.policy` field), `stage3.py` eval path as reference | `run_policy_sweep(candidate, policies=("orca","social_force")) -> dict[str, ProxyMetrics]` — same eval loop as the existing H-sweep, just varying `config.humans.policy` instead of `human_num` | Smoke test with `stage3_use_stub=True`-equivalent stub trainer, asserts both policy keys present in output |
| A4.2 | Robustness scalar | `amfrs/robustness.py` | A4.1, `selection.navigation_scalar_from_dict` (reuse, don't reimplement) | `robustness_scalar(by_policy: dict) = min over policies of navigation_scalar_from_dict(metrics)` — worst-case, not average, matching the existing worst-H philosophy in `h_profile_scalar` | Unit test: one bad policy result drags the scalar down to that policy's value even if the other is excellent |
| A4.3 | Adversarial scenario definitions | `amfrs/scenarios/*.yaml` | Check `crowd_sim`'s `test_case` / scenario-override mechanism (`generate_robot_humans`, `reset(phase, test_case=...)` per `AUDIT.md` §1) before writing YAML — confirm exactly which fields are overridable | 3 YAML scenario files: narrow corridor (humans forced into a tight passage), dense crossing (many humans converging near robot's straight-line path), bottleneck swap (two humans forced to swap sides near the robot) | Each YAML loads and produces a valid `test_case` override without crashing `env.reset` |
| A4.4 | Adversarial evaluation script | `amfrs/robustness.py` + `scripts/eval_amfrs_checkpoint.py`-style extension | A4.3 | Given a finalist checkpoint + reward, run the 3 adversarial scenarios N episodes each, report SR/CR per scenario | Manual run once GPU/checkpoint available; not blocked on this for the rest of the roadmap |
| A4.5 | Wire into Stage-III finalist selection | `amfrs/pipeline2.py` | A4.2, existing `selection.select_top_k_finalists`/`attach_h_profile` as the pattern to mirror | New `attach_robustness_profile(candidate, by_policy) -> RewardCandidate` (same `replace(...)` pattern as `attach_h_profile`), and a `pick_best_by_robustness` mirroring `pick_best_by_h_profile` | Unit test structurally identical to existing `test_selection.py` cases, just swapping H-sweep dict for policy-sweep dict |

### 7.3 Notes

- **Do this one only among finalists**, exactly like the existing H-sweep already restricts itself
  to `stage3_h_sweep_max_finalists` (default 2) — running a second full policy on every Stage-I/II
  candidate would multiply your compute budget for a signal that only matters once you're choosing
  a winner.
- A4.1/A4.2 (multi-policy sweep) is the cheap, high-value half of this axis — build it first.
  A4.3/A4.4 (adversarial YAML scenarios) require you to first confirm exactly how `crowd_sim`'s
  `test_case` override works (grep `generate_robot_humans` and the `reset()` signature in
  `crowd_sim/envs/crowd_sim.py` before writing the YAML schema) — treat A4.3 as a half-day spike to
  nail the override mechanism before committing to the YAML format.

---

## 8. Axis 5 — Static / Symbolic Safety Gate

### 8.1 Problem with the current pipeline

`sandbox/ast_policy.py` + `sandbox/config.py` are excellent at *code* safety (no `import os`, no
`eval`, no `__import__`, no attribute reflection — confirmed `isinstance` is correctly in
`_SAFE_BUILTINS` in `sandbox/runtime.py`, so the Gen0 bug from the audit is fixed). They say
**nothing** about whether the reward function is *sane as a reward* — e.g. a candidate that
returns a constant, ignores `state.humans` entirely, or has a discontinuity that blows up near
`dmin=0` currently sails through the sandbox and only gets caught (expensively) after real Stage-I
or Stage-II compute is spent on it, if at all. This axis adds a **second, reward-domain gate**
after the existing code-safety gate.

### 8.2 Task table

| ID | Task | New file | Depends on | Expected result | Validation |
|---|---|---|---|---|---|
| A5.1 | Synthetic `RewardState` sampler | `amfrs/static_gate.py` | `state.py` dataclasses (exact field list already known — see §"grounding" in this doc's header) | `sample_synthetic_states(n=200, seed=...) -> list[RewardState]` covering edge cases: 0 humans, 1 human at exactly `discomfort_dist`, human overlapping robot, `dmin` sweeping from 0 to `sensor_range`, `timeout`/`collision`/`reaching_goal` each True in isolation | `test_static_gate.py`: sampler produces states that satisfy `state.py`'s own dataclass field types/ranges |
| A5.2 | Bound/finite check | `amfrs/static_gate.py` | A5.1, `sandbox/runtime.compile_compute_reward` (reuse the compiled function, don't re-implement sandboxed exec) | `check_bounded(compute_fn, states) -> StaticReport(all_finite: bool, min_val, max_val, offending_state_idx)` | Constant-reward and NaN-producing synthetic candidates are correctly flagged in a unit test |
| A5.3 | Monotonicity-in-danger check | `amfrs/static_gate.py` | A5.1, A5.2 | For paired states differing only in `dmin` (all else equal, sampled inside the discomfort zone), assert reward is non-increasing as `dmin` decreases; flag candidates that reward getting closer to humans | Unit test with a hand-written "bad" reward (`+dmin` instead of `-dmin`) is correctly flagged |
| A5.4 | Human-blindness check | `amfrs/static_gate.py` | A5.1 | Run the compiled function on two states identical except `state.humans` is empty vs. populated; if reward is bit-identical across **all** sampled pairs, flag `ignores_humans=True` | Unit test: a reward that only reads `state.robot`/`state.dmin` (not `state.humans` directly, but note `dmin` already encodes human proximity — flag should specifically check for a candidate that also ignores `dmin`, i.e. truly crowd-blind) |
| A5.5 | Wire as a pipeline gate, non-fatal by default | `amfrs/pipeline2.py` | A5.2–A5.4 | `StaticGateReport` attached to `candidate.metadata["static_report"]`; candidates with `all_finite=False` are hard-rejected (never reach F0); others get a soft warning surfaced in logs/manifest but still proceed — mirrors the existing "log, never silently drop" philosophy already used for sandbox repair failures in `stage2.refine_candidate` | Integration test: a NaN-producing candidate never reaches `scoring.score1_for_dataset`; a monotonicity-violating candidate does reach it but with a logged warning |

### 8.3 Why non-fatal by default for A5.3/A5.4

Being too aggressive here risks false-positive-rejecting a legitimately good but unusual reward
(e.g. one that's deliberately flat in a region because a different term dominates there). Only
`all_finite=False` (A5.2) is an unambiguous hard-reject — NaN/Inf is never valid regardless of
design intent. Monotonicity and human-blindness are diagnostic signal, not disqualifying by
themselves; feed them into the LLM critique loop (Axis 3) as free, zero-simulation-cost diagnostic
text rather than using them as a gate at first. Revisit making A5.3/A5.4 hard gates only after
you've observed, empirically, that flagged candidates reliably turn out bad at F1+ (i.e. validate
the gate's precision before trusting it to reject anything outright).

---

## 9. `AMFRS2Pipeline` — how the axes plug together

This is pseudocode, not literal code to paste in — it shows call order and data flow so Cursor
builds the pieces in a way that actually composes, instead of five disconnected modules.

```python
# amfrs/pipeline2.py (shape, not final code)

class AMFRS2Pipeline:
    def __init__(self, config: AMFRS2RunConfig):
        self.validator = RewardValidator(...)          # reused as-is from sandbox/
        self.static_gate = StaticGate(...)              # A5
        self.ladder = build_fidelity_ladder(config)      # A1
        self.halving = SuccessiveHalvingScheduler(self.ladder, config.halving)  # A1
        self.bandit = UCB1Allocator(cost_aware=True) if config.use_bandit else None  # A1
        self.archive = EliteGrid(shape=config.archive_shape)  # A2
        self.memory = RewardMemory(config.memory_db_path)      # A3
        self.llm_a = build_llm_client(config.llm_provider_a)    # A3, e.g. groq
        self.llm_b = build_llm_client(config.llm_provider_b)    # A3, e.g. ollama, for ensemble critique

    def run(self) -> AMFRS2Artifacts:
        seed_candidates = self._generate_initial_population()   # uses primitives.py-aware prompts (A3.3)
        for cand in seed_candidates:
            ok, err = self.validator.validate_code(cand.code)
            if not ok:
                continue  # existing repair-or-drop path, reuse evolver._attempt_candidate logic
            report = self.static_gate.check(cand)                # A5
            cand.metadata["static_report"] = report
            if not report.all_finite:
                continue  # hard reject, never spends a fidelity-rung evaluation

        # Successive halving climbs the ladder; each surviving candidate that reaches F1+ gets a
        # behavior descriptor and is inserted into the archive, not just ranked by scalar.
        survivors = self.halving.run(
            seed_candidates,
            on_rung_complete=self._on_rung_complete,   # attaches descriptor (A2.5), logs to memory (A3.6)
        )

        # New generations are illuminated from the archive, not just "mutate the top-2":
        for generation in range(config.generations):
            parents = self.archive.all_elites()  # or a sampled subset, weighted toward sparse cells
            children = self._propose_children(parents)  # semantic crossover (A3.4) + primitive-aware mutation (A3.1-3.3)
            children = self._sandbox_and_static_filter(children)  # A5 again, every generation
            survivors = self.halving.run(children, on_rung_complete=self._on_rung_complete)
            # ensemble-critique-gated refinement happens inside the F1/F2 rungs, analogous to
            # stage2._maybe_accept_refine but using critique agreement (A3.5) as the extra gate.

        finalists = self.archive.all_elites()  # already diverse — no separate top-k step needed
        finalists = [attach_robustness_profile(c, run_policy_sweep(c)) for c in finalists]  # A4
        best = pick_best_by_robustness(finalists)  # A4, mirrors pick_best_by_h_profile

        self.archive.to_json(os.path.join(config.output_dir, "map_elites_archive.json"))  # A2.4
        return AMFRS2Artifacts(archive=self.archive, best=best, ...)
```

Key composition point to get right: **the archive (A2) is the population**, not a side artifact.
Once F1-level evaluation is possible, every candidate that beats its cell's current elite goes into
the archive and becomes a future parent; the halving scheduler (A1) decides *how much budget* each
candidate earns on its way to that point, and the memory (A3) and static gate (A5) are consulted
*before* any budget is spent. Robustness (A4) is deliberately the very last step, evaluated only
over the (already small, already diverse) set of final archive elites.

---

## 10. Phased implementation order (milestones for Cursor)

Build in this order — each milestone is independently runnable and testable before starting the
next; do not try to build all five axes in parallel.

| Milestone | Scope | Tasks | Exit criteria |
|---|---|---|---|
| **M0** | Scaffolding, zero behavior change | Create `amfrs/` package skeleton, `AMFRS2RunConfig`, empty `AMFRS2Pipeline.run()` that just calls the existing `RewardValidator` and returns | `pytest amfrs_env/crowd_nav/reward_search/amfrs/tests -q` passes, `run_amfrs2.py --fast` exits 0 |
| **M1** | Axis 5 (static gate) | A5.1–A5.5 | Static gate correctly hard-rejects a hand-written NaN candidate and soft-flags a hand-written human-blind candidate, both as unit tests with no simulator |
| **M2** | Axis 1 (fidelity + halving) | A1.1, A1.2, A1.4, A1.5 (skip A1.3 bandit for now) | A fast/stub run shows a population shrinking rung-over-rung with logged promotion decisions in the manifest |
| **M3** | Axis 2 (MAP-Elites) | A2.1–A2.3, A2.5 | After M2's halved survivors, archive `coverage()` > 0 and increases over a few generations in a fast/stub run |
| **M4** | Axis 3 (LLM smarts) | A3.1–A3.3 first (primitives + prompts), then A3.6–A3.7 (memory), then A3.5 (ensemble critique) last since it needs two live LLM backends to be meaningful | Scripted-LLM tests pass for all of A3.1–A3.7; one real (non-stub) small run with Groq+Ollama shows a logged agree/disagree decision |
| **M5** | Axis 4 (robustness) | A4.1, A4.2 first (cheap), A4.3/A4.4 (adversarial YAML) as a stretch goal | A finalist's `robustness_scalar` visibly differs from its plain `navigation_scalar`, proving the two orca/social_force runs produced different metrics |
| **M6** | Axis 1 bandit refinement + full integration | A1.3, A1.6, `compare_amfrs_runs.py` | One full paper-scale-equivalent `run_amfrs2.py` run completes and its manifest + archive JSON + cost-vs-quality plot are all produced |

M1 is deliberately first because it's the cheapest, has zero dependency on anything else, and
immediately gives you a concrete artifact (a rejected NaN candidate, a flagged human-blind
candidate) that's easy to sanity-check by hand before anything more complex is built on top.

---

## 11. Thesis deliverables mapping

| Axis | What ends up in the thesis |
|---|---|
| 1 | A figure: cumulative compute cost vs. best score found, EvoNav-faithful baseline vs. AMFRS2, showing AMFRS2 reaches equal/better quality for less cost (or better quality for equal cost) |
| 2 | The MAP-Elites grid heatmap (A2.4) — a qualitative figure showing a spread of distinct navigation styles, plus a `qd_score`/`coverage` table compared against a scalar-only ablation (`selection_mode="scalar"` run) |
| 3 | An ablation table: with/without primitive DSL, with/without retrieval memory, with/without ensemble critique — each row a full small run, comparing generations-to-convergence or final archive `qd_score` |
| 4 | A robustness table: each finalist's `navigation_scalar` under ORCA vs. `robustness_scalar` (worst-case across policies) — showing which candidates that looked good under the baseline's single-policy H-sweep are actually policy-overfit |
| 5 | A short methods-section paragraph + one number: "N candidates were rejected pre-simulation by the static gate across the full run, saving an estimated X GPU-hours" — cheap to state, strengthens the "trustworthy pipeline" narrative alongside Phases 1–4 of the original master plan |

---

## 12. Risks and fallbacks

- **4GB VRAM ceiling.** Axis 1's whole point is to reduce wasted GPU-hours, but F2/F3 rungs still
  need real training. If `num_processes`/batch sizes from the existing `AMFRSRunConfig` defaults
  OOM under the new scheduler's concurrent-candidate bookkeeping, keep evaluation strictly
  sequential per rung (no parallel env workers across *candidates*, only within one candidate's
  vec-env as today) — the halving scheduler's value is in *which* candidates get funded, not in
  running them concurrently.
- **Archive under-population.** With population sizes of 8–16 (current `stage1_population`
  default), a 2D grid with 5×5 bins (25 cells) may stay sparse for a long time. If `coverage()`
  is still low after several generations in early testing, shrink the grid (e.g. 3×3) before
  concluding the algorithm doesn't work — this is a hyperparameter issue, not a correctness issue.
- **Retrieval memory retrieving garbage.** If AST-bag similarity (D-D) turns out to retrieve
  semantically unrelated candidates in practice (validate this manually on ~20 real past candidates
  once Stage-I real data exists), the fallback is retrieval by *outcome similarity* instead of
  *code similarity* (i.e., "candidates with similar `ProxyMetrics` profiles," which is already
  computable with zero new infrastructure) rather than reaching for an external embedding API.
- **Ensemble critique doubles LLM cost.** Every gated refinement now costs 2 LLM calls
  (Groq + Ollama) instead of 1. Ollama is local/free, so the marginal cost is really just Groq's
  existing single call plus one local Ollama call — check this is actually true for your current
  `--llm groq` runs (i.e. confirm Ollama is being called as the *second* critic, not as a second
  *paid* Groq call) before assuming this is free.

---

## 13. What NOT to build (scope discipline, mirrors §13 of the old master plan)

- Do not build a learned surrogate model for any axis here — none of axes 1–5 need one, and the
  old master plan's own D-6/T19 reasoning (insufficient labeled data from full runs) still applies
  even in a "not faithful to EvoNav" framing.
- Do not let Axis 3's primitive DSL fully replace free-form `compute_reward` generation — keep both
  paths available (`AMFRS2RunConfig.require_primitives: bool = False` default) so the LLM can still
  invent something outside the registry; primitives are a *bias toward* good structure, not a hard
  constraint, at least until you have evidence the registry is expressive enough to not need an
  escape hatch.
- Do not implement Axis 4's adversarial scenarios (A4.3/A4.4) before A4.1/A4.2 (multi-policy sweep)
  are working and validated — the policy sweep is strictly cheaper and already tells you most of
  what you need about generalization.
- Do not touch `pipeline.py`/`AMFRSPipeline` at all during any milestone above (§1, non-negotiable).

---

*End of roadmap. Suggested first Cursor prompt: "Implement Milestone M0 exactly as specified in
§10 of AMFRS_V2_INNOVATION_ROADMAP.md — package skeleton and AMFRS2RunConfig only, no algorithm
logic yet."*
