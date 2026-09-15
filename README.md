# AMFRS

**AMFRS** is our research codebase for robot crowd navigation with evolved
reward functions (multi-fidelity search, MAP-Elites archive, static gate,
robustness sweep).

## Status

Single pipeline path: `scripts/run_amfrs.py` → `AMFRSPipeline`
(`crowd_nav/reward_search/amfrs/`). Shared CrowdNav++ / Stage II–III trainers
remain as libraries under `reward_search/`.

## Repository layout

| Directory | Role |
|-----------|------|
| `amfrs_env/` | Main project tree (CrowdNav++ simulator + AMFRS reward search) |
| `baselines_openai/` | Trimmed OpenAI Baselines (vec_env / logger / bench only) |

`amfrs_env` is a **derivative work** of
[CrowdNav_Prediction_AttnGraph](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)
(MIT — see `amfrs_env/LICENSE` and `amfrs_env/NOTICE.md`).

## Quick start

```bash
cd amfrs_env
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements_pinned.txt
pip install -e ../baselines_openai --no-build-isolation
# Install PyTorch (pinned) and Python-RVO2 per amfrs_env/README.md

# Fast wiring test (~seconds)
python scripts/run_amfrs.py --fast --output-dir results/amfrs_fast

# Tests
pytest crowd_nav/reward_search/tests crowd_nav/reward_search/amfrs/tests -m "not slow"
```

See **`amfrs_env/README_AMFRS.md`** for runs, assets, and API keys.
Simulator train/test docs: **`amfrs_env/README.md`**. Architecture notes: **`amfrs_env/AUDIT.md`**.

## Groq API keys

Copy `amfrs_env/groq_keys.json.example` → `amfrs_env/groq_keys.json` (gitignored). Never commit real keys.

## Prior work

Built on CrowdNav++ (ICRA 2023) and ideas from EvoNav-style reward evolution.
The dual “Algorithm 1 baseline vs AMFRS2” entry points were consolidated into
this single AMFRS path.
