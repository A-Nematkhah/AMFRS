# AMFRS

**AMFRS** is our research codebase for robot crowd navigation with evolved
reward functions.

[describe AMFRS's novel contribution here]

## Status

AMFRS is under active development; the current codebase is a renamed fork of
an EvoNav Algorithm 1 baseline reproduction, to be extended with our own
contributions.

## Repository layout

| Directory | Role |
|-----------|------|
| `amfrs_env/` | Main project tree (CrowdNav++ simulator + reward-search pipeline) |
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
pytest crowd_nav/reward_search/tests -m "not slow"
```

See **`amfrs_env/README_AMFRS.md`** for Algorithm 1 runs, paper-scale budgets, and API keys.
Simulator train/test docs: **`amfrs_env/README.md`**. Architecture notes: **`amfrs_env/AUDIT.md`**.

## Groq API keys

Copy `amfrs_env/groq_keys.json.example` → `amfrs_env/groq_keys.json` (gitignored). Never commit real keys.

## Baseline / Prior work

This repository currently embeds a faithful replication of **EvoNav Algorithm 1**
(arXiv:2605.11859) on the CrowdNav++ simulator. That baseline remains intact as
a reference and comparison point while AMFRS contributions are developed on top.

We will cite the following prior works in our paper:

```bibtex
@article{evonav2026,
  title   = {EvoNav},
  eprint  = {arXiv:2605.11859},
  year    = {2026}
}

@inproceedings{liu2023crowdnavpp,
  title     = {Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph},
  author    = {Liu, Shuijing and Chang, Peixin and Huang, Zhe and others},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2023},
  pages     = {12015--12021}
}
```
