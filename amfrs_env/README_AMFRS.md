# AMFRS (this fork)

Main entry docs for the **AMFRS** project tree under `amfrs_env/`.

[describe AMFRS's novel contribution here]

## Status

AMFRS is under active development; the current codebase is a renamed fork of
an EvoNav Algorithm 1 baseline reproduction, to be extended with our own
contributions.

Upstream simulator docs: `README.md` in this directory. Architecture audit: `AUDIT.md`.

## Install

```bash
cd amfrs_env
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements_pinned.txt
.venv\Scripts\python.exe -m pip install git+https://github.com/sybrenstuvel/Python-RVO2.git
.venv\Scripts\python.exe -m pip install -e ../baselines_openai --no-build-isolation
```

Use this single `.venv` for the project. Do not mix it with system Python or
another environment, and do not use a `PYTHONPATH` workaround.

The pinned GPU wheel is `torch==2.11.0+cu128`. Adjust the `cu1xx` suffix and
the official PyTorch index URL in `requirements_pinned.txt` to match the local
CUDA driver before installing on another machine.

## Ollama (local, no API key)

For predictable latency, use the non-thinking model variant:

```bash
ollama pull frob/qwen3.5-instruct:4b
```

The default Ollama model is the reasoning model `qwen3.5:4b`, which can emit
`<think>` content before the code fence. If using it explicitly:

```bash
ollama pull qwen3.5:4b
python scripts/run_amfrs.py --llm ollama --llm-model qwen3.5:4b --output-dir results/ollama_run
```

The server URL defaults to `http://localhost:11434/v1`; override it with
`OLLAMA_BASE_URL`. If Gen0 rejection rates are unexpectedly high, inspect
`results/.../gen0_rejections.jsonl` for the truncation-specific
`no closing code fence found` error before assuming a sandbox problem.

## API keys (Groq)

1. Copy `groq_keys.json.example` → `groq_keys.json`
2. Add keys: `{"keys": ["gsk_...", "..."]}`
3. Or set `GROQ_API_KEY` for a single key (pool disabled)

## Run matrix

| Goal | Command | Hardware | Time |
|------|---------|----------|------|
| Wiring smoke | `python scripts/run_amfrs.py --fast` | CPU | seconds |
| Stage I dataset (M=100) | `python scripts/collect_stage1_dataset.py --regime without_random` | CPU | ~tens of min |
| Local validation | `python scripts/run_amfrs.py --llm groq --device cuda --regime without_random --stage1-dataset data/stage1_dataset --stage3-train-steps 500000` | GPU + Groq | hours |
| Paper scale | `python scripts/run_amfrs_paper_scale.py --device cuda --llm groq` | GPU + Groq | days (K3=1e7 × seeds) |

Defaults (AUDIT.md §8): `without_random`, Stage II/III `predict_method=inferred`, GST `...-seed_1000/sj`.

## Tests

```bash
pytest crowd_nav/reward_search/tests -m "not slow"   # CI default, 74 tests
pytest crowd_nav/reward_search/tests -m slow         # 1 real-env collect test
```

## Baseline checkpoints

Pretrained ORCA/SF/GST under `trained_models/` (see `scripts/report.py`). GST weights under `gst_updated/results/`.

## Baseline / Prior work

The runnable Algorithm 1 pipeline in this tree is currently a renamed
reproduction of **EvoNav** (arXiv:2605.11859) on CrowdNav++. It is kept as the
comparison baseline while AMFRS-specific contributions are added.

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
