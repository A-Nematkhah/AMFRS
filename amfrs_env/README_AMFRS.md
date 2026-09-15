# AMFRS (this fork)

Main entry docs for the **AMFRS** project tree under `amfrs_env/`.

## Status

Single pipeline: multi-fidelity reward search with MAP-Elites, static gate,
and multi-policy robustness (`crowd_nav/reward_search/amfrs/` →
`scripts/run_amfrs.py`).

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

```bash
ollama pull frob/qwen3.5-instruct:4b
python scripts/run_amfrs.py --llm ollama --llm-model frob/qwen3.5-instruct:4b --output-dir results/ollama_run
```

Override the server with `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`).

## API keys (Groq)

1. Copy `groq_keys.json.example` → `groq_keys.json`
2. Add keys: `{"keys": ["gsk_...", "..."]}`
3. Or set `GROQ_API_KEY` for a single key (pool disabled)

## Run matrix

| Goal | Command | Hardware | Time |
|------|---------|----------|------|
| Wiring smoke | `python scripts/run_amfrs.py --fast` | CPU | seconds |
| Check GST / dataset | `python scripts/check_amfrs_assets.py` | CPU | seconds |
| Stage I dataset (M=100) | `python scripts/collect_stage1_dataset.py --regime without_random` | CPU | ~tens of min |
| Local (stub trainers) | `python scripts/run_amfrs.py --use-stub --allow-seed-llm --predict-method none` | CPU | minutes |
| Local (real + GST) | `python scripts/run_amfrs.py --llm groq --device cuda --predict-method inferred` | GPU + Groq | hours+ |
| Plot archive | `python scripts/plot_amfrs_archive.py --run-dir results/...` | CPU | seconds |

Defaults: `without_random`, `predict_method=inferred` (unless `--fast` / `--predict-method none`).

## Tests

```bash
pytest crowd_nav/reward_search/tests crowd_nav/reward_search/amfrs/tests -m "not slow"
```

## Assets

Pretrained ORCA/SF/GST under `trained_models/` / `gst_updated/results/`.
Fetch GST if missing: `python scripts/fetch_gst_weights.py`.
