#!/usr/bin/env bash
set -euo pipefail
export PATH=/mingw64/bin:/usr/bin:$PATH
# shellcheck disable=SC1091
source /c/Users/nematkhah/amfrs-mingw-venv/bin/activate
cd /i/Code/AMFRS/amfrs_env
export PYTHONPATH="/i/Code/AMFRS/amfrs_env:/i/Code/AMFRS/baselines_openai:${PYTHONPATH:-}"
python -c 'import rvo2, gym, numpy; print("imports OK", gym.__version__, numpy.__version__)'
python scripts/collect_stage1_dataset.py --n-scenarios 3 --n-traj 4 --out data/stage1_smoke --regime without_random
python -c 'from crowd_nav.reward_search.amfrs.assets import check_stage1_dataset; print(check_stage1_dataset("data/stage1_smoke"))'
