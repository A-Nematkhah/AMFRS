#!/usr/bin/env bash
set -euo pipefail
export PATH=/mingw64/bin:/usr/bin:$PATH
cd /i/Code/AMFRS/amfrs_env/_rvo2_src
# shellcheck disable=SC1091
source /c/Users/nematkhah/amfrs-mingw-venv/bin/activate
rm -rf build
pip install --no-build-isolation .
python -c 'import rvo2; print("rvo2 OK", rvo2)'
