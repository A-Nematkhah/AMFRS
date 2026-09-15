#!/usr/bin/env bash
set -euo pipefail
export PATH=/mingw64/bin:/usr/bin:$PATH

# Recreate venv so pacman numpy/scipy are visible
rm -rf /c/Users/nematkhah/amfrs-mingw-venv
python -m venv --system-site-packages /c/Users/nematkhah/amfrs-mingw-venv
# shellcheck disable=SC1091
source /c/Users/nematkhah/amfrs-mingw-venv/bin/activate
python -c 'import numpy,scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'
pip install --no-build-isolation /i/Code/AMFRS/amfrs_env/_rvo2_src
pip install --no-deps 'gym==0.15.7' cloudpickle pyglet six future
python -c 'import rvo2,gym,numpy; print("OK", gym.__version__, numpy.__version__)'
