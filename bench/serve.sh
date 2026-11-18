#!/usr/bin/env bash
# turing_server.py with the same env as server/run.sh. IRODORI_* env is passed through.
cd "$(dirname "$0")/.."
unset LD_LIBRARY_PATH
if [ -f ~/.secrets/huggingface.env ]; then set -a; . ~/.secrets/huggingface.env; set +a; fi
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONWARNINGS=ignore IRODORI_SDPA_CUDNN=${IRODORI_SDPA_CUDNN:-0}
exec .venv/bin/python turing_server.py "$@"
