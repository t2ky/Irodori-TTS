#!/usr/bin/env bash
# bench wrapper: same env as server/run.sh
cd "$(dirname "$0")/.."
unset LD_LIBRARY_PATH
if [ -f ~/.secrets/huggingface.env ]; then set -a; . ~/.secrets/huggingface.env; set +a; fi
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONWARNINGS=ignore
exec .venv/bin/python bench/${BENCH_SCRIPT:-bench.py} "$@" 2>&1 | grep -v -E "Fetching|WeightNorm|codec\] dacvae|ckpt path"
