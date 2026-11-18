#!/usr/bin/env bash
cd "$(dirname "$0")/.."
FAST="--precision fp16 --decoder-fp16 --graphs --compile-decoder --compile-blocks --fast --length-check"
G=bench/out/grid; mkdir -p $G
run() { tag=$1; shift; echo "=== $tag $*"; IRODORI_SDPA_CUDNN=0 bench/run.sh "$@" --batch ${BATCHES:-1,8} --repeat 2 --seeds 3 --stages --wav-dir $G/$tag 2>&1 | grep -E "^\[B=|length_check|段ごと|max_mem"; }
# done: BATCHES=1 run up_small40_fp32_lc --model small --length-check
run small40_lc --model small $FAST
run small16_sway_lc --model small $FAST --steps 16 --sway -1
run small12_sway_lc --model small $FAST --steps 12 --sway -1
run small12_lc --model small $FAST --steps 12
