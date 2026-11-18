#!/usr/bin/env bash
# Small/MF variants: B=8 timing + 3 seeds of wavs into bench/out/grid/<tag>
cd "$(dirname "$0")/.."
FAST="--precision fp16 --decoder-fp16 --graphs --compile-decoder --compile-blocks --fast"
G=bench/out/grid; mkdir -p $G
run() { tag=$1; shift; echo "=== $tag $*"; IRODORI_SDPA_CUDNN=0 bench/run.sh "$@" --batch 8 --repeat 2 --seeds 3 --wav-dir $G/$tag 2>&1 | grep -E "^\[B=|max_mem"; }
run up_small40_fp32 --model small
run up_mf4_fp32 --model mf
run mf4 --model mf $FAST
run small40 --model small $FAST
run small24 --model small $FAST --steps 24
run small16 --model small $FAST --steps 16
run small12 --model small $FAST --steps 12
run small16_sway --model small $FAST --steps 16 --sway -1
run small12_sway --model small $FAST --steps 12 --sway -1
run small40_cfg07 --model small $FAST --cfg-min-t 0.7
run small16_cfg07 --model small $FAST --steps 16 --cfg-min-t 0.7
run small40_alt --model small $FAST --cfg-mode alternating
run small16_alt --model small $FAST --steps 16 --cfg-mode alternating
