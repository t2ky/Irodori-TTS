"""Latency bench for Turing work: one-by-one vs batched synthesis, with per-stage timings.

  .venv/bin/python bench/bench.py --model mf --batch 1,8 [--precision fp32] [--save out.npz | --compare out.npz]

Uses bench/ref30.wav (not in git) as the reference voice. Watermarking is off unless --watermark,
so outputs can be compared bit-for-bit (SNR) against a saved run.
"""
from __future__ import annotations

import argparse
import collections
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
)

MODELS = {"small": "Aratako/Irodori-TTS-v4.1-Small", "mf": "Aratako/Irodori-TTS-v4.1-Small-MF"}

# Chat-like replies, 3-48 chars.
TEXTS = [
    "うん。",
    "おはよう！今日もいい天気だね。",
    "えっと、それってどういう意味なのかな？もう少し詳しく教えてほしいな。",
    "ふふっ、そんなに褒めても何も出ないよ？",
    "ごめんね、昨日は返事できなくて。ちょっとバタバタしてたの。",
    "やったー！",
    "そっか……大変だったんだね。無理しないで、ゆっくり休んでね。",
    "じゃあ、また明日ね。おやすみなさい。",
]


def make_runtime(args) -> InferenceRuntime:
    ckpt = download_hf_checkpoint(MODELS[args.model])
    return InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=ckpt,
            model_device="cuda",
            model_precision=args.precision,
            codec_device="cuda",
            codec_precision=args.codec_precision,
            compile_model=args.compile,
        )
    )


def make_req(text: str, seed: int, args) -> SamplingRequest:
    extra = {}
    if getattr(args, "cfg_min_t", None) is not None:
        extra["cfg_min_t"] = args.cfg_min_t
    if getattr(args, "cfg_mode", None):
        extra["cfg_guidance_mode"] = args.cfg_mode
    if getattr(args, "sway", None) is not None:
        extra["t_schedule_mode"] = "sway"
        extra["sway_coeff"] = args.sway
    return SamplingRequest(text=text, ref_wav=str(ROOT / "bench" / "ref30.wav"),
                           seed=seed + 7919 * getattr(args, "seed_round", 0),
                           num_steps=args.steps, **extra)


def run(rt, texts, batch: int, args):
    """Returns (wall seconds per call list, stage sums, audios)."""
    walls, stages, audios = [], collections.defaultdict(float), []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        reqs = [make_req(t, 1000 + i + j, args) for j, t in enumerate(chunk)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if getattr(args, "length_check", False) and args.batcher is not None:
            res = args.batcher.synthesize_batch(reqs, length_check=True)
            args.lc_info.extend(args.batcher.last_length_check)
        elif getattr(args, "length_check", False):
            from irodori_tts import length_check as lc
            res = []
            for r in reqs:
                out, info = lc.synthesize(rt, r)
                res.append(out)
                args.lc_info.append(info)
        else:
            res = (args.batcher or rt).synthesize_batch(reqs)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
        for r in res:
            audios.append(r.audio.float().numpy().reshape(-1))
        # stage timings are per request but the batched stages are shared: count each once per call
        seen = set()
        for r in res:
            for name, sec in r.stage_timings:
                key = (name, sec)
                if key in seen:
                    continue
                seen.add(key)
                stages[name] += sec
    return walls, stages, audios


def snr_db(ref: np.ndarray, x: np.ndarray) -> float:
    n = min(len(ref), len(x))
    if len(ref) != len(x):
        return float("nan")
    err = ref[:n] - x[:n]
    p = float((ref[:n] ** 2).mean())
    e = float((err ** 2).mean())
    return float("inf") if e == 0 else 10 * np.log10(p / e)


_MEL = None


def mel_db(ref: np.ndarray, x: np.ndarray) -> float:
    """Mean absolute log-mel difference (dB) over the common length; a perceptual-ish distance."""
    global _MEL
    import torchaudio
    if _MEL is None:
        _MEL = torchaudio.transforms.MelSpectrogram(48000, n_fft=2048, hop_length=480, n_mels=80)
    n = min(len(ref), len(x))
    a = _MEL(torch.from_numpy(ref[:n].copy()))
    b = _MEL(torch.from_numpy(x[:n].copy()))
    return float((10 * torch.log10(a + 1e-5) - 10 * torch.log10(b + 1e-5)).abs().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="mf")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--codec-precision", default="fp32")
    ap.add_argument("--batch", default="1,8")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--watermark", action="store_true")
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--stages", action="store_true")
    ap.add_argument("--wav-dir")
    ap.add_argument("--decoder-fp16", action="store_true")
    ap.add_argument("--graphs", action="store_true")
    ap.add_argument("--cfg-min-t", type=float)
    ap.add_argument("--length-check", action="store_true")
    ap.add_argument("--cfg-mode")
    ap.add_argument("--sway", type=float)
    ap.add_argument("--seeds", type=int, default=1, help="with --wav-dir: write this many seeds per text")
    ap.add_argument("--compile-decoder", action="store_true")
    ap.add_argument("--compile-blocks", action="store_true")
    ap.add_argument("--watermark-fp16", action="store_true")
    ap.add_argument("--fast", action="store_true", help="use irodori_tts.turing.TuringBatcher")
    ap.add_argument("--latent-multiple", type=int, default=1)
    args = ap.parse_args()
    args.lc_info = []

    t0 = time.perf_counter()
    rt = make_runtime(args)
    if not args.watermark:
        class _NoWatermark:
            ready = False
        rt.watermarker = _NoWatermark()
    args.batcher = None
    if args.fast:
        from irodori_tts.turing import TuringBatcher
        args.batcher = TuringBatcher(rt, latent_multiple=args.latent_multiple,
                                     decoder_fp16=args.decoder_fp16, cuda_graphs=args.graphs,
                                     compile_decoder=args.compile_decoder,
                                     compile_blocks=args.compile_blocks,
                                     watermark_fp16=args.watermark_fp16)
    print(f"load {time.perf_counter() - t0:.1f}s  cudnn_sdpa={os.environ.get('IRODORI_SDPA_CUDNN', '1')}",
          flush=True)
    # warm-up: every shape once
    t0 = time.perf_counter()
    for b in [int(x) for x in args.batch.split(",")]:
        run(rt, TEXTS, b, args)
    print(f"warmup {time.perf_counter() - t0:.1f}s", flush=True)

    ref = np.load(args.compare) if args.compare else None
    for b in [int(x) for x in args.batch.split(",")]:
        per_rep, stage_rep, audios = [], [], None
        for _ in range(args.repeat):
            walls, stages, audios = run(rt, TEXTS, b, args)
            per_rep.append(walls)
            stage_rep.append(stages)
        calls = list(zip(*per_rep))
        med = [statistics.median(c) for c in calls]
        if b == 1:
            desc = "  ".join(f"{len(t)}字 {m:.3f}" for t, m in zip(TEXTS, med))
            print(f"[B=1] 1 本ずつ 平均 {statistics.mean(med):.3f}s 最大 {max(med):.3f}s  合計 {sum(med):.3f}s | {desc}")
        else:
            print(f"[B={b}] 1 呼び出し {' / '.join(f'{m:.3f}' for m in med)}s  合計 {sum(med):.3f}s")
        if args.length_check:
            last = args.lc_info[-len(TEXTS):]
            print("   length_check: " + " ".join(
                f"{i['length_check'][:5]}:{i.get('first_frames', '-')}->{i.get('estimated_frames', '-')}" for i in last))
        if args.stages:
            names = list(stage_rep[0].keys())
            print("   段ごと（1 回あたり合計 ms）: " + "  ".join(
                f"{n} {statistics.median(s[n] for s in stage_rep) * 1000:.0f}" for n in names))
        if ref is not None:
            snrs = [snr_db(ref[f"a{i}"], a) for i, a in enumerate(audios)]
            mels = [mel_db(ref[f"a{i}"], a) for i, a in enumerate(audios)]
            print("   メル差(dB) vs 基準: " + " ".join(f"{m:.2f}" for m in mels) + f"  平均 {np.mean(mels):.2f}")
            print("   SNR(dB) vs 基準: " + " ".join(f"{s:.0f}" for s in snrs)
                  + "  長さ一致 " + str(sum(len(ref[f'a{i}']) == len(a) for i, a in enumerate(audios))) + f"/{len(audios)}")
        if args.wav_dir:
            import soundfile as sf
            d = Path(args.wav_dir); d.mkdir(parents=True, exist_ok=True)
            for k in range(args.seeds):
                if k:
                    args.seed_round = k
                    _, _, audios = run(rt, TEXTS, b, args)
                for i, a in enumerate(audios):
                    sf.write(d / f"b{b}_s{k}_{i}.wav", a, 48000)
            args.seed_round = 0
        if args.save and b == 1:
            np.savez(args.save, **{f"a{i}": a for i, a in enumerate(audios)})
            print(f"   saved {args.save}")
    if args.batcher is not None and args.batcher.graphs is not None:
        g = args.batcher.graphs
        print(f"graphs: {len(g.graphs)} kept, {g.captures} captures, {g.replays} replays")
    print(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB "
          f"reserved {torch.cuda.memory_reserved() / 2**30:.2f}GiB")


if __name__ == "__main__":
    main()
