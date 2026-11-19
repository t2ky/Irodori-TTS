"""Precompute a reference latent (<name>.pt) from a reference wav, for IRODORI_VOICES_DIR.

The latent is exactly what the runtime would compute from the wav (same codec, fp32, same
max_ref_seconds trimming), so a server given <name>.pt sounds the same as one given <name>.wav but
never runs the codec encoder.

  bench/run.sh ... (BENCH_SCRIPT=make_latent.py) <in.wav> [<in2.wav> ...] --out-dir <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime, RuntimeKey, SamplingRequest, download_hf_checkpoint,
)

MODELS = {"small": "Aratako/Irodori-TTS-v4.1-Small", "small-mf": "Aratako/Irodori-TTS-v4.1-Small-MF"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="small-mf", choices=MODELS, help="only the codec is used; both share it")
    a = ap.parse_args()
    rt = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=download_hf_checkpoint(MODELS[a.model]), model_device="cuda", model_precision="fp32",
        codec_device="cuda", codec_precision="fp32"))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for wav in a.wavs:
        req = SamplingRequest(text="-", ref_wav=wav)
        max_ref = float(rt.default_max_ref_seconds)
        steps = None
        if max_ref > 0:
            import math
            steps = max(1, math.ceil(max_ref * float(rt.codec.sample_rate) / float(int(rt.codec.model.hop_length))))
        latent, _ = rt._encode_reference_wavs(req=req, wav_paths=[wav], max_ref_seconds=max_ref,
                                              max_ref_latent_steps=steps)
        dst = out / (Path(wav).stem + ".pt")
        torch.save(latent[0].float().cpu().contiguous(), dst)
        print(f"{wav} -> {dst} {tuple(latent[0].shape)}")


if __name__ == "__main__":
    main()
