"""Wrap model/codec entry points with synchronized timers and run bench texts at B=1 and B=8."""
import collections, sys, time, functools
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv += [] 
import bench as B

T = collections.defaultdict(list)
def wrap(obj, name, label):
    f = getattr(obj, name)
    @functools.wraps(f)
    def g(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        r = f(*a, **k)
        torch.cuda.synchronize(); T[label].append(time.perf_counter() - t)
        return r
    setattr(obj, name, g)

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="mf"); ap.add_argument("--precision", default="fp32")
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    args = argparse.Namespace(model=a.model, precision=a.precision, codec_precision="fp32", compile=False, steps=a.steps)
    rt = B.make_runtime(args)
    class NW: ready = False
    rt.watermarker = NW()
    m = rt.model
    wrap(m, "encode_conditions", "encode_conditions")
    wrap(m, "build_context_kv_cache", "build_kv")
    wrap(m, "forward_with_encoded_conditions", "dit_forward")
    wrap(m, "predict_duration_log_frames", "duration_head")
    wrap(rt.codec, "decode_latent", "codec_decode")
    wrap(m.speaker_encoder, "forward", "speaker_encoder")
    wrap(m.pretrained_text_backbone if hasattr(m, "pretrained_text_backbone") else m.text_encoder, "forward", "text_backbone")
    for b in (1, 8):
        B.run(rt, B.TEXTS, b, args)  # warm
        T.clear()
        walls, _, _ = B.run(rt, B.TEXTS, b, args)
        print(f"== B={b} wall total {sum(walls)*1000:.0f}ms")
        for k, v in T.items():
            print(f"   {k:18s} n={len(v):3d} sum={sum(v)*1000:7.1f}ms  mean={sum(v)/len(v)*1000:6.2f}ms")
main()
