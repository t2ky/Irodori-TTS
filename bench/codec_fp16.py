import sys, copy, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bench as B
import argparse
args = argparse.Namespace(model="mf", precision="fp16", codec_precision="fp32", compile=False, steps=None)
rt = B.make_runtime(args)
class NW: ready=False
rt.watermarker = NW()
from irodori_tts.turing import TuringBatcher
tb = TuringBatcher(rt)
zs = []
orig = rt.codec.decode_latent
def cap(z):
    zs.append(z.clone()); return orig(z)
rt.codec.decode_latent = cap
tb.synthesize_batch([B.make_req(t, 1000+i, args) for i, t in enumerate(B.TEXTS)])
rt.codec.decode_latent = orig
dec32 = rt.codec.model
dec16 = copy.deepcopy(dec32).half()
print("latent absmax", max(float(z.abs().max()) for z in zs))
for mode in ("fp16", "fp16-autocast"):
    res = []
    for z in zs:
        with torch.inference_mode():
            a = dec32.decode(z.transpose(1,2).contiguous().to(dtype=torch.float32)).float()
            if mode == "fp16":
                b = dec16.decode(z.transpose(1,2).contiguous().half()).float()
            else:
                with torch.autocast("cuda", dtype=torch.float16):
                    b = dec32.decode(z.transpose(1,2).contiguous().float()).float()
        a = a.cpu().numpy().reshape(-1); b = b.cpu().numpy().reshape(-1)
        res.append((B.snr_db(a, b), B.mel_db(a, b), np.isfinite(b).all(), float(np.abs(b).max())))
    print(mode, " ".join(f"snr{r[0]:.0f}/mel{r[1]:.2f}/{'ok' if r[2] else 'NONFINITE'}" for r in res))
    zz = zs[2].transpose(1,2).contiguous()
    for dt, m in ((torch.float32, dec32), (torch.float16, dec16)):
        for _ in range(3): m.decode(zz.to(dt))
        torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(10): m.decode(zz.to(dt))
        torch.cuda.synchronize(); print(dt, f"{(time.perf_counter()-t)/10*1000:.1f}ms")
    break
