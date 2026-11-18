"""Batched watermark vs one-by-one: speed, difference and detection (payload IRDTS) on bench outputs."""
import sys, os, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, soundfile as sf, numpy as np
from irodori_tts.watermark import SilentCipherWatermarker
wm = SilentCipherWatermarker(device="cuda")
files = sorted(glob.glob(sys.argv[1] + "/b8_s0_*.wav"))
audios = [torch.from_numpy(sf.read(f, dtype="float32")[0]).view(1, -1) for f in files]
def timed(f):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(3): r = f()
    torch.cuda.synchronize(); return r, (time.perf_counter() - t) / 3 * 1000
seq, ts = timed(lambda: wm.encode_batch(audios, sample_rate=48000))
print(f"fp32 one-by-one {ts:.0f}ms ({len(audios)} rows)")
for f, a, s_ in zip(files, audios, seq):
    ds = wm.model.decode_wav(s_.view(-1), 48000, phase_shift_decoding=False)
    dn = wm.model.decode_wav(a.view(-1), 48000, phase_shift_decoding=False)
    print(f"{os.path.basename(f)} fp32 {ds['status']} {ds['messages'][0] if ds['status'] else ''} "
          f"{ds['confidences'][0] if ds['status'] else 0:.2f} | without watermark {dn['status']}")
print("--- fp16 autocast, one-by-one")
def enc16():
    with torch.autocast("cuda", dtype=torch.float16):
        return wm.encode_batch(audios, sample_rate=48000)
h16, th = timed(enc16)
print(f"autocast fp16 {th:.0f}ms")
for f, a, s_, b in zip(files, audios, seq, h16):
    a, s_, b = a.view(-1).numpy(), s_.view(-1).numpy(), b.view(-1).numpy()
    wm_s, wm_b = s_ - a, b - a
    diff = 10 * np.log10((wm_s ** 2).mean() / max(((wm_s - wm_b) ** 2).mean(), 1e-20))
    db = wm.model.decode_wav(torch.from_numpy(b), 48000, phase_shift_decoding=False)
    print(f"{os.path.basename(f)} wm差 {diff:5.1f}dB nonfinite={not np.isfinite(b).all()} | fp16 {db['status']} {db['messages'][0] if db['status'] else ''} {db['confidences'][0] if db['status'] else 0:.2f}")
