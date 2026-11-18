import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import time, torch, torchaudio
from irodori_tts.watermark import SilentCipherWatermarker
wm = SilentCipherWatermarker(device="cuda")
x = torch.randn(48000*3) * 0.1
for _ in range(3): wm.encode_one(x, sample_rate=48000)
def t(f, n=10):
    torch.cuda.synchronize(); s=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-s)/n*1000
print("encode_one 3s:", round(t(lambda: wm.encode_one(x, sample_rate=48000)),1), "ms")
y = x.cuda()
print("resample 48->44.1 gpu:", round(t(lambda: torchaudio.functional.resample(y.view(1,-1), 48000, 44100)),1), "ms")
r = torchaudio.transforms.Resample(48000, 44100).cuda()
print("cached Resample gpu:", round(t(lambda: r(y.view(1,-1))),1), "ms")
a = torchaudio.functional.resample(y.view(1,-1), 48000, 44100); b = r(y.view(1,-1)); print("same:", torch.equal(a,b), float((a-b).abs().max()))
