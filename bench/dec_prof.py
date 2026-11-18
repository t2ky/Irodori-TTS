import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from irodori_tts.codec import DACVAECodec
codec = DACVAECodec.load(device="cuda")
dac = codec.model
for m in dac.modules():
    if hasattr(m, "weight_g") and isinstance(getattr(m, "weight", None), torch.Tensor): m.weight = m.weight.detach()
import copy
op = copy.deepcopy(dac.quantizer.out_proj).half(); dec = copy.deepcopy(dac.decoder).half()
def run(T, B=1):
    z = torch.randn(B, 32, T, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        for _ in range(3): dec(op(z))
        torch.cuda.synchronize(); s = time.perf_counter()
        for _ in range(10): dec(op(z))
        torch.cuda.synchronize(); return (time.perf_counter() - s) / 10 * 1000
for bm in (False, True):
    torch.backends.cudnn.benchmark = bm
    print("cudnn.benchmark", bm, " ".join(f"T{T}:{run(T):.1f}ms" for T in (25, 50, 100, 200)), f" B4xT100:{run(100,4):.1f}ms")
# where is the time: per top-level block
torch.backends.cudnn.benchmark = False
z = torch.randn(1, 32, 100, device="cuda", dtype=torch.float16)
with torch.inference_mode():
    x = op(z)
    for i, layer in enumerate(dec.model):
        torch.cuda.synchronize(); s = time.perf_counter()
        for _ in range(10): y = layer(x)
        torch.cuda.synchronize(); print(i, type(layer).__name__, tuple(y.shape), f"{(time.perf_counter()-s)/10*1000:.2f}ms"); x = y
print("--- compile")
import torch._dynamo
dec_c = torch.compile(dec, dynamic=True)
ref = None
with torch.inference_mode():
    for T in (25, 50, 100, 200):
        z = torch.randn(1, 32, T, device="cuda", dtype=torch.float16)
        s = time.perf_counter(); out = dec_c(op(z)); torch.cuda.synchronize(); first = time.perf_counter() - s
        for _ in range(3): dec_c(op(z))
        torch.cuda.synchronize(); s = time.perf_counter()
        for _ in range(10): dec_c(op(z))
        torch.cuda.synchronize(); t = (time.perf_counter() - s) / 10 * 1000
        e = dec(op(z)).float(); c = dec_c(op(z)).float()
        snr = 10 * torch.log10((e**2).mean() / ((e - c)**2).mean())
        print(f"T{T}: {t:.1f}ms (first call {first:.1f}s) snr vs eager {float(snr):.1f}dB")
