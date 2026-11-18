"""CER (faster-whisper large-v3, ja) and speaker similarity (WavLM-SV vs the reference) of wav dirs.

  ~/latentlink/.venv/bin/python bench/eval_quality.py <dir> [<dir> ...]
Files are b<B>_s<seed>_<text index>.wav from bench.py --wav-dir; texts are bench.TEXTS.
"""
import sys, re, unicodedata, glob, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TEXTS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "texts.json")))

def norm(s):
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"[\s、。，．,.！!？?…・「」『』（）()〜~ー\-—♡♪…]", "", s)

def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))

MODE = sys.argv[1]  # asr (faster-whisper venv) | sv (irodori venv) | report
DIRS = sys.argv[2:]
HERE = os.path.dirname(os.path.abspath(__file__))

def wavs(d):
    return sorted(glob.glob(os.path.join(d, "b*_s*_*.wav")))

if MODE == "asr":
    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
    for d in DIRS:
        out = {}
        for p in wavs(d):
            segs, _ = asr.transcribe(p, language="ja", beam_size=5)
            out[os.path.basename(p)] = "".join(s.text for s in segs)
        json.dump(out, open(os.path.join(d, "asr.json"), "w"), ensure_ascii=False, indent=0)
        print("asr", d, len(out))
elif MODE == "sv":
    import torch, soundfile as sf
    import torchaudio
    from transformers import AutoFeatureExtractor, WavLMForXVector
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
    def emb(path):
        a, sr = sf.read(path, dtype="float32")
        if a.ndim > 1: a = a.mean(1)
        a = torchaudio.functional.resample(torch.from_numpy(a), sr, 16000).numpy()
        with torch.no_grad():
            e = sv(**fe(a, sampling_rate=16000, return_tensors="pt")).embeddings[0]
        return torch.nn.functional.normalize(e, dim=0)
    ref = emb(os.path.join(HERE, "ref30.wav"))
    for d in DIRS:
        out = {os.path.basename(p): float(emb(p) @ ref) for p in wavs(d)}
        json.dump(out, open(os.path.join(d, "sv.json"), "w"), indent=0)
        print("sv", d, len(out))
else:
    import soundfile as sf
    for d in DIRS:
        asr = json.load(open(os.path.join(d, "asr.json")))
        svs = json.load(open(os.path.join(d, "sv.json")))
        rows = []
        for name, hyp in asr.items():
            i = int(re.search(r"_(\d+)\.wav$", name).group(1))
            if str(i) in os.environ.get("EVAL_EXCLUDE", "").split(","):
                continue
            rows.append((i, cer(TEXTS[i], hyp), svs[name], sf.info(os.path.join(d, name)).duration, hyp, name))
        c = np.array([r[1] for r in rows]); sv_ = np.array([r[2] for r in rows]); du = np.array([r[3] for r in rows])
        bad = [r for r in rows if r[1] > 0.3]
        print(f"{os.path.basename(d.rstrip('/')):18s} n={len(rows)} CER 平均 {c.mean()*100:5.1f}% 中央 {np.median(c)*100:4.1f}% 30%超 {len(bad):2d}  類似度 平均 {sv_.mean():.3f} 最小 {sv_.min():.3f}  長さ合計 {du.sum():5.1f}s")
        if "-v" in os.environ.get("EVAL_FLAGS", ""):
            for r in bad:
                print(f"     {r[5]} CER {r[1]*100:.0f}%  {TEXTS[r[0]]} -> {r[4]}")
