"""Peak GPU memory of one synthesize_batch call vs batch size, text length and reference length."""
import sys, os, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, bench as B
from irodori_tts.inference_runtime import SamplingRequest
from irodori_tts.turing import TuringBatcher

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="mf"); a = ap.parse_args()
args = argparse.Namespace(model=a.model, precision="fp16", codec_precision="fp32", compile=False, steps=None)
rt = B.make_runtime(args)
tb = TuringBatcher(rt, decoder_fp16=True, watermark_fp16=True, cuda_graphs=True, compile_decoder=True,
                   compile_blocks=True, latent_multiple=16, speaker_multiple=64)
GiB = 2 ** 30
def smi():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return sum(int(l.split(",")[1]) for l in out.strip().splitlines() if int(l.split(",")[0]) == os.getpid()) / 1024
torch.cuda.synchronize()
base = torch.cuda.memory_allocated()
print(f"重み等（読み込み後） allocated {base/GiB:.2f}GiB  プロセス全体(nvidia-smi) {smi():.2f}GiB")
TEXT = {"短い文(~2s)": "おはよう！今日もいい天気だね。",
        "普通の文(~5s)": "えっと、それってどういう意味なのかな？もう少し詳しく教えてほしいな。",
        "長い文(~12s)": "今日はね、朝から雨が降っていたから、駅まで歩くのがすごく大変だったんだけど、帰りにはすっかり晴れて、きれいな夕焼けが見えたんだよ。"}
REFS = {"参照30秒": "bench/ref30.wav", "参照120秒": "bench/ref120.wav"}
lc = rt.model_cfg.flow_parameterization != "meanflow"
for rname, ref in REFS.items():
    for tname, text in TEXT.items():
        row = []
        for b in (1, 2, 4, 8):
            reqs = [SamplingRequest(text=text, ref_wav=ref, seed=i) for i in range(b)]
            tb.synthesize_batch(reqs, length_check=lc)  # warm (graphs / compile for this shape)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            res = tb.synthesize_batch(reqs, length_check=lc)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            row.append(f"B={b}: +{(peak-base)/GiB:.2f}")
        secs = res[0].audio.shape[-1] / 48000
        print(f"{rname} {tname} 音声{secs:.1f}s | ピーク（重み以外） " + "  ".join(row) + f" GiB | reserved {torch.cuda.memory_reserved()/GiB:.2f} 全体 {smi():.2f}GiB")
print(f"最終: allocated {torch.cuda.memory_allocated()/GiB:.2f} reserved {torch.cuda.memory_reserved()/GiB:.2f} プロセス全体 {smi():.2f}GiB  graphs {len(tb.graphs.graphs)}")
