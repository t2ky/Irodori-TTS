"""Batching HTTP server for Irodori-TTS on one GPU (see TURING.md).

Same contract as the local demo server (POST /synth, GET /health), but concurrent requests are
synthesized together: the worker takes the first waiting request, waits up to IRODORI_BATCH_WINDOW_MS
for more (at most IRODORI_MAX_BATCH), and runs the compatible ones as one TuringBatcher batch
(fp16, CUDA graphs, compiled blocks / decoder, batched length check).

  POST /synth  {"text", "voice"?(name in IRODORI_VOICES_DIR) | "ref_audio"?(base64 wav),
                "caption"?, "seed"?, "num_steps"?, "speaker_kv_scale"?,
                "cfg_scale_text"?, "cfg_scale_caption"?, "cfg_scale_speaker"?, "watermark"?}
            -> {"sample_rate", "audio"(base64 int16 LE mono), "seconds", "batch_size",
                "audio_seconds", "model", "seed", "watermarked"}
  POST /tts    same body with a whole reply ("text" up to 2000 chars, "split"?=true) -> audio/wav
               (IRODORI_OUTPUT_SR=24000, mono, PCM 16), what the app plays. The reply is split into
               sentence-sized pieces (reply_split.py: IRODORI_SPLIT_TARGET=30 / _MIN=8 / _MAX=70),
               synthesized as one batch and joined with IRODORI_GAP_MS=200 of silence (IRODORI_TRIM=1 also
               cuts each piece's lead-in / tail first; off by default); X-Pieces / X-Segments (start of each piece in ms)
  GET  /health

env: IRODORI_MODEL=small|small-mf  IRODORI_PORT=8097  IRODORI_HOST=127.0.0.1
     IRODORI_PRECISION=fp16|fp32  IRODORI_MAX_BATCH=8  IRODORI_BATCH_WINDOW_MS=10
     IRODORI_LENGTH_CHECK=1  IRODORI_DEFAULT_SEED=0  IRODORI_DEFAULT_STEPS / IRODORI_SWAY (RF only)
     IRODORI_WARMUP=1  IRODORI_WARMUP_REF=<wav>  IRODORI_COMPILE=1  IRODORI_GRAPHS=1
     IRODORI_LATENT_MULTIPLE=16  IRODORI_SPEAKER_MULTIPLE=64  IRODORI_GRAPH_MAX_ROWS=8
     IRODORI_VOICES_DIR=<dir of <name>.wav | <name>.pt | <name>.speaker.safetensors (Speaker Inversion),
     and voices.json with per-voice cfg_scale_speaker / speaker_kv_scale / num_steps>: voices selectable by name (no upload per
     request; the first one also warms up the server). <name>.pt is a reference latent made by
     bench/make_latent.py: the server then never runs the codec encoder (no VRAM peak for it)
     IRODORI_ALLOW_REF_AUDIO=0: refuse "ref_audio" uploads (registered voices only)
     IRODORI_ROW_STREAM=0: answer a batch's requests together (default: each as soon as it is decoded)
     IRODORI_GRAPH_FREEZE=0: keep capturing graphs / growing their slots after warm-up
     IRODORI_COMPILE_FREEZE=0: allow torch.compile recompiles after warm-up
     IRODORI_WARMUP_ROUNDS=2: passes over batch sizes MAX_BATCH..1 with varied sentences at warm-up
     IRODORI_SDPA_CUDNN=1: allow cuDNN attention (default off here: a plan per new length, see below)
     IRODORI_CUDA_MEMORY_MB=<n>: cap PyTorch's allocator (sharing a GPU; the CUDA context is extra)
     IRODORI_GRAPH_ENCODER=0: no CUDA graphs for the text encoder (default: shapes up to
     IRODORI_ENCODER_GRAPH_TOKENS=96 tokens captured at warm-up; longer sentences run eagerly)
     IRODORI_FAST_WATERMARK=0: the packaged SilentCipher encode (default: same output, no host round trips)
     IRODORI_GRAPH_WARM_ROWS=4 / _TOKENS=64 / _SECONDS=15: the DiT shapes captured at warm-up (with
     IRODORI_GRAPHS=1 and registered voices); other shapes run eagerly after the freeze
     IRODORI_WARMUP_LONG_TEXT: the longest text of the warm-up; it sizes the graphs (see _warmup)
     IRODORI_SPEECH_CHECK=1: on /tts, score every piece against its text (speech_check.py: phoneme CTC +
     forced alignment) and synthesize a piece that scores below IRODORI_SPEECH_CHECK_MIN=-2.0 once more
     with another seed, keeping the better one. Counts and scores (no text) go to the log and to
     IRODORI_SPEECH_CHECK_LOG (JSON lines); /health has the totals. Each line has hashes of the reply / piece
     (find a reported reply with speech_check_lookup.py)

Privacy: request text / caption are never logged or written to disk; reference audio is written
once (named by its sha256) to a private tmpfs dir for the runtime's latent cache and removed at
shutdown. Error responses carry only the exception type.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import concurrent.futures
import dataclasses
import gc
import hashlib
import io
import json
import logging
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from irodori_tts import length_check  # noqa: E402
from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    batch_compatibility_key,
    download_hf_checkpoint,
    resolve_cfg_scales,
)
from irodori_tts.turing import TuringBatcher  # noqa: E402

import reply_split  # noqa: E402
import speech_check  # noqa: E402

log = logging.getLogger("turing_server")

# cuDNN's attention builds an execution plan for every new shape: on sm80+ (it is not used on Turing)
# the first request of each new length paid ~0.7 s (RTX 2000 Ada, MF; the same call again 0.1 s).
# Without it SDPA falls back to flash / memory-efficient attention at the same speed. Set
# IRODORI_SDPA_CUDNN=1 to keep cuDNN.
os.environ.setdefault("IRODORI_SDPA_CUDNN", "0")

MODELS = {"small": "Aratako/Irodori-TTS-v4.1-Small", "small-mf": "Aratako/Irodori-TTS-v4.1-Small-MF"}
MODEL = os.environ.get("IRODORI_MODEL", "small-mf")
PRECISION = os.environ.get("IRODORI_PRECISION", "fp16")
MAX_BATCH = int(os.environ.get("IRODORI_MAX_BATCH", "8"))
WINDOW = float(os.environ.get("IRODORI_BATCH_WINDOW_MS", "10")) / 1000.0
LENGTH_CHECK = os.environ.get("IRODORI_LENGTH_CHECK", "1") not in ("0", "false", "off")
DEFAULT_SEED = int(os.environ.get("IRODORI_DEFAULT_SEED", "0"))
DEFAULT_STEPS = int(os.environ["IRODORI_DEFAULT_STEPS"]) if os.environ.get("IRODORI_DEFAULT_STEPS") else None
SWAY = float(os.environ["IRODORI_SWAY"]) if os.environ.get("IRODORI_SWAY") else None  # RF only
FLAG = lambda name, default="1": os.environ.get(name, default) not in ("0", "false", "off")  # noqa: E731
VOICES_DIR = Path(os.environ["IRODORI_VOICES_DIR"]) if os.environ.get("IRODORI_VOICES_DIR") else None
MAX_TEXT_CHARS = 300
# IRODORI_ALLOW_REF_AUDIO=0: refuse uploaded references (encoding one is a transient VRAM peak of
# ~1.4GB for 30-120 s of audio, too much next to another tenant on the same GPU)
ALLOW_REF_AUDIO = FLAG("IRODORI_ALLOW_REF_AUDIO")
# IRODORI_ROW_STREAM=1 (default): answer each request of a batch as soon as its row is decoded and
# watermarked (shortest first) instead of when the whole batch is done
ROW_STREAM = FLAG("IRODORI_ROW_STREAM")
# sizes the graphs at warm-up (see _warmup); sentences longer than this run without CUDA graphs
WARMUP_LONG_TEXT = os.environ.get("IRODORI_WARMUP_LONG_TEXT", (
    "今日はね、朝から雨が降っていたから、駅まで歩くのがすごく大変だったんだけど、"
    "帰りにはすっかり晴れて、きれいな夕焼けが見えたんだよ。あなたにも見せたかったな。"))
WARMUP_TEXTS = [
    "うん。", "そうなんだ！", "えへへ、ありがと。", "おはよう、よく眠れた？", "またね！",
    "ちょっと待って、今考えてるから。", "それ、すごく面白そうだね！私もやってみたいな。",
    "ねえ、聞いてくれる？今日ね、すごく嬉しいことがあったんだ。",
    "えっと、それってどういう意味なのかな？もう少し詳しく教えてほしいな。",
    "ごめんね、昨日は返事できなくて。ちょっとバタバタしてて、気づいたら寝ちゃってたの。",
    "雨の日って、なんだか落ち着くよね。窓の外を眺めながら、あったかいココアを飲むのが好きなんだ。",
]


_EMBED_SUFFIX = ".speaker.safetensors"


def _voice_files() -> dict[str, Path]:
    """name -> file in IRODORI_VOICES_DIR. A Speaker Inversion embedding (<name>.speaker.safetensors)
    wins over a precomputed latent (<name>.pt), which wins over <name>.wav."""
    if VOICES_DIR is None:
        return {}
    out: dict[str, Path] = {}
    for p in (sorted(VOICES_DIR.glob("*.wav")) + sorted(VOICES_DIR.glob("*.pt"))
              + sorted(VOICES_DIR.glob("*" + _EMBED_SUFFIX))):
        name = p.name[: -len(_EMBED_SUFFIX)] if p.name.endswith(_EMBED_SUFFIX) else p.stem
        out[name] = p
    return dict(sorted(out.items()))


_voice_settings_cache: dict | None = None


def _voice_settings() -> dict[str, dict]:
    """Per-voice defaults from IRODORI_VOICES_DIR/voices.json:
    {"<name>": {"cfg_scale_speaker": 2.0, "speaker_kv_scale": 1.5, "num_steps": 10}, ...}.
    A request's own values win. A voice learned with Speaker Inversion is read with a weak speaker CFG
    and a K/V scale (the setting the voices were chosen with)."""
    global _voice_settings_cache
    if _voice_settings_cache is None:
        p = VOICES_DIR / "voices.json" if VOICES_DIR is not None else None
        _voice_settings_cache = json.loads(p.read_text()) if p is not None and p.exists() else {}
    return _voice_settings_cache


def _voice_name(path: str | None) -> str | None:
    if path is None:
        return None
    for name, p in _voice_files().items():
        if str(p) == path:
            return name
    return None
MAX_REF_BYTES = 30 * 1024 * 1024

_state: dict = {}
_ref_dir: Path | None = None


class _NoWatermark:
    ready = False


@dataclasses.dataclass
class _Job:
    req: SamplingRequest
    watermark: bool
    future: concurrent.futures.Future
    enqueued: float


class BatchWorker:
    """One thread owns the GPU; requests are grouped by compatibility and watermark flag."""

    def __init__(self, rt: InferenceRuntime, batcher: TuringBatcher) -> None:
        self.rt, self.batcher = rt, batcher
        self.jobs: queue.Queue[_Job] = queue.Queue()
        self.pending: list[_Job] = []
        self.thread = threading.Thread(target=self._loop, name="tts-batch", daemon=True)
        self.batches = 0
        self.oom_retries = 0
        self.thread.start()

    def submit(self, req: SamplingRequest, watermark: bool) -> concurrent.futures.Future:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self.jobs.put(_Job(req, watermark, fut, time.perf_counter()))
        return fut

    def _gather(self) -> list[_Job]:
        if not self.pending:
            self.pending.append(self.jobs.get())
        deadline = time.perf_counter() + WINDOW
        while len(self.pending) < MAX_BATCH * 2:
            left = deadline - time.perf_counter()
            if left <= 0:
                break
            try:
                self.pending.append(self.jobs.get(timeout=left))
            except queue.Empty:
                break
        while True:  # drain whatever else already arrived
            try:
                self.pending.append(self.jobs.get_nowait())
            except queue.Empty:
                break
        head = self.pending[0]
        key = (self._key(head.req), head.watermark)
        batch = [j for j in self.pending if (self._key(j.req), j.watermark) == key][:MAX_BATCH]
        for j in batch:
            self.pending.remove(j)
        return batch

    def _key(self, req: SamplingRequest):
        return (batch_compatibility_key(req), self.batcher.supports(req))

    def _loop(self) -> None:
        while True:
            batch = self._gather()
            try:
                self._run_jobs(batch)
            except Exception as exc:  # noqa: BLE001
                log.warning("batch failed: %s", type(exc).__name__)
                exc.__traceback__ = None  # the futures keep the exception; not the failed run's tensors
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(exc)
                del exc
                gc.collect()
                torch.cuda.empty_cache()
            self.batches += 1

    def _run_jobs(self, batch: list[_Job], retried: bool = False) -> None:
        """Run a batch and answer its jobs. Under IRODORI_CUDA_MEMORY_MB a batch can run out of memory
        (the peak depends on fragmentation and on the length check's second pass, not only on the
        batch's size): then give the cache back and run the jobs not answered yet in two halves, down
        to one job, which gets one more try before it fails. The other jobs wait in the meantime."""
        def on_row(i, res):  # answer each row as soon as it is decoded
            job = batch[i]
            if not job.future.done():
                job.future.set_result((res, time.perf_counter() - job.enqueued, len(batch)))

        results = None
        try:
            results = self.run([j.req for j in batch], watermark=batch[0].watermark,
                               on_row=on_row if ROW_STREAM else None)
        except torch.OutOfMemoryError:
            pass  # retried below: inside this block the traceback still holds the failed run's tensors
        if results is None:
            gc.collect()
            torch.cuda.empty_cache()
            left = [j for j in batch if not j.future.done()]
            self.oom_retries += 1
            if len(left) > 1:
                log.warning("batch of %d hit the memory limit; running the %d left in halves",
                            len(batch), len(left))
                half = (len(left) + 1) // 2
                self._run_jobs(left[:half])
                self._run_jobs(left[half:])
                return
            if left and not retried:
                log.warning("one row hit the memory limit; retrying after emptying the cache")
                self._run_jobs(left, retried=True)
                return
            if left:
                raise torch.OutOfMemoryError("one row does not fit under IRODORI_CUDA_MEMORY_MB")
            return
        now = time.perf_counter()
        for job, res in zip(batch, results, strict=True):
            if not job.future.done():
                job.future.set_result((res, now - job.enqueued, len(batch)))

    def run(self, reqs: list[SamplingRequest], *, watermark: bool = True, on_row=None):
        rt = self.rt
        saved = rt.watermarker
        if not watermark:
            rt.watermarker = _NoWatermark()
        try:
            if all(self.batcher.supports(r) for r in reqs):
                return self.batcher.synthesize_batch(reqs, length_check=LENGTH_CHECK, on_row=on_row)
            # outside the fast path (e.g. no reference): upstream, one by one
            out = []
            for r in reqs:
                if LENGTH_CHECK:
                    out.append(length_check.synthesize(rt, r)[0])
                else:
                    out.append(rt.synthesize(r))
            return out
        finally:
            rt.watermarker = saved


def _warmup(worker: BatchWorker, ref_wav: str | None) -> None:
    """Capture graphs / compile for batch sizes 1..MAX_BATCH with short and long texts."""
    # the longest first: the batch of MAX_BATCH rows with it sizes the graphs' input slots, which
    # are frozen afterwards (longer calls then run without graphs instead of dropping them all)
    texts = [WARMUP_LONG_TEXT] + WARMUP_TEXTS
    t0 = time.perf_counter()
    # largest first: the graphs' shared input buffers are sized once instead of growing (each
    # growth drops the graphs captured so far). Then every batch size with sentences of varied
    # lengths, IRODORI_WARMUP_ROUNDS times, so that the shapes a chat produces are captured /
    # compiled here and not in front of a user (after warm-up neither happens any more, see below)
    rounds = int(os.environ.get("IRODORI_WARMUP_ROUNDS", "2"))
    worker.run([_request(WARMUP_LONG_TEXT, ref_wav, None, i, None, {}) for i in range(MAX_BATCH)], watermark=True)
    k = 0
    for _ in range(rounds):
        for n in range(MAX_BATCH, 0, -1):
            reqs = [_request(texts[(k + i) % len(texts)], ref_wav, None, k + i, None, {}) for i in range(n)]
            worker.run(reqs, watermark=True)
            k += n
    if worker.batcher.graphs is not None and FLAG("IRODORI_GRAPH_FREEZE") and _voice_files():
        # every DiT shape a chat produces, so that none runs eagerly after the freeze (on Turing an
        # eager DiT call costs ~3x a replay; without this grid every sentence missed the graphs)
        t1 = time.perf_counter()
        voices = [str(v) for v in _voice_files().values()]
        n = worker.batcher.warm_dit(
            lambda text, seed: [_request(text, v, None, seed, None, {}) for v in voices], max_rows=int(os.environ.get("IRODORI_GRAPH_WARM_ROWS", "4")),
            max_tokens=int(os.environ.get("IRODORI_GRAPH_WARM_TOKENS", "64")),
            max_seconds=float(os.environ.get("IRODORI_GRAPH_WARM_SECONDS", "15")))
        log.warning("DiT graphs: %d in %.1fs (speaker padded to %d)", n, time.perf_counter() - t1,
                    worker.batcher.speaker_pad_to)
    t1 = time.perf_counter()
    # tokens are ~2 characters each: 96 covers sentences up to ~190 characters; longer ones run the
    # encoder eagerly. Every captured shape keeps its output in the graph pool, which the allocator
    # cannot give back (all 256 lengths took ~370 MB and ran a long reply into the memory cap)
    n_enc = worker.batcher.warm_encoder(MAX_BATCH, int(os.environ.get("IRODORI_ENCODER_GRAPH_TOKENS", "96")))
    if n_enc:
        log.warning("text encoder graphs: %d in %.1fs", n_enc, time.perf_counter() - t1)
    graphs = worker.batcher.graphs
    if graphs is not None and FLAG("IRODORI_GRAPH_FREEZE"):
        graphs.freeze()
    if FLAG("IRODORI_COMPILE_FREEZE") and FLAG("IRODORI_COMPILE"):
        # a shape that would need a recompile runs that compiled region eagerly instead (a
        # recompile is seconds of latency for the request that hits it)
        torch.compiler.set_stance("eager_on_recompile")
    log.warning("warm-up %.1fs (%d graphs)", time.perf_counter() - t0,
                len(graphs.graphs) if graphs else 0)


def _request(text, ref_wav, caption, seed, num_steps, cfg, kv_scale=None) -> SamplingRequest:
    rt = _state["rt"]
    caption = caption.strip() if caption and caption.strip() else None
    use_spk = bool(rt.model_cfg.use_speaker_condition_resolved and ref_wav is not None)
    # the registered voice's defaults (voices.json), under the request's own values
    vs = _voice_settings().get(_voice_name(ref_wav) or "", {})
    cfg = {"speaker": vs["cfg_scale_speaker"], **cfg} if "cfg_scale_speaker" in vs else cfg
    kv_scale = vs.get("speaker_kv_scale") if kv_scale is None else kv_scale
    num_steps = vs.get("num_steps") if num_steps is None else num_steps
    # a voice may be a Speaker Inversion embedding (<name>.speaker.safetensors) or a precomputed
    # reference latent (<name>.pt, see bench/make_latent.py): no codec encode on the server, so no
    # transient VRAM peak for the reference
    ref_kw = ({} if ref_wav is None else
              {"ref_embed": ref_wav} if ref_wav.endswith(_EMBED_SUFFIX) else
              {"ref_latent": ref_wav} if ref_wav.endswith(".pt") else {"ref_wav": ref_wav})
    t, c, s, _ = resolve_cfg_scales(
        cfg_guidance_mode="independent", cfg_scale_text=cfg.get("text", 3.0),
        cfg_scale_caption=cfg.get("caption", 3.0), cfg_scale_speaker=cfg.get("speaker", 5.0),
        cfg_scale=None, use_caption_condition=bool(rt.model_cfg.use_caption_condition and caption),
        use_speaker_condition=use_spk,
    )
    steps = num_steps if num_steps is not None else (
        DEFAULT_STEPS if rt.model_cfg.flow_parameterization != "meanflow" else None)
    sway = {}
    if SWAY is not None and rt.model_cfg.flow_parameterization != "meanflow":
        sway = {"t_schedule_mode": "sway", "sway_coeff": SWAY}
    return SamplingRequest(
        text=text, caption=caption, **ref_kw, no_ref=ref_wav is None,
        cfg_scale_text=t, cfg_scale_caption=c, cfg_scale_speaker=s, seed=seed, num_steps=steps,
        # RF only (MeanFlow ignores it); a batch shares one value (it is in the compatibility key)
        speaker_kv_scale=(kv_scale if use_spk and rt.model_cfg.flow_parameterization != "meanflow"
                          else None),
        **sway,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ref_dir
    base = Path("/dev/shm") if os.path.isdir("/dev/shm") else Path(tempfile.gettempdir())
    _ref_dir = base / f"irodori-turing-ref-{os.getuid()}-{os.environ.get('IRODORI_PORT', '8097')}"
    shutil.rmtree(_ref_dir, ignore_errors=True)
    _ref_dir.mkdir(mode=0o700)
    t0 = time.perf_counter()
    limit_mb = int(os.environ.get("IRODORI_CUDA_MEMORY_MB", "0"))
    if limit_mb > 0:
        # a hard ceiling for PyTorch's caching allocator: at the limit it frees its cache and
        # retries before failing, so the process stops growing instead of taking another tenant's
        # VRAM (the CUDA context, ~0.3-0.5 GB, is outside this limit)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, limit_mb * 2**20 / total))
    rt = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=download_hf_checkpoint(MODELS[MODEL]), model_device="cuda",
        model_precision=PRECISION, codec_device="cuda", codec_precision="fp32",
    ))
    fp16 = PRECISION == "fp16"
    batcher = TuringBatcher(
        rt, decoder_fp16=fp16, watermark_fp16=fp16, cuda_graphs=FLAG("IRODORI_GRAPHS"),
        compile_decoder=fp16 and FLAG("IRODORI_COMPILE"), compile_blocks=FLAG("IRODORI_COMPILE"),
        latent_multiple=int(os.environ.get("IRODORI_LATENT_MULTIPLE", "16")),
        speaker_multiple=int(os.environ.get("IRODORI_SPEAKER_MULTIPLE", "64")),
        graph_max_rows=int(os.environ.get("IRODORI_GRAPH_MAX_ROWS", "8")),
        graph_max_graphs=int(os.environ.get("IRODORI_GRAPH_MAX_GRAPHS", "512")),
        graph_encoder=FLAG("IRODORI_GRAPH_ENCODER"),
        fast_watermark=FLAG("IRODORI_FAST_WATERMARK"),
        sync_timings=False,
    )
    _state.update(rt=rt, batcher=batcher)
    if FLAG("IRODORI_SPEECH_CHECK", "0"):
        _state["check"] = speech_check.SpeechCheck("cuda")
    worker = BatchWorker(rt, batcher)
    _state["worker"] = worker
    log.warning("loaded %s (%s) in %.1fs", MODEL, PRECISION, time.perf_counter() - t0)
    if FLAG("IRODORI_WARMUP"):
        warm_ref = os.environ.get("IRODORI_WARMUP_REF")
        voices = _voice_files()
        if not warm_ref and voices:
            warm_ref = str(next(iter(voices.values())))
        if not warm_ref:
            # a synthetic 10 s reference so that warm-up exercises the fast (reference) path
            sr = 48000
            tt = np.arange(sr * 10) / sr
            wav = (0.1 * np.sin(2 * np.pi * 180 * tt) * (1 + 0.5 * np.sin(2 * np.pi * 3 * tt))).astype(np.float32)
            warm_ref = str(_ref_dir / "warmup.wav")
            sf.write(warm_ref, wav, sr)
        _warmup(worker, warm_ref)
    log.warning("cuda memory allocated=%.0fMiB reserved=%.0fMiB",
                torch.cuda.memory_allocated() / 2**20, torch.cuda.memory_reserved() / 2**20)
    yield
    shutil.rmtree(_ref_dir, ignore_errors=True)


app = FastAPI(lifespan=lifespan)


class SynthIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    voice: str | None = Field(None, pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    ref_audio: str | None = None
    caption: str | None = None
    seed: int | None = None
    model: str | None = None
    num_steps: int | None = None
    cfg_scale_text: float | None = Field(None, ge=0, le=20)
    cfg_scale_caption: float | None = Field(None, ge=0, le=20)
    cfg_scale_speaker: float | None = Field(None, ge=0, le=20)
    speaker_kv_scale: float | None = Field(None, gt=0, le=3)
    watermark: bool = True


def _ref_path(b64: str) -> str:
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "ref_audio is not valid base64")
    if len(raw) > MAX_REF_BYTES:
        raise HTTPException(413, "ref_audio too large")
    try:
        sf.info(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "ref_audio is not a readable audio file")
    p = _ref_dir / (hashlib.sha256(raw).hexdigest() + ".wav")
    if not p.exists():
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(raw)
        os.chmod(tmp, 0o600)
        tmp.replace(p)
    return str(p)


@app.get("/health")
def health():
    rt = _state.get("rt")
    return {
        "ok": rt is not None,
        "models": {MODEL: MODELS[MODEL]},
        "default_model": MODEL,
        "precision": PRECISION,
        "length_check": LENGTH_CHECK,
        "watermark": bool(rt is not None and rt.watermarker.ready),
        "max_batch": MAX_BATCH,
        "batch_window_ms": WINDOW * 1000.0,
        "batches": _state["worker"].batches if "worker" in _state else 0,
        "oom_retries": _state["worker"].oom_retries if "worker" in _state else 0,
        "split": {"target": reply_split.TARGET, "min": reply_split.MIN_LEN, "max": reply_split.HARD_MAX,
                  "weights": reply_split.WEIGHTS,
                  "gap_ms": GAP_MS, "trim": TRIM},
        "graphs": ({"kept": len(g.graphs), "captures": g.captures, "replays": g.replays,
                    "eager_calls": g.eager_calls, "frozen": g.frozen}
                   if (g := getattr(_state.get("batcher"), "graphs", None)) is not None else None),
        "voices": sorted(_voice_files()),
        "speech_check": ({"min_score": speech_check.MIN_SCORE, **c.counts}
                         if (c := _state.get("check")) is not None else None),
    }


async def _synthesize(body: SynthIn):
    """(result, elapsed seconds, batch size) for a /synth or /tts request."""
    if body.model not in (None, MODEL):
        raise HTTPException(400, f"this server runs model {MODEL!r} only")
    if body.voice is not None:
        path = _voice_files().get(body.voice)
        if path is None:
            raise HTTPException(400, "unknown voice")
        ref = str(path)
    elif body.ref_audio and not ALLOW_REF_AUDIO:
        raise HTTPException(400, "ref_audio is disabled on this server (use a registered voice)")
    else:
        ref = (await asyncio.to_thread(_ref_path, body.ref_audio)) if body.ref_audio else None
    seed = DEFAULT_SEED if body.seed is None else int(body.seed)
    cfg = {k: v for k, v in (("text", body.cfg_scale_text), ("caption", body.cfg_scale_caption),
                             ("speaker", body.cfg_scale_speaker)) if v is not None}
    t0 = time.perf_counter()
    try:
        req = _request(body.text, ref, body.caption, seed, body.num_steps, cfg, body.speaker_kv_scale)
        result, waited, batch_size = await asyncio.wrap_future(_state["worker"].submit(req, body.watermark))
    except HTTPException:
        raise
    except Exception as exc:  # do not echo text
        log.warning("synth failed: %s", type(exc).__name__)
        raise HTTPException(500, f"synthesis failed ({type(exc).__name__})")
    return result, time.perf_counter() - t0, batch_size


# /tts: the WAV the app plays as it is (mono, 16-bit PCM). 24 kHz like VOICEVOX: the app's lip-sync
# analysis runs on the main thread and costs ~2x at 48 kHz, and the file is half the size
OUTPUT_SR = int(os.environ.get("IRODORI_OUTPUT_SR", "24000"))


# the pause between the pieces of a reply (the old VOICEVOX path of coreapi used 200 ms), and the level
# below a piece's peak that counts as silence when its lead-in / tail is trimmed before the pause
GAP_MS = int(os.environ.get("IRODORI_GAP_MS", "200"))
# trimming is off by default: kiyo found the pause after a trimmed piece unnatural (2026-11-20). The
# model's own lead-in / tail is kept and GAP_MS is added on top
TRIM = FLAG("IRODORI_TRIM", "0")
TRIM_DB = float(os.environ.get("IRODORI_TRIM_DB", "-40"))
TRIM_MARGIN_MS = int(os.environ.get("IRODORI_TRIM_MARGIN_MS", "30"))
MAX_REPLY_CHARS = 2000


def _pcm(audio: torch.Tensor, sr: int) -> tuple[np.ndarray, int]:
    """float audio -> int16 at IRODORI_OUTPUT_SR"""
    import torchaudio.functional as AF

    x = audio.detach().float().reshape(1, -1).cpu()
    if OUTPUT_SR and sr != OUTPUT_SR:
        x = AF.resample(x, sr, OUTPUT_SR)
        sr = OUTPUT_SR
    return (np.clip(x.reshape(-1).numpy(), -1.0, 1.0) * 32767.0).astype("<i2"), sr


def _trim(pcm: np.ndarray, sr: int) -> np.ndarray:
    """Cut the silence before and after a piece (a piece ends with ~0.2 s of it on average, up to
    1.1 s), keeping TRIM_MARGIN_MS, so that the pause between pieces is GAP_MS and not more."""
    x = pcm.astype(np.float32)
    win = max(1, sr // 100)  # 10 ms
    env = np.sqrt(np.convolve(x * x, np.ones(win) / win, "same"))
    loud = np.flatnonzero(env > env.max() * 10 ** (TRIM_DB / 20)) if env.max() > 0 else []
    if len(loud) == 0:
        return pcm
    m = sr * TRIM_MARGIN_MS // 1000
    return pcm[max(int(loud[0]) - m, 0): int(loud[-1]) + m]


def _wav(pcm: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _join(results) -> tuple[bytes, list[int]]:
    """Pieces -> one WAV with GAP_MS between them, and the start of each piece in ms."""
    pieces = [_pcm(r.audio, int(r.sample_rate)) for r in results]
    sr = pieces[0][1]
    if len(pieces) == 1:
        return _wav(pieces[0][0], sr), [0]
    gap = np.zeros(sr * GAP_MS // 1000, dtype="<i2")
    out, starts, pos = [], [], 0
    for k, (pcm, _) in enumerate(pieces):
        if TRIM:
            pcm = _trim(pcm, sr)
        if k:
            out.append(gap)
            pos += len(gap)
        starts.append(pos * 1000 // sr)
        out.append(pcm)
        pos += len(pcm)
    return _wav(np.concatenate(out), sr), starts


class TtsIn(SynthIn):
    # a whole reply: it is split into pieces of about IRODORI_SPLIT_TARGET characters (reply_split)
    text: str = Field(..., min_length=1, max_length=MAX_REPLY_CHARS)
    split: bool = True


@app.post("/tts")
async def tts(body: TtsIn):
    """A reply -> audio/wav (IRODORI_OUTPUT_SR, mono, PCM 16), what the app plays.

    The text is split into sentence-sized pieces (reply_split; Irodori reads long text worse and clips
    it at 30 s). The pieces go into the queue together, so they are synthesized as one batch (with any
    other requests that arrive at the same time), then joined with GAP_MS of silence (IRODORI_TRIM=1 trims
    each piece first). X-Segments has the start of each piece in ms. split=false reads the text as one piece."""
    texts = reply_split.split_reply(body.text) if body.split else [body.text]
    if not texts or any(len(t) > MAX_TEXT_CHARS for t in texts):
        raise HTTPException(422, f"a piece is longer than {MAX_TEXT_CHARS} characters (split it)")
    t0 = time.perf_counter()
    outs = await asyncio.gather(*(_synthesize(body.model_copy(update={"text": t})) for t in texts))
    results = [o[0] for o in outs]
    retried = 0
    if (check := _state.get("check")) is not None:
        results, retried = await _check_pieces(check, body, texts, results, t0)
    wav, starts = await asyncio.to_thread(_join, results)
    return Response(wav, media_type="audio/wav", headers={
        "X-Synth-Seconds": f"{time.perf_counter() - t0:.4f}",
        "X-Batch-Size": str(max(o[2] for o in outs)),
        "X-Pieces": str(len(texts)),
        "X-Segments": ",".join(map(str, starts)),
        "X-Retried": str(retried),
    })


def _score_piece(check, res, phonemes):
    sr = int(res.sample_rate)
    score, heard = check.score(res.audio, sr, phonemes)
    return score, heard, round(res.audio.numel() / sr, 3)


async def _check_pieces(check, body, texts, results, t0):
    """Score each piece (speech_check.py); synthesize the low ones once more with another seed and keep
    the better take. One JSON line per piece: counts and scores only, never the text."""
    req_id = os.urandom(4).hex()
    reply_hash = speech_check.text_hash(body.text)
    synth_ms = round((time.perf_counter() - t0) * 1000)
    phs = [check.expected(t) for t in texts]
    skips = [speech_check.reason_to_skip(t, len(p)) for t, p in zip(texts, phs)]
    # one thread for all the pieces: a thread per piece fought the batch worker for the GIL
    tc = time.perf_counter()
    scored = await asyncio.to_thread(lambda: [_score_piece(check, r, p) for r, p in zip(results, phs)])
    check_ms = round((time.perf_counter() - tc) * 1000)
    low = [i for i, ((sc, _, _), sk) in enumerate(zip(scored, skips)) if sk is None and sc < speech_check.MIN_SCORE]
    second = {}
    if low:
        base = DEFAULT_SEED if body.seed is None else int(body.seed)
        redo = await asyncio.gather(*(_synthesize(body.model_copy(update={"text": texts[i], "seed": base + 7919 * (i + 1)}))
                                      for i in low), return_exceptions=True)
        for i, o in zip(low, redo):
            if isinstance(o, BaseException):
                continue
            second[i] = (o[0], *(await asyncio.to_thread(_score_piece, check, o[0], phs[i])))
    out = list(results)
    for i, (t, p, sk, (sc, heard, dur)) in enumerate(zip(texts, phs, skips, scored)):
        rec = {"ts": round(time.time(), 3), "req": req_id, "reply_hash": reply_hash, "piece_hash": speech_check.text_hash(t),
               "voice": body.voice, "piece": i, "pieces": len(texts),
               "chars": len(t), "phonemes": len(p), "heard": heard, "sec": dur, "score": round(sc, 3), "skip": sk,
               "synth_ms": synth_ms, "check_ms": check_ms}
        check.count("skipped" if sk else "checked")
        if i in low:
            check.count("flagged")
            rec["retried"] = i in second
            if i in second:
                res2, sc2, heard2, dur2 = second[i]
                check.count("retried")
                rec.update(score2=round(sc2, 3), heard2=heard2, sec2=dur2, kept=2 if sc2 > sc else 1)
                rec["redo_ms"] = round((time.perf_counter() - t0) * 1000) - synth_ms - check_ms
                if sc2 > sc:
                    out[i] = res2
                check.count("recovered" if max(sc, sc2) >= speech_check.MIN_SCORE else "still_low")
        check.write(rec)
    redo_ms = round((time.perf_counter() - t0) * 1000) - synth_ms - check_ms
    if low:
        log.warning("speech check: %d of %d pieces low, %d retried (%.2fs in all)", len(low), len(texts),
                    len(second), time.perf_counter() - t0)
    return out, len(second)


@app.post("/synth")
async def synth(body: SynthIn):
    result, elapsed, batch_size = await _synthesize(body)
    audio = result.audio.detach().float().reshape(-1).numpy()
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return JSONResponse({
        "sample_rate": int(result.sample_rate),
        "audio": base64.b64encode(pcm).decode("ascii"),
        "seconds": round(elapsed, 4),
        "batch_size": batch_size,
        "audio_seconds": round(audio.shape[0] / result.sample_rate, 3),
        "model": MODEL,
        "seed": int(result.used_seed),
        "watermarked": bool(body.watermark and _state["rt"].watermarker.ready),
    })


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    uvicorn.run(app, host=os.environ.get("IRODORI_HOST", "127.0.0.1"),
                port=int(os.environ.get("IRODORI_PORT", "8097")), access_log=False, log_level="warning")
