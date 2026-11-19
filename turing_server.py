"""Batching HTTP server for Irodori-TTS on one GPU (see TURING.md).

Same contract as the local demo server (POST /synth, GET /health), but concurrent requests are
synthesized together: the worker takes the first waiting request, waits up to IRODORI_BATCH_WINDOW_MS
for more (at most IRODORI_MAX_BATCH), and runs the compatible ones as one TuringBatcher batch
(fp16, CUDA graphs, compiled blocks / decoder, batched length check).

  POST /synth  {"text", "voice"?(name in IRODORI_VOICES_DIR) | "ref_audio"?(base64 wav),
                "caption"?, "seed"?, "num_steps"?,
                "cfg_scale_text"?, "cfg_scale_caption"?, "cfg_scale_speaker"?, "watermark"?}
            -> {"sample_rate", "audio"(base64 int16 LE mono), "seconds", "batch_size",
                "audio_seconds", "model", "seed", "watermarked"}
  GET  /health

env: IRODORI_MODEL=small|small-mf  IRODORI_PORT=8097  IRODORI_HOST=127.0.0.1
     IRODORI_PRECISION=fp16|fp32  IRODORI_MAX_BATCH=8  IRODORI_BATCH_WINDOW_MS=10
     IRODORI_LENGTH_CHECK=1  IRODORI_DEFAULT_SEED=0  IRODORI_DEFAULT_STEPS / IRODORI_SWAY (RF only)
     IRODORI_WARMUP=1  IRODORI_WARMUP_REF=<wav>  IRODORI_COMPILE=1  IRODORI_GRAPHS=1
     IRODORI_LATENT_MULTIPLE=16  IRODORI_SPEAKER_MULTIPLE=64  IRODORI_GRAPH_MAX_ROWS=8
     IRODORI_VOICES_DIR=<dir of <name>.wav | <name>.pt>: voices selectable by name (no upload per
     request; the first one also warms up the server). <name>.pt is a reference latent made by
     bench/make_latent.py: the server then never runs the codec encoder (no VRAM peak for it)
     IRODORI_ALLOW_REF_AUDIO=0: refuse "ref_audio" uploads (registered voices only)

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
import hashlib
import io
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
from fastapi.responses import JSONResponse
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

log = logging.getLogger("turing_server")

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


def _voice_files() -> dict[str, Path]:
    """name -> file in IRODORI_VOICES_DIR; a precomputed latent (<name>.pt) wins over <name>.wav."""
    if VOICES_DIR is None:
        return {}
    out: dict[str, Path] = {}
    for p in sorted(VOICES_DIR.glob("*.wav")) + sorted(VOICES_DIR.glob("*.pt")):
        out[p.stem] = p
    return dict(sorted(out.items()))
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
                results = self.run([j.req for j in batch], watermark=batch[0].watermark)
                now = time.perf_counter()
                for job, res in zip(batch, results, strict=True):
                    job.future.set_result((res, now - job.enqueued, len(batch)))
            except Exception as exc:  # noqa: BLE001
                log.warning("batch failed: %s", type(exc).__name__)
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(exc)
            self.batches += 1

    def run(self, reqs: list[SamplingRequest], *, watermark: bool = True):
        rt = self.rt
        saved = rt.watermarker
        if not watermark:
            rt.watermarker = _NoWatermark()
        try:
            if all(self.batcher.supports(r) for r in reqs):
                return self.batcher.synthesize_batch(reqs, length_check=LENGTH_CHECK)
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
    texts = ["えっと、それってどういう意味なのかな？もう少し詳しく教えてほしいな。",
             "おはよう！今日もいい天気だね。", "うん。"]
    t0 = time.perf_counter()
    # largest first: the graphs' shared input buffers are sized once instead of growing (each
    # growth drops the graphs captured so far)
    for n in range(MAX_BATCH, 0, -1):
        reqs = [_request(texts[i % len(texts)], ref_wav, None, i, None, {}) for i in range(n)]
        worker.run(reqs, watermark=True)
    log.warning("warm-up %.1fs (%d graphs)", time.perf_counter() - t0,
                len(worker.batcher.graphs.graphs) if worker.batcher.graphs else 0)


def _request(text, ref_wav, caption, seed, num_steps, cfg) -> SamplingRequest:
    rt = _state["rt"]
    caption = caption.strip() if caption and caption.strip() else None
    use_spk = bool(rt.model_cfg.use_speaker_condition_resolved and ref_wav is not None)
    # a voice may be a precomputed reference latent (<name>.pt, see bench/make_latent.py): no codec
    # encode on the server, so no transient VRAM peak for the reference
    ref_kw = ({} if ref_wav is None else
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
    )
    _state.update(rt=rt, batcher=batcher)
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
        "voices": sorted(_voice_files()),
    }


@app.post("/synth")
async def synth(body: SynthIn):
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
        req = _request(body.text, ref, body.caption, seed, body.num_steps, cfg)
        result, waited, batch_size = await asyncio.wrap_future(_state["worker"].submit(req, body.watermark))
    except HTTPException:
        raise
    except Exception as exc:  # do not echo text
        log.warning("synth failed: %s", type(exc).__name__)
        raise HTTPException(500, f"synthesis failed ({type(exc).__name__})")
    elapsed = time.perf_counter() - t0
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
