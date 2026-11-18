"""Batched fast path for serving many short requests at once (see TURING.md).

``TuringBatcher(runtime).synthesize_batch(requests)`` returns the same ``SamplingResult`` list as
``InferenceRuntime.synthesize_batch`` but does the per-request work once for the whole batch:

* text / caption token padding is trimmed to the longest row (rounded up to ``pad_multiple``)
  instead of the checkpoint maximum (256 / 512 tokens with the rare-token splitter). Padded keys are
  masked everywhere, so this only removes work: ModernBERT runs on ~32 tokens instead of 256 + 512 and
  every DiT step attends over ~250 context keys instead of ~1000.
* an empty caption is not run through ModernBERT: its state is all zeros (its mask is all False, so
  attention and the duration predictor's masked pooling never read it).
* the speaker state of each reference is computed once and cached (the same voice is reused across
  requests), then passed as ``speaker_state_override``.
* conditions are encoded once for the batch and shared between the duration predictor and the
  sampler (upstream encodes once per request for the duration and once more for sampling).
* the codec decodes the whole batch in one call.

Requests the fast path does not cover (LoRA, several candidates, speaker embeddings, no reference,
noise speaker-uncond mode) fall back to ``runtime.synthesize_batch``.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import math
import secrets
from collections import OrderedDict
from collections.abc import Callable, Sequence

import torch

from .duration import build_duration_features
from .inference_runtime import (
    InferenceRuntime,
    SamplingRequest,
    SamplingResult,
    _measure_end,
    _measure_start,
    _reference_latent_key,
    _right_pad_cat,
    batch_compatibility_key,
    find_flattening_point,
)
from .meanflow import sample_euler_meanflow
from .codec import unpatchify_latent
from .model import patch_sequence_with_mask
from .rf import draw_initial_noise, sample_euler_rf_cfg

_SPEAKER_CACHE_SIZE = 32
_MISSING = object()


def _trim(ids: torch.Tensor, mask: torch.Tensor, multiple: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut right padding down to the longest row, rounded up to ``multiple``."""
    longest = max(1, int(mask.sum(dim=1).max()))
    keep = min(ids.shape[1], int(math.ceil(longest / multiple) * multiple))
    return ids[:, :keep].contiguous(), mask[:, :keep].contiguous()


class _BatchPointer:
    """length_check.PointerRecorder for every row of the batch at once: for each conditional row
    (the first ``rows`` of the call), the text position each latent frame attends to most."""

    def __init__(self, model, rows: int) -> None:
        from . import length_check as lc

        self.attention = model.blocks[lc.POINTER_LAYER].attention
        self.heads = list(lc.POINTER_HEADS)
        self.rows = rows
        self.pointer: torch.Tensor | None = None  # (rows, S) on CPU
        self.text_len: list[int] = []

    @torch.no_grad()
    def hook(self, mod, args, kwargs):
        n = self.rows
        x = kwargs["x"][:n]
        seq_len = x.shape[1]
        q = mod.wq(x).reshape(n, seq_len, mod.heads, mod.head_dim)
        q = mod._apply_rotary_half(mod.q_norm(q), kwargs["freqs_cis"][:seq_len])
        context_kv = kwargs.get("context_kv")
        if context_kv is not None:
            k_text = context_kv[0][:n]
        else:
            text = kwargs["text_context"][:n]
            k_text = mod.k_norm(mod.wk_text(text).reshape(n, text.shape[1], mod.heads, mod.head_dim))
        valid = kwargs["text_mask"][:n].bool()
        logits = torch.einsum(
            "bshd,bthd->bhst", q[:, :, self.heads].float(), k_text[:, :, self.heads].float()
        )
        logits = logits.masked_fill(~valid[:, None, None, :], float("-inf"))
        self.pointer = logits.argmax(-1).amax(1).cpu()  # (rows, S)
        self.text_len = [int(v) for v in valid.sum(dim=1).tolist()]


class _HookLastCall:
    """DiT forward that runs the ``last_call``-th call eagerly with the pointer hook attached (the
    hook would only fire while a CUDA graph is captured). Independent CFG makes one call per step."""

    def __init__(self, forward, eager, recorder: _BatchPointer, last_call: int, eager_blocks) -> None:
        self.forward, self.eager, self.recorder, self.last_call = forward, eager, recorder, last_call
        self.eager_blocks = eager_blocks
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        if self.calls != self.last_call:
            return self.forward(**kwargs)
        handle = self.recorder.attention.register_forward_pre_hook(self.recorder.hook, with_kwargs=True)
        try:
            with self.eager_blocks():
                out = self.eager(**kwargs)
        finally:
            handle.remove()
        if self.recorder.pointer is None:
            raise RuntimeError("length check: the pointer hook did not run")
        return out


class TuringBatcher:
    def __init__(
        self,
        runtime: InferenceRuntime,
        *,
        pad_multiple: int = 16,
        latent_multiple: int = 1,
        decode_batch: int = 1,
        decoder_fp16: bool = False,
        cuda_graphs: bool = False,
        graph_max_rows: int | None = 8,
        speaker_multiple: int = 1,
        compile_decoder: bool = False,
        compile_blocks: bool = False,
        watermark_fp16: bool = False,
    ) -> None:
        self.rt = runtime
        self.pad_multiple = int(pad_multiple)
        # Round the batch's latent length up (masked frames) so DiT shapes repeat; 1 = exact.
        self.latent_multiple = int(latent_multiple)
        # Rows decoded per codec call: the 48 kHz decoder's activations grow with the batch (8 rows of
        # ~4 s take ~1 GB in one conv), so decode in groups of rows of similar length.
        self.decode_batch = max(1, int(decode_batch))
        # fp16 copy of the codec's decode path only (quantizer.out_proj + decoder). Encoding the
        # reference stays in the codec's own precision: an fp16 encode changes the speaker state
        # enough to move durations (6/8 lengths changed), while an fp16 decode of the same latent
        # is within ~59 dB SNR of fp32 and ~1.5x faster.
        # (block, eager forward) of the compiled DiT blocks, see _eager_blocks
        self._compiled_blocks: list[tuple[torch.nn.Module, Callable]] = []
        if compile_blocks:
            # Fuses the norms / AdaLN / RoPE / SwiGLU element-wise ops of each DiT block. Dynamic
            # shapes, so new lengths do not recompile; works under the CUDA graphs below.
            for block in runtime.model.blocks:
                eager_forward = block.forward
                block.forward = torch.compile(eager_forward, dynamic=True)
                self._compiled_blocks.append((block, eager_forward))
        # SilentCipher's networks under fp16 autocast: ~3x faster (3090: 226 -> 72 ms for 8 rows)
        # with the same decoded payloads and confidences as fp32. Batching the rows (padded) gives
        # nothing: the spectrogram convolutions are compute-bound.
        self.watermark_fp16 = bool(watermark_fp16)
        self.graphs = None
        if cuda_graphs:
            from .cuda_graphs import GraphedDiT

            self.graphs = GraphedDiT(runtime.model, max_rows=graph_max_rows)
        # Pad the speaker state (masked) to a multiple of this many tokens so that references of
        # different lengths share DiT shapes (CUDA graphs are per shape). 1 = exact.
        self.speaker_multiple = max(1, int(speaker_multiple))
        self._decoder16 = None
        if decoder_fp16:
            dac = runtime.codec.model
            # Old-style weight_norm keeps the derived `weight` as a non-leaf attribute, which
            # deepcopy refuses; it is recomputed from weight_g / weight_v on every forward anyway.
            for module in dac.modules():
                if hasattr(module, "weight_g") and isinstance(getattr(module, "weight", None), torch.Tensor):
                    module.weight = module.weight.detach()
            decoder = copy.deepcopy(dac.decoder).half()
            if compile_decoder:
                # The 48 kHz stages are bound by the element-wise Snake activations; fusing them
                # takes ~23% off (3090: 15.0 -> 11.6 ms for 4 s of audio). Dynamic lengths.
                decoder = torch.compile(decoder, dynamic=True)
            self._decoder16 = (copy.deepcopy(dac.quantizer.out_proj).half(), decoder)
        self._speaker_cache: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self.last_duration_tokens: list[dict | None] = []

    # ---------------------------------------------------------------- eligibility
    def supports(self, req: SamplingRequest) -> bool:
        rt = self.rt
        return (
            req.lora_adapter is None
            and int(req.num_candidates or 1) == 1
            and req.ref_embed is None
            and not req.no_ref
            and (req.ref_wav is not None or bool(req.ref_wavs))
            and str(req.speaker_uncond_mode).strip().lower() == "mask"
            and rt.model_cfg.use_speaker_condition_resolved
            and getattr(rt.model, "speaker_inversion", None) is None
        )

    # ---------------------------------------------------------------- pieces
    def _speaker_state(self, req: SamplingRequest, messages: list[str]):
        """(1, S, D) speaker state and (1, S) mask of the request's reference, cached."""
        rt = self.rt
        max_ref_seconds = (
            rt.default_max_ref_seconds if req.max_ref_seconds is None else float(req.max_ref_seconds)
        )
        wav_paths = ([req.ref_wav] if req.ref_wav is not None else []) + list(req.ref_wavs or [])
        key = (_reference_latent_key(req, wav_paths, max_ref_seconds), rt.model.dtype)
        hit = self._speaker_cache.get(key)
        if hit is not None:
            self._speaker_cache.move_to_end(key)
            return hit
        ref_latent, ref_mask = rt._load_reference_latent(req=req, batch_size=1, messages=messages)
        model = rt.model
        latent, mask = patch_sequence_with_mask(
            seq=ref_latent, mask=ref_mask, patch_size=model.cfg.speaker_patch_size
        )
        if int(mask[0].sum()) < 1:
            raise ValueError("Reference sequence too short for the speaker encoder.")
        state = model.speaker_norm(model.speaker_encoder(latent, mask))
        state, mask = model._prepend_masked_mean_token(state, mask)
        self._speaker_cache[key] = (state, mask)
        if len(self._speaker_cache) > _SPEAKER_CACHE_SIZE:
            self._speaker_cache.popitem(last=False)
        return state, mask

    @contextlib.contextmanager
    def _eager_blocks(self):
        """Run the DiT blocks uncompiled: a compiled block inlines its attention module, so hooks
        added to it afterwards (the length-check pointer) would never fire."""
        compiled = [(block, block.forward) for block, _ in self._compiled_blocks]
        for block, eager_forward in self._compiled_blocks:
            block.forward = eager_forward
        try:
            yield
        finally:
            for block, fn in compiled:
                block.forward = fn

    @contextlib.contextmanager
    def _zero_caption_encoder(self):
        """Replace the caption encoder with zeros while every caption is empty."""
        model = self.rt.model
        original = model.caption_encoder
        width = int(model.caption_norm.weight.shape[0])

        class _Zero(torch.nn.Module):
            def forward(self, *args):
                ids = args[-2]
                return torch.zeros(
                    (ids.shape[0], ids.shape[1], width), device=ids.device, dtype=model.dtype
                )

        model.caption_encoder = _Zero()
        try:
            yield
        finally:
            model.caption_encoder = original

    @contextlib.contextmanager
    def _reuse_encoded(self, encoded, *, use_graphs: bool = True, recorder=None, last_call: int = 0):
        """Make the sampler's single encode_conditions call return the batch's encoded conditions,
        and route its DiT calls through the CUDA graphs when enabled. With ``recorder`` the
        ``last_call``-th DiT call runs eagerly with the pointer hook attached."""
        model = self.rt.model
        patches = {"encode_conditions": lambda *a, **k: encoded}
        eager = model.forward_with_encoded_conditions
        forward = eager
        if self.graphs is not None and use_graphs:
            self.graphs.begin()
            forward = self.graphs
        if recorder is not None:
            forward = _HookLastCall(forward, eager, recorder, last_call, self._eager_blocks)
        if forward is not eager:
            patches["forward_with_encoded_conditions"] = forward
        saved = {name: model.__dict__.get(name, _MISSING) for name in patches}
        for name, fn in patches.items():
            setattr(model, name, fn)
        try:
            yield
        finally:
            for name, old in saved.items():
                if old is _MISSING:
                    delattr(model, name)
                else:
                    setattr(model, name, old)

    # ---------------------------------------------------------------- sampling
    def _sample(self, requests, timings, log, *, record_pointer: bool = False):
        """Plans and sampled latents (B, T, D) of the requests; with ``record_pointer`` also the
        reading pointer of every row at the last DiT call (see length_check.PointerRecorder)."""
        rt = self.rt
        plans = [rt._plan_request(req, log_fn=log) for req in requests]
        model = rt.model
        device = rt.model_device
        _log = log
        # --- tokens
        t0 = _measure_start(device)
        rows = [
            rt.tokenizer.batch_encode([p.normalized_text], max_length=p.text_max_len) for p in plans
        ]
        text_ids, text_mask = _trim(
            _right_pad_cat([r[0] for r in rows]), _right_pad_cat([r[1] for r in rows]),
            self.pad_multiple,
        )
        text_ids, text_mask = text_ids.to(device), text_mask.to(device)
        caption_ids = caption_mask = None
        any_caption = False
        if model.cfg.use_caption_condition:
            cap_rows = []
            for p in plans:
                cap = "" if p.req.caption is None else str(p.req.caption).strip()
                ids, mask = rt.caption_tokenizer.batch_encode([cap], max_length=p.caption_max_len)
                if cap == "":
                    mask.zero_()
                else:
                    any_caption = True
                cap_rows.append((ids, mask))
            caption_ids = _right_pad_cat([r[0] for r in cap_rows])
            caption_mask = _right_pad_cat([r[1] for r in cap_rows])
            caption_ids, caption_mask = _trim(caption_ids, caption_mask, self.pad_multiple)
            caption_ids, caption_mask = caption_ids.to(device), caption_mask.to(device)
        timings.append(("tokenize_text", _measure_end(device, t0)))

        # --- speaker
        t0 = _measure_start(device)
        spk = [self._speaker_state(p.req, p.messages) for p in plans]
        speaker_state = _right_pad_cat([s for s, _ in spk])
        speaker_mask = _right_pad_cat([m for _, m in spk])
        extra = -speaker_state.shape[1] % self.speaker_multiple
        if extra:
            speaker_state = torch.nn.functional.pad(speaker_state, (0, 0, 0, extra))
            speaker_mask = torch.nn.functional.pad(speaker_mask, (0, extra))
        timings.append(("prepare_reference", _measure_end(device, t0)))

        # --- encode once
        t0 = _measure_start(device)
        caption_ctx = contextlib.nullcontext() if any_caption else self._zero_caption_encoder()
        with caption_ctx:
            encoded = model.encode_conditions(
                text_input_ids=text_ids,
                text_mask=text_mask,
                ref_latent=None,
                ref_mask=None,
                caption_input_ids=caption_ids,
                caption_mask=caption_mask,
                speaker_state_override=speaker_state,
                speaker_mask_override=speaker_mask,
                speaker_uncond_mode="mask",
            )
        timings.append(("encode_conditions", _measure_end(device, t0)))

        # --- duration
        t0 = _measure_start(device)
        self._plan_lengths(plans, encoded, text_ids, text_mask, speaker_mask, _log)
        timings.append(("predict_duration", _measure_end(device, t0)))

        # --- sample
        seq_len = max(p.patched_steps for p in plans)
        if self.latent_multiple > 1:
            seq_len = int(math.ceil(seq_len / self.latent_multiple) * self.latent_multiple)
        latent_mask = _right_pad_cat(
            [torch.ones((1, p.patched_steps), dtype=torch.bool, device=device) for p in plans]
        )
        if latent_mask.shape[1] < seq_len:
            latent_mask = torch.nn.functional.pad(latent_mask, (0, seq_len - latent_mask.shape[1]))
        noise = _right_pad_cat(
            [
                draw_initial_noise(
                    p.used_seed,
                    (1, p.patched_steps, rt.model_cfg.patched_latent_dim),
                    device=model.device,
                    dtype=model.dtype,
                )[0]
                for p in plans
            ]
        )
        if noise.shape[1] < seq_len:
            noise = torch.nn.functional.pad(noise, (0, 0, 0, seq_len - noise.shape[1]))
        batch_kwargs = {"sequence_length": seq_len, "latent_mask": latent_mask,
                        "initial_noise": noise}
        if len(plans) == 1 and seq_len == plans[0].patched_steps:
            # identical to upstream's single-request call (no mask)
            batch_kwargs = {"sequence_length": seq_len, "initial_noise": noise}
        first = plans[0]
        t0 = _measure_start(device)
        # speaker_kv_scale rescales the context K/V in place between steps: keep it eager
        use_graphs = first.speaker_kv_scale is None
        recorder = _BatchPointer(model, len(plans)) if record_pointer else None
        with self._reuse_encoded(encoded, use_graphs=use_graphs,
                                 recorder=recorder, last_call=first.num_steps):
            if rt.model_cfg.flow_parameterization == "meanflow":
                z_patched = sample_euler_meanflow(
                    model=model, text_input_ids=text_ids, text_mask=text_mask,
                    ref_latent=None, ref_mask=None, caption_input_ids=caption_ids,
                    caption_mask=caption_mask, speaker_state_override=speaker_state,
                    speaker_mask_override=speaker_mask, speaker_uncond_mode="mask",
                    num_steps=first.num_steps, **batch_kwargs,
                )
            else:
                z_patched = sample_euler_rf_cfg(
                    model=model, text_input_ids=text_ids, text_mask=text_mask,
                    ref_latent=None, ref_mask=None, caption_input_ids=caption_ids,
                    caption_mask=caption_mask, speaker_state_override=speaker_state,
                    speaker_mask_override=speaker_mask, speaker_uncond_mode="mask",
                    num_steps=first.num_steps,
                    cfg_scale_text=first.cfg_scale_text,
                    cfg_scale_caption=first.cfg_scale_caption,
                    cfg_scale_speaker=first.cfg_scale_speaker,
                    cfg_guidance_mode=first.cfg_mode,
                    cfg_min_t=float(first.req.cfg_min_t),
                    cfg_max_t=float(first.req.cfg_max_t),
                    truncation_factor=first.truncation_factor,
                    rescale_k=first.rescale_k,
                    rescale_sigma=first.rescale_sigma,
                    use_context_kv_cache=bool(first.req.context_kv_cache),
                    speaker_kv_scale=first.speaker_kv_scale,
                    speaker_kv_max_layers=first.speaker_kv_max_layers,
                    speaker_kv_min_t=first.speaker_kv_min_t,
                    t_schedule_mode=str(first.req.t_schedule_mode),
                    sway_coeff=float(first.req.sway_coeff),
                    **batch_kwargs,
                )
        stage = "sample_meanflow" if rt.model_cfg.flow_parameterization == "meanflow" else "sample_rf"
        timings.append((stage, _measure_end(device, t0)))

        z = unpatchify_latent(
            z_patched, patch_size=rt.model_cfg.latent_patch_size, latent_dim=rt.model_cfg.latent_dim
        )
        pointers = None
        if recorder is not None and recorder.pointer is not None:
            pointers = [
                (recorder.pointer[i, : p.patched_steps], int(recorder.text_len[i]))
                for i, p in enumerate(plans)
            ]
        return plans, z, pointers

    # ---------------------------------------------------------------- decoding
    def _finish(self, plans, z_rows, timings, post_load_t0):
        rt = self.rt
        device = rt.model_device
        # --- decode (row by row by default, see decode_batch)
        t0 = _measure_start(device, rt.codec_device)
        hop = int(rt.codec.model.hop_length)
        order = sorted(range(len(plans)), key=lambda i: plans[i].latent_steps)
        decoded: dict[int, torch.Tensor] = {}
        for g in range(0, len(order), self.decode_batch):
            group = order[g:g + self.decode_batch]
            longest = max(plans[i].latent_steps for i in group)
            out = self._decode(
                torch.cat([z_rows[i][None, :longest] for i in group], dim=0)
            ).cpu()
            for row, i in enumerate(group):
                decoded[i] = out[row]
        audios = []
        for i, p in enumerate(plans):
            max_samples = p.target_samples
            if bool(p.req.trim_tail):
                fp = find_flattening_point(
                    z_rows[i][: p.latent_steps],
                    window_size=max(1, int(p.req.tail_window_size)),
                    std_threshold=float(p.req.tail_std_threshold),
                    mean_threshold=float(p.req.tail_mean_threshold),
                )
                if fp * hop > 0:
                    max_samples = min(max_samples, fp * hop)
            audios.append(decoded[i][:, :max_samples])
        timings.append(("decode_latent", _measure_end(device, t0, rt.codec_device)))

        if rt.watermarker.ready:
            t0 = _measure_start(rt.codec_device)
            amp = (torch.autocast("cuda", dtype=torch.float16)
                   if self.watermark_fp16 and rt.codec_device.type == "cuda" else contextlib.nullcontext())
            with amp:
                audios = rt.watermarker.encode_batch(audios, sample_rate=int(rt.codec.sample_rate))
            timings.append(("silentcipher_watermark", _measure_end(rt.codec_device, t0)))

        total = _measure_end(device, post_load_t0, rt.codec_device)
        return [
            SamplingResult(
                audio=a, audios=[a], sample_rate=int(rt.codec.sample_rate),
                stage_timings=list(timings), total_to_decode=total, used_seed=p.used_seed,
                messages=p.messages,
            )
            for p, a in zip(plans, audios, strict=True)
        ]

    # ---------------------------------------------------------------- main
    def synthesize_batch(
        self,
        requests: Sequence[SamplingRequest],
        *,
        log_fn: Callable[[str], None] | None = None,
        length_check: bool = False,
    ) -> list[SamplingResult]:
        """``length_check``: the two-pass length check of length_check.py, batched (RF v4.1 Small,
        independent CFG). Rows whose first pass is clearly longer than the reading are synthesized
        again together, at the estimated length and with the same seed; the rest keep their first
        pass. Only the kept latents are decoded."""
        rt = self.rt
        if not requests:
            raise ValueError("synthesize_batch requires at least one request.")
        if not all(self.supports(r) for r in requests):
            return rt.synthesize_batch(requests, log_fn=log_fn)
        first_key = batch_compatibility_key(requests[0])
        if first_key is None or any(batch_compatibility_key(r) != first_key for r in requests[1:]):
            raise ValueError("Requests cannot share a batch; see batch_compatibility_key.")

        def _log(msg: str) -> None:
            if log_fn is not None:
                log_fn(msg)

        from . import length_check as lc

        check = (
            length_check
            and lc.supported(rt.model_cfg)
            and lc.single_call_per_step(requests[0])
        )
        if check:
            # fix the seeds so a second pass starts from the same noise
            requests = [
                r if r.seed is not None else dataclasses.replace(r, seed=int(secrets.randbits(63)))
                for r in requests
            ]
        timings: list[tuple[str, float]] = []
        post_load_t0 = _measure_start(rt.model_device, rt.codec_device)
        with rt._infer_lock, torch.inference_mode():
            plans, z, pointers = self._sample(requests, timings, _log, record_pointer=check)
            z_rows = [z[i] for i in range(len(plans))]
            self.last_length_check = [{"length_check": "skipped"}] * len(plans)
            if check and pointers is not None:
                self._length_check(requests, plans, z_rows, pointers, timings, _log)
            results = self._finish(plans, z_rows, timings, post_load_t0)
        return results

    def _length_check(self, requests, plans, z_rows, pointers, timings, log) -> None:
        """Replace rows of ``plans`` / ``z_rows`` by second passes where length_check.py would."""
        from . import length_check as lc

        rt = self.rt
        sr = int(rt.codec.sample_rate)
        hop = int(rt.codec.model.hop_length)
        patch = int(rt.model_cfg.latent_patch_size)
        fps = sr / hop
        tokens = list(self.last_duration_tokens)
        info = [{"length_check": "skipped"} for _ in plans]
        redo: dict[int, int] = {}  # row -> frames
        first_frames: dict[int, int] = {}
        punct: dict[int, bool] = {}
        for i, (req, (pointer, text_len)) in enumerate(zip(requests, pointers, strict=True)):
            if req.seconds is not None or int(req.num_candidates or 1) != 1:
                continue
            tok = tokens[i] if tokens and tokens[i] and len(tokens[i].get("frames") or ()) == text_len else None
            ff = int(pointer.shape[0]) * patch
            est = lc.estimated_frames(pointer, text_len, patch, fps, req.text, tok)
            info[i] = {"length_check": "kept", "first_frames": ff, "estimated_frames": est}
            if est is not None and est < ff * (1 - lc.RESYNTH_SHRINK):
                redo[i] = est
                first_frames[i] = ff
                punct[i] = lc.last_token_is_punct(req.text, tok)
        for attempt in range(1 + lc.RETRIES):
            if not redo:
                break
            rows = sorted(redo)
            t0 = _measure_start(rt.model_device)
            sub = [dataclasses.replace(requests[i], seconds=redo[i] / fps) for i in rows]
            sub_plans, sub_z, sub_ptr = self._sample(sub, [], log, record_pointer=True)
            timings.append((f"length_check_pass{attempt + 2}", _measure_end(rt.model_device, t0)))
            nxt: dict[int, int] = {}
            for k, i in enumerate(rows):
                pointer, text_len = sub_ptr[k]
                if lc.reading_end(pointer, text_len, punct[i]) is not None:
                    plans[i], z_rows[i] = sub_plans[k], sub_z[k]
                    info[i] = {"length_check": "resynthesized", "first_frames": first_frames[i],
                               "estimated_frames": redo[i]}
                    if attempt:
                        info[i]["retries"] = attempt
                    log(f"[length_check] row {i}: {first_frames[i]} -> {redo[i]} frames")
                    continue
                longer = int(math.ceil(redo[i] * lc.RETRY_GROWTH))
                if attempt < lc.RETRIES and longer < first_frames[i] * (1 - lc.MIN_SHRINK):
                    nxt[i] = longer
                else:
                    info[i] = {"length_check": "fallback", "first_frames": first_frames[i],
                               "estimated_frames": redo[i]}
            redo = nxt
        self.last_length_check = info

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, T, D) latent -> (B, 1, samples) float32 waveform."""
        if self._decoder16 is None:
            return self.rt.codec.decode_latent(z)
        out_proj, decoder = self._decoder16
        frames = z.transpose(1, 2).contiguous().to(device=self.rt.codec_device, dtype=torch.float16)
        return decoder(out_proj(frames)).float()

    def _plan_lengths(self, plans, encoded, text_ids, text_mask, speaker_mask, log) -> None:
        """Set latent_steps / target_samples / patched_steps of every plan (one predictor call)."""
        rt = self.rt
        model = rt.model
        hop = int(rt.codec.model.hop_length)
        sr = rt.codec.sample_rate
        need = [i for i, p in enumerate(plans) if p.manual_seconds is None]
        frames: dict[int, float] = {}
        self.last_duration_tokens = [None] * len(plans)
        if need and model.cfg.use_duration_predictor:
            text_state, tmask, spk_state, spk_mask, cap_state, cap_mask = encoded
            idx = torch.tensor(need, device=text_ids.device)

            def take(x):
                return None if x is None else x.index_select(0, idx)

            has_speaker = take(speaker_mask).any(dim=1)
            feats = build_duration_features(
                [plans[i].normalized_text for i in need],
                token_counts=take(text_mask).sum(dim=1),
                max_text_len=plans[need[0]].text_max_len,
                has_speaker=has_speaker,
            ).to(text_ids.device)
            token_sum = (
                getattr(model.duration_predictor, "token_out_proj", None) is not None
                and text_state.shape[1] == text_ids.shape[1]
            )
            out = model.predict_duration_log_frames(
                text_state=take(text_state), text_mask=take(tmask),
                speaker_state=take(spk_state), speaker_mask=take(spk_mask),
                caption_state=take(cap_state), caption_mask=take(cap_mask),
                duration_features=feats, has_speaker=has_speaker,
                has_caption=torch.tensor(
                    [plans[i].has_caption_text for i in need], dtype=torch.bool, device=text_ids.device
                ) if model.cfg.use_caption_condition else None,
                return_token_frames=token_sum,
            )
            if token_sum:
                out, token_frames = out
                ids_n, mask_n = take(text_ids), take(tmask)
                for row, i in enumerate(need):
                    self.last_duration_tokens[i] = rt._duration_tokens(
                        ids_n[row:row + 1], mask_n[row:row + 1], token_frames[row:row + 1]
                    )
            pred = torch.expm1(out).float().cpu().tolist()
            frames = dict(zip(need, pred, strict=True))
        for i, p in enumerate(plans):
            if p.manual_seconds is not None:
                seconds = min(p.max_seconds, max(p.min_seconds, p.manual_seconds))
                latent_steps = math.ceil(max(1, int(seconds * sr)) / hop)
            elif i in frames:
                lo = max(1, math.ceil(p.min_seconds * sr / hop))
                hi = max(1, math.floor(p.max_seconds * sr / hop))
                latent_steps = max(lo, min(hi, int(round(frames[i] * p.duration_scale))))
                p.messages.append(f"info: predicted duration frames={frames[i]:.1f}, using_frames={latent_steps}.")
            else:
                latent_steps = math.ceil(int(30.0 * sr) / hop)
            p.latent_steps = latent_steps
            p.target_samples = int(latent_steps * hop) if p.manual_seconds is None else max(
                1, int(min(p.max_seconds, max(p.min_seconds, p.manual_seconds)) * sr)
            )
            p.patched_steps = math.ceil(latent_steps / rt.model_cfg.latent_patch_size)
        if self.last_duration_tokens:
            rt.last_duration_tokens = self.last_duration_tokens[0]
