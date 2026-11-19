"""CUDA graphs for the DiT forward (``forward_with_encoded_conditions``), see TURING.md.

A single request is launch-bound: one DiT call is ~1,500 small kernels and takes ~16 ms on a 3090
whatever the text length. Replaying a captured graph removes the per-kernel CPU cost.

* One graph per input signature (shapes and dtypes of every tensor argument, including each layer's
  context K/V). Callers keep the number of signatures small by bucketing lengths (TuringBatcher pads
  text / caption tokens and, optionally, latent frames to multiples).
* Inputs live in one flat buffer per argument slot that every graph shares (a contiguous prefix view
  of the right shape), so a new signature costs graph-private activations only, not another copy of
  the context K/V. Before a replay each slot is refreshed unless it already holds that same tensor
  (the context K/V and conditions stay the same for all steps of a synthesis, so only ``x_t`` / ``t``
  are copied per step).
* All graphs share one memory pool; they never run concurrently.
* ``freeze()`` (after warm-up): the slots no longer grow and no new graph is captured. A call that
  does not fit the slots or has a shape not seen during warm-up runs eagerly. Growing the slots
  drops every captured graph (a server seeing one long sentence then paid a capture again for every
  shape), and a capture costs a request several eager calls (on RTX 2000 Ada the graphs save ~10%
  of a call, a capture costs ~0.5 s of tail latency).
* Forward hooks (length_check's pointer recorder) only run while a graph is captured, so the eager
  forward is used whenever the first block's attention has hooks.

The attention dispatch must be capture-safe: true without FlashAttention-3 (Hopper only), whose plan
packing needs a device sync.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

# Always copied before a replay (a new tensor every step, whose id / address can repeat).
_PER_STEP = {"x_t", "t", "delta_t"}

_ARG_ORDER = (
    "x_t", "t", "text_state", "text_mask", "speaker_state", "speaker_mask",
    "caption_state", "caption_mask", "latent_mask", "delta_t",
)


_SIDE_STREAM: torch.cuda.Stream | None = None


def _side_stream() -> torch.cuda.Stream:
    """One side stream for every capture's warm-up runs. cuBLAS keeps a workspace per stream, so a
    new stream per capture leaked one each (48 text-encoder graphs held ~290 MB, ~6 MB a capture)."""
    global _SIDE_STREAM
    if _SIDE_STREAM is None:
        _SIDE_STREAM = torch.cuda.Stream()
    return _SIDE_STREAM


class _Slot:
    """A flat device buffer shared by all graphs for one argument position."""

    def __init__(self) -> None:
        self.buf: torch.Tensor | None = None
        self.holds: tuple | None = None  # (id, data_ptr, version, shape) of the tensor copied in

    def view(self, like: torch.Tensor) -> torch.Tensor:
        return self.buf[: like.numel()].view(like.shape)


class GraphedDiT:
    def __init__(self, model, *, max_graphs: int = 128, max_rows: int | None = None) -> None:
        self.model = model
        self.eager = model.forward_with_encoded_conditions
        self.max_graphs = int(max_graphs)
        # Larger calls are compute-bound (graphs gave 2-4% at 8 requests) and would only multiply
        # the number of captured shapes: run them eagerly.
        self.max_rows = max_rows
        self.graphs: OrderedDict[tuple, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = OrderedDict()
        self.slots: dict[str, _Slot] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.captures = 0
        self.replays = 0
        self.eager_calls = 0
        self.frozen = False
        # The RoPE table is cached on first use; grow it now so no capture allocates it.
        model._rope_freqs(8192, model.device)

    def freeze(self) -> None:
        """Keep the slots and the captured graphs as they are; other calls run eagerly."""
        self.frozen = True

    def _fits(self, leaves) -> bool:
        for name, tensor in leaves:
            if tensor is None:
                continue
            slot = self.slots.get(name)
            if (slot is None or slot.buf is None or slot.buf.numel() < tensor.numel()
                    or slot.buf.dtype != tensor.dtype or slot.buf.device != tensor.device):
                return False
        return True

    # -------------------------------------------------------------- helpers
    def _hooked(self) -> bool:
        attn = self.model.blocks[0].attention
        return bool(attn._forward_pre_hooks or attn._forward_hooks)

    @staticmethod
    def _flatten(kwargs) -> list[tuple[str, torch.Tensor | None]]:
        leaves = [(name, kwargs.get(name)) for name in _ARG_ORDER]
        cache = kwargs.get("context_kv_cache")
        if cache is not None:
            for li, layer in enumerate(cache):
                for ti, tensor in enumerate(layer):
                    leaves.append((f"kv{li}.{ti}", tensor))
        return leaves

    def _ensure_slots(self, leaves) -> bool:
        """Make every slot large enough; returns True when a buffer was (re)allocated."""
        grown = False
        for name, tensor in leaves:
            if tensor is None:
                continue
            slot = self.slots.setdefault(name, _Slot())
            if (slot.buf is None or slot.buf.numel() < tensor.numel()
                    or slot.buf.dtype != tensor.dtype or slot.buf.device != tensor.device):
                # headroom so that warming up with growing shapes does not drop graphs every time
                size = tensor.numel() if slot.buf is None else max(
                    tensor.numel(), int(slot.buf.numel() * 1.5)
                )
                slot.buf = torch.empty(size, dtype=tensor.dtype, device=tensor.device)
                slot.holds = None
                grown = True
        return grown

    def _load(self, leaves) -> None:
        for name, tensor in leaves:
            if tensor is None:
                continue
            slot = self.slots[name]
            # inference tensors have no version counter; nothing writes the conditions in place
            # within a synthesis (TuringBatcher bypasses graphs for speaker_kv_scale, which does)
            tag = (id(tensor), tensor.data_ptr(), tuple(tensor.shape))
            if name not in _PER_STEP and slot.holds == tag:
                continue
            slot.view(tensor).copy_(tensor)
            slot.holds = tag

    def _static_kwargs(self, kwargs, leaves):
        static = {name: (None if t is None else self.slots[name].view(t))
                  for name, t in leaves if not name.startswith("kv")}
        cache = kwargs.get("context_kv_cache")
        if cache is not None:
            static["context_kv_cache"] = [
                tuple(self.slots[f"kv{li}.{ti}"].view(t) for ti, t in enumerate(layer))
                for li, layer in enumerate(cache)
            ]
        else:
            static["context_kv_cache"] = None
        return static

    def begin(self) -> None:
        """Call at the start of every synthesis: tensors of a finished one may reuse ids/addresses."""
        for slot in self.slots.values():
            slot.holds = None

    # -------------------------------------------------------------- call
    def __call__(self, **kwargs) -> torch.Tensor:
        x_t = kwargs["x_t"]
        if (x_t.device.type != "cuda" or self._hooked() or torch.cuda.is_current_stream_capturing()
                or (self.max_rows is not None and x_t.shape[0] > self.max_rows)):
            self.eager_calls += 1
            return self.eager(**kwargs)
        leaves = self._flatten(kwargs)
        if self.frozen and not self._fits(leaves):
            self.eager_calls += 1
            return self.eager(**kwargs)
        key = tuple((n, None) if t is None else (n, tuple(t.shape), t.dtype) for n, t in leaves)
        if self.frozen and key not in self.graphs:
            self.eager_calls += 1
            return self.eager(**kwargs)
        if self._ensure_slots(leaves):
            # buffers moved: graphs captured against the old addresses are invalid. A pool whose
            # graphs are all gone cannot be captured into again, so start a new one.
            self.graphs.clear()
            self.pool = torch.cuda.graph_pool_handle()
        self._load(leaves)
        entry = self.graphs.get(key)
        if entry is None:
            entry = self._capture(kwargs, leaves)
            self.graphs[key] = entry
            if len(self.graphs) > self.max_graphs:
                self.graphs.popitem(last=False)
        else:
            self.graphs.move_to_end(key)
        graph, out = entry
        graph.replay()
        self.replays += 1
        return out.clone()

    def _capture(self, kwargs, leaves):
        static = self._static_kwargs(kwargs, leaves)
        stream = _side_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # warm-up: lazy inits, cuBLAS handles, SDPA backend choice
                self.eager(**static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = self.eager(**static)
        self.captures += 1
        return graph, out


class GraphedCall:
    """CUDA graphs for a plain function of tensors (e.g. the text encoder), one per input signature.

    Unlike GraphedDiT, each graph owns its static inputs (the inputs are small: token ids and masks)
    and all graphs share one memory pool. Non-tensor arguments are passed through unchanged and must
    be the same objects on every call (they are baked into the graph). ``freeze()`` stops capturing:
    signatures not captured by then run eagerly.
    """

    def __init__(self, fn, *, max_graphs: int = 256) -> None:
        self.fn = fn
        self.max_graphs = int(max_graphs)
        self.graphs: OrderedDict[tuple, tuple[list[torch.Tensor], torch.cuda.CUDAGraph, torch.Tensor]] = OrderedDict()
        self.pool = torch.cuda.graph_pool_handle()
        self.captures = 0
        self.replays = 0
        self.eager_calls = 0
        self.frozen = False

    def freeze(self) -> None:
        self.frozen = True

    def __call__(self, *args):
        tensors = [(i, a) for i, a in enumerate(args) if isinstance(a, torch.Tensor)]
        if (not tensors or any(a.device.type != "cuda" for _, a in tensors)
                or torch.cuda.is_current_stream_capturing()):
            self.eager_calls += 1
            return self.fn(*args)
        key = tuple((i, tuple(a.shape), a.dtype) for i, a in tensors)
        entry = self.graphs.get(key)
        if entry is None:
            if self.frozen:
                self.eager_calls += 1
                return self.fn(*args)
            entry = self._capture(args, tensors)
            self.graphs[key] = entry
            if len(self.graphs) > self.max_graphs:
                self.graphs.popitem(last=False)
        else:
            self.graphs.move_to_end(key)
        static_in, graph, out = entry
        for (_, a), s in zip(tensors, static_in):
            s.copy_(a)
        graph.replay()
        self.replays += 1
        return out.clone()

    def _capture(self, args, tensors):
        static_in = [a.clone() for _, a in tensors]
        sargs = list(args)
        for (i, _), s in zip(tensors, static_in):
            sargs[i] = s
        stream = _side_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.fn(*sargs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = self.fn(*sargs)
        self.captures += 1
        return static_in, graph, out
