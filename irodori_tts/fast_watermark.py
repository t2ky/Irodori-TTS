"""SilentCipher's encode_wav for one mono row, the same computation without host round trips.

The packaged ``Model.encode_wav`` stops the CPU several times per row while the GPU waits: the
silence check reads the power back (``if original_power == 0``), the STFT window is copied from the
CPU on every call, the resampling kernels are rebuilt on every call, the message is built in numpy
and uploaded, and it prints a line whenever the default SDR is used. On a 3090 one 3 s row took
27 ms of which the GPU was busy for 10 ms; on a slow CPU (Xeon E5-2695 v2) it is worse.

Here the input stays on the GPU (straight from the decoder), the resampling kernels (built once
exactly as ``functional.resample`` builds them, on the device in float32; ``transforms.Resample``
builds them on the CPU and differs by ~1e-4), the window and
the message tensor are cached, and silence is handled with ``torch.where`` on the device. The output
is the same as ``encode_wav(..., calc_sdr=False)`` for the configuration SilentCipher ships
(44.1 kHz, utterance-level normalization, ensure_negative_message); other configurations fall back
to the packaged code.
"""
from __future__ import annotations

from collections.abc import Iterable

import math

import torch
from torchaudio.functional.functional import _apply_sinc_resample_kernel, _get_sinc_resample_kernel

from .watermark import IRODORI_WATERMARK_PAYLOAD


class _Resample:
    """torchaudio.functional.resample with its kernel kept (same kernel, same application)."""

    def __init__(self, orig: int, new: int, device: torch.device) -> None:
        self.orig, self.new = int(orig), int(new)
        self.gcd = math.gcd(self.orig, self.new)
        self.kernel, self.width = _get_sinc_resample_kernel(
            self.orig, self.new, self.gcd, 6, 0.99, "sinc_interp_hann", None, device, torch.float32)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_sinc_resample_kernel(x, self.orig, self.new, self.gcd, self.kernel, self.width)


def _binary_encode(message: Iterable[int]) -> list[int]:
    bits = "".join("{0:08b}".format(m) for m in message)
    return [int(bits[i * 2:i * 2 + 2], 2) for i in range(len(bits) // 2)]


class FastSilentCipher:
    def __init__(self, model, sample_rate: int, payload: Iterable[int] = IRODORI_WATERMARK_PAYLOAD) -> None:
        self.m = model
        cfg = model.config
        self.supported = bool(
            cfg.utterance_level_normalization and not cfg.frame_level_normalization
            and cfg.ensure_negative_message and int(model.n_messages) == 1
        )
        dev = torch.device(model.device)
        self.device = dev
        self.sample_rate = int(sample_rate)
        self.down = self.up = None
        if self.sample_rate != int(model.sr):
            self.down = _Resample(self.sample_rate, int(model.sr), dev)
            self.up = _Resample(int(model.sr), self.sample_rate, dev)
        stft = model.stft
        self.n_fft, self.hop, self.win = int(stft.filter_length), int(stft.hop_len), int(stft.win_len)
        self.window = stft.window.to(dev)
        self.energy = torch.tensor(model.average_energy_VCTK, device=dev)
        self.sdr = cfg.message_sdr
        self.message = _binary_encode(payload)
        self._msg_cache: dict[int, torch.Tensor] = {}

    def _msg(self, frames: int) -> torch.Tensor:
        t = self._msg_cache.get(frames)
        if t is None:
            msgs, _ = self.m.letters_encoding(frames, [self.message])
            t = torch.tensor(msgs, device=self.device, dtype=torch.float32).unsqueeze(0)
            self._msg_cache[frames] = t
        return t

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """``audio``: any shape holding one mono row (on any device); returns a 1-D float32 tensor on
        the model's device (no host synchronization)."""
        m = self.m
        orig = audio.detach().reshape(-1).to(device=self.device, dtype=torch.float32)
        seq_len = orig.shape[0]
        y = orig if self.down is None else self.down(orig.view(1, -1)).squeeze()
        power = torch.mean(y ** 2)
        silent = power == 0
        safe = torch.where(silent, torch.ones_like(power), power)
        y = y * torch.sqrt(self.energy / safe)
        x = y.view(1, -1)
        # STFT.transform
        x = torch.nn.functional.pad(x, (0, self.win - x.shape[1] % self.win))
        fft = torch.stft(x, self.n_fft, self.hop, self.win, window=self.window, return_complex=True)
        real_part, imag_part = fft.real, fft.imag
        squared = real_part ** 2 + imag_part ** 2
        additive_epsilon = torch.ones_like(squared) * (squared == 0).float() * 1e-24
        magnitude = torch.sqrt(squared + additive_epsilon) - torch.sqrt(additive_epsilon)
        phase = torch.atan2(imag_part, real_part).float()
        carrier = magnitude[:, None]
        carrier_phase = phase[:, None]
        msg_enc = self._msg(carrier.shape[3])
        carrier_enc = m.enc_c(carrier)
        msg_enc = m.enc_c.transform_message(msg_enc)
        merged_enc = torch.cat((carrier_enc, carrier.repeat(1, 32, 1, 1), msg_enc.repeat(1, 32, 1, 1)), dim=1)
        message_info = m.dec_c(merged_enc, self.sdr)
        message_info = message_info * (torch.mean(carrier ** 2, dim=(2, 3), keepdim=True) ** 0.5)
        message_info = -message_info
        carrier_reconst = torch.nn.functional.relu(message_info + carrier)
        # STFT.inverse
        mag, ph = carrier_reconst.squeeze(1), carrier_phase.squeeze(1)
        recombined = mag * torch.cos(ph) + 1j * mag * torch.sin(ph)
        inv = torch.istft(recombined, self.n_fft, hop_length=self.hop, win_length=self.win,
                          window=self.window).unsqueeze(1)
        num_samples = y.shape[0]
        padding = self.win - (num_samples % self.win)
        out = inv[:, :, :-padding][0, 0]
        out = out * torch.sqrt(safe / self.energy)
        if self.up is not None:
            out = self.up(out.view(1, -1)).squeeze()[:seq_len]
        if out.shape[0] == seq_len:
            out = torch.where(silent, orig, out)
        return out
