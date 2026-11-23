"""Catch pieces of a reply that were read short (words skipped, only the head read, a sentence cut off).

Irodori now and then drops part of a piece (kiyo 2026-11-22: "reads only the first letter of a word,
skips words"). It is a random failure: the same text with another seed reads fine. The text is known,
so the check compares what should have been said with the audio:

  expected  pyopenjtalk.g2p(piece) -> phonemes (pau / sil dropped), ~0.4 ms
  heard     prj-beatrice/japanese-hubert-base-phoneme-ctc-v4 (Apache-2.0, HuBERT base fine-tuned for
            Japanese phoneme CTC; its targets are pyopenjtalk-plus readings, so the phoneme names match)
  score     the expected phonemes are force-aligned to the CTC output (torchaudio forced_align) and the
            mean log-probability of the aligned (non-blank) frames is the score. Audio that lacks part of
            the text has to squeeze the missing phonemes in and scores low

Measured on 1158 pieces of v29 replies (2026-11-23, 3090): score < -2.0 flags 1.4 % of correct pieces,
87 % of pieces with the middle 40 % of the speech cut out and 99 % of pieces cut to their first 35 %.
0.5 s of silence is padded on both sides (with 0.25 s the model missed the first morae and flagged twice
as many correct pieces). Pieces that are only breath / interjections ("はぁっ…はぁっ…！") and pieces with
Latin letters (their g2p reading is unreliable) are scored and logged but never retried.

Privacy: nothing of the text leaves this module; the log carries counts and scores only.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("speech_check")

MODEL_ID = os.environ.get("IRODORI_SPEECH_CHECK_MODEL", "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4")
MIN_SCORE = float(os.environ.get("IRODORI_SPEECH_CHECK_MIN", "-2.0"))
MIN_PHONEMES = int(os.environ.get("IRODORI_SPEECH_CHECK_MIN_PHONEMES", "6"))
PAD_S = 0.5
LOG_PATH = os.environ.get("IRODORI_SPEECH_CHECK_LOG")  # JSON lines appended here as well as to stdout

# pieces that are only breath, interjections and marks: their "expected" phonemes say little
_BREATH = re.compile(r"[\sぁあァアぃいィイぅうゥウぇえェエぉおォオはハひヒふフへヘほホんンっッーゃゅょ〜～…‥・、。，．！？!?♡♥❤️♪「」『』（）()]+")
_LATIN = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]")


def reason_to_skip(piece: str, n_phonemes: int) -> str | None:
    if _LATIN.search(piece):
        return "latin"
    if _BREATH.fullmatch(piece):
        return "breath"
    if n_phonemes < MIN_PHONEMES:
        return "short"
    return None


class SpeechCheck:
    def __init__(self, device: str = "cuda") -> None:
        import pyopenjtalk
        from transformers import AutoProcessor, HubertForCTC

        t0 = time.perf_counter()
        self._g2p = pyopenjtalk.g2p
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = HubertForCTC.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
        self.device = device
        self.vocab = self.proc.tokenizer.get_vocab()
        self.inv = {i: t for t, i in self.vocab.items()}
        self.blank = self.model.config.pad_token_id
        self._not_phoneme = torch.zeros(len(self.inv), dtype=torch.bool, device=device)
        for t in ("pau", "sil", "SOS", "EOS", "UNK", "PAD"):
            if t in self.vocab:
                self._not_phoneme[self.vocab[t]] = True
        self._not_phoneme[self.blank] = True
        self.lock = threading.Lock()
        self._log_lock = threading.Lock()
        self.counts = {"checked": 0, "skipped": 0, "flagged": 0, "retried": 0, "recovered": 0, "still_low": 0}
        # the first call pays ~0.3 s of CUDA kernel set-up: do it here, not on a user's reply
        ph = self.expected("ウォームアップの文です")
        rng = np.random.default_rng(0)
        for sec in (1.0, 2.5, 4.0):
            self.score((rng.standard_normal(int(16000 * sec)) * 1000).astype(np.int16), 16000, ph)
        log.warning("speech check: %s loaded in %.1fs (min score %.1f)", MODEL_ID, time.perf_counter() - t0, MIN_SCORE)

    def expected(self, piece: str) -> list[str]:
        return [p for p in self._g2p(piece).split() if p not in ("pau", "sil") and p in self.vocab]

    @torch.inference_mode()
    def score(self, audio, sr: int, phonemes: list[str]) -> tuple[float, int]:
        """(score, number of phonemes the CTC greedy decode heard) for mono audio (a tensor, or an int16 /
        float numpy array). All of it runs on the GPU: the resampling, the normalization, the CTC and the
        alignment, so the thread holds the GIL only to launch kernels (a CPU pipeline here slowed the
        batch worker's synthesis of the other requests by ~0.2 s per reply)."""
        import torchaudio.functional as AF

        if isinstance(audio, np.ndarray):
            x = torch.from_numpy(audio.astype(np.float32) / (32768.0 if audio.dtype == np.int16 else 1.0))
        else:
            x = audio.detach().reshape(-1).float()
        with self.lock:
            x = x.to(self.device, non_blocking=True)
            if sr != 16000:
                x = AF.resample(x, sr, 16000)
            x = torch.nn.functional.pad(x, (int(PAD_S * 16000), int(PAD_S * 16000)))
            # the model's feature extractor does not normalize (do_normalize: false): the raw waveform goes in
            lp = torch.log_softmax(self.model(x.view(1, -1).half()).logits[0].float(), -1)
            ids = lp.argmax(-1)
            keep = torch.ones_like(ids, dtype=torch.bool)
            keep[1:] = ids[1:] != ids[:-1]
            heard = int((keep & ~self._not_phoneme[ids]).sum())
            if not phonemes or lp.shape[0] < len(phonemes):
                return -99.0, heard
            tgt = torch.tensor([[self.vocab[p] for p in phonemes]], device=self.device)
            toks, sc = AF.forced_align(lp.unsqueeze(0), tgt, blank=self.blank)
            nb = toks[0] != self.blank
            return (float(sc[0][nb].mean()) if bool(nb.any()) else -99.0), heard

    def count(self, key: str, n: int = 1) -> None:
        with self._log_lock:
            self.counts[key] += n

    def write(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
        log.warning("speech_check %s", line)
        if LOG_PATH:
            with self._log_lock:
                try:
                    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
                    with open(LOG_PATH, "a") as f:
                        f.write(line + "\n")
                except OSError as exc:
                    log.warning("speech check log not written: %s", type(exc).__name__)
