"""Split a chat reply into short pieces for /tts (one sentence or so each).

Irodori reads a long text worse than its sentences one by one: the audio gets muffled / boxy / noisy
as the text grows, and from about 120 Japanese characters the output is clipped at the model's
max_seconds (30 s). Splitting a reply into sentences and joining them with a short pause raised
DNSMOS OVRL by ~0.3 on 80-180 character replies (2026-11-20). See TURING.md "Replies".

A break mark has a strength w (0..1). A piece is cut after a mark when its length L reaches
target * (1 - w): a full stop always cuts, "?" / "!" almost always, "…" from 25% of the target and
"、" from 70%. A text no longer than the target is not split at all. Pieces shorter than min_len are not cut (they run on into the next one), and a piece
that would pass hard_max is cut at the strongest break seen so far (or, without any, right there).
"""
from __future__ import annotations

import json
import os

# "…" is 0.75 (was 0.6): "こんばんは、かな…ふふっ、でもね、…" ran on to the full stop as one 32-character
# piece and came out muffled (kiyo 2026-11-21); at 0.75 it cuts after "かな…" (9 characters).
# IRODORI_SPLIT_WEIGHTS='{"…": 0.8}' overrides single marks
WEIGHTS = {"。": 1.0, "．": 1.0, "？": 0.9, "?": 0.9, "！": 0.9, "!": 0.9, "…": 0.75, "‥": 0.75,
           "、": 0.3, "，": 0.3, ",": 0.3}
WEIGHTS.update({k: float(v) for k, v in json.loads(os.environ.get("IRODORI_SPLIT_WEIGHTS", "{}")).items()})
# kept on the piece before the cut: closing brackets and trailing marks that belong to the sentence
TRAIL = "」』）)】〉》♡♥♪〜～ー"

TARGET = int(os.environ.get("IRODORI_SPLIT_TARGET", "30"))
MIN_LEN = int(os.environ.get("IRODORI_SPLIT_MIN", "8"))
HARD_MAX = int(os.environ.get("IRODORI_SPLIT_MAX", "70"))


def split_reply(text: str, target: int = TARGET, min_len: int = MIN_LEN, hard_max: int = HARD_MAX) -> list[str]:
    text = text.strip()
    if len(text) <= target:  # short lines (tap reactions) read well as they are
        return [text] if text else []
    parts: list[str] = []
    start, i, n = 0, 0, len(text)
    best: tuple[float, int] | None = None  # (strength, cut position) of the best break not taken
    while i < n:
        if text[i] in WEIGHTS:
            j = i
            while j < n and (text[j] in WEIGHTS or text[j] in TRAIL):  # "…！", "。」" stay together
                j += 1
            w = max(WEIGHTS[c] for c in text[i:j] if c in WEIGHTS)
            length = j - start
            if length >= min_len and length >= target * (1 - w):
                parts.append(text[start:j])
                start, best = j, None
            elif length >= min_len and (best is None or w >= best[0]):
                best = (w, j)
            i = j
            continue
        if i - start + 1 > hard_max:
            cut = best[1] if best is not None else i
            parts.append(text[start:cut])
            start, best = cut, None
        i += 1
    if start < n:
        rest = text[start:]
        if parts and len(rest.strip()) < min_len:
            parts[-1] += rest
        else:
            parts.append(rest)
    return [p for p in (x.strip() for x in parts) if p]
