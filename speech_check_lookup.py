"""Find a reply in the speech check log (IRODORI_SPEECH_CHECK_LOG) by its text.

  IRODORI_SPEECH_CHECK_HMAC_KEY=<the server's key> python speech_check_lookup.py <log.jsonl> "<reply text>" ["<reply text>" ...]

The text is hashed the way the server does (speech_check.text_hash: NFKC, no whitespace, HMAC-SHA256). It is
matched against the whole reply and against each piece, so a part of a reply (one sentence) also finds it.
Prints the matching lines (scores and counts; the log has no text).
"""
import json
import sys

import speech_check

log_path, texts = sys.argv[1], sys.argv[2:]
want = {}
for t in texts:
    h = speech_check.text_hash(t)
    if h is None:
        sys.exit("set IRODORI_SPEECH_CHECK_HMAC_KEY")
    want[h] = t
hits = 0
for line in open(log_path):
    r = json.loads(line)
    for k in ("reply_hash", "piece_hash"):
        if r.get(k) in want:
            hits += 1
            print(f"[{k[:-5]}: {want[r[k]][:30]}] " + json.dumps(r, ensure_ascii=False))
            break
print(f"{hits} lines", file=sys.stderr)
