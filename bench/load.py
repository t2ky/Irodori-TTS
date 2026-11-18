"""Load test of turing_server.py: response time of /synth under concurrent requests.

  python3 bench/load.py --url http://127.0.0.1:8099 burst --n 8 --rounds 10
  python3 bench/load.py --url http://127.0.0.1:8099 poisson --rate 6 --seconds 60

burst: n requests at the same instant, `rounds` times (one round after the other finished).
poisson: requests arrive at `rate` per second (exponential gaps) for `seconds`.
Texts are bench/texts.json, the voice is bench/ref30.wav.
"""
import argparse
import base64
import json
import random
import statistics
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXTS = json.loads((HERE / "texts.json").read_text())
REF = base64.b64encode((HERE / "ref30.wav").read_bytes()).decode()


VOICE = None


def call(url, text, seed):
    body = json.dumps({"text": text, "seed": seed, **({"voice": VOICE} if VOICE else {"ref_audio": REF})}).encode()
    req = urllib.request.Request(url + "/synth", body, {"Content-Type": "application/json"})
    t = time.perf_counter()
    out = json.load(urllib.request.urlopen(req, timeout=60))
    return time.perf_counter() - t, out["batch_size"], out["audio_seconds"]


def report(name, rows):
    lat = sorted(r[0] for r in rows)
    q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]  # noqa: E731
    sizes = [r[1] for r in rows]
    print(f"{name}: n={len(lat)} 平均 {statistics.mean(lat):.3f}s 中央 {q(0.5):.3f}s p90 {q(0.9):.3f}s "
          f"p95 {q(0.95):.3f}s 最大 {lat[-1]:.3f}s | バッチ平均 {statistics.mean(sizes):.1f} "
          f"| 0.8 秒以内 {sum(x <= 0.8 for x in lat)}/{len(lat)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--voice", help="name of a voice registered on the server instead of uploading ref30.wav")
    sub = ap.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("burst"); b.add_argument("--n", type=int, default=8); b.add_argument("--rounds", type=int, default=10)
    p = sub.add_parser("poisson"); p.add_argument("--rate", type=float, default=6.0); p.add_argument("--seconds", type=float, default=60)
    a = ap.parse_args()
    global VOICE
    VOICE = a.voice
    call(a.url, TEXTS[1], 0)  # wake up
    rows, lock = [], threading.Lock()

    def one(text, seed):
        r = call(a.url, text, seed)
        with lock:
            rows.append(r)

    if a.mode == "burst":
        for k in range(a.rounds):
            ts = [threading.Thread(target=one, args=(TEXTS[(i + k) % len(TEXTS)], 100 * k + i)) for i in range(a.n)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        report(f"burst n={a.n}", rows)
    else:
        rnd = random.Random(0)
        ts, t_end, i = [], time.perf_counter() + a.seconds, 0
        while time.perf_counter() < t_end:
            t = threading.Thread(target=one, args=(rnd.choice(TEXTS), i)); t.start(); ts.append(t); i += 1
            time.sleep(rnd.expovariate(a.rate))
        for t in ts:
            t.join()
        report(f"poisson {a.rate}/s", rows)


if __name__ == "__main__":
    main()
