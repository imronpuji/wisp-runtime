"""Eager fused (WISP_FUSED=2) vs CUDA-graph decode, same prompts, plus token agreement.
python scripts/bench_graph.py --bits 4 --vram-gb 8"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("WISP_FUSED", "2")
from wisp import runner
from wisp.budget import plan_slots
from wisp.graphdec import GraphDecoder
from wisp.model import configure, nbytes_of

ap = argparse.ArgumentParser()
ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--vram-gb", type=float, default=8)
ap.add_argument("--n-new", type=int, default=64)
ap.add_argument("--n-prompts", type=int, default=6)
ap.add_argument("--model-dir", default=os.environ.get("WISP_MODEL"))
a = ap.parse_args()

model, cfg, rt, tok = runner.setup(a.model_dir, a.bits, cpu_threads=12)
assert rt.fused == 2, "run with WISP_FUSED=2"
slots = plan_slots(a.vram_gb, nbytes_of(model), rt.store, cfg, 1024)["slots"]
configure(rt, slots, "lru", "transfer", vram_cap_gb=a.vram_gb)
prompts = runner.load_prompts("data/prompts.jsonl", tok, a.n_prompts + 2, 128)
dec = GraphDecoder(model, rt, tmax=128 + a.n_new + 8)
for _, ids in prompts[:2]:  # warm the expert cache + compile
    runner.generate_timed(model, ids, 24)
print("[graph] capturing ...", flush=True)
runner.generate_graph(dec, model, prompts[0][1], 8)
print("[graph] captured", flush=True)
te = tg = 0.0
ne = ng = 0
agree = []
for i, (_, ids) in enumerate(prompts[2:]):
    order = ("e", "g") if i % 2 == 0 else ("g", "e")
    res = {}
    for w in order:
        f = runner.generate_timed if w == "e" else lambda m, x, n: runner.generate_graph(dec, m, x, n)
        _, dt, n, toks = f(model, ids, a.n_new)
        res[w] = (dt, n, toks)
    te += res["e"][0]
    ne += res["e"][1]
    tg += res["g"][0]
    ng += res["g"][1]
    m = (res["e"][2] == res["g"][2]).int()
    pref = int(m.cumprod(0).sum())
    agree.append(pref)
    print(
        f"prompt {i}: eager {res['e'][1] / res['e'][0]:.1f} tok/s | graph {res['g'][1] / res['g'][0]:.1f} tok/s | "
        f"identical prefix {pref}/{a.n_new} tokens",
        flush=True,
    )
print(f"\neager decode_tok_s: {ne / te:.2f}\ngraph decode_tok_s: {ng / tg:.2f}\nspeedup: {(ng / tg) / (ne / te):.2f}x")
