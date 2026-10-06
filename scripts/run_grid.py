"""Hypothesis runner. Loads the expert store ONCE per bit-width, then loops configs x repeats.
Resumable: every finished cell is appended to results/<phase>.jsonl and skipped on restart.

  python scripts/run_grid.py --phase h1
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

from wisp import runner
from wisp.budget import plan_slots
from wisp.model import configure, nbytes_of

PHASES = {
    # H1: replication. BF16 experts (paper setting), 22 GB budget. static = profile-then-pin baselines.
    "h1": dict(
        bits=[16],
        cells=[
            dict(vram_gb=22, policy="static", miss="transfer"),  # fixed resident set, misses streamed over PCIe
            dict(
                vram_gb=22, policy="static", miss="hybrid"
            ),  # fixed resident set, misses computed on CPU (llama.cpp-like)
            dict(vram_gb=22, policy="lru", miss="transfer"),
            dict(vram_gb=22, policy="wsa", miss="transfer"),  # WiSP-style working-set paging
        ],
    ),
    # H2+H3: bytes per token vs speed, across bit-widths and budgets (paging, transfer)
    "h3": dict(bits=[16, 8, 4, 3, 2], cells=[dict(vram_gb=v, policy="lru", miss="transfer") for v in (8, 12, 16, 22)]),
    # H3b: compressed transfer, decoded once on arrival into a bf16 cache
    "h3b": dict(
        bits=[8, 4, 3, 2],
        cells=[dict(vram_gb=v, policy="lru", miss="transfer", decode_on_load=True) for v in (8, 12, 16, 22)],
    ),
    # H4: CPU-computes-misses vs transfer at small budgets
    "h4": dict(
        bits=[16, 4], cells=[dict(vram_gb=v, policy="lru", miss=m) for v in (8, 12, 16) for m in ("transfer", "hybrid")]
    ),
    # H5: VRAM split between expert cache and KV for long contexts (prompt_tokens set per ctx)
    "h5": dict(
        bits=[4],
        cells=[
            dict(vram_gb=12, policy="wsa", miss="transfer", ctx=c, kv_mode=k, prompt_tokens=c)
            for c in (8192, 32768)
            for k in ("fixed", "adaptive")
        ],
    ),
    "prefetch": dict(
        bits=[16], cells=[dict(vram_gb=22, policy="wsa", miss="transfer", prefetch=p) for p in (False, True)]
    ),
}

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True, choices=list(PHASES))
ap.add_argument("--bits", default=None, help="override, e.g. 16,4")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--n-prompts", type=int, default=6)
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--n-new", type=int, default=64)
ap.add_argument("--prompt-tokens", type=int, default=128)
ap.add_argument("--profile-prompts", type=int, default=16)
ap.add_argument("--cpu-threads", type=int, default=12)
ap.add_argument("--model-dir", default=os.environ.get("WISP_MODEL"))
a = ap.parse_args()

spec = PHASES[a.phase]
bits_list = [int(b) for b in a.bits.split(",")] if a.bits else spec["bits"]
if len(bits_list) > 1:
    # one process per bit-width: PyTorch caches pinned host memory and never returns it to the OS,
    # so several stores in one process exhaust RAM (observed: silent OOM kill after bits=16 + bits=8).
    import subprocess

    rc = 0
    for b in bits_list:
        argv = [sys.executable, __file__] + [x for x in sys.argv[1:]] + ["--bits", str(b)]
        if "--bits" in sys.argv[1:]:
            i = sys.argv.index("--bits")
            argv = [sys.executable, __file__] + sys.argv[1:i] + sys.argv[i + 2 :] + ["--bits", str(b)]
        print(time.strftime("%H:%M:%S"), "spawn", " ".join(argv[1:]), flush=True)
        r = subprocess.run(argv)
        print(time.strftime("%H:%M:%S"), f"bits={b} exit code {r.returncode}", flush=True)
        rc |= r.returncode
    sys.exit(rc)
out = f"results/{a.phase}.jsonl"
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)
done = set()
if os.path.exists(out):
    for l in open(out):
        r = json.loads(l)
        if r.get("status") == "ok":
            done.add((r["cell_id"], r["rep"]))


def cell_id(bits, c):
    return hashlib.md5(json.dumps(dict(bits=bits, **c), sort_keys=True).encode()).hexdigest()[:10]


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


for bits in bits_list:
    todo = [(c, r) for c in spec["cells"] for r in range(a.repeats) if (cell_id(bits, c), r) not in done]
    if not todo:
        log(f"bits={bits}: all cells done")
        continue
    model, cfg, rt, tok = runner.setup(a.model_dir, bits, cpu_threads=a.cpu_threads)
    L, E = cfg.num_hidden_layers, cfg.num_experts

    # --- profiling trace (disjoint prompts, offset 100) for static pinning & the simulator ---
    tpath = f"data/trace_b{bits}.npz"
    if os.path.exists(tpath):
        trace = np.load(tpath)["ids"]
    else:
        slots = plan_slots(22, nbytes_of(model), rt.store, cfg, 1024)["slots"]
        configure(rt, slots, "lru", "transfer", vram_cap_gb=22)
        steps = []
        for _, ids in runner.load_prompts("data/prompts.jsonl", tok, a.profile_prompts, a.prompt_tokens, offset=100):
            rt.recorder = []
            runner.generate_timed(model, ids, a.n_new)
            steps.append(
                np.stack([s for (l, s) in rt.recorder if s.shape[0] == 1]).reshape(-1, L, cfg.num_experts_per_tok)
            )
        rt.recorder = None
        trace = np.concatenate(steps).astype(np.int16)
        np.savez_compressed(tpath, ids=trace, L=L, E=E)
        log(f"trace saved {tpath} {trace.shape}")

    cache_prompts = {}
    for c, rep in todo:
        pt = c.get("prompt_tokens", a.prompt_tokens)
        if pt not in cache_prompts:
            cache_prompts[pt] = runner.load_prompts("data/prompts.jsonl", tok, a.n_prompts + a.warm, pt)
        prompts = cache_prompts[pt]
        cc = dict(n_new=a.n_new, warm=a.warm, ctx=max(2048, pt + a.n_new), **c)
        t0 = time.time()
        try:
            m = runner.run_config(model, cfg, rt, prompts, cc, trace_ids=trace)
            m.pop("per_prompt")
            status = "ok"
        except Exception as e:
            m = dict(error=f"{type(e).__name__}: {str(e)[:300]}")
            status = "error"
            traceback.print_exc()
        rec = dict(
            phase=a.phase,
            cell_id=cell_id(bits, c),
            rep=rep,
            bits=bits,
            status=status,
            wall_s=time.time() - t0,
            **cc,
            **m,
        )
        open(out, "a").write(json.dumps(rec) + "\n")
        log(
            f"bits={bits} {c} rep={rep}: "
            + (
                f"{m['decode_tok_s']:.2f} tok/s hit={m['hit_rate']:.3f} "
                f"dec_h2d={m['dec_h2d_mb_per_token']:.0f}MB/tok dec_miss/tok={m['dec_misses_per_token']:.1f} "
                f"dec_cpu/tok={m['dec_cpu_experts_per_token']:.1f} slots={m['slots']}"
                if status == "ok"
                else m["error"]
            )
        )
    del model, rt
    torch.cuda.empty_cache()
log("phase done")
