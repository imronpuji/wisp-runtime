"""Record which experts are used at every decode step (routing trace) -> data/trace_b{bits}.npz"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from wisp import runner
from wisp.budget import plan_slots
from wisp.model import configure, nbytes_of


def hit_rate(st):
    return st["hits"] / max(1, st["hits"] + st["misses"])


ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default=os.environ.get("WISP_MODEL"))
ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--vram-gb", type=float, default=22)
ap.add_argument("--n-prompts", type=int, default=64)
ap.add_argument("--n-new", type=int, default=96)
ap.add_argument("--prompt-tokens", type=int, default=128)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--prompts", default="data/prompts.jsonl")
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or f"data/trace_b{a.bits}.npz"

model, cfg, rt, tok = runner.setup(a.model_dir, a.bits, cpu_threads=12)
slots = plan_slots(a.vram_gb, nbytes_of(model), rt.store, cfg, 1024)["slots"]
configure(rt, slots, "lru", "transfer", vram_cap_gb=a.vram_gb)
prompts = runner.load_prompts(a.prompts, tok, a.n_prompts, a.prompt_tokens, a.offset)
L, k = cfg.num_hidden_layers, cfg.num_experts_per_tok
steps, pid = [], []
for i, (dom, ids) in enumerate(prompts):
    rt.recorder = []
    runner.generate_timed(model, ids, a.n_new)
    rec = [s for (l, s) in rt.recorder if s.shape[0] == 1]  # decode calls only
    arr = np.stack(rec).reshape(-1, L, k)  # [tokens, L, k]
    steps.append(arr.astype(np.int16))
    pid += [i] * arr.shape[0]
    print(
        f"[trace] prompt {i + 1}/{len(prompts)} ({dom}): {arr.shape[0]} decode steps, "
        f"hit_rate={hit_rate(rt.cache.core.stats):.3f}",
        flush=True,
    )
rt.recorder = None
ids = np.concatenate(steps)
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez_compressed(out, ids=ids, L=L, E=cfg.num_experts, prompt_id=np.array(pid), bits=a.bits)
print("saved", out, ids.shape)
