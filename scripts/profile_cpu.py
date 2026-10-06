"""Where does the CPU spend its time during decode? cProfile over N decode steps (GPU synchronised at the end).
python scripts/profile_cpu.py --bits 4 --vram-gb 8"""

import argparse
import cProfile
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("WISP_FUSED", "2")
import torch

from wisp import runner
from wisp.budget import plan_slots
from wisp.model import configure, nbytes_of

ap = argparse.ArgumentParser()
ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--vram-gb", type=float, default=8)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--model-dir", default=os.environ.get("WISP_MODEL"))
a = ap.parse_args()

model, cfg, rt, tok = runner.setup(a.model_dir, a.bits, cpu_threads=12)
slots = plan_slots(a.vram_gb, nbytes_of(model), rt.store, cfg, 1024)["slots"]
configure(rt, slots, "lru", "transfer", vram_cap_gb=a.vram_gb)
prompts = runner.load_prompts("data/prompts.jsonl", tok, 3, 128)
for _, ids in prompts[:2]:
    runner.generate_timed(model, ids, 24)  # warm cache + compile

ids = prompts[2][1].cuda()
with torch.no_grad():
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    pk, nxt = out.past_key_values, out.logits[:, -1].argmax(-1)
    torch.cuda.synchronize()
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    for _ in range(a.steps):
        out = model(input_ids=nxt[:, None], past_key_values=pk, use_cache=True)
        pk = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
    torch.cuda.synchronize()
    pr.disable()
    dt = time.perf_counter() - t0
print(f"\n{a.steps} steps in {dt:.2f}s = {a.steps / dt:.1f} tok/s  ({dt / a.steps * 1000:.1f} ms/token)\n")
st = pstats.Stats(pr)
st.sort_stats("tottime").print_stats(22)
