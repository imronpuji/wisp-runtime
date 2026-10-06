"""Where does one decode token go? Times attention vs MoE block per layer (synchronised) + top CUDA ops."""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from wisp import runner
from wisp.model import configure, nbytes_of
from wisp.budget import plan_slots

ap = argparse.ArgumentParser()
ap.add_argument("--bits", type=int, default=4); ap.add_argument("--vram-gb", type=float, default=22)
ap.add_argument("--model-dir", default=os.environ.get("WISP_MODEL", "/workspace/models/Qwen3-30B-A3B"))
a = ap.parse_args()
model, cfg, rt, tok = runner.setup(a.model_dir, a.bits, cpu_threads=12)
slots = plan_slots(a.vram_gb, nbytes_of(model), rt.store, cfg, 1024)["slots"]
configure(rt, slots, "lru", "transfer", vram_cap_gb=a.vram_gb)
prompts = runner.load_prompts("data/prompts.jsonl", tok, 3, 128)
for _, ids in prompts[:2]:
    runner.generate_timed(model, ids, 24)      # warm caches + compile

T = dict(attn=0.0, moe=0.0)
def wrap(mod, key):
    fwd = mod.forward
    def f(*args, **kw):
        torch.cuda.synchronize(); t = time.perf_counter()
        r = fwd(*args, **kw)
        torch.cuda.synchronize(); T[key] += time.perf_counter() - t
        return r
    mod.forward = f
for l in model.model.layers:
    wrap(l.self_attn, "attn"); wrap(l.mlp, "moe")
ids = prompts[2][1]
ttft, dt, nt, _ = runner.generate_timed(model, ids, 41)
print(f"decode {nt/dt:.1f} tok/s = {dt/nt*1e3:.1f} ms/token")
print(f"  attention (48 layers): {T['attn']/(nt+1)*1e3:.1f} ms/token (incl. prefill share)")
print(f"  moe block (48 layers): {T['moe']/(nt+1)*1e3:.1f} ms/token")
print(f"  other (embed/norm/lm_head/python): {(dt/nt - (T['attn']+T['moe'])/(nt+1))*1e3:.1f} ms/token")
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    runner.generate_timed(model, ids, 9)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8, max_name_column_width=50))
