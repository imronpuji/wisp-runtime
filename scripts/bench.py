"""Single-configuration benchmark. Example:
  python scripts/bench.py --bits 4 --vram-gb 12 --policy wsa --miss transfer
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from wisp import runner

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default=os.environ.get("WISP_MODEL", "/workspace/models/Qwen3-30B-A3B"))
ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--vram-gb", type=float, default=22)
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--policy", default="wsa", choices=["lru", "wsa", "static"])
ap.add_argument("--miss", default="transfer", choices=["transfer", "hybrid"])
ap.add_argument("--prefetch", action="store_true")
ap.add_argument("--cpu-shadow", action="store_true", help="hybrid: CPU computes misses from a bf16 copy")
ap.add_argument("--kv-mode", default="adaptive", choices=["fixed", "adaptive"])
ap.add_argument("--n-prompts", type=int, default=6); ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--n-new", type=int, default=64); ap.add_argument("--prompt-tokens", type=int, default=128)
ap.add_argument("--prompts", default="data/prompts.jsonl")
ap.add_argument("--trace", default=None)
ap.add_argument("--cpu-threads", type=int, default=12)
ap.add_argument("--show-text", action="store_true")
ap.add_argument("--out", default="results/bench.jsonl")
a = ap.parse_args()

model, cfg, rt, tok = runner.setup(a.model_dir, a.bits, cpu_threads=a.cpu_threads)
shadow = runner.build_cpu_shadow(a.model_dir, cfg) if a.cpu_shadow else None
prompts = runner.load_prompts(a.prompts, tok, a.n_prompts + a.warm, a.prompt_tokens)
trace = None
if a.trace:
    from wisp.sim import load_trace
    trace = load_trace(a.trace)[0]
c = dict(vram_gb=a.vram_gb, ctx=a.ctx, policy=a.policy, miss=a.miss, prefetch=a.prefetch, kv_mode=a.kv_mode,
         n_new=a.n_new, warm=a.warm, cpu_shadow=a.cpu_shadow)
if a.show_text:
    from wisp.model import configure
    from wisp.budget import plan_slots
    from wisp.model import nbytes_of
    slots = plan_slots(a.vram_gb, nbytes_of(model), rt.store, cfg, a.ctx)["slots"]
    configure(rt, slots, a.policy, a.miss, vram_cap_gb=a.vram_gb)
    ids = prompts[0][1]
    _, _, _, out = runner.generate_timed(model, ids, 60)
    print("PROMPT :", tok.decode(ids[0][-60:]))
    print("OUTPUT :", tok.decode(out))
res = runner.run_config(model, cfg, rt, prompts, c, trace_ids=trace, cpu_shadow=shadow)
res.pop("per_prompt")
rec = dict(bits=a.bits, **c, **res)
print(json.dumps(rec, indent=1))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(a.out, "a").write(json.dumps(rec) + "\n")
