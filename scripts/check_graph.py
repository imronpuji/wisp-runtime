"""Is the CUDA-graph decoder numerically the same model? Teacher-forced: both paths get the SAME tokens, we compare logits.
  WISP_FUSED=2 PYTHONPATH=. python scripts/check_graph.py --bits 4 --vram-gb 8"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from wisp import runner
from wisp.model import configure, nbytes_of
from wisp.budget import plan_slots
from wisp.graphdec import GraphDecoder

ap = argparse.ArgumentParser()
ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--vram-gb", type=float, default=8)
ap.add_argument("--steps", type=int, default=48)
ap.add_argument("--n-prompts", type=int, default=6)
ap.add_argument("--model-dir", default=os.environ.get("WISP_MODEL"))
a = ap.parse_args()
model, cfg, rt, tok = runner.setup(a.model_dir, a.bits, cpu_threads=12)
slots = plan_slots(a.vram_gb, nbytes_of(model), rt.store, cfg, 1024)["slots"]
configure(rt, slots, "lru", "transfer", vram_cap_gb=a.vram_gb)
prompts = runner.load_prompts("data/prompts.jsonl", tok, a.n_prompts, 128)
dec = GraphDecoder(model, rt, tmax=128 + a.steps + 8)
tot_agree = tot = 0; worst = 0.0; cosmin = 1.0; top5 = 0
with torch.no_grad():
    for p, (_, ids) in enumerate(prompts):
        ids = ids.cuda()
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        pk, t = out.past_key_values, out.logits[:, -1].argmax(-1)
        dec.load_kv(pk, ids.shape[1])
        ag = 0
        for _ in range(a.steps):
            ref = model(input_ids=t[:, None], past_key_values=pk, use_cache=True)
            pk = ref.past_key_values
            r = ref.logits[:, -1].float(); g = dec.next(t).float()
            ag += int((r.argmax(-1) == g.argmax(-1)).item())
            top5 += int(g.argmax(-1).item() in r.topk(5, -1).indices[0].tolist())
            worst = max(worst, (r - g).abs().max().item())
            cosmin = min(cosmin, torch.nn.functional.cosine_similarity(r, g).item())
            t = r.argmax(-1)                      # same token to both
        tot_agree += ag; tot += a.steps
        print(f"prompt {p}: top-1 agreement {ag}/{a.steps}", flush=True)
print(f"\ntop-1 agreement: {tot_agree/tot:.1%}   graph top-1 inside eager top-5: {top5/tot:.1%}")
print(f"max |logit diff|: {worst:.3f}   min cosine similarity: {cosmin:.5f}")
