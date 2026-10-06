"""Quality of the CUDA-graph decoder on REAL text: teacher-forced NLL of the true next token, eager vs graph (decode path only).
  WISP_FUSED=2 PYTHONPATH=. python scripts/ppl_graph.py --bits 4 --vram-gb 8"""
import argparse, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn.functional as F
from wisp import runner
from wisp.model import configure, nbytes_of
from wisp.budget import plan_slots
from wisp.graphdec import GraphDecoder

ap = argparse.ArgumentParser()
ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--vram-gb", type=float, default=8)
ap.add_argument("--prefix", type=int, default=48)
ap.add_argument("--total", type=int, default=128)
ap.add_argument("--n-prompts", type=int, default=12)
ap.add_argument("--model-dir", default=os.environ.get("WISP_MODEL"))
a = ap.parse_args()
model, cfg, rt, tok = runner.setup(a.model_dir, a.bits, cpu_threads=12)
slots = plan_slots(a.vram_gb, nbytes_of(model), rt.store, cfg, 1024)["slots"]
configure(rt, slots, "lru", "transfer", vram_cap_gb=a.vram_gb)
prompts = runner.load_prompts("data/prompts.jsonl", tok, a.n_prompts, a.total)
dec = GraphDecoder(model, rt, tmax=a.total + 8)
ne = ng = n = 0; nllE = nllG = 0.0; cos = []
with torch.no_grad():
    for p, (_, ids) in enumerate(prompts):
        if ids.shape[1] < a.total:
            continue
        ids = ids.cuda()
        out = model(input_ids=ids[:, :a.prefix], use_cache=True, logits_to_keep=1)
        pk = out.past_key_values
        dec.load_kv(pk, a.prefix)
        lr = out.logits[:, -1]
        for t in range(a.prefix, a.total):
            tok_in = ids[:, t:t + 1]                  # true token t is fed, predicts token t+1 (if any)
            ref = model(input_ids=tok_in, past_key_values=pk, use_cache=True)
            pk = ref.past_key_values
            r = ref.logits[:, -1].float(); g = dec.next(tok_in[:, 0]).float()
            if t + 1 < a.total:
                tgt = ids[:, t + 1]
                nllE += F.cross_entropy(r, tgt).item(); nllG += F.cross_entropy(g, tgt).item(); n += 1
            cos.append(F.cosine_similarity(r, g).item())
cos = torch.tensor(cos)
print(f"\ntokens scored: {n}")
print(f"mean NLL  eager {nllE/n:.4f}  (ppl {math.exp(nllE/n):.3f})   graph {nllG/n:.4f}  (ppl {math.exp(nllG/n):.3f})")
print(f"cosine(eager,graph): median {cos.median():.5f}   p5 {cos.quantile(0.05):.5f}   min {cos.min():.5f}")
