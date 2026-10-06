"""Shows the ACTUAL error of the paged model vs HF reference on a tiny model, plus a sabotage control."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import test_paged_moe as T
import torch

from wisp.cache import ExpertCache

for name, kw in [
    ("all-resident", dict(nslots=24)),
    ("thrash lru", dict(nslots=6)),
    ("hybrid", dict(nslots=6, miss="hybrid")),
    ("prefetch", dict(nslots=8, prefetch=True)),
]:
    model, cfg = T.tiny()
    ids = torch.randint(0, 256, (1, 14), device="cuda")
    with torch.no_grad():
        ref = model(ids).logits.float()
    rt = T.build(model, cfg, 16, **kw)
    got = T.logits(model, ids, 9)
    e = (got - ref).abs().max().item()
    s = rt.stats
    print(
        f"{name:13s} max|err|={e:.4f} ref_scale={ref.abs().max().item():.2f} "
        f"misses={rt.cache.core.stats['misses']} cpu_experts={s['cpu_experts']} prefetched={s['prefetched']}"
    )

model, cfg = T.tiny()
ids = torch.randint(0, 256, (1, 14), device="cuda")
with torch.no_grad():
    ref = model(ids).logits.float()
rt = T.build(model, cfg, 16, 24)
rt.store.host[:] = rt.store.host[torch.randperm(rt.store.host.shape[0])]
rt.cache.free()
rt.cache = ExpertCache(rt.store, 24)
with torch.no_grad():
    bad = model(ids).logits.float()
print(f"SABOTAGED (experts shuffled) max|err|={(bad - ref).abs().max().item():.3f}  <- must be much larger than above")
