"""Microbench on GPU: fused 4-bit kernel vs dequant_fast + bmm, one MoE layer (k=8 experts, 1 token)."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

from wisp.kernels import moe_decode_q3_fast, moe_decode_q4, moe_decode_q4_fast
from wisp.quant import dequant_fast, quantize

H, I, k, E = 2048, 768, 8, 64
n = H * I
dev = "cuda"
w3 = (torch.randn(E, 3, n, device=dev) * 0.02).to(torch.bfloat16)
pool = torch.stack([quantize(w3[e].cpu(), 4, 128) for e in range(E)]).to(dev)
slots = torch.randperm(E, device=dev)[:k]
x = torch.randn(H, device=dev, dtype=torch.bfloat16)
w = torch.softmax(torch.randn(k, device=dev), 0)


def ref():
    Wb = dequant_fast(pool.index_select(0, slots), n, 4, 128)
    xg = x.view(1, 1, H).expand(k, 1, H)
    g = torch.bmm(xg, Wb[:, 0].view(k, I, H).transpose(1, 2))
    u = torch.bmm(xg, Wb[:, 1].view(k, I, H).transpose(1, 2))
    y = torch.bmm(torch.nn.functional.silu(g) * u, Wb[:, 2].view(k, H, I).transpose(1, 2))
    return (y * w.view(k, 1, 1)).sum(0)[0]


def t(f, it=200):
    for _ in range(20):
        f()
    torch.cuda.synchronize()
    a = time.perf_counter()
    for _ in range(it):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - a) / it * 1e3


fu = lambda: moe_decode_q4(pool, slots, x, w, n, H, I)
err = ((fu().float() - ref().float()).abs().max() / ref().float().abs().max()).item()
print(f"rel.err {err:.4f}")
print(f"ref  (dequant+bmm): {t(ref):.3f} ms/layer -> x48 = {t(ref) * 48:.1f} ms/token")
for br in (16, 32, 64):
    f = lambda: moe_decode_q4(pool, slots, x, w, n, H, I, block_r=br)
    print(f"fused block_r={br}: {t(f):.3f} ms/layer -> x48 = {t(f) * 48:.1f} ms/token")

meta = torch.cat([slots.float(), w.float()])
fe = (moe_decode_q4_fast(pool, meta, x, k, n, H, I).float() - ref().float()).abs().max() / ref().float().abs().max()
print(f"fast rel.err {fe.item():.4f}")
for gu, dn in ((32, 8), (32, 16), (32, 4), (16, 8)):
    f = lambda: moe_decode_q4_fast(pool, meta, x, k, n, H, I, block_gu=gu, block_down=dn)
    print(f"fast gu={gu} down={dn}: {t(f):.3f} ms/layer -> x48 = {t(f) * 48:.1f} ms/token")

# ---- 3-bit ----
from wisp.quant import dequant

pool3 = torch.stack([quantize(w3[e].cpu(), 3, 128) for e in range(E)]).to(dev)


def ref3():
    Wb = dequant(pool3.index_select(0, slots), n, 3, 128)
    xg = x.view(1, 1, H).expand(k, 1, H)
    g = torch.bmm(xg, Wb[:, 0].view(k, I, H).transpose(1, 2))
    u = torch.bmm(xg, Wb[:, 1].view(k, I, H).transpose(1, 2))
    y = torch.bmm(torch.nn.functional.silu(g) * u, Wb[:, 2].view(k, H, I).transpose(1, 2))
    return (y * w.view(k, 1, 1)).sum(0)[0]


e3 = (
    (moe_decode_q3_fast(pool3, meta, x, k, n, H, I).float() - ref3().float()).abs().max() / ref3().float().abs().max()
).item()
print(f"q3 rel.err {e3:.4f}")
print(f"q3 ref (dequant eager+bmm): {t(ref3):.3f} ms/layer")
for gu, dn in ((8, 8), (4, 8), (2, 8), (4, 16), (8, 16)):
    f = lambda: moe_decode_q3_fast(pool3, meta, x, k, n, H, I, block_gu=gu, block_down=dn)
    print(f"q3 fast gu={gu} down={dn}: {t(f):.3f} ms/layer -> x48 = {t(f) * 48:.1f} ms/token")
