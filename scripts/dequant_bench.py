import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from wisp.quant import quantize, dequant

n = 2048 * 768
def t(fn, it=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / it

for bits in (8, 4, 3, 2):
    pool = quantize(torch.randn(3, n, device="cuda") * 0.02, bits, 128)[None].repeat(8, 1).contiguous()
    eager = lambda: dequant(pool, n, bits, 128)
    try:
        comp = torch.compile(lambda b: dequant(b, n, bits, 128), dynamic=True)
        ref, got = eager(), comp(pool)
        ok = torch.equal(ref, got)
        te, tc = t(eager), t(lambda: comp(pool))
        print(f"bits={bits}: eager {te/8*1e3:.3f} ms/expert | compiled {tc/8*1e3:.3f} ms/expert | identical={ok}", flush=True)
    except Exception as e:
        print(f"bits={bits}: compile failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
