"""Hardware microbenchmarks: PCIe H2D bandwidth, GPU per-expert cost, CPU per-expert cost."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import statistics
from wisp.quant import quantize, dequant, dequant_fast, expert_nbytes
from wisp.moe import _mlp

ap = argparse.ArgumentParser()
ap.add_argument("--H", type=int, default=2048); ap.add_argument("--I", type=int, default=768)
ap.add_argument("--bits", default="16,8,4,3,2"); ap.add_argument("--threads", default="4,8,12,24")
ap.add_argument("--out", default="results/microbench.json")
a = ap.parse_args()
n = a.H * a.I
res = dict(h2d={}, gpu_expert_ms={}, cpu_expert_ms={})

def timeit(fn, it=30, warm=5, sync=False):
    """median seconds per call (robust to noisy neighbours); fn may take the iteration index."""
    import inspect
    takes_i = len(inspect.signature(fn).parameters) == 1
    call = (lambda i: fn(i)) if takes_i else (lambda i: fn())
    for i in range(warm): call(i)
    if sync: torch.cuda.synchronize()
    ts = []
    for i in range(it):
        t = time.perf_counter(); call(i)
        if sync: torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)

# 1) pinned host -> device bandwidth
for mb in (1, 2.4, 9.4, 64, 256):
    b = int(mb * 2 ** 20)
    h = torch.empty(b, dtype=torch.uint8).pin_memory(); d = torch.empty(b, dtype=torch.uint8, device="cuda")
    dt = timeit(lambda: d.copy_(h, non_blocking=True), it=50 if mb < 100 else 10, sync=True)
    res["h2d"][str(mb)] = dict(gb_s=b / dt / 1e9, ms=dt * 1e3)
    print(f"H2D {mb:>6} MiB: {b/dt/1e9:5.1f} GB/s", flush=True)

# 2) GPU cost per expert (dequant + 3 GEMV) for a batch of 8 resident experts, decode (1 token)
x = torch.randn(1, a.H, device="cuda", dtype=torch.bfloat16)
for bits in map(int, a.bits.split(",")):
    w = torch.randn(3, n, device="cuda") * 0.02
    buf = quantize(w, bits, 128)
    pool = buf[None].repeat(8, 1).contiguous()
    def run():
        W = list(dequant_fast(pool, n, bits, 128).unbind(0))
        for W3 in W: _mlp(x, W3, a.I, a.H)
    dt = timeit(run, sync=True) / 8
    res["gpu_expert_ms"][str(bits)] = dt * 1e3
    print(f"GPU expert bits={bits:>2}: {dt*1e3:.3f} ms  ({expert_nbytes(n,bits,128)/2**20:.2f} MiB)", flush=True)

# 3) CPU cost per expert for 1 token, by thread count
xc = torch.randn(1, a.H, dtype=torch.bfloat16)
for th in map(int, a.threads.split(",")):
    torch.set_num_threads(th)
    for bits in map(int, a.bits.split(",")):
        K = 96   # distinct experts, round-robin => reads come from DRAM (K * 9 MiB >> L3), like real cache misses
        pool = [quantize(torch.randn(3, n) * 0.02, bits, 128) for _ in range(K)]
        def run(i):
            buf = pool[i % K]
            W3 = buf.view(torch.bfloat16).view(3, n) if bits == 16 else dequant(buf[None], n, bits, 128)[0]
            _mlp(xc, W3, a.I, a.H)
        dt = timeit(run, it=60, warm=4)
        res["cpu_expert_ms"][f"{th}t_{bits}b"] = dt * 1e3
        print(f"CPU expert threads={th:>2} bits={bits:>2}: {dt*1e3:.3f} ms", flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
