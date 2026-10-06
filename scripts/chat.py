"""Interactive chat with the paged model. Example:
  WISP_FUSED=2 PYTHONPATH=. python scripts/chat.py --bits 4 --vram-gb 8 --ctx 32768 --graph
Type 'exit' to quit, '/reset' to clear the conversation.

Elastic context (Qwen3, WISP_FUSED=2, 3/4-bit): --ctx is only the MAXIMUM. VRAM for the KV cache is taken only as the
conversation actually grows (capacity steps 4K -> 8K -> 16K -> 32K ...), and the expert cache gets the rest, so a
short chat runs as fast as with a small --ctx. The KV of previous turns is reused: each turn only prefills new tokens."""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from wisp import runner
from wisp.model import configure, nbytes_of
from wisp.budget import plan_slots

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default=os.environ.get("WISP_MODEL"))
ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--vram-gb", type=float, default=8)
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--max-new", type=int, default=300)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--graph", action="store_true", help="CUDA-graph decode (needs WISP_FUSED=2)")
ap.add_argument("--effort", default="low", help="gpt-oss reasoning effort: low|medium|high")
ap.add_argument("--think", action="store_true", help="enable Qwen3 thinking mode (slower, longer)")
a = ap.parse_args()

model, cfg, rt, tok = runner.setup(a.model_dir, a.bits, cpu_threads=12)
GPT = cfg.model_type == "gpt_oss"
ELASTIC = (not GPT) and rt.fused == 2 and a.bits in (3, 4)
NONEXP = nbytes_of(model)
TOTAL = rt.store.L * rt.store.E


def slots_for(cap):
    return plan_slots(a.vram_gb, NONEXP, rt.store, cfg, cap, kv_mode="adaptive")["slots"]


dec = None
if ELASTIC:
    from wisp.graphdec import GraphDecoder
    cap = min(4096, a.ctx)
    configure(rt, slots_for(cap), "lru", "transfer", vram_cap_gb=a.vram_gb)
    dec = GraphDecoder(model, rt, tmax=cap, use_graph=a.graph)
    cached = []                                   # token ids whose K/V sit in dec's buffers, in order
else:
    slots = plan_slots(a.vram_gb, NONEXP, rt.store, cfg, a.ctx)["slots"]
    configure(rt, slots, "lru", "transfer", vram_cap_gb=a.vram_gb)
    if a.graph:
        from wisp.graphdec import GraphDecoder
        dec = GraphDecoder(model, rt, tmax=a.ctx)
if GPT:
    assert not a.graph, "--graph is Qwen3-only"
    stop = {tok.convert_tokens_to_ids(t) for t in ("<|return|>", "<|call|>")} | {tok.eos_token_id}
else:
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
hist = []
print(f"\nSiap. {a.bits}-bit, VRAM dibatasi {a.vram_gb} GB, fused={rt.fused}, konteks maks {a.ctx}"
      f"{' (elastis)' if ELASTIC else ''}. Ketik 'exit' untuk keluar.\n")


def set_capacity(need):
    """Smallest capacity step >= need; shrink the expert cache BEFORE growing KV (and the reverse), so the two never
    both hold their larger size at once."""
    cap = 4096
    while cap < need:
        cap *= 2
    cap = min(max(cap, need), a.ctx)
    if cap == dec.tmax:
        return
    ns = slots_for(cap)
    if ns < 2 * cfg.num_experts_per_tok:
        raise SystemExit(f"VRAM {a.vram_gb} GB tidak cukup untuk konteks {cap}: sisa slot expert {ns}")
    t = time.perf_counter()
    if cap > dec.tmax:
        rt.cache.resize(ns); dec.resize(cap)
    else:
        dec.resize(cap); rt.cache.resize(ns)
    print(f"[konteks] kapasitas {cap} token, cache expert {ns} slot ({ns/TOTAL:.0%}) "
          f"[{time.perf_counter()-t:.1f}s]", flush=True)


@torch.no_grad()
def reply_elastic(messages):
    global cached
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=a.think)
    ids = tok(text).input_ids[-(a.ctx - a.max_new):]
    p = 0
    while p < min(len(ids), len(cached)) and ids[p] == cached[p]:
        p += 1
    p = min(p, len(ids) - 1)                       # always run at least one token to get logits
    set_capacity(len(ids) + a.max_new)
    t0 = time.perf_counter()
    logits = dec.prefill(torch.tensor(ids[p:]), p)
    nxt = sample(logits)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    gen, shown = [], ""
    for i in range(a.max_new):
        t = nxt.item()
        if t in stop:
            break
        gen.append(t)
        s = tok.decode(gen)
        if not s.endswith("\ufffd"):
            print(s[len(shown):], end="", flush=True); shown = s
        nxt = sample(dec.next(nxt))
    dt = time.perf_counter() - t1
    cached = ids + gen
    print(f"\n[{len(gen)} token, {len(gen)/max(dt,1e-9):.1f} tok/s, prefill {len(ids)-p} token baru "
          f"(pakai ulang {p}) {t1-t0:.1f}s, konteks terpakai {len(cached)}/{dec.tmax}]\n")
    return tok.decode(gen)


def sample(logits):
    if a.temp <= 0:
        return logits.argmax(-1)
    p = torch.softmax(logits.float() / a.temp, -1)
    top, idx = p.topk(40, -1)
    return idx.gather(-1, torch.multinomial(top / top.sum(-1, keepdim=True), 1)).squeeze(-1)


@torch.no_grad()
def reply(messages):
    kw = dict(reasoning_effort=a.effort) if GPT else dict(enable_thinking=a.think)
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
    ids = tok(text, return_tensors="pt").input_ids[:, -(a.ctx - a.max_new):].cuda()
    t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    nxt = sample(out.logits[:, -1]); pk = out.past_key_values
    if dec is not None:
        dec.load_kv(pk, ids.shape[1])
    gen, shown, t1 = [], "", time.perf_counter()
    for i in range(a.max_new):
        t = nxt.item()
        if t in stop:
            break
        gen.append(t)
        s = tok.decode(gen)
        if not s.endswith("�"):
            print(s[len(shown):], end="", flush=True); shown = s
        if dec is not None:
            nxt = sample(dec.next(nxt))
            continue
        out = model(input_ids=nxt[:, None], past_key_values=pk, use_cache=True)
        pk = out.past_key_values
        nxt = sample(out.logits[:, -1])
    dt = time.perf_counter() - t1
    print(f"\n[{len(gen)} token, {len(gen)/max(dt,1e-9):.1f} tok/s, prefill {t1-t0:.1f}s]\n")
    full = tok.decode(gen, skip_special_tokens=False)
    if GPT and "<|channel|>final<|message|>" in full:      # keep only the answer in the history, not the reasoning
        return full.split("<|channel|>final<|message|>")[-1].replace("<|return|>", "").strip()
    return tok.decode(gen)


while True:
    try:
        q = input("Kamu: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if q in ("exit", "quit"):
        break
    if q == "/reset":
        hist = []
        if ELASTIC:
            cached = []; dec.pos = 0; set_capacity(1)
        print("(percakapan dikosongkan)\n"); continue
    if not q:
        continue
    hist.append({"role": "user", "content": q})
    print("Model: ", end="", flush=True)
    hist.append({"role": "assistant", "content": (reply_elastic if ELASTIC else reply)(hist)})
