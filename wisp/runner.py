"""Shared helpers for the scripts: setup, timed generation, metric snapshots."""

import json
import os
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from .model import load_skeleton, nbytes_of, patch_fast_attention, patch_fast_norm
from .store import ExpertStore


def setup(
    model_dir,
    bits,
    group=128,
    cache_dir=None,
    cpu_threads=None,
    device="cuda",
):
    """Load a Qwen3-MoE (or gpt-oss) model with paged experts. Returns (model, config, runtime, tokenizer).

    The quantized expert store is cached in `cache_dir` (default: $WISP_STORES, else ./stores) so later runs load it
    in seconds instead of re-quantizing the checkpoint."""
    if not model_dir:
        raise SystemExit("no model directory: pass --model-dir or set WISP_MODEL")
    if cache_dir is None:
        cache_dir = os.environ.get("WISP_STORES", "stores")
    if os.environ.get("WISP_FASTNORM", "1") == "1":
        patch_fast_norm()
    if os.environ.get("WISP_FASTATTN", "1") == "1":
        patch_fast_attention()
    cfg = AutoConfig.from_pretrained(model_dir)
    if cfg.model_type == "gpt_oss":
        from .gptoss import build_store

        store = build_store(model_dir, cfg, bits, group, device, cache_dir)
    else:
        store = ExpertStore.from_safetensors(model_dir, cfg, bits, group, device, cache_dir)
    from .moe import Runtime

    rt = Runtime(store, cpu_threads=cpu_threads)
    model, cfg = load_skeleton(model_dir, rt, device)
    tok = AutoTokenizer.from_pretrained(model_dir)
    return model, cfg, rt, tok


def load_prompts(path, tok, n, prompt_tokens, offset=0):
    rows = [json.loads(l) for l in open(path)]
    out = []
    for r in rows[offset : offset + n]:
        ids = tok(r["text"], return_tensors="pt").input_ids[:, :prompt_tokens]
        out.append((r.get("domain", "?"), ids))
    return out


@torch.no_grad()
def generate_timed(model, ids, n_new, rt=None):
    """Greedy decode. Returns (ttft_s, decode_s, decode_tokens, generated ids).
    If rt is given, rt._after_prefill is set to a metric snapshot taken between prefill and decode."""
    ids = ids.cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    nxt = out.logits[:, -1].argmax(-1)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    if rt is not None:
        rt._after_prefill = snap(rt)
    pk, toks = out.past_key_values, [nxt]
    t1 = time.perf_counter()
    for _ in range(n_new - 1):
        out = model(input_ids=nxt[:, None], past_key_values=pk, use_cache=True)
        pk = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        toks.append(nxt)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t1
    return ttft, dt, n_new - 1, torch.cat(toks)


def snap(rt):
    c = rt.cache
    return dict(
        hits=c.core.stats["hits"], misses=c.core.stats["misses"], bytes=c.bytes_h2d, loads=c.n_loads, **rt.stats
    )


def bench_prompts(model, rt, prompts, n_new, warm=2):
    """Run `warm` prompts (cache warm-up, not reported) then time the rest. Returns aggregate metrics."""
    for _, ids in prompts[:warm]:
        generate_timed(model, ids, min(n_new, 32))
    torch.cuda.reset_peak_memory_stats()
    rows, s0 = [], snap(rt)
    dc = {}  # decode-only counters (prefill excluded)
    for dom, ids in prompts[warm:]:
        ttft, dt, nt, _ = generate_timed(model, ids, n_new, rt)
        end = snap(rt)
        for k in end:
            dc[k] = dc.get(k, 0) + end[k] - rt._after_prefill[k]
        rows.append(dict(domain=dom, prompt_len=int(ids.shape[1]), ttft=ttft, decode_s=dt, decode_tokens=nt))
    s1 = snap(rt)
    d = {k: s1[k] - s0[k] for k in s0}
    tok = sum(r["decode_tokens"] for r in rows)
    dec = sum(r["decode_s"] for r in rows)
    calls = max(1, d["layer_calls"])
    return dict(
        decode_tok_s=tok / dec,
        ttft_s=float(np.mean([r["ttft"] for r in rows])),
        prefill_tok_s=float(np.mean([r["prompt_len"] / r["ttft"] for r in rows])),
        hit_rate=d["hits"] / max(1, d["hits"] + d["misses"]),
        expert_misses_per_layer_call=d["misses"] / calls,
        h2d_mb_per_token=d["bytes"] / 2**20 / max(1, tok + len(rows)),
        cpu_experts_per_token=d["cpu_experts"] / max(1, tok + len(rows)),
        cpu_time_frac=d["cpu_time"] / max(1e-9, dec + sum(r["ttft"] for r in rows)),
        peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
        n_prompts=len(rows),
        decode_tokens=tok,
        dec_h2d_mb_per_token=dc["bytes"] / 2**20 / max(1, tok),
        dec_misses_per_token=dc["misses"] / max(1, tok),
        dec_hit_rate=dc["hits"] / max(1, dc["hits"] + dc["misses"]),
        dec_cpu_experts_per_token=dc["cpu_experts"] / max(1, tok),
        per_prompt=rows,
    )


def build_cpu_shadow(model_dir, cfg, group=128):
    """bf16 copy of all experts in (unpinned) host RAM, used only for CPU-side compute of misses."""
    return ExpertStore.from_safetensors(model_dir, cfg, 16, group, "cuda", cache_dir=None, pin=False)


def run_config(model, cfg, rt, prompts, c, trace_ids=None, cpu_shadow=None):
    """One experiment cell. `c` keys: vram_gb, ctx, policy, miss, prefetch, kv_mode, n_new, warm, ..."""
    from .budget import plan_slots
    from .model import configure
    from .sim import top_keys

    chunk = c.get("chunk", 8)
    plan = plan_slots(
        c["vram_gb"],
        nbytes_of(model),
        rt.store,
        cfg,
        c.get("ctx", 2048),
        c.get("kv_mode", "adaptive"),
        c.get("kv_frac", 0.3),
        chunk,
        decode_on_load=c.get("decode_on_load", False),
    )
    slots = c.get("slots") or plan["slots"]
    n_static, preload = 0, None
    if c["policy"] == "static":
        n_static = max(0, slots - c.get("scratch", 16))
        assert trace_ids is not None, "static policy needs a profiling trace"
        preload = top_keys(trace_ids, cfg.num_hidden_layers, cfg.num_experts, n_static)
    if slots - n_static < chunk:
        raise ValueError(f"budget too small: {slots} slots")

    configure(
        rt,
        slots,
        c["policy"],
        c.get("miss", "transfer"),
        c.get("decay", 0.9995),
        n_static,
        c.get("admit_after", 2.0),
        preload,
        c.get("prefetch", False),
        c.get("cpu_threshold", 4),
        chunk,
        vram_cap_gb=c["vram_gb"],
        cpu_store=cpu_shadow if c.get("cpu_shadow") else None,
        decode_on_load=c.get("decode_on_load", False),
    )
    m = bench_prompts(model, rt, prompts, c.get("n_new", 64), c.get("warm", 2))
    m.update(slots=slots, kv_gib=plan["kv_gib"], expert_mib=plan["expert_mib"], fits_all=plan["fits_all"])
    return m


@torch.no_grad()
def generate_graph(dec, model, ids, n_new):
    """Same as generate_timed but decode runs through the CUDA-graph GraphDecoder. Returns (ttft, decode_s, n, ids)."""
    ids = ids.cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    nxt = out.logits[:, -1].argmax(-1)
    dec.load_kv(out.past_key_values, ids.shape[1])
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    toks = [nxt]
    t1 = time.perf_counter()
    for _ in range(n_new - 1):
        nxt = dec.next(nxt).argmax(-1)
        toks.append(nxt)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t1
    return ttft, dt, n_new - 1, torch.cat(toks)
