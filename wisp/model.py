"""Build a Qwen3-MoE model whose experts live in an ExpertStore; only non-expert weights go to the GPU."""

import json
import os
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from .cache import ExpertCache
from .moe import Runtime, attach


def nbytes_of(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        b.numel() * b.element_size() for b in model.buffers()
    )


def load_skeleton(model_dir, rt, device="cuda"):
    """Meta-init the HF model, attach paged blocks, then load all NON-expert tensors straight to `device`."""
    from safetensors import safe_open

    cfg = AutoConfig.from_pretrained(model_dir)
    if cfg.model_type == "gpt_oss":
        from .gptoss import load_skeleton_gptoss

        return load_skeleton_gptoss(model_dir, rt, device)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    for layer in model.model.layers:  # give every gate real storage so attach() keeps a real module
        layer.mlp.gate = torch.nn.Linear(
            cfg.hidden_size, cfg.num_experts, bias=False, device=device, dtype=torch.bfloat16
        )
    attach(model, rt)
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    wanted = {k: f for k, f in idx.items() if ".mlp.experts." not in k}
    sd = {}
    t0 = time.time()
    for f in sorted(set(wanted.values())):
        with safe_open(os.path.join(model_dir, f), framework="pt", device=device) as h:
            for k in h.keys():
                if k in wanted:
                    sd[k] = h.get_tensor(k)
    missing = model.load_state_dict(sd, strict=False, assign=True)
    bad = [k for k in missing.missing_keys if "rotary" not in k]
    assert not bad, f"missing keys: {bad[:5]}"
    rot = type(model.model.rotary_emb)(config=cfg, device=device)
    model.model.rotary_emb = rot
    model.eval()
    print(
        f"[model] non-expert weights on {device}: {nbytes_of(model) / 2**30:.2f} GiB ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return model, cfg


def make_runtime(store, **kw):
    return Runtime(store, cache=None, **kw)


def configure(
    rt,
    nslots,
    policy="lru",
    miss_policy="transfer",
    decay=0.9995,
    n_static=0,
    admit_after=2.0,
    preload_keys=None,
    prefetch=False,
    cpu_threshold=4,
    chunk=8,
    vram_cap_gb=None,
    cpu_store=None,
    decode_on_load=False,
):
    """(Re)build the GPU cache with a new setting, optionally capping the allocator to emulate a smaller GPU."""
    if rt.cache is not None:
        rt.cache.free()
        rt.cache = None
    torch.cuda.empty_cache()
    if vram_cap_gb is not None:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, vram_cap_gb * 2**30 / total), 0)
    rt.cache = ExpertCache(
        rt.store, nslots, "cuda", policy, decay, n_static, admit_after, preload_keys, cpu_store, decode_on_load
    )
    rt.miss_policy, rt.prefetch, rt.cpu_threshold, rt.chunk = miss_policy, prefetch, cpu_threshold, chunk
    rt.reset_stats()
    return rt.cache


def patch_fast_norm():
    """Qwen3MoeRMSNorm as 2 ops (F.rms_norm + weight) instead of ~8 small ones.
    Same math: fp32 inside, cast, then weight.
    CPU launch overhead dominated decode on slow-CPU hosts. Disable with WISP_FASTNORM=0."""
    import torch.nn.functional as F
    from transformers.models.qwen3_moe import modeling_qwen3_moe as m

    def forward(self, x):
        return self.weight * F.rms_norm(x, self.weight.shape, None, self.variance_epsilon)

    if not getattr(m.Qwen3MoeRMSNorm, "_wisp_patched", False):
        m.Qwen3MoeRMSNorm.forward = forward
        m.Qwen3MoeRMSNorm._wisp_patched = True
        print("[patch] Qwen3MoeRMSNorm -> F.rms_norm", flush=True)


def patch_fast_attention():
    """Cut per-layer CPU launches in decode (B=1, T=1): one-kernel rotary, and GQA attention without repeat_kv copies
    (the 8 query heads of a KV group are treated as 8 'positions' of one head, which is exactly equivalent
    for a single query).
    Prefill and any other shape use the original code. Disable with WISP_FASTATTN=0."""
    import torch.nn.functional as F
    from transformers.models.qwen3_moe import modeling_qwen3_moe as m

    from .kernels import rope_decode

    if getattr(m, "_wisp_attn_patched", False):
        return
    orig_rope = m.apply_rotary_pos_emb

    def rope(q, k, cos, sin, unsqueeze_dim=1):
        if q.shape[0] == 1 and q.shape[2] == 1 and q.dim() == 4 and unsqueeze_dim == 1:
            return rope_decode(q, k, cos, sin)
        return orig_rope(q, k, cos, sin, unsqueeze_dim)

    m.apply_rotary_pos_emb = rope
    try:
        orig_sdpa = m.ALL_ATTENTION_FUNCTIONS["sdpa"]
    except Exception:
        orig_sdpa = None
    if orig_sdpa is not None:

        def sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
            if (
                query.shape[0] == 1
                and query.shape[2] == 1
                and getattr(module, "sliding_window", None) is None
                and query.shape[1] % key.shape[1] == 0
                and not kw.get("output_attentions", False)
            ):
                B, Hq, _, D = query.shape
                Hk = key.shape[1]
                o = F.scaled_dot_product_attention(
                    query.reshape(B, Hk, Hq // Hk, D), key, value, scale=scaling, is_causal=False
                )
                return o.reshape(B, Hq, 1, D).transpose(1, 2).contiguous(), None
            return orig_sdpa(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kw)

        try:
            m.ALL_ATTENTION_FUNCTIONS.register("sdpa", sdpa)
        except Exception:
            m.ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa
    m._wisp_attn_patched = True
    print("[patch] rotary + GQA decode attention", flush=True)
