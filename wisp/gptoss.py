"""gpt-oss adapter (openai/gpt-oss-20b and -120b): MXFP4 checkpoint -> WiSP expert store, and a paged MoE block.

An expert is gate_up [H, 2I] (gate/up interleaved on the last axis), down [I, H], plus two biases. With I == H both
halves of gate_up and down are n = H*I elements, so an expert fits the framework's flat [3, n] layout as
    [ gate_up.flat[:n] | gate_up.flat[n:] | down.flat ]
and the whole store/cache/quantizer is reused unchanged. The (tiny) biases stay on the GPU for all experts.
This block uses plain torch ops on the dequantized experts (no fused kernel yet): correct first, fast later."""
import json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .store import ExpertStore

FP4 = [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def mxfp4_to_bf16(blocks, scales):
    """blocks [..., G, 16] uint8 (two fp4 per byte, low nibble first), scales [..., G] uint8 (power-of-two exponents,
    bias 127) -> [..., G*32] bfloat16."""
    lut = torch.tensor(FP4, dtype=torch.float32, device=blocks.device)
    lo, hi = lut[(blocks & 0xF).long()], lut[(blocks >> 4).long()]
    v = torch.stack([lo, hi], -1).reshape(*blocks.shape[:-1], 32)
    v = v * torch.exp2(scales.float() - 127.0).unsqueeze(-1)
    return v.reshape(*blocks.shape[:-2], -1).to(torch.bfloat16)


def expert_w3(gate_up, down):
    """gate_up [H, 2I], down [I, H] -> [3, n] (n = H*I)."""
    H, I2 = gate_up.shape
    n = H * (I2 // 2)
    f = gate_up.reshape(-1)
    return torch.stack([f[:n], f[n:], down.reshape(-1)])


def _layer_tensors(get, has, l, E, device):
    """Dequantized gate_up [E,H,2I], down [E,I,H], biases, for one layer (MXFP4 or plain bf16 checkpoints)."""
    p = f"model.layers.{l}.mlp.experts."
    if has(p + "gate_up_proj_blocks"):
        gu = mxfp4_to_bf16(get(p + "gate_up_proj_blocks").to(device), get(p + "gate_up_proj_scales").to(device))
        dn = mxfp4_to_bf16(get(p + "down_proj_blocks").to(device), get(p + "down_proj_scales").to(device))
        gu, dn = gu.transpose(1, 2).contiguous(), dn.transpose(1, 2).contiguous()   # [E,out,K] -> [E,K,out]
    else:
        gu, dn = get(p + "gate_up_proj").to(device), get(p + "down_proj").to(device)
    return gu, dn, get(p + "gate_up_proj_bias"), get(p + "down_proj_bias")


def _reader(model_dir):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(name):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(model_dir, f), framework="pt", device="cpu")
        return handles[f].get_tensor(name)

    return get, (lambda name: name in idx), idx


def build_store(model_dir, cfg, bits, group=128, device="cuda", cache_dir=None, pin=True):
    L, E, H, I = cfg.num_hidden_layers, cfg.num_local_experts, cfg.hidden_size, cfg.intermediate_size
    assert H == I, "adapter assumes intermediate_size == hidden_size (true for gpt-oss)"
    n = H * I
    st = ExpertStore(L, E, n, bits, group, pin)
    path = None
    if cache_dir and bits < 16:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"store_gptoss_b{bits}_g{group}_{L}x{E}x{n}.u8")
        if os.path.exists(path) and os.path.getsize(path) == st.host.numel():
            st._load(path)
            return st
    get, has, _ = _reader(model_dir)
    t0 = time.time()
    for l in range(L):
        gu, dn, _, _ = _layer_tensors(get, has, l, E, device)
        for e in range(E):
            st.put(l, e, expert_w3(gu[e], dn[e]), device)
        del gu, dn
        if l % 4 == 3 or l == L - 1:
            print(f"[store] gpt-oss bits={bits} layer {l+1}/{L}  {time.time()-t0:.0f}s", flush=True)
    if path:
        st._save(path)
    return st


class PagedGptOssBlock(nn.Module):
    """Drop-in for GptOssMLP: router stays resident, experts are paged. Returns (output, router_scores) like HF."""

    def __init__(self, router, layer_idx, rt, cfg, device="cuda"):
        super().__init__()
        self.router, self.layer_idx, self.rt = router, layer_idx, rt
        self.E, self.top_k = cfg.num_local_experts, cfg.num_experts_per_tok
        self.H, self.I = cfg.hidden_size, cfg.intermediate_size
        self.alpha, self.limit = 1.702, float(getattr(cfg, "swiglu_limit", 7.0))
        self.register_buffer("gub", torch.zeros(self.E, 2 * self.I, dtype=torch.bfloat16, device=device), persistent=False)
        self.register_buffer("db", torch.zeros(self.E, self.H, dtype=torch.bfloat16, device=device), persistent=False)

    def _act(self, gu):
        gate, up = gu[..., ::2].clamp(max=self.limit), gu[..., 1::2].clamp(min=-self.limit, max=self.limit)
        return (up + 1) * (gate * torch.sigmoid(gate * self.alpha))

    def forward(self, hidden_states):
        rt, cache = self.rt, self.rt.cache
        B, T, H = hidden_states.shape
        x = hidden_states.reshape(-1, H)
        N, k, I = x.shape[0], self.top_k, self.I
        logits = F.linear(x, self.router.weight, self.router.bias)
        tv, ti = torch.topk(logits, k, dim=-1)
        sc = F.softmax(tv, dim=1, dtype=tv.dtype)
        base = self.layer_idx * self.E
        rt.stats["layer_calls"] += 1
        if rt.recorder is not None:
            rt.recorder.append((self.layer_idx, ti.cpu().numpy().copy()))
        if N == 1:                                           # decode: one sync, all k experts in 2 bmm
            ex = ti[0].tolist()
            slots = cache.ensure([base + e for e in ex])
            W = cache.weights_batched(torch.as_tensor(slots, device=x.device))            # [k,3,n]
            gu = torch.bmm(x.unsqueeze(0).expand(k, 1, H), W[:, :2].reshape(k, H, 2 * I)) + self.gub[ti[0]].unsqueeze(1)
            out = torch.bmm(self._act(gu), W[:, 2].reshape(k, I, H)) + self.db[ti[0]].unsqueeze(1)
            y = (out[:, 0] * sc[0].unsqueeze(-1)).sum(0, keepdim=True)
            rt.stats["gpu_experts"] += k
        else:                                                # prefill: group tokens per expert, page in chunks
            y = torch.zeros_like(x)
            uniq = torch.unique(ti).tolist()
            for i in range(0, len(uniq), rt.chunk):
                ce = uniq[i:i + rt.chunk]
                Ws = cache.weights(cache.ensure([base + e for e in ce]))
                for e, W3 in zip(ce, Ws):
                    tok, pos = torch.where(ti == e)
                    gu = x[tok] @ torch.cat([W3[0], W3[1]]).view(H, 2 * I) + self.gub[e]
                    out = self._act(gu) @ W3[2].view(I, H) + self.db[e]
                    y.index_add_(0, tok, (out * sc[tok, pos, None]).to(x.dtype))
                    rt.stats["gpu_experts"] += 1
        return y.view(B, T, H), sc


def load_skeleton_gptoss(model_dir, rt, device="cuda"):
    """Meta-init HF gpt-oss, swap every MLP for a paged block, load all non-expert tensors and the expert biases."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from .model import nbytes_of
    cfg = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    blocks = []
    for i, layer in enumerate(model.model.layers):
        blk = PagedGptOssBlock(layer.mlp.router, i, rt, cfg, device)
        layer.mlp = blk
        blocks.append(blk)
    get, has, idx = _reader(model_dir)
    from safetensors import safe_open
    wanted = {k: f for k, f in idx.items() if ".mlp.experts." not in k}
    sd, t0 = {}, time.time()
    for f in sorted(set(wanted.values())):
        with safe_open(os.path.join(model_dir, f), framework="pt", device=device) as h:
            for k in h.keys():
                if k in wanted:
                    sd[k] = h.get_tensor(k)
    res = model.load_state_dict(sd, strict=False, assign=True)
    bad = [k for k in res.missing_keys if "rotary" not in k]
    assert not bad, f"missing keys: {bad[:5]}"
    for l, blk in enumerate(blocks):
        p = f"model.layers.{l}.mlp.experts."
        blk.gub.copy_(get(p + "gate_up_proj_bias").to(device))
        blk.db.copy_(get(p + "down_proj_bias").to(device))
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg, device=device)
    model.eval()
    print(f"[model] gpt-oss non-expert weights on {device}: {nbytes_of(model)/2**30:.2f} GiB ({time.time()-t0:.0f}s)", flush=True)
    return model, cfg
