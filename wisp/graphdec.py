"""CUDA-graph decode for 1-token steps (B=1).

Why: after the fused kernels, decode is limited by CPU kernel-launch overhead (~1.1 ms/layer, GPU idle in between).
A layer needs exactly one host decision (which experts to page in), so a token is cut at those 48 sync points and
everything BETWEEN them is recorded once as a CUDA graph and replayed with one launch:

    graph[0]   : embed -> layer0 attention -> router(0)                         -> row
    (host)       read row, cache.ensure(experts), write slots/weights into meta
    graph[i]   : MoE(i-1) kernels -> residual -> layer i attention -> router(i) -> row      (i = 1..L-1)
    graph[L]   : MoE(L-1) kernels -> residual -> final norm -> lm_head          -> logits

Everything a graph reads is a static buffer (static KV cache [1,Hk,Tmax,D], position/token in `ctl`, routing in `meta`),
so nothing in a graph depends on the current length: attention runs over all Tmax slots with a mask (<= pos).
Prefill still uses the normal HF path; its KV is copied into the static buffers with load_kv().

Usage:   dec = GraphDecoder(model, rt, tmax); logits = prefill with HF; dec.load_kv(pk, T);
         then  logits = dec.next(token_tensor)   # repeated
Rebuilds itself if the expert cache is re-created (configure()).  Set use_graph=False to run the same code eagerly.
"""
import torch
import torch.nn.functional as F


def _get_kv(pk, i):
    if hasattr(pk, "layers"):
        l = pk.layers[i]
        return l.keys, l.values
    if hasattr(pk, "key_cache"):
        return pk.key_cache[i], pk.value_cache[i]
    kv = pk[i]
    return kv[0], kv[1]


class GraphDecoder:
    def __init__(self, model, rt=None, tmax=2048, use_graph=True):
        self.model, self.rt = model, rt
        cfg = model.config
        self.L, self.H = cfg.num_hidden_layers, cfg.hidden_size
        self.Hq, self.Hk = cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or self.H // self.Hq
        self.G = self.Hq // self.Hk
        self.tmax = (tmax + 63) // 64 * 64
        self.layers = list(model.model.layers)
        assert all(hasattr(l.mlp, "top_k") for l in self.layers), "GraphDecoder needs an all-MoE model"
        self.k = self.layers[0].mlp.top_k
        self.E = self.layers[0].mlp.E if hasattr(self.layers[0].mlp, "E") else cfg.num_experts
        w = model.model.embed_tokens.weight
        self.dev, dt = w.device, w.dtype
        self.cuda = self.dev.type == "cuda"
        self.use_graph = use_graph and self.cuda
        self.dt = dt
        d = dict(device=self.dev, dtype=dt)
        self.kc = self.vc = None
        self.ctl = torch.zeros(2, dtype=torch.long, device=self.dev)       # [token, position]
        self.pos = 0
        self._alloc(self.tmax)
        self.hm = torch.zeros(1, 1, self.H, **d)                           # residual stream after attention
        self.xn = torch.zeros(1, 1, self.H, **d)                           # MoE input
        self.prefetch = bool(rt is not None and getattr(rt, "prefetch", False))
        self.row = torch.zeros(1, (3 if self.prefetch else 2) * self.k, dtype=torch.float32, device=self.dev)
        self.logits = torch.zeros(1, cfg.vocab_size, **d)
        self.meta_dev = torch.zeros(2 * self.k, dtype=torch.float32, device=self.dev)
        self.meta_host = torch.zeros(2 * self.k, dtype=torch.float32, pin_memory=self.cuda)
        self.graphs, self._key, self.pos = None, None, 0

    def _alloc(self, tmax):
        """(Re)allocate the static KV buffers for `tmax` positions, keeping the first `pos` entries."""
        d = dict(device=self.dev, dtype=self.dt)
        old_k, old_v, keep = self.kc, self.vc, min(self.pos, tmax)
        self.kc, self.vc = [], []
        for i in range(self.L):
            k = torch.zeros(1, self.Hk, tmax, self.D, **d)
            v = torch.zeros(1, self.Hk, tmax, self.D, **d)
            if old_k is not None and keep:
                k[:, :, :keep].copy_(old_k[i][:, :, :keep]); v[:, :, :keep].copy_(old_v[i][:, :, :keep])
                old_k[i] = old_v[i] = None                     # free layer by layer: peak = old + one layer
            self.kc.append(k); self.vc.append(v)
        self.tmax, self.pos = tmax, keep
        self.ar = torch.arange(tmax, device=self.dev)
        pos = torch.arange(tmax, device=self.dev)[None]
        cos, sin = self.model.model.rotary_emb(torch.zeros(1, 1, self.H, **d), pos)
        self.cos, self.sin = cos[0].contiguous(), sin[0].contiguous()      # [tmax, D]

    def resize(self, tmax):
        """Change the context capacity (graphs are re-captured on the next step)."""
        tmax = (tmax + 63) // 64 * 64
        if tmax == self.tmax:
            return
        self.graphs = None
        if self.cuda:
            torch.cuda.synchronize()
        self._alloc(tmax)
        if self.cuda:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def prefill(self, ids, start=None, chunk=1024):
        """Run tokens `ids` (LongTensor [T]) at positions start..start+T-1, writing K/V straight into the static
        buffers (no second KV copy), in chunks. KV before `start` is reused. Returns logits [1, V] of the last token."""
        from transformers.models.qwen3_moe import modeling_qwen3_moe as m
        start = self.pos if start is None else start
        ids = ids.reshape(-1).to(self.dev)
        assert start + ids.numel() <= self.tmax, f"context {start + ids.numel()} > capacity {self.tmax}"
        try:
            from torch.nn.attention.bias import causal_lower_right
        except Exception:                                    # pragma: no cover
            causal_lower_right = None
        Hq, Hk, D, G = self.Hq, self.Hk, self.D, self.G
        logits = None
        for c0 in range(0, ids.numel(), chunk):
            part = ids[c0:c0 + chunk]
            T, s0 = part.numel(), start + c0
            Lk = s0 + T
            h = self.model.model.embed_tokens(part.view(1, T))
            cos, sin = self.cos[s0:Lk][None], self.sin[s0:Lk][None]
            if causal_lower_right is not None:
                bias = causal_lower_right(T, Lk)
            else:
                bias = (torch.arange(Lk, device=self.dev)[None] <= torch.arange(s0, Lk, device=self.dev)[:, None])
            for i, L in enumerate(self.layers):
                at = L.self_attn
                xn = L.input_layernorm(h)
                q = at.q_norm(at.q_proj(xn).view(1, T, Hq, D)).transpose(1, 2)
                k = at.k_norm(at.k_proj(xn).view(1, T, Hk, D)).transpose(1, 2)
                v = at.v_proj(xn).view(1, T, Hk, D).transpose(1, 2)
                q, k = m.apply_rotary_pos_emb(q, k, cos, sin)
                self.kc[i][:, :, s0:Lk].copy_(k)
                self.vc[i][:, :, s0:Lk].copy_(v)
                outs = []
                for g in range(Hk):                          # one KV head at a time: no repeat_kv copy of the whole cache
                    kk = self.kc[i][:, g:g + 1, :Lk].expand(1, G, Lk, D).contiguous()
                    vv = self.vc[i][:, g:g + 1, :Lk].expand(1, G, Lk, D).contiguous()
                    outs.append(F.scaled_dot_product_attention(q[:, g * G:(g + 1) * G].contiguous(), kk, vv,
                                                               attn_mask=bias, scale=getattr(at, "scaling", D ** -0.5)))
                o = torch.cat(outs, 1).transpose(1, 2).reshape(1, T, Hq * D)
                h = h + at.o_proj(o)
                y = L.mlp(L.post_attention_layernorm(h))
                h = h + (y[0] if isinstance(y, tuple) else y)
            logits = self.model.lm_head(self.model.model.norm(h[:, -1:])).view(1, -1)
        self.pos = start + ids.numel()
        return logits

    # ---------------------------------------------------------------- graph body
    def _attn_router(self, i, h):
        L = self.layers[i]
        at, blk = L.self_attn, L.mlp
        Hq, Hk, D, H, G, k = self.Hq, self.Hk, self.D, self.H, self.G, self.k
        xn = L.input_layernorm(h)
        q = at.q_norm(at.q_proj(xn).view(1, 1, Hq, D)).transpose(1, 2)
        kk = at.k_norm(at.k_proj(xn).view(1, 1, Hk, D)).transpose(1, 2)
        v = at.v_proj(xn).view(1, 1, Hk, D).transpose(1, 2).contiguous()
        pos = self.ctl[1:2]
        cos = self.cos.index_select(0, pos).view(1, 1, D)
        sin = self.sin.index_select(0, pos).view(1, 1, D)
        from .kernels import rope_decode
        q, kk = rope_decode(q, kk, cos, sin)
        self.kc[i].index_copy_(2, pos, kk)
        self.vc[i].index_copy_(2, pos, v)
        mask = (self.ar <= pos).view(1, 1, 1, -1)
        o = F.scaled_dot_product_attention(q.reshape(1, Hk, G, D), self.kc[i], self.vc[i], attn_mask=mask,
                                           scale=getattr(at, "scaling", D ** -0.5))
        hm = h + at.o_proj(o.reshape(1, 1, Hq * D))
        xn2 = L.post_attention_layernorm(hm)
        rw = F.softmax(blk.gate(xn2.view(1, H)), dim=1, dtype=torch.float32)
        rw, sel = torch.topk(rw, k, dim=-1)
        if blk.norm_topk:
            rw = rw / rw.sum(dim=-1, keepdim=True)
        rw = rw.to(xn2.dtype)                                    # same rounding as the eager path
        parts = [sel.float(), rw.float()]
        if self.prefetch:
            ng = getattr(blk, "next_gate", None)
            parts.append(torch.topk(ng(xn2.view(1, H)), k, dim=-1).indices.float() if ng is not None
                         else torch.zeros(1, k, device=self.dev))
        self.hm.copy_(hm)
        self.xn.copy_(xn2)
        self.row.copy_(torch.cat(parts, 1))

    def _moe(self, i):
        """MoE of layer i from the static buffers -> [1,1,H]. Overridable for tests."""
        from .kernels import moe_decode_q4_fast, moe_decode_q3_fast
        c = self.rt.cache
        fn = moe_decode_q4_fast if c.store.bits == 4 else moe_decode_q3_fast
        y = fn(c.gpu, self.meta_dev, self.xn.view(self.H), self.k, c.store.n, self.H, self.layers[i].mlp.I,
               c.store.group)
        return y.view(1, 1, self.H)

    def _seg(self, j):
        if j == 0:
            h = self.model.model.embed_tokens(self.ctl[0:1].view(1, 1))
        else:
            h = self.hm + self._moe(j - 1)
        if j < self.L:
            self._attn_router(j, h)
        else:
            self.logits.copy_(self.model.lm_head(self.model.model.norm(h)).view(1, -1))

    # ---------------------------------------------------------------- host side of one layer
    def _prepare(self, i, row):
        rt, cache, k = self.rt, self.rt.cache, self.k
        ex = [int(v) for v in row[:k]]
        base = i * self.E
        if rt.recorder is not None:
            import numpy as np
            rt.recorder.append((i, np.array([ex])))
        st = rt.stats
        st["layer_calls"] += 1
        slots = cache.ensure([base + e for e in ex])
        hn = self.meta_host.numpy()
        hn[:k] = slots
        hn[k:] = row[k:2 * k]
        self.meta_dev.copy_(self.meta_host, non_blocking=True)  # ordered before the next replay on this stream
        st["gpu_experts"] += k
        if self.prefetch and i < self.L - 1:
            nb = (i + 1) * self.E
            st["prefetched"] += cache.prefetch([nb + int(e) for e in set(row[2 * k:3 * k])], list(slots))

    # ---------------------------------------------------------------- capture / run
    def _cache_key(self):
        c = self.rt.cache
        return (c.gpu.data_ptr(), c.store.bits, self.prefetch)

    @torch.no_grad()
    def capture(self):
        rt, k = self.rt, self.k
        self.prefetch = bool(getattr(rt, "prefetch", False))
        if self.row.shape[1] != (3 if self.prefetch else 2) * k:
            self.row = torch.zeros(1, (3 if self.prefetch else 2) * k, dtype=torch.float32, device=self.dev)
        self.graphs = None
        # the warm-up below really executes the segments, which writes K/V at `ctl[1]`: aim it at the last slot and put
        # whatever was there back, so capturing in the middle of a conversation cannot corrupt the loaded context
        wp = self.tmax - 1
        saved = [(self.kc[i][:, :, wp].clone(), self.vc[i][:, :, wp].clone()) for i in range(self.L)]
        self.ctl[0] = 0
        self.ctl[1] = wp
        hn = self.meta_host.numpy()
        hn[:k] = range(k)
        hn[k:] = 1.0 / k
        self.meta_dev.copy_(self.meta_host)
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):                       # warm-up: compiles the Triton kernels, inits cuBLAS
            for _ in range(2):
                for j in range(self.L + 1):
                    self._seg(j)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        for i, (kk, vv) in enumerate(saved):
            self.kc[i][:, :, wp].copy_(kk)
            self.vc[i][:, :, wp].copy_(vv)
        pool = torch.cuda.graph_pool_handle()
        graphs = []
        for j in range(self.L + 1):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self._seg(j)
            graphs.append(g)
        self.graphs, self._key = graphs, self._cache_key()
        torch.cuda.synchronize()

    def _run(self, j):
        if self.graphs is not None:
            self.graphs[j].replay()
        else:
            self._seg(j)

    @torch.no_grad()
    def load_kv(self, pk, T):
        assert T < self.tmax, f"context {T} >= tmax {self.tmax}"
        for i in range(self.L):
            k, v = _get_kv(pk, i)
            self.kc[i][:, :, :T].copy_(k)
            self.vc[i][:, :, :T].copy_(v)
        self.pos = T

    @torch.no_grad()
    def next(self, token):
        """token: LongTensor [1] (or [1,1]) on the device -> logits [1, V] for the following position."""
        assert self.pos < self.tmax, "context full"
        if self.use_graph and (self.graphs is None or self._key != self._cache_key()):
            self.capture()
        self.ctl[0:1].copy_(token.reshape(1))
        self.ctl[1].fill_(self.pos)
        self._run(0)
        for i in range(self.L):
            self._prepare(i, self.row.cpu()[0].tolist())     # the one sync per layer
            self._run(i + 1)
        self.pos += 1
        return self.logits
