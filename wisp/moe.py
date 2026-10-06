"""Paged MoE block (drop-in for Qwen3MoeSparseMoeBlock) and the shared Runtime."""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class Runtime:
    """State shared by all paged blocks. Mutable so a grid runner can reconfigure without reloading the store."""

    def __init__(self, store, cache=None, miss_policy="transfer", cpu_threshold=4, prefetch=False,
                 chunk=8, cpu_threads=None, fused=False):
        self.fused = int(fused) or int(os.environ.get("WISP_FUSED", "0") or 0)                      # fused 4-bit dequant+GEMV Triton kernel for 1-token decode
        self.store, self.cache = store, cache
        self.miss_policy = miss_policy          # 'transfer' | 'hybrid' (CPU computes misses during decode)
        self.cpu_threshold = cpu_threshold      # tokens per forward that count as 'decode'
        self.prefetch = prefetch
        self.chunk = chunk
        self.recorder = None                    # list -> collects (layer, sel[N,k]) for trace collection
        self.reset_stats()
        if cpu_threads:
            torch.set_num_threads(cpu_threads)

    def reset_stats(self):
        self.stats = dict(layer_calls=0, cpu_experts=0, gpu_experts=0, cpu_time=0.0, admitted=0, prefetched=0)


def _mlp(x, W3, I, H):
    g = F.linear(x, W3[0].view(I, H))
    u = F.linear(x, W3[1].view(I, H))
    return F.linear(F.silu(g) * u, W3[2].view(H, I))


class PagedMoEBlock(nn.Module):
    def __init__(self, gate, layer_idx, rt, cfg):
        super().__init__()
        self.gate = gate
        self.layer_idx, self.rt = layer_idx, rt
        self.E, self.top_k = cfg.num_experts, cfg.num_experts_per_tok
        self.norm_topk = cfg.norm_topk_prob
        self.H, self.I = cfg.hidden_size, cfg.moe_intermediate_size
        self.next_gate = None   # prefetch predictor (next layer's gate); set via set_next_gate()

    def set_next_gate(self, gate):
        # bypass nn.Module registration: it is another block's parameter, not ours (would break state_dict loading)
        object.__setattr__(self, "next_gate", gate)

    def forward(self, hidden_states):
        rt, cache = self.rt, self.rt.cache
        B, T, H = hidden_states.shape
        x = hidden_states.reshape(-1, H)
        N, E = x.shape[0], self.E
        logits = self.gate(x)
        rw = F.softmax(logits, dim=1, dtype=torch.float32)
        rw, sel = torch.topk(rw, self.top_k, dim=-1)
        if self.norm_topk:
            rw = rw / rw.sum(dim=-1, keepdim=True)
        rw = rw.to(x.dtype)

        decode = N <= rt.cpu_threshold
        use_cpu = decode and rt.miss_policy == "hybrid"
        do_pf = rt.prefetch and decode and self.next_gate is not None
        pred = torch.topk(self.next_gate(x), self.top_k, dim=-1).indices if do_pf else None

        if (rt.fused == 2 and N == 1 and not use_cpu and cache.store.bits in (3, 4) and not cache.decode_on_load):
            return self._forward_fast(x, rw, sel, pred, logits, B, T, H)

        # ---- the single GPU->CPU sync of this layer: routing decisions -----
        parts = [sel] + ([pred] if do_pf else [])
        both = torch.cat(parts, 1).cpu()
        sel_cpu = both[:, :self.top_k]
        if rt.recorder is not None:
            rt.recorder.append((self.layer_idx, sel_cpu.numpy().copy()))
        uniq = torch.unique(sel_cpu).tolist()
        base = self.layer_idx * E
        keys = [base + e for e in uniq]
        if use_cpu:
            x_cpu, rw_cpu = x.cpu(), rw.cpu()

        st = rt.stats
        st["layer_calls"] += 1
        out = torch.zeros_like(x)
        comb = torch.zeros(N, E, dtype=x.dtype, device=x.device).scatter_(1, sel, rw) if decode else None

        # ---- which experts run on the GPU now ------------------------------
        if use_cpu:
            hit_keys, hit_slots, miss_keys = cache.split(keys)
            gpu_plan = [(hit_keys, hit_slots)]
        else:
            miss_keys, hit_slots = [], []
            gpu_plan = None

        def run(chunk_keys, slots):
            if decode:      # few tokens: all experts of the chunk in 3 batched GEMMs (~10 launches instead of ~40)
                k = len(chunk_keys)
                meta = torch.tensor(list(slots) + [kk - base for kk in chunk_keys], device=x.device)
                if rt.fused == 1 and N == 1 and cache.store.bits == 4 and not cache.decode_on_load:
                    from .kernels import moe_decode_q4
                    cols = meta[k:]
                    out.add_(moe_decode_q4(cache.gpu, meta[:k], x[0], comb[0, cols], cache.store.n, H, self.I,
                                           cache.store.group).unsqueeze(0))
                    st["gpu_experts"] += k
                    return
                Wb = cache.weights_batched(meta[:k])
                cols = meta[k:]
                xg = x.unsqueeze(0).expand(k, N, H)
                g = torch.bmm(xg, Wb[:, 0].view(k, self.I, H).transpose(1, 2))
                u = torch.bmm(xg, Wb[:, 1].view(k, self.I, H).transpose(1, 2))
                y = torch.bmm(F.silu(g) * u, Wb[:, 2].view(k, H, self.I).transpose(1, 2))   # [k, N, H]
                out.add_((y * comb[:, cols].t().unsqueeze(-1)).sum(0))
                st["gpu_experts"] += k
                return
            Ws = cache.weights(slots)
            for kk, W3 in zip(chunk_keys, Ws):
                e = kk - base
                tok, pos = torch.where(sel == e)
                y = _mlp(x[tok], W3, self.I, H) * rw[tok, pos].unsqueeze(-1)
                out.index_add_(0, tok, y)
                st["gpu_experts"] += 1

        used_slots = []
        if gpu_plan is not None:
            for ck, cs in gpu_plan:
                if ck:
                    cache.sync_compute()
                    run(ck, cs)
                used_slots += list(cs)
        else:
            for i in range(0, len(keys), rt.chunk):
                ck = keys[i:i + rt.chunk]
                cs = cache.ensure(ck)
                run(ck, cs)
                used_slots = list(cs)   # only the newest chunk can be evicted-protected; older chunks are queued already

        # ---- CPU side of the hybrid policy (overlaps queued GPU work) -----
        if use_cpu and miss_keys:
            st["admitted"] += cache.admit_async(miss_keys, used_slots)
            t0 = time.perf_counter()
            ycpu = torch.zeros_like(x_cpu)
            comb_cpu = torch.zeros(N, E, dtype=x_cpu.dtype).scatter_(1, sel_cpu, rw_cpu)
            for k in miss_keys:
                e = k - base
                ycpu += _mlp(x_cpu, cache.cpu_weights(k), self.I, H) * comb_cpu[:, e:e + 1]
                st["cpu_experts"] += 1
            out.add_(ycpu.to(out.device))
            st["cpu_time"] += time.perf_counter() - t0

        if do_pf:
            nxt = both[:, self.top_k:]
            nb = (self.layer_idx + 1) * E
            pk = [nb + e for e in torch.unique(nxt).tolist()]
            st["prefetched"] += cache.prefetch(pk, used_slots)

        return out.view(B, T, H), logits


    def _forward_fast(self, x, rw, sel, pred, logits, B, T, H):
        """1-token decode, 4-bit: one GPU->CPU sync, one packed host->GPU routing copy, two kernel launches."""
        from .kernels import moe_decode_q4_fast, moe_decode_q3_fast
        import numpy as np
        rt, cache, k = self.rt, self.rt.cache, self.top_k
        parts = [sel.float(), rw.float()] + ([pred.float()] if pred is not None else [])
        row = torch.cat(parts, 1).cpu()[0].tolist()          # the single sync of this layer
        ex = [int(v) for v in row[:k]]
        base = self.layer_idx * self.E
        if rt.recorder is not None:
            rt.recorder.append((self.layer_idx, np.array([ex])))
        st = rt.stats
        st["layer_calls"] += 1
        slots = cache.ensure([base + e for e in ex])
        mm = getattr(rt, "_meta", None)
        if mm is None or mm[0].numel() != 2 * k:
            host = torch.empty(2 * k, dtype=torch.float32, pin_memory=True)
            mm = rt._meta = (host, host.numpy(), torch.empty(2 * k, dtype=torch.float32, device=x.device))
        host, hnp, dev = mm
        hnp[:k] = slots                                       # same-stream order makes reusing this buffer safe:
        hnp[k:] = row[k:2 * k]                                # last layer's copy finished before this layer's sync
        dev.copy_(host, non_blocking=True)
        fn = moe_decode_q4_fast if cache.store.bits == 4 else moe_decode_q3_fast
        y = fn(cache.gpu, dev, x[0], k, cache.store.n, H, self.I, cache.store.group)
        st["gpu_experts"] += k
        if pred is not None:
            nb = (self.layer_idx + 1) * self.E
            st["prefetched"] += cache.prefetch([nb + int(e) for e in set(row[2 * k:])], list(slots))
        return y.view(B, T, H), logits


def attach(model, rt):
    """Replace every sparse-MoE block of a HF Qwen3-MoE model by a PagedMoEBlock (keeps the gate)."""
    cfg = model.config
    blocks = []
    for i, layer in enumerate(model.model.layers):
        old = layer.mlp
        if not hasattr(old, "gate"):
            blocks.append(None)
            continue
        blk = PagedMoEBlock(old.gate, i, rt, cfg)
        layer.mlp = blk
        blocks.append(blk)
    for i, b in enumerate(blocks[:-1]):
        if b is not None:
            b.set_next_gate(blocks[i + 1].gate if blocks[i + 1] is not None else None)
    return model
