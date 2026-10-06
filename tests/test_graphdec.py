import os

os.environ["TRITON_INTERPRET"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from wisp.graphdec import GraphDecoder


class RefMoE(nn.Module):
    """Tiny stand-in for PagedMoEBlock: same attributes the decoder reads, dense reference experts."""

    def __init__(self, H, I, E, k):
        super().__init__()
        self.gate = nn.Linear(H, E, bias=False)
        self.w1, self.w3, self.w2 = (
            nn.Parameter(torch.randn(E, I, H) * 0.1),
            nn.Parameter(torch.randn(E, I, H) * 0.1),
            nn.Parameter(torch.randn(E, H, I) * 0.1),
        )
        self.top_k, self.norm_topk, self.E, self.I, self.next_gate = k, True, E, I, None

    def _mix(self, x, sel, rw):
        y = torch.zeros_like(x)
        for e, w in zip(sel, rw, strict=True):
            y = y + w * (F.silu(x @ self.w1[e].T) * (x @ self.w3[e].T)) @ self.w2[e].T
        return y

    def forward(self, h):
        x = h.reshape(-1, h.shape[-1])
        rw = F.softmax(self.gate(x), dim=1, dtype=torch.float32)
        rw, sel = torch.topk(rw, self.top_k, -1)
        rw = (rw / rw.sum(-1, keepdim=True)).to(x.dtype)
        return torch.stack([self._mix(x[i : i + 1], sel[i], rw[i])[0] for i in range(x.shape[0])]).view_as(h)


class RefDecoder(GraphDecoder):
    def _prepare(self, i, row):
        k = self.k
        self.sel, self.w = [int(v) for v in row[:k]], row[k : 2 * k]

    def _moe(self, i):
        m = self.layers[i].mlp
        return m._mix(self.xn.view(1, -1), self.sel, [torch.tensor(w).to(self.xn.dtype) for w in self.w]).view(1, 1, -1)


def test_decoder_matches_hf():
    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        max_position_embeddings=256,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )
    model = Qwen3MoeForCausalLM(cfg).eval()
    for l in model.model.layers:
        l.mlp = RefMoE(64, 32, 8, 2)
    ids = torch.randint(0, 100, (1, 7))
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        pk = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        dec = RefDecoder(model, None, tmax=64, use_graph=False)
        dec.load_kv(pk, 7)
        t = nxt.clone()
        for _ in range(6):
            ref = model(input_ids=t[:, None], past_key_values=pk, use_cache=True)
            pk = ref.past_key_values
            got = dec.next(t)
            assert torch.allclose(got, ref.logits[:, -1], atol=2e-3, rtol=2e-3), (got - ref.logits[:, -1]).abs().max()
            t = ref.logits[:, -1].argmax(-1)


def _tiny():
    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        max_position_embeddings=512,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )
    model = Qwen3MoeForCausalLM(cfg).eval()
    for l in model.model.layers:
        l.mlp = RefMoE(64, 32, 8, 2)
    return model


def test_chunked_prefill_reuse_and_resize():
    """Static-buffer prefill (in chunks, reusing a prefix) + growing the buffer mid-conversation == plain HF."""
    model = _tiny()
    ids = torch.randint(0, 100, (1, 40))
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=True)
        dec = RefDecoder(model, None, tmax=64, use_graph=False)
        lg = dec.prefill(ids[0, :25], 0, chunk=7)  # turn 1
        lg = dec.prefill(ids[0, 25:], 25, chunk=6)  # turn 2 reuses the first 25 positions
        assert torch.allclose(lg, ref.logits[:, -1], atol=2e-3, rtol=2e-3)
        dec.resize(200)  # grow capacity, KV must survive
        pk, t = ref.past_key_values, ref.logits[:, -1].argmax(-1)
        for _ in range(4):
            r = model(input_ids=t[:, None], past_key_values=pk, use_cache=True)
            pk = r.past_key_values
            assert torch.allclose(dec.next(t), r.logits[:, -1], atol=2e-3, rtol=2e-3)
            t = r.logits[:, -1].argmax(-1)
        # roll back to a shorter shared prefix (edited history) and re-prefill from there
        lg = dec.prefill(ids[0, 10:40], 10, chunk=8)
        assert torch.allclose(lg, ref.logits[:, -1], atol=2e-3, rtol=2e-3)


def test_cachecore_resized_keeps_hottest():
    from wisp.cache_core import CacheCore

    c = CacheCore(6, "lru")
    for keys in ([1, 2], [3, 4], [5, 6], [1, 3]):
        c.ensure(keys)
    nc, loads = c.resized(3)
    assert sorted(k for k, _ in loads) == [1, 3, 6] or sorted(k for k, _ in loads) == [1, 3, 5]
    assert len(nc.free) == 0 and all(nc.lookup(k) == s for k, s in loads)
    nc2, loads2 = nc.resized(8)
    assert len(loads2) == 3 and len(nc2.free) == 5
    slots, new = nc2.ensure([1, 9])
    assert len(new) == 1 and new[0][0] == 9
