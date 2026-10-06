import torch
from transformers import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
from wisp.gptoss import mxfp4_to_bf16, expert_w3, PagedGptOssBlock
from wisp.quant import dequant, quantize


def test_mxfp4_matches_hf():
    torch.manual_seed(0)
    blocks = torch.randint(0, 256, (3, 8, 5, 16), dtype=torch.uint8)
    scales = torch.randint(120, 130, (3, 8, 5), dtype=torch.uint8)
    ref = convert_moe_packed_tensors(blocks, scales)                          # HF returns [E, K, out]
    got = mxfp4_to_bf16(blocks, scales).transpose(1, 2)
    assert torch.equal(ref, got)


class FakeRT:
    def __init__(self): self.stats = dict(layer_calls=0, gpu_experts=0); self.recorder = None; self.chunk = 4; self.cache = None


class FakeCache:
    """CPU stand-in for ExpertCache: every expert 'resident', slot == key; real quant/dequant round trip."""
    def __init__(self, rows, n, bits, group): self.rows, self.n, self.bits, self.g = rows, n, bits, group
    def ensure(self, keys): return list(keys)
    def weights_batched(self, idx): return dequant(self.rows[idx], self.n, self.bits, self.g)
    def weights(self, slots): return list(self.weights_batched(torch.as_tensor(slots)).unbind(0))


def _check(N, bits):
    torch.manual_seed(1)
    cfg = GptOssConfig(hidden_size=64, intermediate_size=64, num_hidden_layers=1, num_local_experts=8,
                       num_experts_per_tok=2, num_attention_heads=2, num_key_value_heads=1, head_dim=32, vocab_size=50)
    ref = GptOssMLP(cfg).to(torch.bfloat16)
    with torch.no_grad():
        for p in ref.parameters(): p.normal_(0, 0.3)
    E, H, I = 8, 64, 64
    rows = torch.stack([quantize(expert_w3(ref.experts.gate_up_proj[e].data, ref.experts.down_proj[e].data), bits, 64)
                        for e in range(E)])
    rt = FakeRT(); rt.cache = FakeCache(rows, H * I, bits, 64)
    blk = PagedGptOssBlock(ref.router, 0, rt, cfg, device="cpu")
    blk.gub.copy_(ref.experts.gate_up_proj_bias.data); blk.db.copy_(ref.experts.down_proj_bias.data)
    x = torch.randn(1, N, H).to(torch.bfloat16)
    with torch.no_grad():
        want, _ = ref(x)
        got, _ = blk(x)
    return (got.float() - want.float()).abs().max().item(), want.float().abs().max().item()


def test_block_exact_with_16bit():
    for N in (1, 5):
        d, m = _check(N, 16)
        assert d < 0.05 * m, (N, d, m)


def test_block_close_with_4bit():
    for N in (1, 5):
        d, m = _check(N, 4)
        assert d < 0.35 * m, (N, d, m)
