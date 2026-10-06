import os
os.environ["TRITON_INTERPRET"] = "1"
import torch
import torch.nn.functional as F
from transformers.models.qwen3_moe import modeling_qwen3_moe as m
from transformers.integrations import sdpa_attention as S
from wisp.kernels import rope_decode


def test_rope_decode_matches_hf():
    torch.manual_seed(0)
    D, Hq, Hk = 128, 32, 4
    q = torch.randn(1, 1, Hq, D).to(torch.bfloat16).transpose(1, 2)     # as HF produces it: [1,Hq,1,D] non-contiguous view
    k = torch.randn(1, 1, Hk, D).to(torch.bfloat16).transpose(1, 2)
    cos = torch.randn(1, 1, D).to(torch.bfloat16); sin = torch.randn(1, 1, D).to(torch.bfloat16)
    qr, kr = m.apply_rotary_pos_emb(q, k, cos, sin)
    qo, ko = rope_decode(q, k, cos, sin)
    assert qo.shape == qr.shape and ko.shape == kr.shape
    assert (qo.float() - qr.float()).abs().max() < 3e-2 * qr.float().abs().max()
    assert (ko.float() - kr.float()).abs().max() < 3e-2 * kr.float().abs().max()


def test_grouped_decode_attention_matches_repeat_kv():
    torch.manual_seed(1)
    B, Hq, Hk, D, T = 1, 32, 4, 128, 37
    q = torch.randn(B, Hq, 1, D); k = torch.randn(B, Hk, T, D); v = torch.randn(B, Hk, T, D)
    ref = F.scaled_dot_product_attention(q, S.repeat_kv(k, Hq // Hk), S.repeat_kv(v, Hq // Hk), scale=D ** -0.5)
    o = F.scaled_dot_product_attention(q.reshape(B, Hk, Hq // Hk, D), k, v, scale=D ** -0.5)
    got = o.reshape(B, Hq, 1, D)
    assert torch.allclose(got, ref, atol=1e-5)
