import torch
from transformers.models.qwen3_moe import modeling_qwen3_moe as m
from wisp.model import patch_fast_norm


def test_fast_norm_matches_reference():
    torch.manual_seed(0)
    for H, dt in ((2048, torch.bfloat16), (128, torch.bfloat16), (256, torch.float32)):
        n = m.Qwen3MoeRMSNorm(H, 1e-6).to(dt)
        n.weight.data = (torch.randn(H) * 0.3 + 1).to(dt)
        x = (torch.randn(3, H) * 2).to(dt)
        orig = m.Qwen3MoeRMSNorm.forward
        ref = orig(n, x)
        patch_fast_norm()
        got = n(x)
        m.Qwen3MoeRMSNorm.forward = orig; m.Qwen3MoeRMSNorm._wisp_patched = False
        tol = 2e-2 if dt == torch.bfloat16 else 1e-5
        assert (got.float() - ref.float()).abs().max() < tol * ref.float().abs().max().clamp_min(1)
        assert got.dtype == ref.dtype
