import os

os.environ["TRITON_INTERPRET"] = "1"
import torch

from wisp.kernels import moe_decode_q4
from wisp.quant import dequant, quantize


def ref(pool, slots, x, w, n, H, I):
    W = dequant(pool[slots], n, 4, 128).float()  # [k,3,n]
    g = W[:, 0].view(-1, I, H)
    u = W[:, 1].view(-1, I, H)
    d = W[:, 2].view(-1, H, I)
    xf = x.float()
    a = torch.nn.functional.silu(g @ xf) * (u @ xf)  # [k,I]
    y = torch.einsum("khi,ki->kh", d, a)
    return (y * w.float()[:, None]).sum(0)


def test_fused_q4_matches_reference():
    torch.manual_seed(0)
    H, I, E, k = 256, 128, 6, 3
    n = H * I
    w3 = torch.randn(E, 3, n) * 0.05
    pool = torch.stack([quantize(w3[e].to(torch.bfloat16), 4, 128) for e in range(E)])
    slots = torch.tensor([4, 1, 2])
    x = torch.randn(H).to(torch.bfloat16)
    w = torch.softmax(torch.randn(k), 0)
    got = moe_decode_q4(pool, slots, x, w, n, H, I).float()
    exp = ref(pool, slots, x, w, n, H, I)
    err = (got - exp).abs().max() / exp.abs().max()
    assert err < 2e-2, err


def test_fast_variant_matches_reference():
    from wisp.kernels import moe_decode_q4_fast

    torch.manual_seed(1)
    H, I, E, k = 256, 128, 6, 3
    n = H * I
    w3 = torch.randn(E, 3, n) * 0.05
    pool = torch.stack([quantize(w3[e].to(torch.bfloat16), 4, 128) for e in range(E)])
    slots = torch.tensor([5, 0, 3])
    x = torch.randn(H).to(torch.bfloat16)
    w = torch.softmax(torch.randn(k), 0)
    meta = torch.cat([slots.float(), w.float()])
    got = moe_decode_q4_fast(pool, meta, x, k, n, H, I).float()
    exp = ref(pool, slots, x, w, n, H, I)
    err = (got - exp).abs().max() / exp.abs().max()
    assert err < 2e-2, err


def test_q3_matches_reference():
    from wisp.kernels import moe_decode_q3_fast

    torch.manual_seed(2)
    H, I, E, k = 256, 128, 6, 3
    n = H * I
    w3 = torch.randn(E, 3, n) * 0.05
    pool = torch.stack([quantize(w3[e].to(torch.bfloat16), 3, 128) for e in range(E)])
    slots = torch.tensor([2, 5, 1])
    x = torch.randn(H).to(torch.bfloat16)
    w = torch.softmax(torch.randn(k), 0)
    meta = torch.cat([slots.float(), w.float()])
    got = moe_decode_q3_fast(pool, meta, x, k, n, H, I).float()
    W = dequant(pool[slots], n, 3, 128).float()
    g = W[:, 0].view(-1, I, H)
    u = W[:, 1].view(-1, I, H)
    d = W[:, 2].view(-1, H, I)
    a = torch.nn.functional.silu(g @ x.float()) * (u @ x.float())
    exp = (torch.einsum("khi,ki->kh", d, a) * w[:, None]).sum(0)
    err = (got - exp).abs().max() / exp.abs().max()
    assert err < 2e-2, err
