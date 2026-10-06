import pytest
import torch

from wisp.quant import _pack, _unpack, dequant, expert_nbytes, quantize

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("bits", [8, 4, 3, 2])
def test_pack_roundtrip(bits):
    q = torch.randint(0, 2**bits, (4096,), dtype=torch.uint8, device=DEV)
    assert torch.equal(_unpack(_pack(q, bits)[None], bits)[0].to(torch.uint8), q)


@pytest.mark.parametrize("bits", [16, 8, 4, 3, 2])
def test_sizes_and_error(bits):
    n, group = 2048, 32
    w = torch.randn(3, n, device=DEV) * 0.02
    buf = quantize(w, bits, group)
    assert buf.numel() == expert_nbytes(n, bits, group)
    r = dequant(buf[None], n, bits, group)[0].float()
    rel = ((r - w).norm() / w.norm()).item()
    bound = {16: 0.01, 8: 0.01, 4: 0.12, 3: 0.25, 2: 0.6}[bits]
    assert rel < bound, (bits, rel)


def test_error_monotonic():
    n, group = 4096, 32
    w = torch.randn(3, n, device=DEV)
    errs = []
    for b in (8, 4, 3, 2):
        r = dequant(quantize(w, b, group)[None], n, b, group)[0].float()
        errs.append(((r - w).norm() / w.norm()).item())
    assert errs == sorted(errs)


def test_compression_ratio():
    n = 1572864
    assert expert_nbytes(n, 16, 128) == 9437184  # 9.0 MiB
    assert abs(expert_nbytes(n, 4, 128) / expert_nbytes(n, 16, 128) - 0.265) < 0.01
