"""Group-wise min-max (RTN) quantization of one expert (gate, up, down) into ONE flat uint8 buffer.

Layout per expert:
  bits == 16 : raw bf16                              -> 3*n*2 bytes
  bits <  16 : [ codes (packed) | scales fp16 | mins fp16 ]
An expert holds 3 matrices of n = I*H elements each (gate [I,H], up [I,H], down [H,I]) -> flat [3, n].
RTN is deliberately the simplest quantizer; better ones (GPTQ/AWQ/QuIP#) can be swapped in later.
"""

import os

import torch

SUPPORTED_BITS = (16, 8, 4, 3, 2)


def expert_nbytes(n: int, bits: int, group: int) -> int:
    if bits == 16:
        return 3 * n * 2
    assert bits in SUPPORTED_BITS, bits
    assert (3 * n) % group == 0 and (3 * n * bits) % 8 == 0, "3*n must be divisible by group and by 8/bits"
    return (3 * n * bits) // 8 + 4 * (3 * n // group)


def _pack(q: torch.Tensor, bits: int) -> torch.Tensor:
    """q: flat uint8 with values < 2**bits -> packed uint8."""
    if bits == 8:
        return q
    if bits == 4:
        c = q.view(-1, 2)
        return c[:, 0] | (c[:, 1] << 4)
    if bits == 2:
        c = q.view(-1, 4)
        return c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)
    if bits == 3:  # 8 codes -> 3 bytes (24 bits, little endian)
        c = q.view(-1, 8).to(torch.int32)
        v = torch.zeros(c.shape[0], dtype=torch.int32, device=q.device)
        for i in range(8):
            v |= c[:, i] << (3 * i)
        return torch.stack([v & 255, (v >> 8) & 255, (v >> 16) & 255], 1).to(torch.uint8).reshape(-1)
    raise ValueError(bits)


def _unpack(c: torch.Tensor, bits: int) -> torch.Tensor:
    """c: [B, nbytes] uint8 -> [B, m] integer codes."""
    B = c.shape[0]
    if bits == 8:
        return c
    if bits == 4:
        return torch.stack([c & 15, c >> 4], -1).reshape(B, -1)
    if bits == 2:
        return torch.stack([(c >> (2 * i)) & 3 for i in range(4)], -1).reshape(B, -1)
    if bits == 3:
        t = c.reshape(B, -1, 3).to(torch.int32)
        v = t[..., 0] | (t[..., 1] << 8) | (t[..., 2] << 16)
        return torch.stack([(v >> (3 * i)) & 7 for i in range(8)], -1).reshape(B, -1)
    raise ValueError(bits)


@torch.no_grad()
def quantize(w3: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    """w3: [3, n] (any float dtype, any device) -> flat uint8 buffer on the same device."""
    if bits == 16:
        return w3.to(torch.bfloat16).contiguous().view(torch.uint8).reshape(-1)
    g = w3.float().reshape(-1, group)
    levels = 2**bits - 1
    mn = g.min(1).values
    mx = g.max(1).values
    scale = ((mx - mn) / levels).clamp_min(1e-5)
    s16, m16 = scale.half(), mn.half()  # store what dequant will actually see
    q = ((g - m16.float()[:, None]) / s16.float()[:, None]).round().clamp_(0, levels).to(torch.uint8)
    packed = _pack(q.reshape(-1), bits)
    return torch.cat([packed, s16.view(torch.uint8).reshape(-1), m16.view(torch.uint8).reshape(-1)])


def dequant(buf: torch.Tensor, n: int, bits: int, group: int = 128) -> torch.Tensor:
    """buf: [B, nbytes] uint8 -> [B, 3, n] bfloat16 (same device as buf)."""
    B = buf.shape[0]
    if bits == 16:
        return buf.contiguous().view(torch.bfloat16).view(B, 3, n)
    ncode = (3 * n * bits) // 8
    ng = 3 * n // group
    sc = buf[:, ncode : ncode + 2 * ng].contiguous().view(torch.float16)
    mn = buf[:, ncode + 2 * ng : ncode + 4 * ng].contiguous().view(torch.float16)
    q = _unpack(buf[:, :ncode], bits)
    w = q.reshape(B, ng, group).to(torch.float32) * sc.float().unsqueeze(-1) + mn.float().unsqueeze(-1)
    return w.to(torch.bfloat16).reshape(B, 3, n)


_FAST = {}
_WARNED = [False]


def dequant_fast(buf, n, bits, group=128):
    """dequant() fused into one kernel by torch.compile (bit-identical, ~4-5x faster on GPU).
    Falls back to eager if compilation is unavailable. Disable with WISP_COMPILE=0."""
    if bits == 16 or not buf.is_cuda or os.environ.get("WISP_COMPILE", "1") == "0":
        return dequant(buf, n, bits, group)
    key = (n, bits, group)
    fn = _FAST.get(key)
    if fn is None:
        fn = torch.compile(lambda b: dequant(b, n, bits, group), dynamic=True)
        _FAST[key] = fn
    try:
        return fn(buf)
    except Exception as e:  # pragma: no cover
        if not _WARNED[0]:
            print(f"[quant] torch.compile failed, using eager dequant: {type(e).__name__}", flush=True)
            _WARNED[0] = True
        _FAST[key] = lambda b: dequant(b, n, bits, group)
        return dequant(buf, n, bits, group)
