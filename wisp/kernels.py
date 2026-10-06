"""Fused 4-bit dequant + GEMV kernels for MoE decode (1 token).

Reads the packed experts straight out of the GPU slot pool (no gather copy, no bf16 temporary):
  kernel 1:  a[j, r] = silu(Wg_j[r,:] . x) * (Wu_j[r,:] . x)      for every expert j in the request
  kernel 2:  y[j, h] = Wd_j[h,:] . a[j,:]
then out = sum_j w_j * y[j].

Buffer layout per expert (see quant.py, bits=4): [codes: 3n/2 bytes | scales fp16 [3n/G] | mins fp16 [3n/G]],
matrices flat in order gate [I,H], up [I,H], down [H,I]; element e of a matrix is the low nibble of byte e//2
when e is even and the high nibble when odd; group g = e // G covers G consecutive elements.

Dequant per group is folded into the dot product:  sum_i (q_i*s + m) x_i = s * (q . x) + m * sum(x).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_up_kernel(
    pool_ptr,
    slots_ptr,
    x_ptr,
    out_ptr,
    NBYTES: tl.constexpr,
    NCODE: tl.constexpr,
    NG: tl.constexpr,
    NHALF: tl.constexpr,
    NGM: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_r = tl.program_id(0)
    j = tl.program_id(1)
    slot = tl.load(slots_ptr + j).to(tl.int64)
    base = pool_ptr + slot * NBYTES
    sc_ptr = (base + NCODE).to(tl.pointer_type(tl.float16))
    mn_ptr = (base + NCODE + 2 * NG).to(tl.pointer_type(tl.float16))
    HALF: tl.constexpr = GROUP // 2
    GPR: tl.constexpr = H // GROUP
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < I
    rows64 = rows.to(tl.int64)
    cols = tl.arange(0, HALF)
    acc_g = tl.zeros([BLOCK_R], dtype=tl.float32)
    acc_u = tl.zeros([BLOCK_R], dtype=tl.float32)
    for c in range(GPR):
        xe = tl.load(x_ptr + c * GROUP + 2 * cols).to(tl.float32)
        xo = tl.load(x_ptr + c * GROUP + 2 * cols + 1).to(tl.float32)
        xs = tl.sum(xe, axis=0) + tl.sum(xo, axis=0)
        boff = rows64[:, None] * (H // 2) + c * HALF + cols[None, :]
        g = rows64 * GPR + c
        # gate (matrix 0)
        b = tl.load(base + boff, mask=rmask[:, None], other=0)
        dot = tl.sum((b & 15).to(tl.float32) * xe[None, :] + (b >> 4).to(tl.float32) * xo[None, :], axis=1)
        s = tl.load(sc_ptr + g, mask=rmask, other=0.0).to(tl.float32)
        m = tl.load(mn_ptr + g, mask=rmask, other=0.0).to(tl.float32)
        acc_g += s * dot + m * xs
        # up (matrix 1)
        b = tl.load(base + NHALF + boff, mask=rmask[:, None], other=0)
        dot = tl.sum((b & 15).to(tl.float32) * xe[None, :] + (b >> 4).to(tl.float32) * xo[None, :], axis=1)
        s = tl.load(sc_ptr + NGM + g, mask=rmask, other=0.0).to(tl.float32)
        m = tl.load(mn_ptr + NGM + g, mask=rmask, other=0.0).to(tl.float32)
        acc_u += s * dot + m * xs
    a = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(out_ptr + j * I + rows, a, mask=rmask)


@triton.jit
def _down_kernel(
    pool_ptr,
    slots_ptr,
    a_ptr,
    out_ptr,
    NBYTES: tl.constexpr,
    NCODE: tl.constexpr,
    NG: tl.constexpr,
    NHALF: tl.constexpr,
    NGM: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_r = tl.program_id(0)
    j = tl.program_id(1)
    slot = tl.load(slots_ptr + j).to(tl.int64)
    base = pool_ptr + slot * NBYTES
    sc_ptr = (base + NCODE).to(tl.pointer_type(tl.float16))
    mn_ptr = (base + NCODE + 2 * NG).to(tl.pointer_type(tl.float16))
    HALF: tl.constexpr = GROUP // 2
    GPR: tl.constexpr = I // GROUP
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < H
    rows64 = rows.to(tl.int64)
    cols = tl.arange(0, HALF)
    acc = tl.zeros([BLOCK_R], dtype=tl.float32)
    for c in range(GPR):
        ae = tl.load(a_ptr + j * I + c * GROUP + 2 * cols)
        ao = tl.load(a_ptr + j * I + c * GROUP + 2 * cols + 1)
        asum = tl.sum(ae, axis=0) + tl.sum(ao, axis=0)
        boff = 2 * NHALF + rows64[:, None] * (I // 2) + c * HALF + cols[None, :]
        b = tl.load(base + boff, mask=rmask[:, None], other=0)
        dot = tl.sum((b & 15).to(tl.float32) * ae[None, :] + (b >> 4).to(tl.float32) * ao[None, :], axis=1)
        g = 2 * NGM + rows64 * GPR + c
        s = tl.load(sc_ptr + g, mask=rmask, other=0.0).to(tl.float32)
        m = tl.load(mn_ptr + g, mask=rmask, other=0.0).to(tl.float32)
        acc += s * dot + m * asum
    tl.store(out_ptr + j * H + rows, acc, mask=rmask)


def moe_decode_q4(pool, slots, x, w, n, H, I, group=128, block_r=32, num_warps=4):
    """pool: [nslots, nbytes] uint8 (4-bit experts); slots: int64 [k]; x: [H] (bf16);
    w: [k] routing weights. Returns [H] in x.dtype."""
    assert H % group == 0 and I % group == 0 and n == H * I
    k = slots.numel()
    nbytes = pool.shape[1]
    meta = dict(
        NBYTES=nbytes,
        NCODE=3 * n // 2,
        NG=3 * n // group,
        NHALF=n // 2,
        NGM=n // group,
        H=H,
        I=I,
        GROUP=group,
        BLOCK_R=block_r,
    )
    x = x.contiguous()
    a = torch.empty((k, I), dtype=torch.float32, device=x.device)
    _gate_up_kernel[(triton.cdiv(I, block_r), k)](pool, slots, x, a, num_warps=num_warps, **meta)
    y = torch.empty((k, H), dtype=torch.float32, device=x.device)
    _down_kernel[(triton.cdiv(H, block_r), k)](pool, slots, a, y, num_warps=num_warps, **meta)
    return (y * w.float().unsqueeze(1)).sum(0).to(x.dtype)


@triton.jit
def _down_sum_kernel(
    pool_ptr,
    meta_ptr,
    a_ptr,
    out_ptr,
    NBYTES: tl.constexpr,
    NCODE: tl.constexpr,
    NG: tl.constexpr,
    NHALF: tl.constexpr,
    NGM: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    GROUP: tl.constexpr,
    K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """out[h] = sum_j w_j * (Wd_j[h,:] . a_j): all experts reduced inside the kernel, result written once."""
    pid_r = tl.program_id(0)
    HALF: tl.constexpr = GROUP // 2
    GPR: tl.constexpr = I // GROUP
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < H
    rows64 = rows.to(tl.int64)
    cols = tl.arange(0, HALF)
    total = tl.zeros([BLOCK_R], dtype=tl.float32)
    for j in range(K):
        slot = tl.load(meta_ptr + j).to(tl.int64)
        wj = tl.load(meta_ptr + K + j)
        base = pool_ptr + slot * NBYTES
        sc_ptr = (base + NCODE).to(tl.pointer_type(tl.float16))
        mn_ptr = (base + NCODE + 2 * NG).to(tl.pointer_type(tl.float16))
        acc = tl.zeros([BLOCK_R], dtype=tl.float32)
        for c in range(GPR):
            ae = tl.load(a_ptr + j * I + c * GROUP + 2 * cols)
            ao = tl.load(a_ptr + j * I + c * GROUP + 2 * cols + 1)
            asum = tl.sum(ae, axis=0) + tl.sum(ao, axis=0)
            boff = 2 * NHALF + rows64[:, None] * (I // 2) + c * HALF + cols[None, :]
            b = tl.load(base + boff, mask=rmask[:, None], other=0)
            dot = tl.sum((b & 15).to(tl.float32) * ae[None, :] + (b >> 4).to(tl.float32) * ao[None, :], axis=1)
            g = 2 * NGM + rows64 * GPR + c
            s = tl.load(sc_ptr + g, mask=rmask, other=0.0).to(tl.float32)
            m = tl.load(mn_ptr + g, mask=rmask, other=0.0).to(tl.float32)
            acc += s * dot + m * asum
        total += wj * acc
    tl.store(out_ptr + rows, total, mask=rmask)


def moe_decode_q4_fast(pool, meta, x, k, n, H, I, group=128, block_gu=32, block_down=8, num_warps=4):
    """Like moe_decode_q4 but with 2 kernel launches in total and one packed routing buffer.
    meta: float32 [2k] on the GPU = [slot_0..slot_{k-1}, w_0..w_{k-1}]. Returns [H] in x.dtype."""
    assert H % group == 0 and I % group == 0 and n == H * I
    nbytes = pool.shape[1]
    meta_kw = dict(
        NBYTES=nbytes, NCODE=3 * n // 2, NG=3 * n // group, NHALF=n // 2, NGM=n // group, H=H, I=I, GROUP=group
    )
    x = x.contiguous()
    a = torch.empty((k, I), dtype=torch.float32, device=x.device)
    _gate_up_kernel[(triton.cdiv(I, block_gu), k)](pool, meta, x, a, num_warps=num_warps, BLOCK_R=block_gu, **meta_kw)
    out = torch.empty((H,), dtype=x.dtype, device=x.device)
    _down_sum_kernel[(triton.cdiv(H, block_down),)](
        pool, meta, a, out, num_warps=num_warps, K=k, BLOCK_R=block_down, **meta_kw
    )
    return out


# ---------------------------------------------------------------------------------------------
# 3-bit variant. Codes are a little-endian bit stream over the 3n elements of an expert (see quant._pack):
# element e lives at bit p = 3e, so it is always inside two adjacent bytes:
#   q = ((b[p>>3] | b[(p>>3)+1] << 8) >> (p & 7)) & 7
# ---------------------------------------------------------------------------------------------
@triton.jit
def _q3(base, e, mask):
    p = e * 3
    b0 = p >> 3
    lo = tl.load(base + b0, mask=mask, other=0).to(tl.int32)
    hi = tl.load(base + b0 + 1, mask=mask, other=0).to(tl.int32)
    return (((hi << 8) | lo) >> (p & 7).to(tl.int32)) & 7


@triton.jit
def _gate_up_kernel3(
    pool_ptr,
    meta_ptr,
    x_ptr,
    out_ptr,
    NBYTES: tl.constexpr,
    NCODE: tl.constexpr,
    NG: tl.constexpr,
    N: tl.constexpr,
    NGM: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_r = tl.program_id(0)
    j = tl.program_id(1)
    slot = tl.load(meta_ptr + j).to(tl.int64)
    base = pool_ptr + slot * NBYTES
    sc_ptr = (base + NCODE).to(tl.pointer_type(tl.float16))
    mn_ptr = (base + NCODE + 2 * NG).to(tl.pointer_type(tl.float16))
    GPR: tl.constexpr = H // GROUP
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < I
    rows64 = rows.to(tl.int64)
    cols = tl.arange(0, GROUP).to(tl.int64)
    acc_g = tl.zeros([BLOCK_R], dtype=tl.float32)
    acc_u = tl.zeros([BLOCK_R], dtype=tl.float32)
    for c in range(GPR):
        xv = tl.load(x_ptr + c * GROUP + cols).to(tl.float32)
        xs = tl.sum(xv, axis=0)
        e = rows64[:, None] * H + c * GROUP + cols[None, :]
        g = rows64 * GPR + c
        q = _q3(base, e, rmask[:, None]).to(tl.float32)
        s = tl.load(sc_ptr + g, mask=rmask, other=0.0).to(tl.float32)
        m = tl.load(mn_ptr + g, mask=rmask, other=0.0).to(tl.float32)
        acc_g += s * tl.sum(q * xv[None, :], axis=1) + m * xs
        q = _q3(base, e + N, rmask[:, None]).to(tl.float32)
        s = tl.load(sc_ptr + NGM + g, mask=rmask, other=0.0).to(tl.float32)
        m = tl.load(mn_ptr + NGM + g, mask=rmask, other=0.0).to(tl.float32)
        acc_u += s * tl.sum(q * xv[None, :], axis=1) + m * xs
    a = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(out_ptr + j * I + rows, a, mask=rmask)


@triton.jit
def _down_sum_kernel3(
    pool_ptr,
    meta_ptr,
    a_ptr,
    out_ptr,
    NBYTES: tl.constexpr,
    NCODE: tl.constexpr,
    NG: tl.constexpr,
    N: tl.constexpr,
    NGM: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    GROUP: tl.constexpr,
    K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_r = tl.program_id(0)
    GPR: tl.constexpr = I // GROUP
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < H
    rows64 = rows.to(tl.int64)
    cols = tl.arange(0, GROUP).to(tl.int64)
    total = tl.zeros([BLOCK_R], dtype=tl.float32)
    for j in range(K):
        slot = tl.load(meta_ptr + j).to(tl.int64)
        wj = tl.load(meta_ptr + K + j)
        base = pool_ptr + slot * NBYTES
        sc_ptr = (base + NCODE).to(tl.pointer_type(tl.float16))
        mn_ptr = (base + NCODE + 2 * NG).to(tl.pointer_type(tl.float16))
        acc = tl.zeros([BLOCK_R], dtype=tl.float32)
        for c in range(GPR):
            av = tl.load(a_ptr + j * I + c * GROUP + cols)
            asum = tl.sum(av, axis=0)
            e = 2 * N + rows64[:, None] * I + c * GROUP + cols[None, :]
            q = _q3(base, e, rmask[:, None]).to(tl.float32)
            g = 2 * NGM + rows64 * GPR + c
            s = tl.load(sc_ptr + g, mask=rmask, other=0.0).to(tl.float32)
            m = tl.load(mn_ptr + g, mask=rmask, other=0.0).to(tl.float32)
            acc += s * tl.sum(q * av[None, :], axis=1) + m * asum
        total += wj * acc
    tl.store(out_ptr + rows, total, mask=rmask)


def moe_decode_q3_fast(pool, meta, x, k, n, H, I, group=128, block_gu=8, block_down=8, num_warps=4):
    """3-bit twin of moe_decode_q4_fast (same meta layout: float32 [slots..., weights...])."""
    assert H % group == 0 and I % group == 0 and n == H * I
    nbytes = pool.shape[1]
    kw = dict(NBYTES=nbytes, NCODE=(9 * n) // 8, NG=3 * n // group, N=n, NGM=n // group, H=H, I=I, GROUP=group)
    x = x.contiguous()
    a = torch.empty((k, I), dtype=torch.float32, device=x.device)
    _gate_up_kernel3[(triton.cdiv(I, block_gu), k)](pool, meta, x, a, num_warps=num_warps, BLOCK_R=block_gu, **kw)
    out = torch.empty((H,), dtype=x.dtype, device=x.device)
    _down_sum_kernel3[(triton.cdiv(H, block_down),)](
        pool, meta, a, out, num_warps=num_warps, K=k, BLOCK_R=block_down, **kw
    )
    return out


# ---------------------------------------------------------------------------------------------
# Rotary embedding for 1-token decode: q and k rotated in ONE launch (HF needs ~10 small kernels).
# ---------------------------------------------------------------------------------------------
@triton.jit
def _rope_kernel(q_ptr, k_ptr, cos_ptr, sin_ptr, qo_ptr, ko_ptr, SQ, SK, HQ: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.arange(0, D)
    half: tl.constexpr = D // 2
    partner = (idx + half) % D
    sign = tl.where(idx < half, -1.0, 1.0)
    c = tl.load(cos_ptr + idx).to(tl.float32)
    s = tl.load(sin_ptr + idx).to(tl.float32)
    is_q = pid < HQ
    is_k = pid >= HQ
    hq = pid
    hk = pid - HQ
    x = tl.load(q_ptr + hq * SQ + idx, mask=is_q, other=0.0).to(tl.float32) + tl.load(
        k_ptr + hk * SK + idx, mask=is_k, other=0.0
    ).to(tl.float32)
    xp = tl.load(q_ptr + hq * SQ + partner, mask=is_q, other=0.0).to(tl.float32) + tl.load(
        k_ptr + hk * SK + partner, mask=is_k, other=0.0
    ).to(tl.float32)
    out = x * c + sign * xp * s
    tl.store(qo_ptr + hq * D + idx, out, mask=is_q)
    tl.store(ko_ptr + hk * D + idx, out, mask=is_k)


def rope_decode(q, k, cos, sin):
    """q: [1,Hq,1,D], k: [1,Hk,1,D], cos/sin: [1,1,D] (HF layout) -> rotated q, k (same shapes)."""
    Hq, D = q.shape[1], q.shape[3]
    Hk = k.shape[1]
    qo = torch.empty((1, Hq, 1, D), dtype=q.dtype, device=q.device)
    ko = torch.empty((1, Hk, 1, D), dtype=k.dtype, device=k.device)
    _rope_kernel[(Hq + Hk,)](
        q,
        k,
        cos.reshape(-1).contiguous(),
        sin.reshape(-1).contiguous(),
        qo,
        ko,
        q.stride(1),
        k.stride(1),
        HQ=Hq,
        D=D,
        num_warps=1,
    )
    return qo, ko
