"""VRAM budget -> number of expert slots. (The allocator cap in model.configure() enforces it for real.)"""
GIB = 2 ** 30


def kv_bytes_per_token(cfg):
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 2  # K+V, bf16


def dequant_temp_bytes(n, chunk, bits):
    return 0 if bits == 16 else int(chunk * 3 * n * 10)   # fp32 intermediates + bf16 output


def plan_slots(budget_gib, nonexpert_bytes, store, cfg, ctx, kv_mode="fixed", kv_frac=0.3,
               chunk=8, misc_gib=1.0, decode_on_load=False, staging_rows=8):
    """Return dict(slots, kv_gib, ...). kv_mode:
       'fixed'    : KV reservation = kv_frac * budget
       'adaptive' : KV reservation = exactly what `ctx` tokens need (+10%): the rest goes to experts
    """
    budget = budget_gib * GIB
    kv_need = kv_bytes_per_token(cfg) * ctx * 1.1
    kv = kv_need if kv_mode == "adaptive" else kv_frac * budget
    kv = max(kv, kv_need)   # admission floor: never starve active requests
    dol = decode_on_load and store.bits < 16
    slot_bytes = 3 * store.n * 2 if dol else store.nbytes
    tmp = (staging_rows * store.nbytes + dequant_temp_bytes(store.n, staging_rows, store.bits)) if dol \
        else dequant_temp_bytes(store.n, chunk, store.bits)
    free = budget - nonexpert_bytes - kv - tmp - misc_gib * GIB
    slots = int(free // slot_bytes)
    total = store.L * store.E
    return dict(slots=max(0, min(slots, total)), kv_gib=kv / GIB, tmp_gib=tmp / GIB,
                free_gib=free / GIB, expert_mib=store.nbytes / 2 ** 20, slot_mib=slot_bytes / 2 ** 20, fits_all=slots >= total)
