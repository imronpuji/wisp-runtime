"""Replay a routing trace through CacheCore to get miss counts without touching a GPU.

Trace file (npz): ids int16 [T, L, k] = expert ids per decode step; meta: L, E.
"""
import numpy as np
from .cache_core import CacheCore


def load_trace(path):
    z = np.load(path)
    return z["ids"], int(z["L"]), int(z["E"])


def top_keys(ids, L, E, n):
    """Most frequently used (layer, expert) keys over the whole trace -> static pin set."""
    cnt = np.zeros(L * E, dtype=np.int64)
    for l in range(L):
        u, c = np.unique(ids[:, l, :], return_counts=True)
        cnt[l * E + u] += c
    return [int(k) for k in np.argsort(-cnt)[:n]]


def simulate(ids, L, E, slots, policy="lru", miss_policy="transfer", decay=0.9995, n_static=0,
             admit_after=2.0, chunk=8, warm_frac=0.25, profile_ids=None):
    """Returns dict with per-token miss stats measured after the first `warm_frac` of steps."""
    T = ids.shape[0]
    core = CacheCore(slots, policy, decay, n_static, admit_after)
    if n_static:
        pin = top_keys(profile_ids if profile_ids is not None else ids, L, E, n_static)
        core.preload(pin)
    warm = int(T * warm_frac)
    miss = np.zeros(T, dtype=np.int64)
    for t in range(T):
        m = 0
        for l in range(L):
            keys = [l * E + int(e) for e in np.unique(ids[t, l])]
            if miss_policy == "hybrid":
                hk, hs, mk = core.split(keys)
                prot = set(hs)
                for k in mk:
                    s = core.admit(k, prot)
                    if s is not None:
                        prot.add(s)   # admitted for future tokens; the CPU still computes it this time
                m += len(mk)
            else:
                _, loads = core.ensure(keys)
                m += len(loads)
        miss[t] = m
    m = miss[warm:]
    total_keys = ids.shape[2] * L
    return dict(slots=slots, policy=policy, miss_policy=miss_policy, misses_per_token=float(m.mean()),
                hit_rate=float(1 - m.mean() / total_keys), tokens=int(len(m)),
                evictions=core.stats["evictions"])


def predict_tok_s(misses_per_token, expert_bytes, bw_gbs, t_compute_s, miss_policy="transfer", t_cpu_expert_s=0.0):
    """t_token = t_compute + misses * cost_per_miss. cost = bytes/BW (transfer) or CPU time (hybrid)."""
    per = expert_bytes / (bw_gbs * 1e9) if miss_policy == "transfer" else t_cpu_expert_s
    return 1.0 / (t_compute_s + misses_per_token * per)
