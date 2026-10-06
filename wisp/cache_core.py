"""Torch-free cache bookkeeping, shared by the real GPU cache and the trace simulator.

Keys are ints (layer * n_experts + expert). Slots [0, n_static) hold a fixed 'static' resident set
(policy 'static' = profile-then-pin baseline); slots [n_static, nslots) are managed dynamically.
Policies for the dynamic region:
  lru : evict least recently used
  wsa : evict lowest exponentially-decayed access frequency (working-set style)
"""
from collections import defaultdict
import numpy as np


class CacheCore:
    def __init__(self, nslots, policy="lru", decay=0.9995, n_static=0, admit_after=2.0):
        assert policy in ("lru", "wsa", "static"), policy
        self.nslots, self.policy, self.decay = nslots, policy, decay
        self.n_static, self.admit_after = n_static, admit_after
        self.slot_key = np.full(nslots, -1, dtype=np.int64)
        self.last = np.zeros(nslots)
        self.score = np.zeros(nslots)
        self.key2slot = {}
        self.free = list(range(nslots - 1, n_static - 1, -1))
        self.tick, self.inc = 0, 1.0
        self.freq = defaultdict(float)
        self.stats = dict(hits=0, misses=0, evictions=0, calls=0)

    # -- clock / frequency ------------------------------------------------
    def step(self):
        self.tick += 1
        self.inc /= self.decay
        if self.inc > 1e100:  # renormalise to avoid overflow
            for k in self.freq:
                self.freq[k] /= 1e100
            self.score /= 1e100
            self.inc /= 1e100

    def note(self, keys):
        for k in keys:
            self.freq[k] += self.inc
            s = self.key2slot.get(k)
            if s is not None:
                self.last[s] = self.tick
                self.score[s] = self.freq[k]

    def norm_freq(self, k):
        return self.freq.get(k, 0.0) / self.inc

    def should_admit(self, k):
        return self.norm_freq(k) >= self.admit_after

    # -- residency ---------------------------------------------------------
    def lookup(self, k):
        return self.key2slot.get(k)

    def preload(self, keys):
        assert len(keys) <= self.n_static
        for s, k in enumerate(keys):
            self.slot_key[s] = k
            self.key2slot[k] = s
        return [(k, s) for s, k in enumerate(keys)]   # (key, slot), same order as ensure()'s loads

    def acquire(self, key, protected):
        """Pick a slot for `key` (evicting if needed, never a slot in `protected`)."""
        if self.free:
            s = self.free.pop()
        else:
            base = self.n_static
            arr = (self.last if self.policy == "lru" else self.score)[base:].copy()
            for p in protected:
                if p >= base:
                    arr[p - base] = np.inf
            j = int(arr.argmin())
            if not np.isfinite(arr[j]):
                raise RuntimeError("no evictable slot: dynamic region smaller than the working request")
            s = base + j
            del self.key2slot[int(self.slot_key[s])]
            self.stats["evictions"] += 1
        self.slot_key[s] = key
        self.key2slot[key] = s
        self.last[s] = self.tick
        self.score[s] = self.freq[key]
        return s

    # -- high level --------------------------------------------------------
    def ensure(self, keys):
        """Transfer policy: make every key resident. Returns (slots aligned with keys, [(key, slot)] to load)."""
        self.step(); self.note(keys)
        self.stats["calls"] += 1
        slots, loads, prot = [None] * len(keys), [], set()
        for i, k in enumerate(keys):
            s = self.key2slot.get(k)
            if s is not None:
                slots[i] = s; prot.add(s); self.stats["hits"] += 1
        for i, k in enumerate(keys):
            if slots[i] is None:
                s = self.acquire(k, prot)
                slots[i] = s; prot.add(s); loads.append((k, s)); self.stats["misses"] += 1
        return slots, loads

    def split(self, keys):
        """Hybrid policy: return (hit_keys, hit_slots, miss_keys); nothing is loaded."""
        self.step(); self.note(keys)
        self.stats["calls"] += 1
        hk, hs, mk = [], [], []
        for k in keys:
            s = self.key2slot.get(k)
            if s is None:
                mk.append(k)
            else:
                hk.append(k); hs.append(s)
        self.stats["hits"] += len(hk); self.stats["misses"] += len(mk)
        return hk, hs, mk

    def admit(self, key, protected):
        """Bring a (CPU-computed) key into the cache for future use. Returns slot or None if not worth it."""
        if key in self.key2slot or not self.should_admit(key):
            return None
        try:
            return self.acquire(key, protected)
        except RuntimeError:
            return None

    # -- resizing (dynamic VRAM split between experts and KV) ---------------
    def resized(self, nslots):
        """New core with `nslots` slots keeping the hottest resident keys (LRU: most recent; WSA: highest score).
        Returns (core, [(key, new_slot)]) -- the caller copies those keys into the new pool."""
        assert self.n_static == 0, "resize is not supported with a static region"
        nc = CacheCore(nslots, self.policy, self.decay, 0, self.admit_after)
        nc.tick, nc.inc, nc.freq, nc.stats = self.tick, self.inc, self.freq, self.stats
        rank = self.last if self.policy == "lru" else self.score
        live = [(int(self.slot_key[s]), s) for s in range(self.nslots) if self.slot_key[s] >= 0]
        live.sort(key=lambda ks: -rank[ks[1]])
        keep = live[:nslots]
        for ns, (k, s) in enumerate(keep):
            nc.slot_key[ns], nc.key2slot[k] = k, ns
            nc.last[ns], nc.score[ns] = self.last[s], self.score[s]
        nc.free = list(range(nslots - 1, len(keep) - 1, -1))
        return nc, [(k, ns) for ns, (k, _) in enumerate(keep)]
