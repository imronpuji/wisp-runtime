"""GPU expert cache: one big uint8 slot pool + async H2D copies on a side stream."""
import torch
from .cache_core import CacheCore
from .quant import dequant, dequant_fast


class ExpertCache:
    def __init__(self, store, nslots, device="cuda", policy="lru", decay=0.9995, n_static=0,
                 admit_after=2.0, preload_keys=None, cpu_store=None, decode_on_load=False, staging_rows=8):
        self.store, self.device = store, torch.device(device)
        # decode_on_load: experts cross PCIe in quantized form, are dequantized ONCE on arrival and kept as bf16.
        # Hits then cost nothing extra; capacity per GiB is that of bf16.
        self.decode_on_load = decode_on_load and store.bits < 16
        self.cpu_store = cpu_store   # optional bf16 'shadow' used for CPU-side compute of misses
        self.core = CacheCore(nslots, policy, decay, n_static, admit_after)
        self.nslots = nslots
        self.slot_bytes = 3 * store.n * 2 if self.decode_on_load else store.nbytes
        self.gpu = torch.empty((nslots, self.slot_bytes), dtype=torch.uint8, device=self.device)
        self.staging = (torch.empty((staging_rows, store.nbytes), dtype=torch.uint8, device=self.device)
                        if self.decode_on_load else None)
        self.gpu_rows = self.gpu.unbind(0)          # pre-built views: no per-copy indexing work on the CPU
        self.host_rows = store.host.unbind(0)
        self.copy_stream = torch.cuda.Stream(self.device)
        self.evt = None
        self.bytes_h2d = 0
        self.n_loads = 0
        if n_static:
            assert preload_keys is not None and len(preload_keys) <= n_static
            self._issue(self.core.preload(list(preload_keys)))
            self.sync_all()

    # ---- copies ---------------------------------------------------------
    def _issue(self, pairs):
        """Async host->GPU copies of [(key, slot)] on the side stream, ordered after queued compute."""
        if not pairs:
            return
        e = torch.cuda.Event()
        e.record(torch.cuda.current_stream(self.device))   # slots may still be read by queued kernels
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(e)
            if not self.decode_on_load:
                gr, hr = self.gpu_rows, self.host_rows
                for k, s in pairs:
                    gr[s].copy_(hr[k], non_blocking=True)
            else:
                st, R = self.store, self.staging.shape[0]
                bf = self.gpu.view(torch.bfloat16).view(self.nslots, 3, st.n)
                for i in range(0, len(pairs), R):
                    part = pairs[i:i + R]
                    for j, (k, _) in enumerate(part):
                        self.staging[j].copy_(st.host[k], non_blocking=True)
                    W = dequant_fast(self.staging[:len(part)], st.n, st.bits, st.group)
                    idx = torch.tensor([s for _, s in part], device=self.device)
                    bf.index_copy_(0, idx, W)
        done = torch.cuda.Event()
        done.record(self.copy_stream)
        self.evt = done
        self.bytes_h2d += len(pairs) * self.store.nbytes
        self.n_loads += len(pairs)

    def sync_compute(self):
        """Make the current compute stream wait for all issued copies."""
        if self.evt is not None:
            torch.cuda.current_stream(self.device).wait_event(self.evt)

    def sync_all(self):
        torch.cuda.synchronize(self.device)

    # ---- policies -------------------------------------------------------
    def ensure(self, keys):
        slots, loads = self.core.ensure(keys)
        self._issue(loads)
        self.sync_compute()
        return slots

    def split(self, keys):
        return self.core.split(keys)

    def admit_async(self, miss_keys, protected_slots):
        prot, pairs = set(protected_slots), []
        for k in miss_keys:
            s = self.core.admit(k, prot)
            if s is not None:
                prot.add(s); pairs.append((k, s))
        self._issue(pairs)     # not waited on here; the next ensure()/sync_compute() will
        return len(pairs)

    def prefetch(self, keys, protected_slots):
        prot, pairs = set(protected_slots), []
        for k in keys:
            if self.core.lookup(k) is not None:
                continue
            try:
                s = self.core.acquire(k, prot)
            except RuntimeError:
                break
            prot.add(s); pairs.append((k, s))
        self._issue(pairs)
        return len(pairs)

    # ---- weights --------------------------------------------------------
    def weights_batched(self, idx):
        """idx: LongTensor of slots on GPU -> [k, 3, n] bf16 (one gather + at most one fused dequant kernel)."""
        st = self.store
        rows = self.gpu.index_select(0, idx)
        if st.bits == 16 or self.decode_on_load:
            return rows.view(torch.bfloat16).view(len(idx), 3, st.n)
        return dequant_fast(rows, st.n, st.bits, st.group)

    def weights(self, slots):
        """List of [3, n] bf16 views/tensors on GPU for the given slots."""
        st = self.store
        if st.bits == 16 or self.decode_on_load:
            return [self.gpu[s].view(torch.bfloat16).view(3, st.n) for s in slots]
        idx = torch.as_tensor(slots, device=self.device)
        return list(dequant_fast(self.gpu.index_select(0, idx), st.n, st.bits, st.group).unbind(0))

    def cpu_weights(self, key):
        st = self.cpu_store if self.cpu_store is not None else self.store
        b = st.host[key]
        if st.bits == 16:
            return b.view(torch.bfloat16).view(3, st.n)
        return dequant(b[None], st.n, st.bits, st.group)[0]

    def resize(self, nslots):
        """Grow/shrink the slot pool in place (keeps the hottest experts). Frees the old pool BEFORE allocating the
        new one, so it works under a tight allocator cap; the kept experts are re-copied from host RAM."""
        assert not self.decode_on_load
        if nslots == self.nslots:
            return
        torch.cuda.synchronize(self.device)
        self.core, loads = self.core.resized(nslots)
        self.gpu_rows, self.gpu, self.evt = None, None, None
        torch.cuda.empty_cache()
        self.nslots = nslots
        self.gpu = torch.empty((nslots, self.slot_bytes), dtype=torch.uint8, device=self.device)
        self.gpu_rows = self.gpu.unbind(0)
        self._issue(loads)
        self.sync_all()

    def free(self):
        del self.gpu
        self.gpu = None
