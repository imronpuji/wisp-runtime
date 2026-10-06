"""Host-side (pinned RAM) store of all experts, quantized to `bits`."""
import json, os, time
import torch
from .quant import expert_nbytes, quantize


class ExpertStore:
    def __init__(self, n_layers, n_experts, n, bits, group=128, pin=True):
        self.L, self.E, self.n, self.bits, self.group = n_layers, n_experts, n, bits, group
        self.nbytes = expert_nbytes(n, bits, group)
        t = time.time()
        try:
            self.host = torch.empty((n_layers * n_experts, self.nbytes), dtype=torch.uint8, pin_memory=pin)
            self.pinned = pin
        except RuntimeError as e:  # pinning refused by the container
            print(f"[store] pin_memory failed ({e}); falling back to pageable memory", flush=True)
            self.host = torch.empty((n_layers * n_experts, self.nbytes), dtype=torch.uint8)
            self.pinned = False
        print(f"[store] allocated {self.host.numel()/2**30:.1f} GiB host (pinned={self.pinned}) in {time.time()-t:.1f}s", flush=True)

    @property
    def total_gib(self):
        return self.host.numel() / 2 ** 30

    def key(self, l, e):
        return l * self.E + e

    @torch.no_grad()
    def put(self, l, e, w3, device):
        buf = quantize(w3.to(device), self.bits, self.group)
        self.host[l * self.E + e].copy_(buf.cpu())

    # ---- builders -------------------------------------------------------
    @classmethod
    def from_experts(cls, layers, bits, group=128, device="cuda", pin=True):
        """layers: list (per layer) of list of (gate_w, up_w, down_w) tensors -- used by tests."""
        L, E = len(layers), len(layers[0])
        n = layers[0][0][0].numel()
        st = cls(L, E, n, bits, group, pin)
        for l, exps in enumerate(layers):
            for e, (g, u, d) in enumerate(exps):
                st.put(l, e, torch.stack([g.reshape(-1), u.reshape(-1), d.reshape(-1)]), device)
        return st

    @classmethod
    def from_safetensors(cls, model_dir, cfg, bits, group=128, device="cuda", cache_dir=None, pin=True):
        from safetensors import safe_open
        L, E = cfg.num_hidden_layers, cfg.num_experts
        n = cfg.moe_intermediate_size * cfg.hidden_size
        st = cls(L, E, n, bits, group, pin)
        path = None
        if cache_dir and bits < 16:
            os.makedirs(cache_dir, exist_ok=True)
            path = os.path.join(cache_dir, f"store_b{bits}_g{group}_{L}x{E}x{n}.u8")
            if os.path.exists(path) and os.path.getsize(path) == st.host.numel():
                st._load(path)
                return st
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        handles = {}

        def get(name):
            f = idx[name]
            if f not in handles:
                handles[f] = safe_open(os.path.join(model_dir, f), framework="pt", device="cpu")
            return handles[f].get_tensor(name)

        t0 = time.time()
        for l in range(L):
            for e in range(E):
                p = f"model.layers.{l}.mlp.experts.{e}."
                w3 = torch.stack([get(p + "gate_proj.weight").reshape(-1),
                                  get(p + "up_proj.weight").reshape(-1),
                                  get(p + "down_proj.weight").reshape(-1)])
                st.put(l, e, w3, device)
            if l % 4 == 3 or l == L - 1:
                print(f"[store] bits={bits} layer {l+1}/{L}  {time.time()-t0:.0f}s", flush=True)
        if path:
            st._save(path)
        return st

    # ---- persistence ----------------------------------------------------
    def _save(self, path, rows=256):
        a = self.host.numpy()
        with open(path + ".tmp", "wb") as f:
            for i in range(0, a.shape[0], rows):
                f.write(memoryview(a[i:i + rows]))
        os.replace(path + ".tmp", path)
        print(f"[store] saved {path}", flush=True)

    def _load(self, path, rows=256):
        a = self.host.numpy()
        t = time.time()
        with open(path, "rb") as f:
            for i in range(0, a.shape[0], rows):
                f.readinto(memoryview(a[i:i + rows]))
        print(f"[store] loaded {path} in {time.time()-t:.0f}s", flush=True)
