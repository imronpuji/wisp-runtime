import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3MoeConfig

from wisp.cache import ExpertCache
from wisp.moe import Runtime, attach
from wisp.store import ExpertStore

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def tiny():
    cfg = Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=True,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():  # make experts matter: default init leaves MoE almost irrelevant to the logits
        for layer in m.model.layers:
            for e in layer.mlp.experts:
                e.gate_proj.weight.mul_(6)
                e.up_proj.weight.mul_(6)
                e.down_proj.weight.mul_(30)
    return m, cfg


def build(model, cfg, bits, nslots, policy="lru", miss="transfer", prefetch=False, thr=4, dol=False):
    layers = [
        [(e.gate_proj.weight, e.up_proj.weight, e.down_proj.weight) for e in l.mlp.experts] for l in model.model.layers
    ]
    store = ExpertStore.from_experts(layers, bits, group=32, device="cuda")
    rt = Runtime(store, miss_policy=miss, cpu_threshold=thr, prefetch=prefetch, chunk=4)
    rt.cache = ExpertCache(store, nslots, "cuda", policy, admit_after=1.0, decode_on_load=dol, staging_rows=3)
    attach(model, rt)
    return rt


def logits(model, ids, split):
    with torch.no_grad():
        out = model(ids[:, :split], use_cache=True)
        pk, res = out.past_key_values, [out.logits]
        for t in range(split, ids.shape[1]):
            o = model(ids[:, t : t + 1], past_key_values=pk, use_cache=True)
            pk = o.past_key_values
            res.append(o.logits)
    return torch.cat(res, 1).float()


@pytest.mark.parametrize(
    "nslots,policy,miss,prefetch",
    [
        (24, "lru", "transfer", False),  # everything resident
        (6, "lru", "transfer", False),  # heavy thrashing
        (6, "wsa", "transfer", False),
        (6, "lru", "hybrid", False),  # CPU computes misses
        (6, "wsa", "hybrid", False),
        (8, "lru", "transfer", True),  # with prefetch
    ],
)
def test_bf16_matches_reference(nslots, policy, miss, prefetch):
    model, cfg = tiny()
    ids = torch.randint(0, 256, (1, 14), device="cuda")
    with torch.no_grad():
        ref = model(ids).logits.float()
    rt = build(model, cfg, 16, nslots, policy, miss, prefetch)
    got = logits(model, ids, split=9)  # 9-token prefill (routed path) + 5 decode steps (dense path)
    err = (got - ref).abs().max().item()
    assert err < 0.01 * ref.abs().max().item(), err
    if nslots < 24:
        assert rt.cache.core.stats["misses"] > 0
    if miss == "hybrid":
        assert rt.stats["cpu_experts"] > 0


def test_quantized_error_ordering():
    errs = {}
    for bits in (8, 4, 2):
        model, cfg = tiny()
        ids = torch.randint(0, 256, (1, 12), device="cuda")
        with torch.no_grad():
            ref = model(ids).logits.float()
        build(model, cfg, bits, 24)
        with torch.no_grad():
            got = model(ids).logits.float()
        errs[bits] = ((got - ref).norm() / ref.norm()).item()
    assert errs[8] < errs[4] < errs[2], errs
    assert errs[8] < 0.15, errs


def test_cache_bookkeeping():
    model, cfg = tiny()
    rt = build(model, cfg, 16, 6, "lru")
    ids = torch.randint(0, 256, (1, 16), device="cuda")
    logits(model, ids, split=8)
    core = rt.cache.core
    assert len(core.key2slot) <= 6
    assert set(core.key2slot.values()) == {int(s) for s in range(6) if core.slot_key[s] >= 0}
    assert rt.cache.bytes_h2d == rt.cache.n_loads * rt.store.nbytes


def test_sabotage_is_detected():
    """Control: if experts are mixed up the comparison MUST fail by a wide margin (guards against a vacuous test)."""
    model, cfg = tiny()
    ids = torch.randint(0, 256, (1, 14), device="cuda")
    with torch.no_grad():
        ref = model(ids).logits.float()
    rt = build(model, cfg, 16, 24)
    rt.store.host[:] = rt.store.host[torch.randperm(rt.store.host.shape[0])]
    rt.cache.free()
    rt.cache = ExpertCache(rt.store, 24)
    with torch.no_grad():
        bad = model(ids).logits.float()
    assert (bad - ref).abs().max().item() > 10 * 0.01 * ref.abs().max().item()


def test_state_dict_has_no_shared_prefetch_gate():
    model, cfg = tiny()
    build(model, cfg, 16, 24)
    assert not [k for k in model.state_dict() if "next_gate" in k]


@pytest.mark.parametrize("miss", ["transfer", "hybrid"])
def test_static_policy_matches_reference(miss):
    model, cfg = tiny()
    ids = torch.randint(0, 256, (1, 14), device="cuda")
    with torch.no_grad():
        ref = model(ids).logits.float()
    layers = [
        [(e.gate_proj.weight, e.up_proj.weight, e.down_proj.weight) for e in l.mlp.experts] for l in model.model.layers
    ]
    store = ExpertStore.from_experts(layers, 16, group=32, device="cuda")
    rt = Runtime(store, miss_policy=miss, chunk=4)
    pin = [23, 5, 17, 0, 9, 12]  # arbitrary keys in scrambled order
    rt.cache = ExpertCache(store, 12, "cuda", "static", n_static=6, preload_keys=pin)
    for s, k in enumerate(pin):  # every pinned slot must hold exactly that expert's bytes
        assert torch.equal(rt.cache.gpu[s].cpu(), store.host[k])
    attach(model, rt)
    got = logits(model, ids, split=9)
    assert (got - ref).abs().max().item() < 0.01 * ref.abs().max().item()


def test_bench_prompts_metrics():
    from wisp import runner

    model, cfg = tiny()
    rt = build(model, cfg, 16, 6)
    prompts = [("x", torch.randint(0, 256, (1, 10))) for _ in range(3)]
    m = runner.bench_prompts(model, rt, prompts, n_new=6, warm=1)
    assert m["decode_tokens"] == 2 * 5 and m["decode_tok_s"] > 0
    assert 0 <= m["dec_hit_rate"] <= 1 and m["dec_misses_per_token"] >= 0
    assert m["dec_h2d_mb_per_token"] <= m["h2d_mb_per_token"] * 12 / 10 + 1e-9  # decode-only is a subset


@pytest.mark.parametrize("bits,nslots", [(8, 24), (4, 6), (2, 5)])
def test_decode_on_load_equals_quantized_path(bits, nslots):
    """Same quantized weights either way -> logits must match the per-use-dequant path (thrashing included)."""
    ids = torch.randint(0, 256, (1, 14), device="cuda")
    model, cfg = tiny()
    build(model, cfg, bits, nslots)
    a = logits(model, ids, split=9)
    model, cfg = tiny()
    rt = build(model, cfg, bits, nslots, dol=True)
    assert rt.cache.gpu.shape[1] == 3 * rt.store.n * 2  # bf16 slots
    b = logits(model, ids, split=9)
    assert (a - b).abs().max().item() < 1e-3 * a.abs().max().item()
    if nslots < 24:
        assert rt.cache.n_loads > nslots  # evictions + staging batches exercised
