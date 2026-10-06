# WiSP Runtime

**Run a 30B Mixture-of-Experts model on a GPU with 8 GB of VRAM, at ~23 tokens/s.**

Large MoE models like Qwen3-30B-A3B have 30B parameters but use only ~3B per token. WiSP Runtime keeps all experts
in CPU RAM (4-bit) and copies to the GPU only the experts a token actually needs, caching the frequently used ones in
VRAM, the way an operating system pages memory. Everything else (attention, embeddings, router) stays on the GPU.

It is an inference runtime in the same family as vLLM or llama.cpp, but much narrower: experimental, single-user,
batch size 1, no API server, and so far one model family (Qwen3 MoE). It runs on top of PyTorch and Hugging Face
Transformers and replaces their MoE, attention and decode loop.

The name comes from the paper it started as a replication of: *WiSP: A Working-Set View of Mixture-of-Experts Serving
on Extremely Low-Resource Hardware* ([arXiv:2606.21868](https://arxiv.org/abs/2606.21868)). This project is an
independent implementation and is not affiliated with the paper's authors.

## Results

Qwen3-30B-A3B, 4-bit experts, LRU expert cache, NVIDIA A100 MIG slice (3g.20gb) **capped to 8 GB** with
`torch.cuda.set_per_process_memory_fraction`, 32 GB host RAM, batch size 1, greedy decode.

| Step | Decode tok/s |
|---|---|
| Baseline paging (dequantize to bf16, then matmul) | 7.5 |
| Fused 4-bit dequant + GEMV Triton kernels | 11.8 |
| One host sync per layer, 2 kernel launches | 14.7 |
| + fast RMSNorm | 16.5 |
| + one-kernel rotary, GQA decode attention without `repeat_kv` | 18.4 |
| + CUDA graphs between the per-layer paging decisions | **22.9** |

Expert-cache hit rate during decode is ~76% at 8 GB; ~216 MB of experts cross PCIe per token. After the fused
kernels the bottleneck is CPU launch overhead, not PCIe or GPU compute.

Quality (perplexity on 30 windows × 512 tokens, MMLU 200 questions; same model, different expert bit-widths):

| Experts | PPL English | PPL Indonesian | MMLU |
|---|---|---|---|
| 4-bit | 12.31 | 7.04 | 79.0% |
| 3-bit | 13.04 | 7.56 | 78.0% |
| 2-bit | 41.77 | 31.79 | 22.0% (broken) |

Caveats: one model and one machine so far; no bf16 reference (does not fit in 32 GB RAM); MMLU has ~±3 points of
noise at 200 questions. The CUDA-graph decoder agrees with the eager path on 98.3% of top-1 tokens
(teacher-forced, 6 prompts × 48 steps); a perplexity comparison (`scripts/ppl_graph.py`) has not been run yet.

## How it works

1. **Expert store** (`wisp/store.py`, `wisp/quant.py`): every expert (gate, up, down) is quantized group-wise
   (RTN, group 128) to 4/3/2-bit into one flat buffer in pinned CPU RAM (14.3 GiB for Qwen3-30B-A3B at 4-bit).
2. **Expert cache** (`wisp/cache.py`, `wisp/cache_core.py`): a pool of GPU slots; missing experts are copied
   asynchronously; LRU or working-set eviction.
3. **Fused kernels** (`wisp/kernels.py`): Triton kernels read 4-bit/3-bit experts straight from the slot pool and
   compute all 8 selected experts of a token in 2 launches, with no bf16 temporary.
4. **CUDA-graph decode** (`wisp/graphdec.py`): a token is cut at the 48 points where the host must decide which
   experts to page in; everything between them is replayed as a CUDA graph over a static KV cache.
5. **Elastic context** (`scripts/chat.py`): VRAM for the KV cache grows only as the conversation grows; the expert
   cache gets the rest. Earlier turns' KV is reused.

## Quick start

Requires an NVIDIA GPU, Linux, ~16 GB free RAM for the 4-bit store (more is better) and the model weights.

```bash
pip install -r requirements.txt
hf download Qwen/Qwen3-30B-A3B --local-dir /data/Qwen3-30B-A3B
export WISP_MODEL=/data/Qwen3-30B-A3B WISP_STORES=/data/stores PYTHONPATH=.

# chat (the first run builds the 4-bit expert store, a few minutes)
WISP_FUSED=2 python scripts/chat.py --bits 4 --vram-gb 8 --graph

# benchmark eager vs CUDA-graph decode
WISP_FUSED=2 python scripts/bench_graph.py --bits 4 --vram-gb 8
```

`--vram-gb` caps how much of the GPU the process may use, so you can emulate a smaller card on a bigger one.
Tests run on CPU (Triton interpreter): `PYTHONPATH=. python -m pytest -q tests`.

Useful flags: `WISP_FUSED=2` (fused kernels, required for `--graph`), `WISP_FASTNORM=0` / `WISP_FASTATTN=0` to disable
the norm/attention patches.

## Status

| Part | Status |
|---|---|
| Qwen3-30B-A3B, 4-bit and 3-bit | measured (numbers above) |
| CUDA-graph decode | measured; quality check partly done |
| Elastic context in `chat.py` | written and unit-tested on CPU, not yet run on a GPU |
| gpt-oss-20b adapter (`wisp/gptoss.py`) | written and unit-tested on CPU, not yet run on a GPU; plain PyTorch path (slow) |
| Long context (32K+), KV quantization or KV paging | not started |
| Comparison with vLLM / llama.cpp offloading | not measured |

## Ringkasan (Bahasa Indonesia)

WiSP Runtime menjalankan model MoE besar (Qwen3-30B-A3B) di GPU dengan VRAM 8 GB. Semua expert disimpan di RAM dalam 4-bit,
dan hanya expert yang dipakai tiap token yang dipindahkan ke GPU, dengan cache untuk yang sering dipakai. Kecepatan
decode naik dari 7,5 ke 22,9 token/detik lewat kernel fused, optimasi overhead CPU, dan CUDA graph. Ini kode riset,
belum siap produksi.

## License

MIT
