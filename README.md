# WiSP Runtime

[![ci](https://github.com/imronpuji/wisp-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/imronpuji/wisp-runtime/actions/workflows/ci.yml)

**Run a 30B Mixture-of-Experts LLM on 8 GB of VRAM at about 23 tokens per second.**

Qwen3-30B-A3B has 30B parameters but uses only about 3B for each token. WiSP Runtime keeps all experts in CPU RAM
(4-bit) and copies to the GPU only the experts a token needs, caching the recently used ones in VRAM, the way an
operating system pages memory. Attention, embeddings and routers stay on the GPU.

It is an inference runtime in the same family as vLLM and llama.cpp, but much narrower: experimental, one user,
batch size 1, no API server, and one model family so far (Qwen3 MoE). It runs on top of PyTorch and Hugging Face
Transformers and replaces their MoE layer, attention and decode loop.

## Results

Qwen3-30B-A3B, 4-bit experts, A100 MIG slice capped to 8 GB, 32 GB host RAM, greedy decode:

| Step | Decode tok/s |
|---|---|
| Expert paging, dequantize then matmul | 7.5 |
| Fused 4-bit dequant + GEMV Triton kernels | 11.8 |
| One host sync per layer, 2 kernel launches | 14.7 |
| + RMSNorm patch | 16.5 |
| + one-kernel rotary, GQA decode attention | 18.4 |
| + CUDA graphs | **22.9** |

Quality holds at 4-bit (MMLU 79.0%) and 3-bit (78.0%); 2-bit breaks (22.0%). The 8 GB limit is emulated on a larger
GPU, and there is no head-to-head comparison with llama.cpp or vLLM yet. Full numbers, methods and caveats:
[docs/results.md](docs/results.md).

## Quick start

Linux, an NVIDIA GPU, and about 15 GB of free RAM for the 4-bit experts.

```bash
git clone https://github.com/imronpuji/wisp-runtime.git && cd wisp-runtime
pip install -e .
hf download Qwen/Qwen3-30B-A3B --local-dir /data/Qwen3-30B-A3B
export WISP_MODEL=/data/Qwen3-30B-A3B WISP_STORES=/data/stores

python scripts/chat.py --bits 4 --vram-gb 8 --graph          # chat (first run builds the 4-bit store)
python scripts/bench_graph.py --bits 4 --vram-gb 8           # benchmark
```

`--vram-gb` limits how much GPU memory the process may use, so a bigger card can emulate a smaller one.
More options, every script and the environment variables: [docs/usage.md](docs/usage.md).

## How it works

1. **Expert store**: every expert is quantized (group-wise, 4/3/2-bit) into one buffer in pinned CPU RAM.
2. **Expert cache**: a pool of GPU slots with LRU eviction; missing experts are copied asynchronously.
3. **Fused kernels**: Triton kernels compute all 8 selected experts straight from the 4-bit slots in 2 launches.
4. **CUDA graphs**: a token has 48 points where the CPU must decide what to page in; everything between them is
   replayed as a CUDA graph.
5. **Elastic context**: KV-cache memory grows with the conversation instead of being reserved up front, and earlier
   turns are not recomputed.

Details: [docs/architecture.md](docs/architecture.md).

## Status

| Part | Status |
|---|---|
| Qwen3-30B-A3B, 4-bit and 3-bit experts | measured |
| CUDA-graph decode | measured; perplexity check pending |
| Elastic context in `chat.py` | unit-tested on CPU, not yet run on a GPU |
| gpt-oss-20b adapter | unit-tested on CPU, not yet run on a GPU; slow PyTorch path |
| Long context (32K+), KV quantization, KV paging | not started |
| Comparison with llama.cpp and vLLM offloading | not measured |

## Project layout

```
wisp/       the runtime (store, cache, paged MoE layer, kernels, CUDA-graph decoder)
scripts/    chat, benchmarks, correctness checks, research experiments
tests/      unit tests (run on CPU with the Triton interpreter)
docs/       architecture, usage, results
```

## Background

The project started as a replication of *WiSP: A Working-Set View of Mixture-of-Experts Serving on Extremely
Low-Resource Hardware* ([arXiv:2606.21868](https://arxiv.org/abs/2606.21868)) and is named after it. It is an
independent implementation and is not affiliated with the paper's authors.

## Ringkasan (Bahasa Indonesia)

WiSP Runtime menjalankan model MoE besar (Qwen3-30B-A3B) di GPU dengan VRAM 8 GB. Semua expert disimpan di RAM dalam
4-bit, dan hanya expert yang dipakai tiap token yang dipindahkan ke GPU, dengan cache untuk yang sering dipakai.
Kecepatan decode naik dari 7,5 ke 22,9 token/detik lewat kernel fused, pengurangan overhead CPU, dan CUDA graph.
Ini kode riset, belum siap produksi.

## License

[MIT](LICENSE)
