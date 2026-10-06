# Results

All numbers are for Qwen3-30B-A3B, batch size 1, 128-token prompts and 64 generated tokens per prompt (greedy),
unless stated otherwise. Two machines were used, and numbers are only comparable within one machine.

| Machine | GPU | CPU / RAM | PCIe (measured) |
|---|---|---|---|
| A: cloud (Vast.ai) | RTX 3090 24 GB | Ryzen 9 7900 (24 threads), 128 GB | 26.7 GB/s |
| B: university server | A100 MIG slice 3g.20gb, capped to 8 GB | slow shared CPU, 32 GB | ≈24 GB/s |

"VRAM X GB" means the process was limited to X GB with `torch.cuda.set_per_process_memory_fraction`, not a card of
that size.

## Decode speed on machine B (8 GB, 4-bit experts, LRU)

| Step | Decode tok/s | Note |
|---|---|---|
| Paging, dequantize to bf16 then matmul | 7.53 | hit rate 76.5% |
| Fused dequant + GEMV kernels (`WISP_FUSED=1`) | 11.78 | hit rate 77.3% |
| One sync per layer, 2 launches (`WISP_FUSED=2`) | 14.73 | hit rate 76.5% |
| + RMSNorm patch | 16.53 | 14.71 without it (+12%) |
| + one-kernel rotary, GQA decode attention | 18.40 | 16.23 without it (+13%) |
| + CUDA graphs | **22.93** | 17.54 eager on the same prompts (1.31×); repeat run 22.82 vs 17.57 |

About 216 MB of experts cross PCIe per token. A single-layer micro-benchmark (8 experts, one token) took 1.65 ms on
the first path and 0.114 ms with the fused kernel, with 0.5% relative difference in the output.

After the fused kernels the GPU was busy only about 1.5 of every 3.2 seconds of decode; the rest was CPU time spent
launching small kernels (profiled with `profile_cpu.py`). The GPU→CPU syncs themselves cost 0.05 s per 40 tokens.
That is why the later steps target launch overhead rather than PCIe.

### 3-bit experts

Measured before the RMSNorm, attention and CUDA-graph steps: 14.82 tok/s at 8 GB, with an 87.5% hit rate and 88 MB
per token over PCIe (4-bit: 76.5% and 216 MB). Speed is the same as 4-bit because the bottleneck is the CPU, not the
transfer; 3-bit should matter more at smaller VRAM, which has not been measured yet.

### CUDA-graph correctness

Teacher-forced comparison of eager and graph decode (6 prompts × 48 steps, same input tokens for both): top-1 token
equal in 98.3% of steps, graph top-1 inside eager top-5 in 100%, largest logit difference 6.56, lowest cosine
similarity 0.92 (a single step). That step is probably a near-tie in expert selection flipped by bf16 rounding, but
this is not confirmed. A perplexity comparison (`ppl_graph.py`) has not been run yet.

## Quality per expert bit-width (machine B)

Perplexity on 30 windows × 512 tokens; MMLU on 200 questions, zero-shot.

| Experts | PPL English | PPL Indonesian | MMLU | Expert size |
|---|---|---|---|---|
| 4-bit | 12.31 | 7.04 | 79.0% | 2.39 MiB |
| 3-bit | 13.04 | 7.56 | 78.0% | 1.83 MiB |
| 2-bit | 41.77 | 31.79 | 22.0% | 1.27 MiB |

2-bit is broken (MMLU below the 25% of random guessing). MMLU on 200 questions has about ±3 points of noise. There is
no bf16 reference because the bf16 model does not fit in 32 GB of RAM.

## Earlier experiments on machine A (non-fused path)

These ran before the fused kernels existed, so absolute speeds are lower than on the path used today.

**H1, paging vs static placement** (bf16 experts, 22 GB, cache holds 2,039 of 6,144 experts; median of 3 runs):

| Policy | Decode tok/s | Hit rate | vs static |
|---|---|---|---|
| Static (pinned from a profile), misses copied | 11.6 | 62.5% | 1.00× |
| Static, misses computed on the CPU | 10.4 | 59.3% | 0.89× |
| Working-set (frequency-based) | 17.7 | 76.9% | 1.52× |
| LRU | 20.0 | 77.5% | 1.72× |

Plain LRU beat the frequency-based policy, so recency matters more than global popularity. All later runs use LRU.

**H2, does PCIe explain the speed?** Modelling time per token as a fixed compute cost plus misses × expert size /
measured PCIe bandwidth predicted 16 held-out configurations (4 VRAM budgets × 5 bit-widths) with a median error of
0.3% (max 2.7%). A free fit gives an effective bandwidth of 25.8 GB/s, R² = 0.998. Compute and transfer did not overlap.

**H3, quantized experts dequantized on every use** (decode tok/s, LRU):

| VRAM | bf16 | 8-bit | 4-bit | 3-bit | 2-bit |
|---|---|---|---|---|---|
| 8 GB | 9.1 | 9.9 | 12.3 | 10.0 | 11.9 |
| 12 GB | 12.1 | 11.5 | 13.3 | 10.4 | 12.1 |
| 16 GB | 14.7 | 12.4 | 13.6 | 10.4 | 12.1 |
| 22 GB | 19.9 | 13.2 | 13.6 | 10.5 | 12.1 |

Quantization only helped at 8 GB: dequantizing on every use raised the fixed cost per token from 33 ms (bf16) to
73 ms (4-bit). This is what the fused kernels later removed.

**H3b, dequantize once on arrival** (experts cross PCIe quantized, are stored as bf16 in the cache): small gains,
for example 3-bit at 16 GB 16.5 tok/s (1.12× bf16), because the per-miss overhead (about 0.15 ms) exceeded the
transfer it saved.

**H4, computing misses on the CPU**: not competitive on machine B, whose CPU is about 10× slower than machine A's.

## Not measured yet

- Comparison with llama.cpp and vLLM CPU offloading on the same machine.
- Speed versus VRAM (4–16 GB) on the fused path, and 3-bit with CUDA graphs.
- Long context (H5), elastic context in `chat.py`, and the gpt-oss adapter on a GPU.
- A real 8 GB card instead of a capped larger GPU.
