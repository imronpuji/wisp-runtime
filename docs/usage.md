# Usage

## Install

Linux with an NVIDIA GPU. The 4-bit expert store of Qwen3-30B-A3B needs about 15 GB of free RAM, plus the model
download (about 60 GB) to build it once.

```bash
git clone https://github.com/imronpuji/wisp-runtime.git
cd wisp-runtime
pip install -e .            # or: pip install -e ".[eval,dev]" for the evaluation scripts and tests

hf download Qwen/Qwen3-30B-A3B --local-dir /data/Qwen3-30B-A3B
export WISP_MODEL=/data/Qwen3-30B-A3B   # model directory
export WISP_STORES=/data/stores         # where quantized expert stores are cached
```

The first run of any script quantizes the experts and saves the store in `$WISP_STORES` (a few minutes). Later runs
load it in seconds. The original checkpoint is still needed for the non-expert weights.

`bash setup_check.sh` prints the GPU, runs the tests, measures PCIe and per-expert costs, and builds the prompt set
used by the benchmarks (`data/prompts.jsonl`).

## Chat

```bash
python scripts/chat.py --bits 4 --vram-gb 8 --ctx 32768 --graph
```

| Option | Default | Meaning |
|---|---|---|
| `--bits` | 4 | expert bit-width (4 recommended, 3 slightly lower quality, 8/16 slower) |
| `--vram-gb` | 8 | GPU memory the process may use (emulates a smaller card on a bigger one) |
| `--ctx` | 8192 | maximum context; memory is only taken as the chat grows |
| `--graph` | off | CUDA-graph decode, the fastest path |
| `--max-new` | 300 | maximum tokens per answer |
| `--temp` | 0.7 | sampling temperature, 0 = greedy |
| `--think` | off | Qwen3 thinking mode |

Type `/reset` to clear the conversation and `exit` to quit. After each answer the chat prints speed, how many prompt
tokens were new versus reused, and the context in use.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `WISP_MODEL` | none | model directory (scripts also accept `--model-dir`) |
| `WISP_STORES` | `./stores` | cache directory for quantized expert stores |
| `WISP_FUSED` | `2` in chat and graph scripts, `0` elsewhere | `0` plain PyTorch path, `1` first fused kernels, `2` fused kernels with one sync per layer |
| `WISP_FASTNORM` | `1` | `0` disables the RMSNorm patch |
| `WISP_FASTATTN` | `1` | `0` disables the rotary and decode-attention patch |
| `WISP_COMPILE` | `1` | `0` disables `torch.compile` for dequantization on the non-fused path |

## Scripts

### Everyday

| Script | What it does |
|---|---|
| `chat.py` | interactive chat |
| `bench_graph.py` | decode speed, eager vs CUDA graph, on the same prompts, plus token agreement |
| `bench.py` | one configuration (bits, VRAM, cache policy, miss policy) → `results/bench.jsonl` |

### Checking correctness

| Script | What it does |
|---|---|
| `check_graph.py` | teacher-forced logits, eager vs CUDA graph (top-1 agreement, cosine similarity) |
| `ppl_graph.py` | perplexity on real text, eager vs CUDA graph |
| `eval_quality.py` | perplexity (English and Indonesian) and MMLU per bit-width → `results/quality.jsonl` |
| `sanity_check.py` | paged MoE vs the Hugging Face reference on a tiny model, plus a sabotage control |

### Profiling and micro-benchmarks

| Script | What it does |
|---|---|
| `profile_cpu.py` | cProfile over decode steps: where the CPU time goes |
| `profile_decode.py` | per-layer attention vs MoE time and top CUDA ops |
| `microbench.py` | PCIe bandwidth, GPU and CPU cost per expert |
| `fused_bench.py` | fused kernels vs dequantize + matmul, one layer |
| `dequant_bench.py` | eager vs compiled dequantization |

### Research experiments (hypotheses H1–H5)

| Script | What it does |
|---|---|
| `make_prompts.py` | builds `data/prompts.jsonl` (English, math, code, Indonesian) |
| `run_grid.py --phase h1` | runs a whole experiment grid (`h1`, `h3`, `h3b`, `h4`, `h5`), resumable, → `results/<phase>.jsonl` |
| `collect_trace.py` | records which experts every decode step uses → `data/trace_b{bits}.npz` |
| `analyze.py` | median and spread per cell, speed-up vs a baseline, Welch t-test |
| `analyze_h2.py` | checks whether decode time is explained by bytes over PCIe |

## Tests

```bash
python -m pytest -q
```

Tests run on the CPU through the Triton interpreter. GPU-only tests are skipped there and run automatically on a
machine with CUDA.

## Using the library directly

```python
import os

os.environ["WISP_FUSED"] = "2"  # fused kernels; read when the runtime is created

from wisp import runner
from wisp.budget import plan_slots
from wisp.graphdec import GraphDecoder
from wisp.model import configure, nbytes_of

model, cfg, rt, tok = runner.setup("/data/Qwen3-30B-A3B", bits=4)
slots = plan_slots(8, nbytes_of(model), rt.store, cfg, 4096, kv_mode="adaptive")["slots"]
configure(rt, slots, "lru", "transfer", vram_cap_gb=8)

dec = GraphDecoder(model, rt, tmax=4096)
ids = tok("The capital of Indonesia is", return_tensors="pt").input_ids[0]
logits = dec.prefill(ids)
for _ in range(20):
    nxt = logits.argmax(-1)
    print(tok.decode(nxt), end="", flush=True)
    logits = dec.next(nxt)
```
