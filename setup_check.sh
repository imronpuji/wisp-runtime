#!/bin/bash
# Quick environment check: GPU, tests, PCIe/GPU/CPU microbenchmark, prompt set.
# Usage:  WISP_MODEL=/path/to/Qwen3-30B-A3B WISP_STORES=/path/to/stores bash setup_check.sh
set -e
: "${WISP_MODEL:?set WISP_MODEL to the Hugging Face model directory}"
: "${WISP_STORES:?set WISP_STORES to a directory for the quantized expert stores}"
cd "$(dirname "$0")"
export PYTHONPATH=.
echo "== 1. GPU"
python -c "import torch; f,t=torch.cuda.mem_get_info(0); print(torch.cuda.get_device_name(0), f'| VRAM {t/2**30:.1f} GiB, used {(t-f)/2**30:.1f} GiB')"
echo "== 2. tests"
python -m pytest -q tests 2>&1 | tail -2
echo "== 3. microbenchmark (PCIe / GPU / CPU)"
mkdir -p results
python scripts/microbench.py --bits 16,4 --threads 8 --out results/microbench.json 2>&1 | grep -E "H2D|GPU expert|CPU expert" || true
echo "== 4. prompt set"
[ -f data/prompts.jsonl ] || python scripts/make_prompts.py 2>&1 | tail -1
echo "done"
