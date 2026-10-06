"""WiSP Runtime: run large Mixture-of-Experts LLMs on small GPUs by paging experts from CPU RAM.

Main pieces: ExpertStore (quantized experts in pinned RAM), ExpertCache (GPU slot pool), PagedMoEBlock
(drop-in MoE layer), fused Triton kernels, and GraphDecoder (CUDA-graph decode with an elastic KV cache).
"""

__version__ = "0.1.0"
