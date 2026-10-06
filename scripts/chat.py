"""Interactive chat with a paged MoE model.

    python scripts/chat.py --bits 4 --vram-gb 8 --ctx 32768 --graph

Commands: 'exit' quits, '/reset' clears the conversation.

For Qwen3 with 3/4-bit experts the chat uses the elastic decoder (wisp/graphdec.py):
  * --ctx is only the maximum. KV-cache memory is taken as the conversation actually grows (capacity steps
    4K -> 8K -> 16K -> ...) and the expert cache gets the rest, so a short chat is as fast as with a small --ctx.
  * The KV of earlier turns is reused: each turn only prefills the new tokens.
  * --graph replays each decode step as CUDA graphs (fastest).
Other models (gpt-oss) or bit-widths use the plain Hugging Face decode loop.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("WISP_FUSED", "2")  # fused Triton kernels unless the user asks otherwise
import torch

from wisp import runner
from wisp.budget import plan_slots
from wisp.model import configure, nbytes_of

FIRST_CAPACITY = 4096


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.environ.get("WISP_MODEL"), help="model directory (default: $WISP_MODEL)")
    ap.add_argument("--bits", type=int, default=4, help="expert bit-width: 4 (default), 3, 8 or 16")
    ap.add_argument("--vram-gb", type=float, default=8, help="GPU memory the process may use")
    ap.add_argument("--ctx", type=int, default=8192, help="maximum context length in tokens")
    ap.add_argument("--max-new", type=int, default=300, help="maximum tokens per answer")
    ap.add_argument("--temp", type=float, default=0.7, help="sampling temperature (0 = greedy)")
    ap.add_argument("--graph", action="store_true", help="CUDA-graph decode (Qwen3, 3/4-bit)")
    ap.add_argument("--think", action="store_true", help="Qwen3 thinking mode (slower, longer answers)")
    ap.add_argument("--effort", default="low", help="gpt-oss reasoning effort: low | medium | high")
    return ap.parse_args()


def sample(logits, temp, top_k=40):
    if temp <= 0:
        return logits.argmax(-1)
    p = torch.softmax(logits.float() / temp, -1)
    top, idx = p.topk(top_k, -1)
    return idx.gather(-1, torch.multinomial(top / top.sum(-1, keepdim=True), 1)).squeeze(-1)


class Streamer:
    """Prints generated tokens as they come, never splitting a multi-byte character."""

    def __init__(self, tok):
        self.tok, self.ids, self.shown = tok, [], ""

    def push(self, t):
        self.ids.append(t)
        s = self.tok.decode(self.ids)
        if not s.endswith("�"):
            print(s[len(self.shown) :], end="", flush=True)
            self.shown = s


class ElasticChat:
    """Qwen3 + fused kernels: static KV buffers that grow with the conversation, KV reuse across turns."""

    def __init__(self, a, model, cfg, rt, tok, stop):
        from wisp.graphdec import GraphDecoder

        self.a, self.cfg, self.rt, self.tok, self.stop = a, cfg, rt, tok, stop
        self.nonexpert = nbytes_of(model)
        self.total = rt.store.L * rt.store.E
        cap = min(FIRST_CAPACITY, a.ctx)
        configure(rt, self.slots_for(cap), "lru", "transfer", vram_cap_gb=a.vram_gb)
        self.dec = GraphDecoder(model, rt, tmax=cap, use_graph=a.graph)
        self.cached = []  # token ids whose K/V are in the decoder's buffers, in order

    def slots_for(self, capacity):
        return plan_slots(self.a.vram_gb, self.nonexpert, self.rt.store, self.cfg, capacity, kv_mode="adaptive")[
            "slots"
        ]

    def set_capacity(self, need):
        """Smallest capacity step >= need. The expert cache shrinks BEFORE the KV grows (and the reverse), so the
        two never hold their larger size at the same time."""
        cap = FIRST_CAPACITY
        while cap < need:
            cap *= 2
        cap = min(max(cap, need), self.a.ctx)
        if cap == self.dec.tmax:
            return
        slots = self.slots_for(cap)
        if slots < 2 * self.cfg.num_experts_per_tok:
            raise SystemExit(f"{self.a.vram_gb} GB of VRAM is not enough for {cap} tokens of context")
        t = time.perf_counter()
        if cap > self.dec.tmax:
            self.rt.cache.resize(slots)
            self.dec.resize(cap)
        else:
            self.dec.resize(cap)
            self.rt.cache.resize(slots)
        print(
            f"[context] capacity {cap} tokens, expert cache {slots} slots ({slots / self.total:.0%}) "
            f"[{time.perf_counter() - t:.1f}s]",
            flush=True,
        )

    def reset(self):
        self.cached = []
        self.dec.pos = 0
        self.set_capacity(1)

    @torch.no_grad()
    def reply(self, messages):
        a, tok = self.a, self.tok
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=a.think)
        ids = tok(text).input_ids[-(a.ctx - a.max_new) :]
        reuse = 0
        while reuse < min(len(ids), len(self.cached)) and ids[reuse] == self.cached[reuse]:
            reuse += 1
        reuse = min(reuse, len(ids) - 1)  # always run at least one token to get logits
        self.set_capacity(len(ids) + a.max_new)

        t0 = time.perf_counter()
        nxt = sample(self.dec.prefill(torch.tensor(ids[reuse:]), reuse), a.temp)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = Streamer(tok)
        for _ in range(a.max_new):
            t = nxt.item()
            if t in self.stop:
                break
            out.push(t)
            nxt = sample(self.dec.next(nxt), a.temp)
        dt = time.perf_counter() - t1
        self.cached = ids + out.ids
        n = len(out.ids)
        print(
            f"\n[{n} tokens, {n / max(dt, 1e-9):.1f} tok/s | prefill {len(ids) - reuse} new tokens "
            f"({reuse} reused) {t1 - t0:.1f}s | context {len(self.cached)}/{self.dec.tmax}]\n"
        )
        return tok.decode(out.ids)


class SimpleChat:
    """Any supported model: Hugging Face decode loop, KV recomputed each turn."""

    def __init__(self, a, model, cfg, rt, tok, stop):
        self.a, self.model, self.tok, self.stop = a, model, tok, stop
        self.gpt_oss = cfg.model_type == "gpt_oss"
        slots = plan_slots(a.vram_gb, nbytes_of(model), rt.store, cfg, a.ctx)["slots"]
        configure(rt, slots, "lru", "transfer", vram_cap_gb=a.vram_gb)

    def reset(self):
        pass

    @torch.no_grad()
    def reply(self, messages):
        a, tok = self.a, self.tok
        kw = dict(reasoning_effort=a.effort) if self.gpt_oss else dict(enable_thinking=a.think)
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        ids = tok(text, return_tensors="pt").input_ids[:, -(a.ctx - a.max_new) :].cuda()
        t0 = time.perf_counter()
        res = self.model(input_ids=ids, use_cache=True, logits_to_keep=1)
        nxt, pk = sample(res.logits[:, -1], a.temp), res.past_key_values
        t1 = time.perf_counter()
        out = Streamer(tok)
        for _ in range(a.max_new):
            t = nxt.item()
            if t in self.stop:
                break
            out.push(t)
            res = self.model(input_ids=nxt[:, None], past_key_values=pk, use_cache=True)
            nxt, pk = sample(res.logits[:, -1], a.temp), res.past_key_values
        dt = time.perf_counter() - t1
        n = len(out.ids)
        print(f"\n[{n} tokens, {n / max(dt, 1e-9):.1f} tok/s | prefill {t1 - t0:.1f}s]\n")
        full = tok.decode(out.ids, skip_special_tokens=False)
        if self.gpt_oss and "<|channel|>final<|message|>" in full:  # keep the answer, drop the reasoning
            return full.split("<|channel|>final<|message|>")[-1].replace("<|return|>", "").strip()
        return tok.decode(out.ids)


def main():
    a = parse_args()
    model, cfg, rt, tok = runner.setup(a.model_dir, a.bits, cpu_threads=12)
    if cfg.model_type == "gpt_oss":
        stop = {tok.convert_tokens_to_ids(t) for t in ("<|return|>", "<|call|>")} | {tok.eos_token_id}
    else:
        stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}

    elastic = cfg.model_type != "gpt_oss" and rt.fused == 2 and a.bits in (3, 4)
    if a.graph and not elastic:
        raise SystemExit("--graph needs a Qwen3 model, 3/4-bit experts and WISP_FUSED=2")
    chat = (ElasticChat if elastic else SimpleChat)(a, model, cfg, rt, tok, stop)

    mode = "elastic context" + (", CUDA graphs" if a.graph else "") if elastic else "simple"
    print(f"\nReady: {a.bits}-bit experts, {a.vram_gb} GB VRAM, max context {a.ctx}, {mode}. Type 'exit' to quit.\n")
    history = []
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("exit", "quit"):
            break
        if q == "/reset":
            history = []
            chat.reset()
            print("(conversation cleared)\n")
            continue
        if not q:
            continue
        history.append({"role": "user", "content": q})
        print("Model: ", end="", flush=True)
        history.append({"role": "assistant", "content": chat.reply(history)})


if __name__ == "__main__":
    main()
