"""Quality per bit-width (paging is lossless, so quality depends only on `bits`).
  - perplexity on WikiText-2 (en) and Indonesian Wikipedia (id): N windows x W tokens
  - MMLU (+ IndoMMLU if loadable) zero-shot multiple choice via next-token logits of ' A'..' D' (or more)
Writes results/quality.jsonl (one line per bits)."""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn.functional as F
from datasets import load_dataset

from wisp import runner
from wisp.budget import plan_slots
from wisp.model import configure, nbytes_of

ap = argparse.ArgumentParser()
ap.add_argument("--bits", default="16,8,4,3,2")
ap.add_argument("--windows", type=int, default=40)
ap.add_argument("--wlen", type=int, default=512)
ap.add_argument("--mmlu", type=int, default=300)
ap.add_argument("--indommlu", type=int, default=300)
ap.add_argument("--model-dir", default=os.environ.get("WISP_MODEL"))
ap.add_argument("--out", default="results/quality.jsonl")
a = ap.parse_args()
random.seed(0)


def windows(texts, tok, n, w):
    ids = tok("\n\n".join(texts), return_tensors="pt").input_ids[0]
    step = max(w, (len(ids) - w) // n)
    return [ids[i : i + w] for i in range(0, len(ids) - w, step)][:n]


@torch.no_grad()
def ppl(model, wins):
    nll, cnt = 0.0, 0
    for w in wins:
        x = w[None].cuda()
        logits = model(input_ids=x).logits.float()
        nll += F.cross_entropy(logits[0, :-1], x[0, 1:], reduction="sum").item()
        cnt += x.shape[1] - 1
    return float(torch.exp(torch.tensor(nll / cnt)))


@torch.no_grad()
def mc(model, tok, items):
    """items: (prompt, n_choices, gold_index). Answer letters A.. scored by next-token logit."""
    letters = "ABCDE"
    lid = [tok(" " + l, add_special_tokens=False).input_ids[-1] for l in letters]
    ok = 0
    for p, k, g in items:
        x = tok(p, return_tensors="pt").input_ids[:, -1024:].cuda()
        lg = model(input_ids=x, logits_to_keep=1).logits[0, -1, lid[:k]]
        ok += int(lg.argmax().item() == g)
    return 100 * ok / len(items)


def fmt(q, ch, subj=None, lang="en"):
    head = (
        f"The following is a multiple choice question about {subj.replace('_', ' ')}.\n\n"
        if lang == "en"
        else "Berikut adalah soal pilihan ganda.\n\n"
    )
    body = q.strip() + "\n" + "".join(f"{'ABCDE'[i]}. {c}\n" for i, c in enumerate(ch))
    return head + body + ("Answer:" if lang == "en" else "Jawaban:")


# ---------- datasets (loaded once) ----------
tok = None
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(a.model_dir)
wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
en_w = windows([t for t in wt if t.strip()], tok, a.windows, a.wlen)
idw = []
for i, r in enumerate(load_dataset("wikimedia/wikipedia", "20231101.id", split="train", streaming=True)):
    if i < 2000:
        continue  # skip the first articles (used for bench prompts)
    idw.append(r["text"])
    if len(idw) >= 300:
        break
id_w = windows(idw, tok, a.windows, a.wlen)
mm = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0).select(range(a.mmlu))
mmlu_items = [(fmt(r["question"], r["choices"], r["subject"]), len(r["choices"]), r["answer"]) for r in mm]
indo_items = []
try:
    d = load_dataset("indolem/IndoMMLU", split="test").shuffle(seed=0)
    for r in d:
        opts = (
            [o.split(".", 1)[-1].strip() for o in eval(r["options"])] if isinstance(r["options"], str) else r["options"]
        )
        key = r["answer"].strip().upper()
        if not 2 <= len(opts) <= 5 or key not in "ABCDE"[: len(opts)]:
            continue
        indo_items.append((fmt(r["question"], opts, lang="id"), len(opts), "ABCDE".index(key)))
        if len(indo_items) >= a.indommlu:
            break
except Exception as e:
    print(f"[skip] IndoMMLU: {type(e).__name__}: {str(e)[:200]}", flush=True)
print(
    f"data: en windows={len(en_w)} id windows={len(id_w)} mmlu={len(mmlu_items)} indommlu={len(indo_items)}", flush=True
)

done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["bits"] for l in open(a.out)}
for bits in [int(b) for b in a.bits.split(",")]:
    if bits in done:
        continue
    t0 = time.time()
    model, cfg, rt, _ = runner.setup(a.model_dir, bits, cpu_threads=12)
    slots = plan_slots(22, nbytes_of(model), rt.store, cfg, a.wlen, chunk=8)["slots"]
    configure(rt, slots, "lru", "transfer", vram_cap_gb=22)
    r = dict(bits=bits, ppl_en=ppl(model, en_w), ppl_id=ppl(model, id_w), mmlu=mc(model, tok, mmlu_items))
    if indo_items:
        r["indommlu"] = mc(model, tok, indo_items)
    r.update(
        n_mmlu=len(mmlu_items), n_indommlu=len(indo_items), n_windows=len(en_w), wlen=a.wlen, eval_s=time.time() - t0
    )
    print(json.dumps(r), flush=True)
    open(a.out, "a").write(json.dumps(r) + "\n")
    del model, rt
    torch.cuda.empty_cache()
