"""Build data/prompts.jsonl: mixed domains (English wiki, math, code, Indonesian). Each source is optional."""

import argparse
import json
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40, help="prompts per domain")
ap.add_argument("--out", default="data/prompts.jsonl")
a = ap.parse_args()
from datasets import load_dataset

random.seed(0)
rows = []


def add(domain, texts):
    texts = [t.strip() for t in texts if t and len(t.strip()) > 400]
    random.shuffle(texts)
    for t in texts[: a.n]:
        rows.append(dict(domain=domain, text=t))
    print(f"{domain}: {min(len(texts), a.n)} prompts", flush=True)


def tryload(name, fn):
    try:
        fn()
    except Exception as e:
        print(f"[skip] {name}: {type(e).__name__}: {str(e)[:150]}", flush=True)


def wiki():
    d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    add("en_wiki", [t for t in d["text"] if not t.strip().startswith("=")])


def math():
    d = load_dataset("openai/gsm8k", "main", split="train")
    qs = list(d["question"])
    random.shuffle(qs)
    add(
        "math",
        [" ".join(qs[i : i + 4]) + "\n\nLet's solve step by step.\n" for i in range(0, len(qs) - 4, 4)][: a.n * 2],
    )


def code():
    d = load_dataset("google-research-datasets/mbpp", "full", split="train")
    add(
        "code",
        [
            f"# Task: {t}\n# Write clean, well-documented Python code for the task above.\n" + (c or "")[:900]
            for t, c in zip(d["text"], d["code"], strict=True)
        ],
    )


def indo():
    d = load_dataset("wikimedia/wikipedia", "20231101.id", split="train", streaming=True)
    buf = []
    for r in d:
        buf.append(r["text"][:1500])
        if len(buf) >= a.n * 3:
            break
    add("id_wiki", buf)


for n, f in [("wikitext", wiki), ("gsm8k", math), ("mbpp", code), ("wikipedia-id", indo)]:
    tryload(n, f)
random.shuffle(rows)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    for i, r in enumerate(rows):
        f.write(json.dumps(dict(id=i, **r), ensure_ascii=False) + "\n")
print("wrote", len(rows), "prompts ->", a.out)
