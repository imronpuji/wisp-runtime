"""Summarise results/<phase>.jsonl: median/IQR per cell, speed-up vs a baseline, Welch t-test.
  python scripts/analyze.py --phase h1 --baseline policy=static,miss=transfer
"""
import argparse, json
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True)
ap.add_argument("--baseline", default=None, help="k=v,k=v selecting the baseline cell")
ap.add_argument("--keys", default="bits,vram_gb,policy,miss,prefetch,ctx,kv_mode")
ap.add_argument("--metric", default="decode_tok_s")
a = ap.parse_args()

df = pd.DataFrame([json.loads(l) for l in open(f"results/{a.phase}.jsonl")])
df = df[df.status == "ok"].copy()
keys = [k for k in a.keys.split(",") if k in df.columns]
for k in keys:
    df[k] = df[k].astype(str)
g = df.groupby(keys)
tab = g.agg(n=(a.metric, "size"), median=(a.metric, "median"),
            q1=(a.metric, lambda s: s.quantile(.25)), q3=(a.metric, lambda s: s.quantile(.75)),
            hit_rate=("hit_rate", "median"), h2d_MB_tok=("h2d_mb_per_token", "median"),
            cpu_exp_tok=("cpu_experts_per_token", "median"), slots=("slots", "median"),
            peak_vram=("peak_vram_gib", "median")).reset_index()
if a.baseline:
    cond = dict(kv.split("=") for kv in a.baseline.split(","))
    m = np.ones(len(df), bool)
    for k, v in cond.items():
        m &= (df[k] == v).values
    base = df[m]
    # baseline per remaining group (e.g. per bits / vram), matched on keys not in the condition
    match = [k for k in keys if k not in cond]
    sp, pv = [], []
    for _, row in tab.iterrows():
        b = base
        for k in match:
            b = b[b[k] == row[k]]
        cur = df
        for k in keys:
            cur = cur[cur[k] == row[k]]
        if len(b) == 0:
            sp.append(np.nan); pv.append(np.nan); continue
        sp.append(row["median"] / b[a.metric].median())
        x, y = cur[a.metric].values, b[a.metric].values
        if len(x) > 1 and len(y) > 1 and (x.std() > 0 or y.std() > 0):
            pv.append(stats.ttest_ind(x, y, equal_var=False).pvalue)
        else:
            pv.append(np.nan)
    tab["speedup"] = sp
    tab["p_welch"] = pv
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print(tab.round(4).to_string(index=False))
tab.to_csv(f"results/{a.phase}_summary.csv", index=False)
