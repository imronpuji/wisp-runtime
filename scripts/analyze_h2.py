"""H2: is decode speed explained by bytes over PCIe?
Model: t_token = t0(bits) + misses_per_token * expert_bytes / BW, BW = measured PCIe (not fitted).
t0(bits) is calibrated on ONE budget (--calib, default 22 GB); the other budgets are out-of-sample predictions.
Also reports a free fit (t0 per bits + one shared slope) -> fitted 'effective bandwidth'."""
import argparse, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--phase", default="h3"); ap.add_argument("--bw", type=float, default=None)
ap.add_argument("--calib", type=float, default=22)
a = ap.parse_args()
bw = a.bw or json.load(open("results/microbench.json"))["h2d"]["9.4"]["gb_s"]
df = pd.DataFrame([json.loads(l) for l in open(f"results/{a.phase}.jsonl")])
df = df[df.status == "ok"]
g = df.groupby(["bits", "vram_gb"]).agg(tok_s=("decode_tok_s", "median"), miss=("dec_misses_per_token", "median"),
                                        mib=("expert_mib", "first"), hit=("dec_hit_rate", "median"),
                                        fits=("fits_all", "first")).reset_index()
g["t_ms"] = 1e3 / g.tok_s
g["xfer_ms"] = g.miss * g.mib * 2 ** 20 / (bw * 1e9) * 1e3
rows = []
for b, gb in g.groupby("bits"):
    cal = gb[gb.vram_gb == a.calib]
    if cal.empty:
        continue
    t0 = float(cal.t_ms.iloc[0] - cal.xfer_ms.iloc[0])
    for _, r in gb.iterrows():
        pred = t0 + r.xfer_ms
        rows.append(dict(bits=b, vram_gb=r.vram_gb, hit=r.hit, miss_tok=r.miss, MB_tok=r.miss * r.mib * 1.048576,
                         xfer_ms=r.xfer_ms, t0_ms=t0, meas_tok_s=r.tok_s, pred_tok_s=1e3 / pred,
                         err_pct=100 * (1e3 / pred - r.tok_s) / r.tok_s, calib=(r.vram_gb == a.calib)))
t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"PCIe bandwidth used: {bw:.1f} GB/s (measured, not fitted)\n")
print(t.round(2).to_string(index=False))
oos = t[~t.calib]
print(f"\nout-of-sample |error|: median {oos.err_pct.abs().median():.1f}%  max {oos.err_pct.abs().max():.1f}%  "
      f"(H2 criterion: every point < 20%) -> {'ACCEPT' if (oos.err_pct.abs() < 20).all() else 'REJECT'}")
# free fit: t = t0_bits + slope * bytes
X = np.column_stack([pd.get_dummies(g.bits).values.astype(float), g.miss * g.mib * 2 ** 20 / 1e9])
coef, *_ = np.linalg.lstsq(X, g.t_ms.values / 1e3, rcond=None)
pred = X @ coef
r2 = 1 - ((g.t_ms / 1e3 - pred) ** 2).sum() / ((g.t_ms / 1e3 - (g.t_ms / 1e3).mean()) ** 2).sum()
print(f"free fit: effective bandwidth = {1/coef[-1]:.1f} GB/s, R^2 = {r2:.3f}")
t.to_csv(f"results/{a.phase}_h2.csv", index=False)
