#!/usr/bin/env python3
"""
timeseries_ensemble_summary_N4.py (v0.1.1)

N=4 companion to the N=8 ensemble summary.
Locks colours to Fig. 2 base (S_perc=C0, F_max(w=0.20)=C4) and uses Fig. 3c / Fig. 3d titles.

Run from Desktop:
  python3 bcqm_vi_spacetime/analysis/timeseries_ensemble_summary_N4.py
"""
from __future__ import annotations
import json
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

W = 100
N = 4

ROOTS = [
    ("n0p4", "n=0.4", "outputs_glue_axes/timeseries_ens_C5/W100/N4/n0p4_s56791_56795/spaceon", "Fig. 3c"),
    ("n0p8", "n=0.8", "outputs_glue_axes/timeseries_ens_C5/W100/N4/n0p8_s56791_56795/spaceon", "Fig. 3d"),
]

C_S = "C0"  # S_perc
C_F = "C4"  # F_max(w=0.20)

def load_metrics(folder: Path):
    files = sorted(Path(p) for p in glob(str(folder / "RUN_METRICS_*.json")))
    if not files:
        raise SystemExit(f"No RUN_METRICS_*.json in {folder}")
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]

def extract_timeseries(m):
    ts = (m.get("timeseries") or {})
    recs = ts.get("records") or []
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    f = pd.json_normalize(df["F_max_by_wstar"]).rename(columns=lambda c: f"Fmax_{c}")
    df = pd.concat([df.drop(columns=["F_max_by_wstar"]), f], axis=1)
    return df

def qstats(x):
    x = np.asarray(x, dtype=float)
    return np.nanmedian(x), np.nanpercentile(x, 25), np.nanpercentile(x, 75)

def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth=0.8)

def plot_band(ax, t, med, q1, q3, label, color):
    t = np.asarray(t, dtype=float)
    med = np.asarray(med, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q3 = np.asarray(q3, dtype=float)
    ax.plot(t, med, label=label, color=color)
    ax.fill_between(t, q1, q3, alpha=0.25, color=color)

def main():
    out_csv_dir = Path("outputs") / "timeseries_ensemble"
    out_fig_dir = Path("figures")
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    for tag, nlabel, rel, figlabel in ROOTS:
        folder = Path(rel)
        mets = load_metrics(folder)
        dfs = [extract_timeseries(m) for m in mets]
        dfs = [d for d in dfs if not d.empty]
        if not dfs:
            raise SystemExit(f"No timeseries records found in {folder}")

        tvals = sorted(set(dfs[0]["t"].astype(int).tolist()))
        rows = []
        for t in tvals:
            vals_S, vals_F, vals_C = [], [], []
            for df in dfs:
                sub = df[df["t"] == t]
                if sub.empty:
                    continue
                vals_S.append(float(sub["S_perc"].iloc[0]))
                vals_F.append(float(sub["Fmax_0.20"].iloc[0]) if "Fmax_0.20" in sub.columns else np.nan)
                vals_C.append(float(sub["comp_size"].iloc[0]))

            medS, q1S, q3S = qstats(vals_S)
            medF, q1F, q3F = qstats(vals_F)
            medC, q1C, q3C = qstats(vals_C)
            rows.append({
                "t": int(t),
                "S_perc_med": medS, "S_perc_q1": q1S, "S_perc_q3": q3S,
                "Fmax020_med": medF, "Fmax020_q1": q1F, "Fmax020_q3": q3F,
                "comp_size_med": medC, "comp_size_q1": q1C, "comp_size_q3": q3C,
                "n_seeds": int(len(vals_S)),
            })

        out = pd.DataFrame(rows)
        out.to_csv(out_csv_dir / f"timeseries_ensemble_{tag}_W{W}_N{N}.csv", index=False)

        plt.figure(figsize=(8.5, 5.5))
        ax = plt.gca()
        plot_band(ax, out["t"], out["S_perc_med"], out["S_perc_q1"], out["S_perc_q3"], "S_perc(t) median±IQR", C_S)
        plot_band(ax, out["t"], out["Fmax020_med"], out["Fmax020_q1"], out["Fmax020_q3"], "F_max(w=0.20) median±IQR", C_F)
        ax.set_title(f"Space vs islands (ensemble) ({nlabel}, W={W}, N={N})")
        ax.set_xlabel("Tick t (binned)")
        ax.set_ylabel("Order parameter (0–1)")
        ax.set_ylim(-0.02, 1.05)
        style_axes(ax)
        ax.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        fname = "fig_3c_space_vs_islands_ensemble_n0p4_W100_N4.pdf" if tag=="n0p4" else "fig_3d_space_vs_islands_ensemble_n0p8_W100_N4.pdf"
        plt.savefig(out_fig_dir / fname, format="pdf")
        plt.close()

    print("Wrote CSVs to outputs/timeseries_ensemble and figures to figs/")

if __name__ == "__main__":
    main()
