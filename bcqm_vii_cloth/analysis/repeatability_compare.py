#!/usr/bin/env python3
"""
BCQM VII Stage-2 repeatability check (two consecutive runs with different seeds).

You should have already run two scan configs (repA and repB) and produced:
  csv/repeatability/repA_run_summary.csv
  csv/repeatability/repB_run_summary.csv
  csv/repeatability/gate4_repA_hopdist_seedwise.csv
  csv/repeatability/gate4_repB_hopdist_seedwise.csv

This script writes:
  csv/repeatability/repeatability_compare_table.csv
  figures/fig_repeatability_repA_vs_repB.pdf
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def apply_bcqm_axes_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth=0.8)

def mean_sd(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return (np.nan, np.nan)
    return (float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repA_summary", default="csv/repeatability/repA_run_summary.csv")
    ap.add_argument("--repB_summary", default="csv/repeatability/repB_run_summary.csv")
    ap.add_argument("--repA_hops", default="csv/repeatability/gate4_repA_hopdist_seedwise.csv")
    ap.add_argument("--repB_hops", default="csv/repeatability/gate4_repB_hopdist_seedwise.csv")
    ap.add_argument("--out_csv", default="csv/repeatability/repeatability_compare_table.csv")
    ap.add_argument("--out_fig", default="figures/fig_repeatability_repA_vs_repB.pdf")
    args = ap.parse_args()

    repA = pd.read_csv(args.repA_summary)
    repB = pd.read_csv(args.repB_summary)

    # Add derived core fraction Phi
    for df in (repA, repB):
        df["N_total_events"] = df["core_events_count"] + df["halo_events_count"]
        df["phi_core"] = df["core_events_count"] / df["N_total_events"]

    # Metrics from run summaries
    metrics = [
        ("phi_core", "Core fraction Φ"),
        ("Q_clock", "Clock quality Q_clock"),
        ("core_events_count", "Core events"),
        ("halo_events_count", "Halo events"),
        ("core_edges_count", "Core edges"),
        ("halo_edges_count", "Halo edges"),
        ("ball_comp_size", "Ball component size"),
    ]

    rows = []
    for col, label in metrics:
        a_m, a_s = mean_sd(repA[col])
        b_m, b_s = mean_sd(repB[col])
        rows.append({
            "metric": col,
            "label": label,
            "repA_mean": a_m, "repA_sd": a_s,
            "repB_mean": b_m, "repB_sd": b_s,
            "abs_diff_means": abs(a_m - b_m) if np.isfinite(a_m) and np.isfinite(b_m) else np.nan,
        })

    # Hop-distance fractions (Gate 4)
    hopA = pd.read_csv(args.repA_hops)
    hopB = pd.read_csv(args.repB_hops)
    hop_cols = ["frac_d0", "frac_d1", "frac_d2", "frac_dge3", "mean_d", "mean_d_cond_change"]
    hop_labels = {
        "frac_d0": "Hop fraction d=0",
        "frac_d1": "Hop fraction d=1",
        "frac_d2": "Hop fraction d=2",
        "frac_dge3": "Hop fraction d≥3",
        "mean_d": "Mean hop distance",
        "mean_d_cond_change": "Mean hop distance | change",
    }
    for col in hop_cols:
        a_m, a_s = mean_sd(hopA[col])
        b_m, b_s = mean_sd(hopB[col])
        rows.append({
            "metric": col,
            "label": hop_labels.get(col, col),
            "repA_mean": a_m, "repA_sd": a_s,
            "repB_mean": b_m, "repB_sd": b_s,
            "abs_diff_means": abs(a_m - b_m) if np.isfinite(a_m) and np.isfinite(b_m) else np.nan,
        })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv}")

    # Figure: compare key headline metrics
    key = table[table["metric"].isin(["phi_core", "Q_clock", "frac_d1", "frac_dge3", "mean_d"])].copy()
    key["x"] = np.arange(len(key))
    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(111)

    # plot as side-by-side errorbars
    w = 0.18
    ax.errorbar(key["x"]-w, key["repA_mean"], yerr=key["repA_sd"], marker="o", linewidth=1.3, capsize=3, label="repA")
    ax.errorbar(key["x"]+w, key["repB_mean"], yerr=key["repB_sd"], marker="s", linewidth=1.3, capsize=3, label="repB")

    ax.set_xticks(key["x"])
    ax.set_xticklabels(key["label"], rotation=25, ha="right")
    ax.set_title("Repeatability check: repA vs repB (N=32, n=0.8, hits1)")
    apply_bcqm_axes_style(ax)
    ax.legend(frameon=False)

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_fig}")

if __name__ == "__main__":
    main()
