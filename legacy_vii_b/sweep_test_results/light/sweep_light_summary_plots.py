#!/usr/bin/env python3
"""
Summarise BCQM VII light N-sweep and generate basic scaling plots.

Inputs:
  sweep_test_output/csv/sweep_N1_128_light_run_summary.csv  (or provide path)

Outputs:
  sweep_test_output/csv/sweep_N1_128_light_summary_by_N.csv
  sweep_test_output/figures_sweep/*.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def apply_bcqm_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth=0.8)

def finalize(fig, title):
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0,0,1,0.92])

def main(in_csv="sweep_test_output/csv/sweep_N1_128_light_run_summary.csv"):
    in_csv = Path(in_csv)
    df = pd.read_csv(in_csv)

    df["total_events"] = df["core_events_count"] + df["halo_events_count"]
    df["phi_core"] = df["core_events_count"] / df["total_events"].replace(0, np.nan)
    df["total_edges"] = df["core_edges_count"] + df["halo_edges_count"]
    df["halo_frac"] = df["halo_events_count"] / df["total_events"].replace(0, np.nan)

    agg_cols = ["phi_core","halo_frac","Q_clock","core_events_count","halo_events_count",
                "core_edges_count","halo_edges_count","total_events","total_edges","ball_comp_size",
                "S_perc","S_junc_w","L"]
    summary = df.groupby("N")[agg_cols].agg(["mean","std"]).reset_index()
    summary.columns = ["N"] + [f"{c}_{stat}" for c, stat in summary.columns[1:]]
    out_csv = in_csv.parent / "sweep_N1_128_light_summary_by_N.csv"
    summary.to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv}")

    fig_dir = in_csv.parent.parent / "figures_sweep"
    fig_dir.mkdir(parents=True, exist_ok=True)
    N = summary["N"].values

    # Phi vs N
    fig = plt.figure(figsize=(8.5,5.5))
    ax = fig.add_subplot(111)
    ax.errorbar(N, summary["phi_core_mean"], yerr=summary["phi_core_std"], marker="o", linewidth=1.5, capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Core fraction Φ")
    apply_bcqm_style(ax)
    finalize(fig, "Core fraction versus N (hits1, n=0.8)")
    fig.savefig(fig_dir/"sweep_phi_vs_N.pdf", bbox_inches="tight")
    plt.close(fig)

    # Q_clock vs N
    fig = plt.figure(figsize=(8.5,5.5))
    ax = fig.add_subplot(111)
    ax.errorbar(N, summary["Q_clock_mean"], yerr=summary["Q_clock_std"], marker="o", linewidth=1.5, capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Q_clock")
    apply_bcqm_style(ax)
    finalize(fig, "Clock quality versus N (hits1, n=0.8)")
    fig.savefig(fig_dir/"sweep_Qclock_vs_N.pdf", bbox_inches="tight")
    plt.close(fig)

    # Core/halo events
    fig = plt.figure(figsize=(8.5,5.5))
    ax = fig.add_subplot(111)
    ax.errorbar(N, summary["core_events_count_mean"], yerr=summary["core_events_count_std"], marker="o", linewidth=1.5, capsize=3, label="Core events")
    ax.errorbar(N, summary["halo_events_count_mean"], yerr=summary["halo_events_count_std"], marker="s", linewidth=1.5, capsize=3, label="Halo events")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Event count")
    apply_bcqm_style(ax)
    ax.legend(frameon=False)
    finalize(fig, "Core and halo event counts versus N (hits1, n=0.8)")
    fig.savefig(fig_dir/"sweep_events_core_halo_vs_N.pdf", bbox_inches="tight")
    plt.close(fig)

    # Core/halo edges
    fig = plt.figure(figsize=(8.5,5.5))
    ax = fig.add_subplot(111)
    ax.errorbar(N, summary["core_edges_count_mean"], yerr=summary["core_edges_count_std"], marker="o", linewidth=1.5, capsize=3, label="Core edges")
    ax.errorbar(N, summary["halo_edges_count_mean"], yerr=summary["halo_edges_count_std"], marker="s", linewidth=1.5, capsize=3, label="Halo edges")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Edge count")
    apply_bcqm_style(ax)
    ax.legend(frameon=False)
    finalize(fig, "Core and halo edge counts versus N (hits1, n=0.8)")
    fig.savefig(fig_dir/"sweep_edges_core_halo_vs_N.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] wrote plots to {fig_dir}")

if __name__ == "__main__":
    main()
