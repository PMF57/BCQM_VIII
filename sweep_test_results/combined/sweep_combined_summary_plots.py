#!/usr/bin/env python3
"""
Combine BCQM VII light+heavy N-sweep summaries and generate scaling plots.

Default behaviour:
- Uses 'heavy' rows for any N present in the heavy CSV (ledger/tracing enabled),
  and 'light' rows for all remaining N.
- Produces a combined per-seed CSV and an aggregated per-N summary, plus PDFs.

Expected inputs (edit paths if needed):
  sweep_test_output/csv/sweep_N1_128_light_run_summary.csv
  sweep_test_output/csv/sweep_N32_128_heavy_run_summary.csv

Outputs:
  sweep_test_output/csv/sweep_N1_128_combined_run_summary.csv
  sweep_test_output/csv/sweep_N1_128_combined_summary_by_N.csv
  sweep_test_output/figures_sweep_combined/*.pdf
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

def main(
    light_csv="sweep_test_output/csv/sweep_N1_128_light_run_summary.csv",
    heavy_csv="sweep_test_output/csv/sweep_N32_128_heavy_run_summary.csv",
):
    light = pd.read_csv(light_csv)
    heavy = pd.read_csv(heavy_csv)

    heavy_Ns = set(heavy["N"].unique().tolist())
    light_sel = light[~light["N"].isin(heavy_Ns)].copy()
    heavy_sel = heavy.copy()
    light_sel["source"] = "light"
    heavy_sel["source"] = "heavy"
    combined = pd.concat([light_sel, heavy_sel], ignore_index=True)

    combined["total_events"] = combined["core_events_count"] + combined["halo_events_count"]
    combined["phi_core"] = combined["core_events_count"] / combined["total_events"].replace(0, np.nan)
    combined["total_edges"] = combined["core_edges_count"] + combined["halo_edges_count"]
    combined["halo_frac"] = combined["halo_events_count"] / combined["total_events"].replace(0, np.nan)

    agg_cols = ["phi_core","halo_frac","Q_clock","core_events_count","halo_events_count",
                "core_edges_count","halo_edges_count","total_events","total_edges",
                "ball_comp_size","S_perc","S_junc_w","L"]
    summary = combined.groupby("N")[agg_cols].agg(["mean","std"]).reset_index()
    summary.columns = ["N"] + [f"{c}_{stat}" for c, stat in summary.columns[1:]]
    summary = summary.sort_values("N")

    out_csv = Path(light_csv).parent
    out_rows = out_csv / "sweep_N1_128_combined_run_summary.csv"
    out_sum = out_csv / "sweep_N1_128_combined_summary_by_N.csv"
    combined.to_csv(out_rows, index=False)
    summary.to_csv(out_sum, index=False)
    print(f"[ok] wrote {out_rows}")
    print(f"[ok] wrote {out_sum}")

    fig_dir = Path(light_csv).parent.parent / "figures_sweep_combined"
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
    finalize(fig, "Core fraction versus N (hits1, n=0.8; combined)")
    fig.savefig(fig_dir/"sweep_phi_vs_N_combined.pdf", bbox_inches="tight")
    plt.close(fig)

    # Q_clock vs N
    fig = plt.figure(figsize=(8.5,5.5))
    ax = fig.add_subplot(111)
    ax.errorbar(N, summary["Q_clock_mean"], yerr=summary["Q_clock_std"], marker="o", linewidth=1.5, capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Q_clock")
    apply_bcqm_style(ax)
    finalize(fig, "Clock quality versus N (hits1, n=0.8; combined)")
    fig.savefig(fig_dir/"sweep_Qclock_vs_N_combined.pdf", bbox_inches="tight")
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
    finalize(fig, "Core and halo event counts versus N (hits1, n=0.8; combined)")
    fig.savefig(fig_dir/"sweep_events_core_halo_vs_N_combined.pdf", bbox_inches="tight")
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
    finalize(fig, "Core and halo edge counts versus N (hits1, n=0.8; combined)")
    fig.savefig(fig_dir/"sweep_edges_core_halo_vs_N_combined.pdf", bbox_inches="tight")
    plt.close(fig)

    # Halo fraction
    fig = plt.figure(figsize=(8.5,5.5))
    ax = fig.add_subplot(111)
    ax.errorbar(N, summary["halo_frac_mean"], yerr=summary["halo_frac_std"], marker="o", linewidth=1.5, capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Halo fraction (1-Φ)")
    apply_bcqm_style(ax)
    finalize(fig, "Halo fraction versus N (hits1, n=0.8; combined)")
    fig.savefig(fig_dir/"sweep_halo_frac_vs_N_combined.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] wrote plots to {fig_dir}")

if __name__ == "__main__":
    main()
