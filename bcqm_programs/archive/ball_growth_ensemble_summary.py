#!/usr/bin/env python3
"""ball_growth_ensemble_summary.py (v0.1.1)

Regenerates the ball-growth ensemble figures WITHOUT embedding figure numbers in the plot titles.

Inputs (CSV; produced by the ball-growth ensemble pipeline):
- ball_growth_frac_N8_n0p4_W100.csv
- ball_growth_frac_N8_n0p8_W100.csv
- ball_growth_frac_N4_n0p4_W100.csv
- ball_growth_frac_N4_n0p8_W100.csv

Outputs (PDF; placed in ./figures/ by default):
- fig_4_ball_growth_frac_ensemble_W100_N8.pdf
- fig_4a_ball_growth_frac_ensemble_W100_N4.pdf

Run from repo root:
  python3 bcqm_vi_spacetime/analysis/ball_growth_ensemble_summary.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Where to read CSVs from (repo root by default)
IN_DIR = Path(".")

# Where to write figures (user keeps figures in a root folder named 'figures')
OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth=0.8)


def plot_curve_with_iqr(ax, df: pd.DataFrame, label: str):
    r = np.asarray(df["r"], dtype=float)
    med = np.asarray(df["frac_med"], dtype=float)
    q1 = np.asarray(df["frac_q1"], dtype=float)
    q3 = np.asarray(df["frac_q3"], dtype=float)
    (line,) = ax.plot(r, med, label=label)
    ax.fill_between(r, q1, q3, alpha=0.25, color=line.get_color())


def make_fig(df04: pd.DataFrame, df08: pd.DataFrame, title: str, outpath: Path):
    plt.figure(figsize=(8.5, 5.5))
    ax = plt.gca()
    plot_curve_with_iqr(ax, df04, "n=0.4 (median±IQR)")
    plot_curve_with_iqr(ax, df08, "n=0.8 (median±IQR)")
    ax.set_title(title)
    ax.set_xlabel("Graph radius r")
    ax.set_ylabel("Fraction covered |B(r)| / |C|")
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.02, 1.05)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, format="pdf")
    plt.close()


def main():
    df_n8_04 = pd.read_csv(IN_DIR / "ball_growth_frac_N8_n0p4_W100.csv")
    df_n8_08 = pd.read_csv(IN_DIR / "ball_growth_frac_N8_n0p8_W100.csv")
    df_n4_04 = pd.read_csv(IN_DIR / "ball_growth_frac_N4_n0p4_W100.csv")
    df_n4_08 = pd.read_csv(IN_DIR / "ball_growth_frac_N4_n0p8_W100.csv")

    for df in (df_n8_04, df_n8_08, df_n4_04, df_n4_08):
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    make_fig(
        df_n8_04, df_n8_08,
        "Ball growth fraction (ensemble) (W=100, N=8)",
        OUT_DIR / "fig_4_ball_growth_frac_ensemble_W100_N8.pdf",
    )
    make_fig(
        df_n4_04, df_n4_08,
        "Ball growth fraction (ensemble) (W=100, N=4)",
        OUT_DIR / "fig_4a_ball_growth_frac_ensemble_W100_N4.pdf",
    )

    print("Wrote:")
    print(" -", OUT_DIR / "fig_4_ball_growth_frac_ensemble_W100_N8.pdf")
    print(" -", OUT_DIR / "fig_4a_ball_growth_frac_ensemble_W100_N4.pdf")


if __name__ == "__main__":
    main()
