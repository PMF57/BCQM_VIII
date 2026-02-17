#!/usr/bin/env python3
"""
BCQM_VII_c plots: knee contour and lock contour.

Inputs (in working folder):
- knee_contour_summary.csv
- lock_contour_summary.csv

Outputs:
- knee_contour.pdf
- lock_contour.pdf

BCQM figure defaults:
- title above chart area
- no "Fig. n" in titles
- hide top/right spines; major grid dashed
"""

from pathlib import Path
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

def main():
    knee = pd.read_csv("knee_contour_summary.csv")
    lock = pd.read_csv("lock_contour_summary.csv")
    knee["n"] = knee["n"].astype(float)
    lock["n"] = lock["n"].astype(float)

    # Knee contour
    kk = knee[pd.notna(knee["Nk_est"])].sort_values("n")
    fig = plt.figure(figsize=(8.5,5.5))
    ax = fig.add_subplot(111)
    ax.plot(kk["n"], kk["Nk_est"], marker="o", linewidth=1.5)
    ax.set_xlabel("n")
    ax.set_ylabel("Nk (Phi=0.5 crossing)")
    apply_bcqm_style(ax)
    finalize(fig, "Knee contour Nk(n) from Phi=0.5 crossing")
    fig.savefig("knee_contour.pdf", bbox_inches="tight")
    plt.close(fig)

    # Lock contour (high-n region only; where Nl_est is defined)
    ll = lock[pd.notna(lock["Nl_est"])].sort_values("n")
    fig = plt.figure(figsize=(8.5,5.5))
    ax = fig.add_subplot(111)
    ax.plot(ll["n"], ll["Nl_est"], marker="o", linewidth=1.5)
    ax.set_xlabel("n")
    ax.set_ylabel("Nl (first N with p_lowQ <= 0.05)")
    apply_bcqm_style(ax)
    finalize(fig, "Lock contour Nl(n) using p_lowQ <= 0.05")
    fig.savefig("lock_contour.pdf", bbox_inches="tight")
    plt.close(fig)

    print("[ok] wrote knee_contour.pdf and lock_contour.pdf")

if __name__ == "__main__":
    main()
