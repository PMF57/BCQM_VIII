#!/usr/bin/env python3
"""
BCQM_VII_c: bootstrap error bars for knee and lock contour plots.

Inputs:
- A directory containing one or more "*_run_summary.csv" files produced by summarise_runs.py
  (each file should contain per-seed rows with at least: N, n, seed, Q_clock,
   core_events_count, halo_events_count).

Outputs:
- knee_contour.pdf                 (Nk(n) with bootstrap CI error bars)
- lock_contour.pdf                 (Nl(n) with bootstrap CI error bars where defined)
- knee_contour_bootstrap.csv        (per-n Nk point estimate + CI)
- lock_contour_bootstrap.csv        (per-n Nl point estimate + CI + defined fraction)

BCQM figure defaults:
- figsize ~ (8.5, 5.5)
- title above chart area
- hide top/right spines
- major grid dashed
- no explicit colour choices

Notes:
- Nk(n) is defined as the linear-interpolation N where mean Phi crosses 0.5.
- In each bootstrap replicate, we resample seeds WITH replacement within each N for that n,
  recompute mean Phi(N), then re-find the crossing and interpolate.
- Nl(n) is defined as the first N (within the tested N grid) where p_lowQ <= p_thresh.
  In each replicate we resample seeds, recompute p_lowQ(N), and re-find first crossing.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, Tuple, Optional, List

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
    fig.tight_layout(rect=[0, 0, 1, 0.92])


def phi_series(df: pd.DataFrame) -> pd.Series:
    tot = df["core_events_count"] + df["halo_events_count"]
    return df["core_events_count"] / tot.replace(0, np.nan)


def find_knee_from_means(Ns: np.ndarray, phi_means: np.ndarray) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """
    Return (Nk, N_low, N_high) from arrays sorted by N.
    """
    idx = None
    for i in range(len(phi_means) - 1):
        y0, y1 = phi_means[i], phi_means[i + 1]
        if (y0 >= 0.5 and y1 <= 0.5) or (y0 <= 0.5 and y1 >= 0.5):
            idx = i
            break
    if idx is None:
        return None, None, None

    N0, N1 = float(Ns[idx]), float(Ns[idx + 1])
    y0, y1 = float(phi_means[idx]), float(phi_means[idx + 1])
    if y1 == y0:
        Nk = 0.5 * (N0 + N1)
    else:
        Nk = N0 + (0.5 - y0) * (N1 - N0) / (y1 - y0)
    return float(Nk), int(N0), int(N1)


def bootstrap_knee(df_n: pd.DataFrame, B: int, rng: np.random.Generator) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
    """
    Bootstrap Nk distribution for a fixed n.
    Resample seeds within each N with replacement.
    Returns (Nk_samples, bracket_low, bracket_high) where bracket is taken from the non-bootstrapped mean.
    """
    df_n = df_n.copy()
    df_n["phi"] = phi_series(df_n)

    # non-bootstrapped bracket for reporting
    g = df_n.groupby("N")["phi"].mean().reset_index().sort_values("N")
    Ns = g["N"].to_numpy()
    means = g["phi"].to_numpy()
    Nk0, Nlow0, Nhigh0 = find_knee_from_means(Ns, means)

    # bootstrap
    Nk_samples = np.full(B, np.nan, dtype=float)
    # Pre-split by N for speed
    byN = {int(N): sub for N, sub in df_n.groupby("N")}

    for b in range(B):
        Ns_b = []
        means_b = []
        for N in sorted(byN.keys()):
            sub = byN[N]
            # resample rows with replacement
            idx = rng.integers(0, len(sub), size=len(sub))
            phi_mean = float(sub["phi"].to_numpy()[idx].mean())
            Ns_b.append(N)
            means_b.append(phi_mean)
        Nk, _, _ = find_knee_from_means(np.array(Ns_b), np.array(means_b))
        if Nk is not None:
            Nk_samples[b] = Nk

    return Nk_samples, Nlow0, Nhigh0


def bootstrap_lock(df_n: pd.DataFrame, B: int, rng: np.random.Generator, p_thresh: float) -> np.ndarray:
    """
    Bootstrap Nl distribution (first N where p_lowQ <= p_thresh) for a fixed n.
    """
    df_n = df_n.copy()
    df_n["lowQ"] = (df_n["Q_clock"] < 3.0).astype(int)

    byN = {int(N): sub for N, sub in df_n.groupby("N")}
    Ns_sorted = sorted(byN.keys())
    Nl_samples = np.full(B, np.nan, dtype=float)

    for b in range(B):
        p_byN = []
        for N in Ns_sorted:
            sub = byN[N]
            idx = rng.integers(0, len(sub), size=len(sub))
            p = float(sub["lowQ"].to_numpy()[idx].mean())
            p_byN.append(p)
        # first N where p <= threshold
        found = None
        for N, p in zip(Ns_sorted, p_byN):
            if p <= p_thresh:
                found = N
                break
        if found is not None:
            Nl_samples[b] = float(found)

    return Nl_samples


def quantiles_ci(x: np.ndarray, lo: float = 0.16, hi: float = 0.84) -> Tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    med = float(np.quantile(x, 0.5))
    qlo = float(np.quantile(x, lo))
    qhi = float(np.quantile(x, hi))
    return med, qlo, qhi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default="csv/stage3_knee_contour", help="Directory with *_run_summary.csv files.")
    ap.add_argument("--out_dir", default="figures/stage3_knee_contour", help="Where to write PDFs and output CSVs.")
    ap.add_argument("--B", type=int, default=2000, help="Bootstrap replicates.")
    ap.add_argument("--rng_seed", type=int, default=0, help="RNG seed.")
    ap.add_argument("--p_thresh", type=float, default=0.05, help="Lock contour threshold for p_lowQ.")
    ap.add_argument("--ci_lo", type=float, default=0.16, help="Lower CI quantile (default 16%).")
    ap.add_argument("--ci_hi", type=float, default=0.84, help="Upper CI quantile (default 84%).")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(csv_dir / "*_run_summary.csv")))
    if not files:
        raise SystemExit(f"No *_run_summary.csv files found in {csv_dir}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Ensure numeric types
    df["n"] = df["n"].astype(float)
    df["N"] = df["N"].astype(int)

    rng = np.random.default_rng(args.rng_seed)

    knee_rows = []
    lock_rows = []

    for nval, sub in df.groupby("n"):
        # Knee
        Nk_samp, Nlow, Nhigh = bootstrap_knee(sub, B=args.B, rng=rng)
        Nk_med, Nk_lo, Nk_hi = quantiles_ci(Nk_samp, lo=args.ci_lo, hi=args.ci_hi)
        knee_rows.append({
            "n": float(nval),
            "Nk_med": Nk_med,
            "Nk_ci_lo": Nk_lo,
            "Nk_ci_hi": Nk_hi,
            "N_bracket_low": Nlow,
            "N_bracket_high": Nhigh,
            "Nk_defined_frac": float(np.isfinite(Nk_samp).mean())
        })

        # Lock contour (optional)
        Nl_samp = bootstrap_lock(sub, B=args.B, rng=rng, p_thresh=args.p_thresh)
        Nl_med, Nl_lo, Nl_hi = quantiles_ci(Nl_samp, lo=args.ci_lo, hi=args.ci_hi)
        lock_rows.append({
            "n": float(nval),
            "Nl_med": Nl_med,
            "Nl_ci_lo": Nl_lo,
            "Nl_ci_hi": Nl_hi,
            "Nl_defined_frac": float(np.isfinite(Nl_samp).mean()),
            "p_thresh": float(args.p_thresh)
        })

    knee_df = pd.DataFrame(knee_rows).sort_values("n")
    lock_df = pd.DataFrame(lock_rows).sort_values("n")

    knee_df.to_csv(out_dir / "knee_contour_bootstrap.csv", index=False)
    lock_df.to_csv(out_dir / "lock_contour_bootstrap.csv", index=False)

    # Plot knee with asymmetric error bars
    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(111)
    x = knee_df["n"].to_numpy()
    y = knee_df["Nk_med"].to_numpy()
    yerr = np.vstack([y - knee_df["Nk_ci_lo"].to_numpy(), knee_df["Nk_ci_hi"].to_numpy() - y])
    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, capsize=3)
    ax.set_xlabel("n")
    ax.set_ylabel("Nk (Phi=0.5 crossing)")
    apply_bcqm_style(ax)
    finalize(fig, "Knee contour Nk(n) with bootstrap CI")
    fig.savefig(out_dir / "knee_contour.pdf", bbox_inches="tight")
    plt.close(fig)

    # Plot lock contour where defined (drop NaNs)
    lock_plot = lock_df[np.isfinite(lock_df["Nl_med"])].copy()
    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(111)
    if len(lock_plot) > 0:
        x = lock_plot["n"].to_numpy()
        y = lock_plot["Nl_med"].to_numpy()
        yerr = np.vstack([y - lock_plot["Nl_ci_lo"].to_numpy(), lock_plot["Nl_ci_hi"].to_numpy() - y])
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, capsize=3)
    ax.set_xlabel("n")
    ax.set_ylabel(f"Nl (first N with p_lowQ <= {args.p_thresh})")
    apply_bcqm_style(ax)
    finalize(fig, "Lock contour Nl(n) with bootstrap CI (where defined)")
    fig.savefig(out_dir / "lock_contour.pdf", bbox_inches="tight")
    plt.close(fig)

    print("[ok] wrote:", out_dir / "knee_contour_bootstrap.csv")
    print("[ok] wrote:", out_dir / "lock_contour_bootstrap.csv")
    print("[ok] wrote:", out_dir / "knee_contour.pdf")
    print("[ok] wrote:", out_dir / "lock_contour.pdf")


if __name__ == "__main__":
    main()
