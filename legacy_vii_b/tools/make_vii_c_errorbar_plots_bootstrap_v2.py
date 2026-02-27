#!/usr/bin/env python3
"""
BCQM_VII_c: bootstrap error bars for knee and lock contour plots (v2).

Change vs v1:
- Plot markers use the *full-sample point estimates* (not bootstrap medians).
- Bootstrap is used only to compute confidence intervals (CI bands).
  This avoids visual shifts in the lock contour caused by the discrete first-crossing statistic.

Inputs:
- A directory containing one or more "*_run_summary.csv" files produced by summarise_runs.py
  (each file should contain per-seed rows with at least: N, n, seed, Q_clock,
   core_events_count, halo_events_count).

Outputs (written to --out_dir):
- knee_contour.pdf
- lock_contour.pdf
- knee_contour_bootstrap.csv   (point estimate + CI + defined fraction)
- lock_contour_bootstrap.csv   (point estimate + CI + defined fraction)

Definitions:
- Phi = N_core / (N_core + N_halo)
- Knee Nk(n): linear interpolation in N between adjacent tested N values whose mean Phi straddles 0.5.
- Lock Nl(n): first tested N where p_lowQ <= p_thresh, where p_lowQ = Pr(Q_clock < 3).

Bootstrap:
- Resample seeds with replacement within each tested N for fixed n.
- Recompute mean Phi(N) and find Nk; likewise recompute p_lowQ(N) and find Nl.
- CI is reported as quantiles (default 16–84%). Use --ci_lo/--ci_hi for 95% intervals.

BCQM figure defaults:
- figsize ~ (8.5, 5.5)
- title above chart area
- hide top/right spines; major grid dashed
- no explicit colour choices
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
    idx = None
    for i in range(len(phi_means) - 1):
        y0, y1 = float(phi_means[i]), float(phi_means[i + 1])
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


def point_estimate_knee(df_n: pd.DataFrame) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    df_n = df_n.copy()
    df_n["phi"] = phi_series(df_n)
    g = df_n.groupby("N")["phi"].mean().reset_index().sort_values("N")
    return find_knee_from_means(g["N"].to_numpy(), g["phi"].to_numpy())


def point_estimate_lock(df_n: pd.DataFrame, p_thresh: float) -> Optional[int]:
    df_n = df_n.copy()
    df_n["lowQ"] = (df_n["Q_clock"] < 3.0).astype(int)
    g = df_n.groupby("N")["lowQ"].mean().reset_index().sort_values("N")
    for N, p in zip(g["N"].to_numpy(), g["lowQ"].to_numpy()):
        if float(p) <= p_thresh:
            return int(N)
    return None


def bootstrap_knee(df_n: pd.DataFrame, B: int, rng: np.random.Generator) -> np.ndarray:
    df_n = df_n.copy()
    df_n["phi"] = phi_series(df_n)
    byN = {int(N): sub for N, sub in df_n.groupby("N")}
    Ns_sorted = sorted(byN.keys())

    Nk_samples = np.full(B, np.nan, dtype=float)
    for b in range(B):
        means_b = []
        for N in Ns_sorted:
            sub = byN[N]
            idx = rng.integers(0, len(sub), size=len(sub))
            means_b.append(float(sub["phi"].to_numpy()[idx].mean()))
        Nk, _, _ = find_knee_from_means(np.array(Ns_sorted), np.array(means_b))
        if Nk is not None:
            Nk_samples[b] = Nk
    return Nk_samples


def bootstrap_lock(df_n: pd.DataFrame, B: int, rng: np.random.Generator, p_thresh: float) -> np.ndarray:
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
            p_byN.append(float(sub["lowQ"].to_numpy()[idx].mean()))
        found = None
        for N, p in zip(Ns_sorted, p_byN):
            if p <= p_thresh:
                found = N
                break
        if found is not None:
            Nl_samples[b] = float(found)
    return Nl_samples


def ci_quantiles(x: np.ndarray, lo: float, hi: float) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return float(np.quantile(x, lo)), float(np.quantile(x, hi))


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
    df["n"] = df["n"].astype(float)
    df["N"] = df["N"].astype(int)

    rng = np.random.default_rng(args.rng_seed)

    knee_rows = []
    lock_rows = []

    for nval, sub in df.groupby("n"):
        # Point estimates
        Nk_pt, Nlow, Nhigh = point_estimate_knee(sub)
        Nl_pt = point_estimate_lock(sub, p_thresh=args.p_thresh)

        # Bootstrap CIs
        Nk_samp = bootstrap_knee(sub, B=args.B, rng=rng)
        Nk_qlo, Nk_qhi = ci_quantiles(Nk_samp, args.ci_lo, args.ci_hi)
        Nk_def = float(np.isfinite(Nk_samp).mean())

        Nl_samp = bootstrap_lock(sub, B=args.B, rng=rng, p_thresh=args.p_thresh)
        Nl_qlo, Nl_qhi = ci_quantiles(Nl_samp, args.ci_lo, args.ci_hi)
        Nl_def = float(np.isfinite(Nl_samp).mean())

        knee_rows.append({
            "n": float(nval),
            "Nk_point": float(Nk_pt) if Nk_pt is not None else np.nan,
            "Nk_ci_lo": Nk_qlo,
            "Nk_ci_hi": Nk_qhi,
            "Nk_defined_frac": Nk_def,
            "N_bracket_low": Nlow,
            "N_bracket_high": Nhigh,
        })

        lock_rows.append({
            "n": float(nval),
            "Nl_point": float(Nl_pt) if Nl_pt is not None else np.nan,
            "Nl_ci_lo": Nl_qlo,
            "Nl_ci_hi": Nl_qhi,
            "Nl_defined_frac": Nl_def,
            "p_thresh": float(args.p_thresh),
        })

    knee_df = pd.DataFrame(knee_rows).sort_values("n")
    lock_df = pd.DataFrame(lock_rows).sort_values("n")

    knee_df.to_csv(out_dir / "knee_contour_bootstrap.csv", index=False)
    lock_df.to_csv(out_dir / "lock_contour_bootstrap.csv", index=False)

    # Knee plot (point estimate markers, bootstrap CI bars)
    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(111)
    kk = knee_df[np.isfinite(knee_df["Nk_point"])].copy()
    x = kk["n"].to_numpy()
    y = kk["Nk_point"].to_numpy()
    # yerr relative to point estimate
    ylo = kk["Nk_ci_lo"].to_numpy()
    yhi = kk["Nk_ci_hi"].to_numpy()
    yerr = np.vstack([np.maximum(0.0, y - ylo), np.maximum(0.0, yhi - y)])
    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, capsize=3)
    ax.set_xlabel("n")
    ax.set_ylabel("Nk (Phi=0.5 crossing)")
    apply_bcqm_style(ax)
    finalize(fig, "Knee contour Nk(n) with bootstrap CI")
    fig.savefig(out_dir / "knee_contour.pdf", bbox_inches="tight")
    plt.close(fig)

    # Lock plot (point estimate markers, bootstrap CI bars; where point defined)
    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(111)
    ll = lock_df[np.isfinite(lock_df["Nl_point"])].copy()
    if len(ll) > 0:
        x = ll["n"].to_numpy()
        y = ll["Nl_point"].to_numpy()
        ylo = ll["Nl_ci_lo"].to_numpy()
        yhi = ll["Nl_ci_hi"].to_numpy()
        yerr = np.vstack([np.maximum(0.0, y - ylo), np.maximum(0.0, yhi - y)])
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, capsize=3)
    ax.set_xlabel("n")
    ax.set_ylabel(f"Nl (first N with p_lowQ <= {args.p_thresh})")
    apply_bcqm_style(ax)
    finalize(fig, "Lock contour Nl(n) with bootstrap CI (point estimate markers)")
    fig.savefig(out_dir / "lock_contour.pdf", bbox_inches="tight")
    plt.close(fig)

    print("[ok] wrote:", out_dir / "knee_contour_bootstrap.csv")
    print("[ok] wrote:", out_dir / "lock_contour_bootstrap.csv")
    print("[ok] wrote:", out_dir / "knee_contour.pdf")
    print("[ok] wrote:", out_dir / "lock_contour.pdf")


if __name__ == "__main__":
    main()
