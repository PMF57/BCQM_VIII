# RUN_LOG — BCQM_VII_c knee contour (Stage-3)
Date: 2026-02-17  
Author: Peter M. Ferguson

## Purpose
Map the Stage-3 **knee contour** in (N, n):

- **Nk(n)**: the N value where the cloth core fraction **Phi = N_core / N_total** crosses **0.5**.
- Optional: **Nl(n)**: first N where **p_lowQ = Pr(Q_clock < 3)** falls to **<= 0.05** (operational threshold; mainly meaningful at higher n).

This run log records the exact configuration families executed (bracket + refine), the outputs produced in `csv_s3kc/`, and where artefacts should be moved in the repo.

## Machine / environment
- Machine: Mac mini Pro M4, 24GB
- OS: macOS (Activity Monitor: memory pressure green; swap 0 during parallel runs)
- Execution style: multiple terminal windows (parallel runs)

## Code provenance (fill in when committing)
- Repo: BCQM_VII_b
- Branch: main
- Commit: 173670f4107dbb8d02939328c7bd5f976e3bd479
- Tag (optional): <fill in>

## Working folders (Desktop)
- Code: `~/Desktop/bcqm_vii_cloth/`
- Configs (working): `~/Desktop/configs_stage3_knee_contour/`
- CSV outputs (working): `~/Desktop/csv_s3kc/`
- Raw outputs (local only): `~/Desktop/sweep_test_output/outputs_cloth/` (do NOT commit)

## Canonical repo destinations
- Configs: `configs_stage3_knee_contour/`
- CSV artefacts: `csv/stage3_knee_contour/`
- Figures: `figures/stage3_knee_contour/`
- Run logs: `provenance/`
- Evidence index: `docs/EVIDENCE_MANIFEST.md` (VII_c section)

## Runs executed

### A) Bracket pass (10 seeds per n, 3 N points each)
Purpose: obtain coarse brackets for Phi crossing.

n=0.65: N = {8, 12, 16}  
n=0.70: N = {12, 14, 16}  
n=0.75: N = {14, 18, 22}  
n=0.80: N = {20, 24, 28}  
n=0.85: N = {24, 28, 32} (later replaced by extended bracket)  
n=0.90: N = {28, 32, 40} (later replaced by extended bracket)

Outputs written by `summarise_runs.py`:
- `csv_s3kc/vii_c_*_run_summary.csv` (one per config tag)

### B) Refinement pass (20 seeds per n, 3 N points each)
Purpose: tighten Nk estimates where bracket exists.

n=0.65: N = {10, 11, 12}  
n=0.70: N = {13, 14, 15}  
n=0.75: N = {16, 17, 18}  
n=0.80: N = {22, 24, 26}

### C) High-n bracket extension (10 seeds per n, 3 N points each)
Purpose: bracket Nk for n=0.85 and n=0.90 (knee beyond initial ranges).

n=0.85: N = {32, 40, 48}  
n=0.90: N = {40, 56, 72}

### D) Final refinement (20 seeds per n, 3 N points each)
Purpose: tighten high-n Nk and tidy n=0.80 near the crossing.

n=0.85: N = {34, 35, 36}  
n=0.90: N = {54, 56, 58}  
n=0.80: N = {23, 24, 25}

## Analysis steps executed

### 1) Per-run summarisation
After each scan:
```bash
python3 bcqm_vii_cloth/analysis/summarise_runs.py --run_dir sweep_test_output/outputs_cloth/<run_dir> --out_dir csv_s3kc --tag <tag>