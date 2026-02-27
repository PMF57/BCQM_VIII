# Stage-3 knee contour sweeps (BCQM_VII_c)

This document describes the Stage-3 sweep campaign to **contour the cloth knee** as a function of coherence n.

## Goal
Map an empirical knee line:

- N_k(n): the N value where the cloth core fraction Phi = N_core / N_total crosses 0.5.

Optionally also map an ordering/clock contour:

- N_l(n): the N value where the low-Q probability p_lowQ = Pr(Q_clock < 3) drops below a chosen level
  (e.g. 0.05 or 0.01).

## Where files go
- YAML configs (inputs): `configs_stage3_knee_contour/`
- Consolidated CSV artefacts (outputs): `csv/stage3_knee_contour/`
- Run logs: `provenance/RUN_LOG_YYYY-MM-DD_VII_c_knee_contour.md`
- Optional plots: `figures/stage3_knee_contour/`

## Working practice (recommended)
1. Bracket then refine for each n:
   - start with a coarse N bracket until Phi > 0.5 and Phi < 0.5 are found,
   - then add 1-2 refined N points to estimate N_k(n).
2. Use 10 seeds for bracketing points, and 20 seeds for the refined knee points.
3. After each batch:
   - copy YAML configs into the repo config folder,
   - copy consolidated CSVs into `csv/stage3_knee_contour/`,
   - write a run log with commit hash + exact commands + outputs,
   - update `docs/EVIDENCE_MANIFEST.md` (VII_c section).

## Minimal outputs to commit
- `*_run_summary.csv` for each batch
- `knee_contour_summary.csv` (N_k vs n, with uncertainty)
- `lock_contour_summary.csv` (optional)
- one short run log per batch (in `provenance/`)
