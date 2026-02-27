# Provenance — BCQM VII_b (Stage‑3 sweeps / candidate BCQM_IV_b)

_Last updated: 17 February 2026_

This file records where the code came from, what was changed, and how the sweep artefacts in `csv/` were produced.

GitHub release: https://github.com/PMF57/BCQM_VII_b/releases/tag/v1.0.0


## BCQM VII_c code/archive (this repo)
- Code/archive DOI (v1.0.0): 10.5281/zenodo.18671297
- GitHub release tag: vii_c-v1.0


## BCQM VII_b paper (this repo)
- Concept DOI: 10.5281/zenodo.18549145
- Version 1 DOI: 10.5281/zenodo.18549146

## Upstream lineage (BCQM VII)
BCQM VII is published/locked. This repo was created as a **new workstream** after BCQM VII v1.0.

Upstream references:
- Paper DOI (v1): 10.5281/zenodo.18497217
- Concept DOI: 10.5281/zenodo.18497216
- Code archive DOI: 10.5281/zenodo.18505934

## Code import
- Source: BCQM VII codebase (copied into `bcqm_vii_cloth/`)
- Import date: 6 Feb 2026
- Import method: filesystem copy (no git history in this repo at import time)
- Machine used for sweeps: Mac mini Pro M4, 24 GB (where noted)

If you later push this repo to GitHub, record:
- initial commit hash: `<fill in>`
- upstream BCQM VII commit/tag used: `<fill in>`

## Analysis scripts that must match the CSVs
Two scripts were modified during the sweep campaign:

1) `bcqm_vii_cloth/analysis/summarise_runs.py`
- Reason: in multi‑N sweeps, seed IDs repeat across N. Pairwise ball‑growth stability must carry (N,n) to be interpretable.
- Required behaviour for published CSVs:
  - write a global pairwise CSV with `N_a,n_a,seed_a,N_b,n_b,seed_b,d_l2`
  - write per‑(N,n) pairwise CSVs: `<tag>_ballgrowth_pairwise_N<NN>_n<nn>.csv`
- Script version: patched “v0.2” (2026‑02‑06).

2) `bcqm_vii_cloth/analysis/ball_growth_ensemble_summary.py`
- Cosmetic/consistency update only (no effect on numbers):
  - titles above chart area
  - no embedded “Fig. n” in plot titles

If the repo currently contains the unpatched versions, apply the patches before claiming full provenance equivalence with the CSVs in `csv/`.

## Sweep configs
All YAML configs used are preserved under `sweep_test_configs/`. Each config encodes:
- N list and n value(s)
- seeds.start and seeds.count
- bins / hits1 settings
- w* (space strength) and other parameters

## Evidence
See `docs/EVIDENCE_MANIFEST.md` for the authoritative list of:
- configs
- run directories (local)
- CSV artefacts
- lab notes and panel notes
