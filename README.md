# BCQM_VIII

BCQM_VIII is an extension of the Boundary-Condition Quantum Mechanics (BCQM) programme.

It moves from **Stage 2** to **Stage 3**: longer runs, more data, and an explicit attempt to achieve **emergent dimensions** from the validated cloth regime.

## What BCQM_VIII is for

Stage 3 work focuses on:

- running longer simulations (hero runs and seed sweeps) once the Stage-2 cloth is validated
- logging the additional observables needed for Stage-3 diagnostics (clock-native temporal structure and metric/geometry diagnostics)
- testing whether emergent dimension-like behaviour can be obtained robustly (without hidden tuning)

## Stage-3 logging (stage3_trace)

BCQM_VIII introduces a Stage-3 logging module, `stage3_trace`, designed to support:

- **clock-native temporal diagnostics** (an operational causal graph \(G_t\) derived from clock phase data)
- **metric/geometry diagnostics** (a metric graph \(G_2\) derived from directed community flows on the cloth)
- **islands and dynamic turnover** diagnostics (required): detecting time-varying occupancy clusters (“islands”) on a persistent cloth, and quantifying entry/exit/stay turnover per bin

The `stage3_trace` module is treated as part of the Stage-3 run definition: its parameters and thresholds are recorded alongside each run to keep evidence sets audit-friendly.

## Provenance

This repository was created by importing the full git history of **BCQM_VII_b** and then starting BCQM_VIII work on top of that baseline.

- See: `PROVENANCE_IMPORT_FROM_BCQM_VII_b.md`
- Archived Stage-2 / VII_b artefacts retained for reference: `legacy_vii_b/`

## Repository layout (in progress)

Core code currently lives under:

- `bcqm_programs/` — simulation programs (imported and being extended for Stage 3)

Stage-3 run artefacts and tooling will be added under clearly separated folders (e.g. `runs_viii/`, `configs_viii/`, `tools_viii/`, `docs/`) so each evidence set remains self-contained and easy to audit.

## Status

Work in progress: setting up the BCQM_VIII Stage-3 run infrastructure and `stage3_trace`, then executing the first Stage-3 run set.