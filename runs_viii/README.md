# runs_viii

Each Stage-3 run lives in its own self-contained folder, named with a timestamp and key parameters, e.g.

`2026-02-27_1300__n0p800__N22_23_24__bins20__seed_sweep/`

Each run folder should contain:
- the exact config used (copy of YAML/JSON)
- a short RUN_REPORT.md (what/why/how)
- logs/
- csv/
- figures/

Large raw outputs should be kept out of git unless explicitly curated as evidence.