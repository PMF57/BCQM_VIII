#!/usr/bin/env python3
"""
build_fingerprint.py (v0.3)

Package-context fingerprint for bcqm_vi_spacetime, with robust sys.path setup.

Key improvement:
- Automatically adds the parent directory of the bcqm_vi_spacetime package to sys.path.
  (No hard-coded "Desktop" path.)

Run from Desktop:
  python3 bcqm_vi_spacetime/analysis/build_fingerprint.py
"""
from __future__ import annotations

import hashlib
import time
import sys
from pathlib import Path


# Ensure imports work regardless of how the script is executed:
# This script lives at: <root>/<pkg>/analysis/build_fingerprint.py
# We add <root> (parent of <pkg>) to sys.path.
_THIS = Path(__file__).resolve()
_PKG_DIR = _THIS.parents[1]          # .../bcqm_vi_spacetime
_ROOT_DIR = _THIS.parents[2]         # .../<root>
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))


KEY_FILES = [
    "engine_vglue.py","event_graph.py","runner.py","cli.py",
    "analysis/sweetspot_check.py","analysis/pathA_summary.py",
    "analysis/scan_from_n_summary.py","analysis/coupled_LS_batch_summary.py",
    "analysis/geometry_scan_summary.py",
]

FEATURE_CHECKS = {
    "p_reuse_mode_support": "p_reuse_mode",
    "multi_wstar_logging": "F_max_by_wstar",
    "bundle_hist_by_wstar": "bundle_hist_by_wstar",
    "glueoff_hook": "glue_decohere",
    "q_base_override": "q_base_override",
    "geometry_hook_present": "estimate_spectral_dimension",
}


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = Path.cwd()
    code = root / "bcqm_vi_spacetime"
    outdir = root / "outputs" / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    outpath = outdir / f"{ts}_build_fingerprint.txt"

    lines=[]
    lines.append(f"BCQM VI build fingerprint — {time.asctime()}\n")
    lines.append(f"Working directory: {root}\n")
    lines.append(f"Code directory: {code}\n")
    lines.append(f"sys.path[0:3]: {sys.path[0:3]}\n\n")

    lines.append("== File hashes (sha256) ==\n")
    for rel in KEY_FILES:
        p = code / rel
        lines.append(f"{rel}: {sha256_file(p) if p.exists() else 'MISSING'}\n")
    lines.append("\n")

    lines.append("== Feature checks (substring) ==\n")
    blob=""
    for rel in ["engine_vglue.py","event_graph.py","runner.py"]:
        p=code/rel
        if p.exists():
            blob += "\n\n# --- "+rel+" ---\n"+p.read_text(encoding="utf-8", errors="replace")
    for name, needle in FEATURE_CHECKS.items():
        lines.append(f"{name}: {'OK' if needle in blob else 'MISSING'} (needle='{needle}')\n")
    lines.append("\n")

    lines.append("== Import smoke test (package context) ==\n")
    try:
        import importlib
        mod=importlib.import_module("bcqm_vi_spacetime.engine_vglue")
        lines.append("import bcqm_vi_spacetime.engine_vglue: OK\n")
        lines.append(f"run_single_v_glue present: {'run_single_v_glue' in dir(mod)}\n")
    except Exception as e:
        lines.append(f"import bcqm_vi_spacetime.engine_vglue: FAIL ({type(e).__name__}: {e})\n")

    outpath.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {outpath}")


if __name__=="__main__":
    main()
