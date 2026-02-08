from __future__ import annotations

import argparse
from pathlib import Path

from .io import load_yaml
from .runner import run_from_config, scan_from_config
from .compat_v5_v6 import import_bcqm_v_config, ImportOptions


def main() -> None:
    parser = argparse.ArgumentParser(prog="bcqmvi", description="bcqm_vi_spacetime CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a config (may include scan lists)")
    p_run.add_argument("--config", required=True, type=str, help="Path to YAML config")

    p_scan = sub.add_parser("scan", help="Run a scan over n, sizes, seeds as defined in YAML")
    p_scan.add_argument("--config", required=True, type=str, help="Path to YAML config")

    p_import = sub.add_parser("import_v", help="Import a BCQM V YAML config and emit an import manifest")
    p_import.add_argument("--config_v", required=True, type=str, help="Path to BCQM V YAML config")
    p_import.add_argument("--out", required=True, type=str, help="Output directory for IMPORT_MANIFEST_*.json")
    p_import.add_argument("--respect_store_states", action="store_true",
                          help="If set, map V diagnostics.store_states to snapshots.enabled")
    p_import.add_argument("--respect_compute_psd", action="store_true",
                          help="If set, map V diagnostics.compute_psd to output.write_timeseries")
    p_import.add_argument("--emit_yamls", action="store_true",
                          help="If set, emit one VI YAML per expanded run (can be many)")

    args = parser.parse_args()

    if args.cmd in ("run", "scan"):
        cfg = load_yaml(Path(args.config))
        if args.cmd == "run":
            run_from_config(cfg)
        else:
            scan_from_config(cfg)
        return

    if args.cmd == "import_v":
        opts = ImportOptions(
            respect_store_states=bool(args.respect_store_states),
            respect_compute_psd=bool(args.respect_compute_psd),
            emit_yaml_per_run=bool(args.emit_yamls),
        )
        manifest_path = import_bcqm_v_config(Path(args.config_v), Path(args.out), opts=opts)
        print(f"Wrote {manifest_path}")
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
