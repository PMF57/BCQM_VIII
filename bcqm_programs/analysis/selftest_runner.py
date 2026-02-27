#!/usr/bin/env python3
"""
selftest_runner.py (v0.1.3)

Runs a small set of selftest configs and asserts key invariants so we can trust the working code.

Run from Desktop:
  python3 bcqm_vi_spacetime/analysis/selftest_runner.py

It will:
- run 4 configs (nospace, space-on, glueoff+space, geometry)
- write a consolidated log to outputs/analysis/<timestamp>_selftest.txt
- exit nonzero if any checks fail
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
# Ensure package imports work when running this file as a script:
# This file lives at: <root>/bcqm_vi_spacetime/analysis/selftest_runner.py
# Add <root> to sys.path.
_THIS = _Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import subprocess
import sys
import time
from pathlib import Path
from glob import glob


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def newest(pattern: str) -> Path | None:
    files = [Path(p) for p in glob(pattern)]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def run_cmd(cmd: list[str], logfile: Path) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logfile.write_text(logfile.read_text(encoding="utf-8") + p.stdout + "\n", encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def assert_true(cond: bool, msg: str, logfile: Path) -> None:
    if not cond:
        logfile.write_text(logfile.read_text(encoding="utf-8") + f"ASSERT FAIL: {msg}\n", encoding="utf-8")
        raise AssertionError(msg)
    logfile.write_text(logfile.read_text(encoding="utf-8") + f"OK: {msg}\n", encoding="utf-8")


def main() -> None:
    root = Path.cwd()
    outdir = root / "outputs" / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log = outdir / f"{ts}_selftest.txt"
    log.write_text(f"BCQM VI SELFTEST — {time.asctime()}\n\n", encoding="utf-8")


    # Validate all YAML configs under configs/ before running anything.
    # This ensures any random config in the repo is schema-compatible.
    import yaml
    from bcqm_vi_spacetime.config_schema import validate as _validate

    def _validate_all_yamls(cfg_root: Path) -> tuple[list[str], list[str]]:
        bad: list[str] = []
        skipped: list[str] = []
        for p in cfg_root.rglob("*.yml"):
            # Skip macOS metadata files.
            if "__MACOSX" in p.parts or p.name.startswith("._"):
                continue
            try:
                cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
                if not isinstance(cfg, dict):
                    bad.append(f"{p}: root not a mapping")
                    continue

                # Only validate VI configs (those declaring a vi_spacetime schema_version).
                sv = cfg.get("schema_version", None)
                if sv is None:
                    skipped.append(f"{p}: no schema_version (treated as non-VI config)")
                    continue
                if not str(sv).startswith("vi_spacetime_config"):
                    skipped.append(f"{p}: schema_version={sv} (non-VI)")
                    continue

                _validate(cfg)
            except Exception as e:
                bad.append(f"{p}: {type(e).__name__}: {e}")
        return bad, skipped

    bad, skipped = _validate_all_yamls(root / "configs")
    log.write_text(log.read_text(encoding="utf-8") + f"OK: skipped {len(skipped)} non-VI YAML(s)\n", encoding="utf-8")
    if skipped:
        log.write_text(
            log.read_text(encoding="utf-8")
            + "== YAML SKIPPED (non-VI) ==\n"
            + "\n".join(skipped)
            + "\n\n",
            encoding="utf-8",
        )

    if bad:
        log.write_text(
            log.read_text(encoding="utf-8")
            + "== YAML VALIDATION FAILURES ==\n"
            + "\n".join(bad)
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"YAML validation failed for {len(bad)} file(s). See log for details.")

    log.write_text(log.read_text(encoding="utf-8") + "OK: all VI YAML configs under configs/ validate()\n\n", encoding="utf-8")

    configs = [
        ("nospace", "configs/selftests/selftest_vglue_nospace.yml"),
        ("space_on", "configs/selftests/selftest_space_on.yml"),
        ("glueoff", "configs/selftests/selftest_glueoff_space_on.yml"),
        ("geometry", "configs/selftests/selftest_geometry.yml"),
    ]

    for tag, cfg in configs:
        log.write_text(log.read_text(encoding="utf-8") + f"== RUN {tag}: {cfg} ==\n", encoding="utf-8")
        run_cmd(["python3", "-m", "bcqm_vi_spacetime.cli", "run", "--config", cfg], log)

        # locate outputs via config file itself
        cfg_obj = read_json(newest("outputs/selftest/**/RUN_CONFIG_*.json") or Path("")).get("resolved", None) if tag!="nospace" else None

    # Now do targeted checks by globbing outputs we know
    # 1) nospace
    m = newest("outputs/selftest/selftest_vglue_nospace/RUN_METRICS_*.json")
    c = newest("outputs/selftest/selftest_vglue_nospace/RUN_CONFIG_*.json")
    assert_true(m is not None and c is not None, "nospace outputs exist", log)
    mj = read_json(m) ; cj = read_json(c)
    assert_true(mj.get("engine_mode") == "v_glue", "nospace engine_mode=v_glue", log)
    assert_true(mj.get("space_state", {}).get("enabled", False) is False, "nospace space_state.enabled=false", log)

    # 2) space_on
    m = newest("outputs/selftest/selftest_space_on/RUN_METRICS_*.json")
    c = newest("outputs/selftest/selftest_space_on/RUN_CONFIG_*.json")
    assert_true(m is not None and c is not None, "space_on outputs exist", log)
    mj = read_json(m) ; cj = read_json(c)
    assert_true(mj.get("space_state", {}).get("enabled", False) is True, "space_on space_state.enabled=true", log)
    assert_true("islands" in mj and "F_max_by_wstar" in mj["islands"], "space_on islands multi-wstar present", log)

    # 3) glueoff
    m = newest("outputs/selftest/selftest_glueoff_space_on/RUN_METRICS_*.json")
    c = newest("outputs/selftest/selftest_glueoff_space_on/RUN_CONFIG_*.json")
    assert_true(m is not None and c is not None, "glueoff outputs exist", log)
    mj = read_json(m) ; cj = read_json(c)
    assert_true(mj.get("space_state", {}).get("enabled", False) is True, "glueoff space_state.enabled=true", log)
    assert_true(cj.get("v_glue", {}).get("ablation", {}).get("glue_decohere", False) is True, "glueoff ablation recorded in RUN_CONFIG", log)

    # 4) geometry
    m = newest("outputs/selftest/selftest_geometry/RUN_METRICS_*.json")
    c = newest("outputs/selftest/selftest_geometry/RUN_CONFIG_*.json")
    assert_true(m is not None and c is not None, "geometry outputs exist", log)
    mj = read_json(m)
    assert_true("geometry" in mj and mj["geometry"] is not None, "geometry block present in RUN_METRICS", log)

    log.write_text(log.read_text(encoding="utf-8") + "\nSELFTEST PASSED\n", encoding="utf-8")
    print(f"Wrote {log}")


if __name__ == "__main__":
    main()