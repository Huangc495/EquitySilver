"""Pre-run environment and data checks.

Run this before anything else on a new machine or cluster. It verifies that
the configured paths resolve, the raw data is where the config expects it,
and the installed packages match `requirements.txt` — the three things that
otherwise fail partway through a notebook with a confusing error.

It also prints the installed versions of the pinned packages, which is what
`requirements.txt` needs reconciling against (DEVIATIONS.md D-01).
"""

from __future__ import annotations

import glob
import importlib.metadata as md
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Config, expand_env, on_databricks

# Distribution names as they appear in requirements.txt.
PINNED = ("numpy", "pandas", "pyarrow", "torch", "matplotlib",
          "openpyxl", "PyYAML", "mlflow", "scipy", "pytest")

EXPECTED_WEATHER_FILES = 21
_REQ_LINE = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*==\s*([^\s#]+)")

OK, FAIL, WARN = "PASS", "FAIL", "WARN"


@dataclass
class Check:
    area: str
    name: str
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status != FAIL


def _check(area: str, name: str, ok: bool, detail: str,
           warn_only: bool = False) -> Check:
    if ok:
        status = OK
    else:
        status = WARN if warn_only else FAIL
    return Check(area, name, status, detail)


def check_environment(cfg: Config) -> list:
    """Where are we running, and is the MLflow target resolvable?"""
    out = [
        Check("environment", "platform", OK,
              f"{sys.platform}, Python {sys.version.split()[0]}"),
        Check("environment", "running on", OK,
              "Databricks" if on_databricks() else "local"),
    ]
    if on_databricks():
        out.append(_check(
            "environment", "DATABRICKS_RUNTIME_VERSION", True,
            os.environ.get("DATABRICKS_RUNTIME_VERSION", ""),
        ))
    try:
        out.append(Check("environment", "mlflow experiment", OK,
                         cfg.mlflow_experiment))
    except KeyError as exc:
        out.append(Check("environment", "mlflow experiment", FAIL,
                         str(exc).strip("'")))
    return out


def check_paths(cfg: Config) -> list:
    """Do the configured paths resolve, and is the raw data actually there?"""
    out: list[Check] = []

    try:
        pattern = str(cfg.resolve("weather_glob"))
    except KeyError as exc:
        return [Check("data", "weather_glob", FAIL, str(exc).strip("'"))]

    files = sorted(glob.glob(pattern))
    out.append(_check(
        "data", "weather CSVs", len(files) == EXPECTED_WEATHER_FILES,
        f"{len(files)} of {EXPECTED_WEATHER_FILES} matched "
        f"{Path(pattern).parent}",
    ))
    if files:
        years = sorted(
            int(m.group(1))
            for m in (re.search(r"_(\d{4})_P1D\.csv$", f) for f in files) if m
        )
        if years:
            gaps = sorted(set(range(years[0], years[-1] + 1)) - set(years))
            out.append(_check(
                "data", "weather years", not gaps,
                f"{years[0]}-{years[-1]}"
                + (f", missing {gaps}" if gaps else ", contiguous"),
            ))

    excel = cfg.resolve("acidity_excel")
    out.append(_check(
        "data", "acidity workbook", excel.exists(),
        f"{excel}" + (f" ({excel.stat().st_size / 1024:,.0f} KB)"
                      if excel.exists() else " - NOT FOUND"),
    ))

    processed = cfg.resolve("processed_dir")
    writable, detail = _writable(processed)
    out.append(_check("data", "processed dir writable", writable, detail))

    reports = cfg.reports_dir
    writable, detail = _writable(reports)
    out.append(_check("data", "reports dir writable", writable, detail,
                      warn_only=True))
    return out


def _writable(directory: Path) -> tuple:
    """Can we create and delete a file here?"""
    probe = directory / ".acidity_lstm_write_probe"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe.write_text("probe", encoding="utf-8")
        probe.unlink()
        return True, str(directory)
    except Exception as exc:                        # noqa: BLE001
        return False, f"{directory} - {type(exc).__name__}: {exc}"


def check_packages(cfg: Config) -> list:
    """Compare installed versions with the pins in requirements.txt."""
    required = _parse_requirements(cfg.repo_root / "requirements.txt")
    out: list[Check] = []

    for name in PINNED:
        try:
            installed = md.version(name)
        except md.PackageNotFoundError:
            out.append(Check("packages", name, FAIL, "not installed"))
            continue

        pinned = required.get(name.lower())
        if pinned is None:
            out.append(Check("packages", name, OK, f"{installed} (unpinned)"))
        elif _same_version(installed, pinned):
            out.append(Check("packages", name, OK, installed))
        else:
            # A mismatch is worth knowing about but rarely fatal, and on a
            # cluster the runtime's own version is the one to trust.
            out.append(Check("packages", name, WARN,
                             f"{installed} installed, {pinned} pinned"))
    return out


def _parse_requirements(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _REQ_LINE.match(line)
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def _same_version(installed: str, pinned: str) -> bool:
    """Compare ignoring local build tags such as torch's "+cpu"."""
    return installed.split("+")[0] == pinned.split("+")[0]


def run_preflight(cfg: Config, verbose: bool = True):
    """Run every check. Returns `(all_passed, DataFrame)`."""
    checks = check_environment(cfg) + check_paths(cfg) + check_packages(cfg)
    table = pd.DataFrame([vars(c) for c in checks])
    passed = all(c.ok for c in checks)

    if verbose:
        width = max(len(c.name) for c in checks) + 2
        current = None
        for c in checks:
            if c.area != current:
                current = c.area
                print(f"\n{current.upper()}")
            print(f"  [{c.status}] {c.name:<{width}} {c.detail}")

        n_fail = sum(c.status == FAIL for c in checks)
        n_warn = sum(c.status == WARN for c in checks)
        print()
        if passed and not n_warn:
            print("All checks passed. Run 00_data_audit next.")
        elif passed:
            print(f"Ready, with {n_warn} warning(s) above. "
                  "Run 00_data_audit next.")
        else:
            print(f"{n_fail} check(s) FAILED - fix these before running the "
                  "notebooks.")
    return passed, table


def environment_report() -> str:
    """Installed versions of the pinned packages, as a requirements block.

    Paste the output back into `requirements.txt` (or to whoever is
    maintaining it) to close DEVIATIONS.md D-01.
    """
    lines = [
        f"# Captured on {'Databricks ' + os.environ.get('DATABRICKS_RUNTIME_VERSION', '') if on_databricks() else 'a local machine'}",
        f"# Python {sys.version.split()[0]} on {sys.platform}",
    ]
    for name in PINNED:
        try:
            lines.append(f"{name}=={md.version(name)}")
        except md.PackageNotFoundError:
            lines.append(f"# {name}: NOT INSTALLED")
    return "\n".join(lines)


# --- Diagnostics ----------------------------------------------------------

VOLUMES_ROOT = "/Volumes"
EXPECTED_NAMES = ("2017 ARD Chemisty_Clean.xlsx", "en_climate_daily_BC_1072692_1997_P1D.csv")


def _spark():
    """The active SparkSession, or None when not on a cluster."""
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return None
    try:
        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception:                                # noqa: BLE001
        return None


def _rows(spark, statement: str):
    """Run a SHOW statement, returning [] if it is not permitted or fails."""
    try:
        return [tuple(r) for r in spark.sql(statement).collect()]
    except Exception as exc:                         # noqa: BLE001
        return [("<error>", str(exc).splitlines()[0][:120])]


def diagnose_volumes(cfg: Config, search: bool = True) -> None:
    """Explain why the configured Volume path holds no data.

    Unity Catalog mounts volumes at /Volumes/<catalog>/<schema>/<volume>, so
    the layout can be walked with ordinary filesystem calls. A schema with no
    volumes does not appear there at all, which is itself the usual answer, so
    the catalog is also queried through Spark when available.
    """
    import os

    configured = str(cfg.paths.get("acidity_excel", ""))
    parts = configured.strip("/").split("/")
    catalog = parts[1] if len(parts) > 2 else ""
    schema = parts[2] if len(parts) > 3 else ""

    print("CONFIGURED")
    print(f"  catalog : {catalog}")
    print(f"  schema  : {schema}")
    print(f"  path    : {configured}")

    if not os.path.isdir(VOLUMES_ROOT):
        print(f"\n{VOLUMES_ROOT} does not exist - this is not a Databricks "
              "cluster with Unity Catalog volumes mounted. Run this notebook "
              "on the cluster.")
        return

    print(f"\nWHAT EXISTS UNDER {VOLUMES_ROOT}")
    catalogs = _safe_listdir(VOLUMES_ROOT)
    if not catalogs:
        print("  (no catalogs visible - check your permissions)")
    for cat in catalogs:
        print(f"  {cat}/")
        for sch in _safe_listdir(f"{VOLUMES_ROOT}/{cat}"):
            vols = _safe_listdir(f"{VOLUMES_ROOT}/{cat}/{sch}")
            print(f"    {sch}/  ->  volumes: {vols or '(none)'}")
            for vol in vols:
                entries = _safe_listdir(f"{VOLUMES_ROOT}/{cat}/{sch}/{vol}")
                shown = entries[:4] + (["..."] if len(entries) > 4 else [])
                print(f"      {vol}/  {len(entries)} entries  {shown}")

    spark = _spark()
    if spark is not None and catalog:
        print(f"\nSCHEMAS IN {catalog} (a schema with no volumes is invisible above)")
        for row in _rows(spark, f"SHOW SCHEMAS IN `{catalog}`"):
            print(f"  {row[0]}")
        if schema:
            print(f"\nVOLUMES IN {catalog}.{schema}")
            found = _rows(spark, f"SHOW VOLUMES IN `{catalog}`.`{schema}`")
            for row in found or [("(none)",)]:
                print(f"  {row}")

    if search:
        print(f"\nSEARCHING {VOLUMES_ROOT} FOR THE EXPECTED FILES")
        hits = _find_expected(VOLUMES_ROOT)
        if not hits:
            print("  Not found anywhere under /Volumes.")
            print("  If you uploaded through the workspace UI the files may be "
                  "in DBFS (/FileStore/...) or a workspace folder rather than "
                  "a Volume - those are different storage.")
        else:
            for path in hits:
                print(f"  FOUND: {path}")
            suggested = sorted({str(Path(p).parent) for p in hits})
            print("\n  Update configs/base.yaml paths.databricks to match, e.g.")
            for directory in suggested:
                print(f"    {directory}/<file>")


def _safe_listdir(path: str) -> list:
    import os

    try:
        return sorted(os.listdir(path))
    except Exception:                                # noqa: BLE001
        return []


def _find_expected(root: str, max_entries: int = 20000) -> list:
    """Look for the known filenames under `root`, bounded so it cannot hang."""
    import os

    hits, seen = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        seen += len(filenames) + len(dirnames)
        if seen > max_entries:
            break
        for name in filenames:
            if name in EXPECTED_NAMES or name.startswith("en_climate_daily_BC_1072692"):
                hits.append(os.path.join(dirpath, name))
                if len(hits) >= 25:
                    return hits
    return hits
