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
