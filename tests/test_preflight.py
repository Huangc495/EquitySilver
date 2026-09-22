"""Preflight check tests.

A preflight that cannot fail is worthless, so most of these drive the failure
paths rather than the happy one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acidity_lstm.config import Config, load_config
from acidity_lstm.preflight import (
    EXPECTED_WEATHER_FILES,
    FAIL,
    OK,
    WARN,
    _parse_requirements,
    _same_version,
    check_environment,
    check_packages,
    check_paths,
    environment_report,
    run_preflight,
)

DATABRICKS_ENV = "DATABRICKS_RUNTIME_VERSION"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def cfg_with_paths(cfg: Config, **overrides) -> Config:
    paths = dict(cfg.paths)
    paths.update(overrides)
    return Config(raw=cfg.raw, paths=paths, repo_root=cfg.repo_root)


def status_of(checks, name: str) -> str:
    return next(c.status for c in checks if c.name == name)


# --- Happy path -----------------------------------------------------------

def test_preflight_passes_on_the_real_local_setup(cfg):
    passed, table = run_preflight(cfg, verbose=False)
    assert passed, table[table["status"] == FAIL].to_string()
    assert set(table["area"]) == {"environment", "data", "packages"}


def test_finds_all_the_weather_files(cfg):
    checks = check_paths(cfg)
    assert status_of(checks, "weather CSVs") == OK
    assert status_of(checks, "weather years") == OK
    assert status_of(checks, "acidity workbook") == OK


# --- Missing data ---------------------------------------------------------

def test_missing_weather_files_fail(cfg, tmp_path):
    bad = cfg_with_paths(cfg, weather_glob=str(tmp_path / "nothing_*.csv"))
    checks = check_paths(bad)
    assert status_of(checks, "weather CSVs") == FAIL
    assert "0 of 21" in next(c.detail for c in checks if c.name == "weather CSVs")


def test_partial_weather_upload_fails(cfg, tmp_path):
    """The realistic failure: an interrupted upload."""
    for year in range(1997, 2007):
        (tmp_path / f"en_climate_daily_BC_1072692_{year}_P1D.csv").write_text("x")
    bad = cfg_with_paths(
        cfg, weather_glob=str(tmp_path / "en_climate_daily_BC_1072692_*_P1D.csv")
    )
    checks = check_paths(bad)
    assert status_of(checks, "weather CSVs") == FAIL
    assert f"10 of {EXPECTED_WEATHER_FILES}" in next(
        c.detail for c in checks if c.name == "weather CSVs"
    )


def test_gap_in_the_weather_years_is_caught(cfg, tmp_path):
    """All 21 files present in count, but a year missing and one duplicated."""
    years = [y for y in range(1997, 2018) if y != 2005] + [2018]
    for year in years:
        (tmp_path / f"en_climate_daily_BC_1072692_{year}_P1D.csv").write_text("x")
    bad = cfg_with_paths(
        cfg, weather_glob=str(tmp_path / "en_climate_daily_BC_1072692_*_P1D.csv")
    )
    checks = check_paths(bad)
    assert status_of(checks, "weather CSVs") == OK, "count is right"
    assert status_of(checks, "weather years") == FAIL, "but 2005 is missing"
    assert "2005" in next(c.detail for c in checks if c.name == "weather years")


def test_missing_workbook_fails(cfg, tmp_path):
    bad = cfg_with_paths(cfg, acidity_excel=str(tmp_path / "absent.xlsx"))
    checks = check_paths(bad)
    assert status_of(checks, "acidity workbook") == FAIL
    assert "NOT FOUND" in next(c.detail for c in checks if c.name == "acidity workbook")


def test_unwritable_processed_dir_fails(cfg):
    """A path that cannot be created at all, e.g. under a regular file."""
    blocker = cfg.repo_root / "requirements.txt"
    bad = cfg_with_paths(cfg, processed_dir=str(blocker / "under_a_file"))
    checks = check_paths(bad)
    assert status_of(checks, "processed dir writable") == FAIL


def test_run_preflight_reports_failure(cfg, tmp_path):
    bad = cfg_with_paths(cfg, acidity_excel=str(tmp_path / "absent.xlsx"))
    passed, table = run_preflight(bad, verbose=False)
    assert passed is False
    assert (table["status"] == FAIL).any()


# --- Environment ----------------------------------------------------------

def test_missing_username_on_databricks_is_reported_not_raised(cfg, monkeypatch):
    """The preflight must report this, not blow up before printing anything."""
    monkeypatch.setenv(DATABRICKS_ENV, "16.4")
    monkeypatch.delenv("DATABRICKS_USERNAME", raising=False)

    checks = check_environment(cfg)
    assert status_of(checks, "mlflow experiment") == FAIL
    assert "DATABRICKS_USERNAME" in next(
        c.detail for c in checks if c.name == "mlflow experiment"
    )


def test_environment_reports_the_runtime(cfg, monkeypatch):
    monkeypatch.setenv(DATABRICKS_ENV, "16.4.x-cpu-ml-scala2.12")
    monkeypatch.setenv("DATABRICKS_USERNAME", "someone@example.com")
    checks = check_environment(cfg)
    assert status_of(checks, "running on") == OK
    assert next(c.detail for c in checks if c.name == "running on") == "Databricks"
    assert status_of(checks, "DATABRICKS_RUNTIME_VERSION") == OK


def test_local_environment_needs_no_variables(cfg, monkeypatch):
    monkeypatch.delenv(DATABRICKS_ENV, raising=False)
    monkeypatch.delenv("DATABRICKS_USERNAME", raising=False)
    checks = check_environment(cfg)
    assert all(c.ok for c in checks)


# --- Packages -------------------------------------------------------------

def test_installed_packages_match_the_pins(cfg):
    checks = check_packages(cfg)
    assert all(c.status != FAIL for c in checks), [
        (c.name, c.detail) for c in checks if c.status == FAIL
    ]


def test_version_mismatch_warns_rather_than_fails(cfg, tmp_path):
    """On a cluster the runtime's version wins, so a mismatch is not fatal."""
    (tmp_path / "requirements.txt").write_text("numpy==0.0.1\n", encoding="utf-8")
    bad = Config(raw=cfg.raw, paths=cfg.paths, repo_root=tmp_path)

    checks = check_packages(bad)
    assert status_of(checks, "numpy") == WARN
    assert "pinned" in next(c.detail for c in checks if c.name == "numpy")
    # WARN must not count towards failure.
    assert all(c.ok for c in checks)


def test_a_warning_alone_does_not_fail_the_run(cfg, monkeypatch):
    """The reports dir check is warn-only; it must never block a run."""
    from acidity_lstm import preflight

    monkeypatch.setattr(
        preflight, "check_packages",
        lambda c: [preflight.Check("packages", "fake", WARN, "mismatched")],
    )
    passed, table = run_preflight(cfg, verbose=False)
    assert passed is True
    assert (table["status"] == WARN).any()


def test_requirements_parsing():
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("# comment\nnumpy==1.26.4\ntorch==2.6.0  # inline\n\nbare-package\n")
        path = Path(fh.name)
    try:
        out = _parse_requirements(path)
        assert out == {"numpy": "1.26.4", "torch": "2.6.0"}
    finally:
        path.unlink()


def test_missing_requirements_file_is_tolerated(tmp_path):
    assert _parse_requirements(tmp_path / "absent.txt") == {}


def test_local_build_tags_are_ignored():
    """torch reports 2.6.0+cpu; the pin says 2.6.0."""
    assert _same_version("2.6.0+cpu", "2.6.0")
    assert _same_version("1.26.4", "1.26.4")
    assert not _same_version("2.6.0", "2.5.1")


# --- Environment report ---------------------------------------------------

def test_environment_report_is_a_requirements_block():
    text = environment_report()
    assert text.startswith("#")
    assert "torch==" in text
    assert "numpy==" in text
    for line in text.splitlines():
        assert line.startswith("#") or "==" in line


# --- Diagnostics ----------------------------------------------------------

def test_find_expected_locates_a_misplaced_upload(tmp_path):
    from acidity_lstm.preflight import _find_expected

    misplaced = tmp_path / "some_catalog" / "some_schema" / "landing"
    misplaced.mkdir(parents=True)
    (misplaced / "2017 ARD Chemisty_Clean.xlsx").write_text("x")
    (misplaced / "en_climate_daily_BC_1072692_1997_P1D.csv").write_text("x")
    (misplaced / "unrelated.txt").write_text("x")

    hits = _find_expected(str(tmp_path))
    assert len(hits) == 2
    assert all("unrelated" not in h for h in hits)


def test_find_expected_returns_nothing_when_absent(tmp_path):
    from acidity_lstm.preflight import _find_expected

    (tmp_path / "empty").mkdir()
    assert _find_expected(str(tmp_path)) == []


def test_safe_listdir_tolerates_a_missing_path(tmp_path):
    from acidity_lstm.preflight import _safe_listdir

    assert _safe_listdir(str(tmp_path / "absent")) == []
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    assert _safe_listdir(str(tmp_path)) == ["a", "b"]


def test_diagnose_is_safe_off_cluster(cfg, capsys):
    """Must explain itself rather than crash when /Volumes does not exist."""
    from acidity_lstm.preflight import diagnose_volumes

    diagnose_volumes(cfg, search=False)
    out = capsys.readouterr().out
    assert "CONFIGURED" in out
    if not Path("/Volumes").is_dir():
        assert "not a Databricks cluster" in out


def test_search_roots_only_returns_existing_paths():
    """Off-cluster none of the Databricks roots exist, so the list is empty."""
    from acidity_lstm.preflight import search_roots

    for path, label in search_roots():
        assert Path(path).is_dir()
        assert label


def test_dbfs_uri_conversion():
    from acidity_lstm.preflight import _to_dbfs_uri

    assert _to_dbfs_uri("/dbfs/FileStore/tables") == "dbfs:/FileStore/tables"
    assert _to_dbfs_uri("/Volumes/cat/sch/raw") == "/Volumes/cat/sch/raw"


def test_search_explains_when_nothing_is_found(capsys):
    from acidity_lstm.preflight import _search_all_storage

    _search_all_storage("mycat", "mysch")
    out = capsys.readouterr().out
    assert "SEARCHING EVERY STORAGE AREA" in out
    if not Path("/Volumes").is_dir():
        assert "run this on the cluster" in out
