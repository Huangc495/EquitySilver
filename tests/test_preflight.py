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

@pytest.mark.integration
def test_preflight_passes_on_the_real_local_setup(cfg):
    passed, table = run_preflight(cfg, verbose=False)
    assert passed, table[table["status"] == FAIL].to_string()
    assert set(table["area"]) == {"environment", "data", "packages"}


@pytest.mark.integration
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


@pytest.mark.integration
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


# --- Compute detection ---------------------------------------------------

def test_ml_runtime_detected_from_the_version_string(monkeypatch):
    from acidity_lstm.preflight import is_ml_runtime

    for version in ("16.4.x-cpu-ml-scala2.12", "15.4.x-gpu-ml-scala2.12",
                    "14.3.x-cpu-ml-scala2.12"):
        monkeypatch.setenv(DATABRICKS_ENV, version)
        assert is_ml_runtime(), version

    for version in ("16.4.x-scala2.12", "15.4.x-photon-scala2.12", ""):
        monkeypatch.setenv(DATABRICKS_ENV, version)
        assert not is_ml_runtime(), version


def test_standard_runtime_fails_with_the_fix(cfg, monkeypatch):
    """The single cause behind 'torch not installed' and 'mlflow not installed'."""
    monkeypatch.setenv(DATABRICKS_ENV, "16.4.x-scala2.12")
    monkeypatch.setenv("DATABRICKS_USERNAME", "someone@example.com")

    checks = check_environment(cfg)
    assert status_of(checks, "compute") == FAIL
    detail = next(c.detail for c in checks if c.name == "compute")
    assert "not ML" in detail
    assert "requirements-databricks.txt" in detail, "serverless is the way out here"


def test_ml_runtime_passes(cfg, monkeypatch):
    monkeypatch.setenv(DATABRICKS_ENV, "16.4.x-cpu-ml-scala2.12")
    monkeypatch.setenv("DATABRICKS_USERNAME", "someone@example.com")
    assert status_of(check_environment(cfg), "compute") == OK


def test_missing_torch_points_at_the_runtime(cfg, monkeypatch):
    from acidity_lstm import preflight

    monkeypatch.setenv(DATABRICKS_ENV, "16.4.x-scala2.12")

    def absent(name):
        if name in ("torch", "mlflow"):
            raise preflight.md.PackageNotFoundError(name)
        return "1.0.0"

    monkeypatch.setattr(preflight.md, "version", absent)
    checks = check_packages(cfg)
    for name in ("torch", "mlflow"):
        assert status_of(checks, name) == FAIL
        assert "Runtime ML" in next(c.detail for c in checks if c.name == name)


def test_missing_package_off_cluster_stays_plain(cfg, monkeypatch):
    """Locally there is no runtime to blame, so keep the message simple."""
    from acidity_lstm import preflight

    monkeypatch.delenv(DATABRICKS_ENV, raising=False)

    def absent(name):
        if name == "torch":
            raise preflight.md.PackageNotFoundError(name)
        return "1.0.0"

    monkeypatch.setattr(preflight.md, "version", absent)
    assert next(c.detail for c in check_packages(cfg) if c.name == "torch") == "not installed"


def test_runtime_check_absent_when_local(cfg, monkeypatch):
    monkeypatch.delenv(DATABRICKS_ENV, raising=False)
    assert not any(c.name == "compute" for c in check_environment(cfg))


# --- Serverless -----------------------------------------------------------

def test_serverless_is_recognised(monkeypatch):
    from acidity_lstm.preflight import is_serverless

    for version in ("client.1.13", "client.2.0", "serverless-1"):
        monkeypatch.setenv(DATABRICKS_ENV, version)
        assert is_serverless(), version

    for version in ("16.4.x-cpu-ml-scala2.12", "16.4.x-scala2.12", ""):
        monkeypatch.setenv(DATABRICKS_ENV, version)
        assert not is_serverless(), version


def test_serverless_environment_version_parsed(monkeypatch):
    from acidity_lstm.preflight import serverless_environment_version

    for version, expected in (("client.4.10", "4"), ("client.1.13", "1"),
                              ("16.4.x-cpu-ml-scala2.12", ""), ("", "")):
        monkeypatch.setenv(DATABRICKS_ENV, version)
        assert serverless_environment_version() == expected, version


def test_serverless_passes_the_compute_check(cfg, monkeypatch):
    """This workspace is serverless-only (D-33): serverless is the normal case."""
    monkeypatch.setenv(DATABRICKS_ENV, "client.4.10")
    monkeypatch.setenv("DATABRICKS_USERNAME", "someone@example.com")

    checks = check_environment(cfg)
    assert status_of(checks, "compute") == OK
    assert "serverless" in next(c.detail for c in checks if c.name == "compute")


def test_serverless_environment_mismatch_warns(cfg, monkeypatch):
    """A different environment version is worth knowing, not a stop."""
    monkeypatch.setenv(DATABRICKS_ENV, "client.2.5")
    monkeypatch.setenv("DATABRICKS_USERNAME", "someone@example.com")

    checks = check_environment(cfg)
    assert status_of(checks, "compute") == WARN
    detail = next(c.detail for c in checks if c.name == "compute")
    assert "serverless_environment_version" in detail


def test_serverless_missing_torch_points_at_the_environment(cfg, monkeypatch):
    """Telling a serverless user to create a cluster is useless: they cannot."""
    from acidity_lstm import preflight

    monkeypatch.setenv(DATABRICKS_ENV, "client.4.10")

    def absent(name):
        if name in ("torch", "mlflow"):
            raise preflight.md.PackageNotFoundError(name)
        return "1.0.0"

    monkeypatch.setattr(preflight.md, "version", absent)
    checks = check_packages(cfg)
    for name in ("torch", "mlflow"):
        assert status_of(checks, name) == FAIL
        detail = next(c.detail for c in checks if c.name == name)
        assert "requirements-databricks.txt" in detail
        assert "cluster" not in detail


def test_standard_runtime_advice_differs_from_serverless(cfg, monkeypatch):
    monkeypatch.setenv(DATABRICKS_ENV, "16.4.x-scala2.12")
    monkeypatch.setenv("DATABRICKS_USERNAME", "someone@example.com")

    detail = next(c.detail for c in check_environment(cfg) if c.name == "compute")
    assert "standard runtime" in detail


def test_databricks_requirements_file_is_tracked(cfg):
    """The serverless environment is only reproducible if its spec is in git."""
    path = cfg.repo_root / cfg["compute"]["requirements"]
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    for name in ("torch", "mlflow", "openpyxl"):
        assert name in text, name
    assert "+cpu" in text or "%2Bcpu" in text, "the CUDA build is ~2.5 GB"


# --- Username fallback ----------------------------------------------------

def test_username_falls_back_to_the_platform(monkeypatch):
    """Serverless has no cluster env vars, so ask Databricks instead."""
    from acidity_lstm import config as config_mod

    monkeypatch.delenv("DATABRICKS_USERNAME", raising=False)
    monkeypatch.setitem(config_mod._ENV_FALLBACKS, "DATABRICKS_USERNAME",
                        lambda: "resolved@example.com")
    assert config_mod.expand_env("/Users/${DATABRICKS_USERNAME}/x") == \
        "/Users/resolved@example.com/x"


def test_environment_variable_wins_over_the_fallback(monkeypatch):
    from acidity_lstm import config as config_mod

    monkeypatch.setenv("DATABRICKS_USERNAME", "explicit@example.com")
    monkeypatch.setitem(config_mod._ENV_FALLBACKS, "DATABRICKS_USERNAME",
                        lambda: "resolved@example.com")
    assert "explicit@example.com" in config_mod.expand_env("${DATABRICKS_USERNAME}")


def test_fallback_returning_nothing_still_raises(monkeypatch):
    from acidity_lstm import config as config_mod

    monkeypatch.delenv("DATABRICKS_USERNAME", raising=False)
    monkeypatch.setitem(config_mod._ENV_FALLBACKS, "DATABRICKS_USERNAME", lambda: "")
    with pytest.raises(KeyError, match="DATABRICKS_USERNAME"):
        config_mod.expand_env("${DATABRICKS_USERNAME}")


def test_current_user_is_empty_off_cluster():
    from acidity_lstm.config import databricks_current_user

    assert databricks_current_user() == ""
