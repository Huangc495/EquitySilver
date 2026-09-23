"""Replication-report assembly tests (CLAUDE.md Phase 11)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acidity_lstm.config import Config, load_config
from acidity_lstm.report_phase11 import (
    HEADING,
    LIKELY_EFFECT,
    count_tests,
    parse_deviations,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# --- Deviation parsing ----------------------------------------------------

def test_every_deviation_is_parsed(cfg):
    df = parse_deviations(cfg)
    assert len(df) >= 30
    assert set(df.columns) >= {"id", "deviation", "status", "likely effect"}
    assert df["id"].is_unique


def test_parsed_ids_match_the_source_file(cfg):
    """The report's list must be the one in DEVIATIONS.md, not a copy."""
    text = (cfg.repo_root / "DEVIATIONS.md").read_text(encoding="utf-8")
    ids_in_file = {m.group(1) for m in HEADING.finditer(text)}
    assert set(parse_deviations(cfg)["id"]) == ids_in_file


def test_every_deviation_has_a_likely_effect(cfg):
    """A new deviation with no effect assessment must fail loudly."""
    df = parse_deviations(cfg)
    assert df["likely effect"].notna().all()
    assert (df["likely effect"].str.len() > 10).all()


def _cfg_rooted_at(cfg, path) -> Config:
    """A Config pointing at a temporary repo root (Config is frozen)."""
    return Config(raw=cfg.raw, paths=cfg.paths, repo_root=path)


def test_missing_effect_assessment_raises(cfg, tmp_path):
    """A new deviation with no effect assessment must fail, not pass silently."""
    (tmp_path / "DEVIATIONS.md").write_text(
        "### D-99 A brand new deviation — *active*\n\nSome reasoning.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="D-99"):
        parse_deviations(_cfg_rooted_at(cfg, tmp_path))


def test_empty_deviations_file_raises(cfg, tmp_path):
    (tmp_path / "DEVIATIONS.md").write_text("# Nothing here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No deviation headings"):
        parse_deviations(_cfg_rooted_at(cfg, tmp_path))


def test_deviations_are_sorted_by_kind_then_number(cfg):
    df = parse_deviations(cfg).reset_index(drop=True)
    kinds = df["id"].str[0].tolist()
    assert kinds == sorted(kinds), "D entries should precede Q entries"

    d_nums = [int(i.split("-")[1]) for i in df["id"] if i.startswith("D")]
    assert d_nums == sorted(d_nums)


def test_no_stale_effect_entries(cfg):
    """Every curated effect should correspond to a real deviation."""
    ids = set(parse_deviations(cfg)["id"])
    assert set(LIKELY_EFFECT) - ids == set(), "LIKELY_EFFECT has stale entries"


def test_heading_regex_handles_the_real_formats():
    samples = [
        "### D-01 Dependency pins are not yet confirmed — *open*",
        "### Q-02 Type B is a strict subset of Type A — *resolved in Phase 6*",
        "### D-04 Markdown tables use `tabulate` — *active*",
    ]
    for line in samples:
        m = HEADING.match(line)
        assert m is not None, line
        assert m.group(1).startswith(("D-", "Q-"))
        assert m.group(2) and m.group(3)


def test_heading_regex_ignores_other_headings():
    for line in ("### Phase 6 headline results", "## Data handling", "# Deviations"):
        assert HEADING.match(line) is None


# --- Test counting --------------------------------------------------------

def test_count_tests_is_plausible(cfg):
    n_tests, n_files = count_tests(cfg)
    assert n_files >= 8
    assert n_tests >= 150
    # This file is one of them, so the count must include it.
    assert (cfg.repo_root / "tests" / "test_report_phase11.py").exists()


# --- Paper targets --------------------------------------------------------

def test_paper_constants_match_claude_md():
    """The targets the report compares against must be the published ones."""
    from acidity_lstm.experiments import PAPER_FC_R, PAPER_FIG6_R, PAPER_LSTM_R, PAPER_TABLE1
    from acidity_lstm.forecast import PAPER_FORECAST

    assert PAPER_TABLE1[("BD", "A")] == {5: 0.79, 10: 0.76, 20: 0.76}
    assert PAPER_TABLE1[("BD", "B")] == {5: 0.59, 10: 0.54, 20: 0.54}
    assert PAPER_TABLE1[("C7", "A")] == {5: 0.85, 10: 0.83, 20: 0.81}
    assert PAPER_TABLE1[("C7", "B")] == {5: 0.77, 10: 0.74, 20: 0.73}
    assert PAPER_FIG6_R == {"BD": 0.70, "C7": 0.51}
    assert PAPER_FC_R == {"BD": 0.64, "C7": 0.42}
    assert PAPER_LSTM_R == {"BD": 0.70, "C7": 0.51}
    assert PAPER_FORECAST == {"train_val_mse": 0.28, "predict_mse": 0.23}


def test_refined_targets_match_claude_md():
    from acidity_lstm.experiments import PAPER_ORIGINAL, PAPER_REFINED

    assert PAPER_REFINED == {"r": 0.86, "mse": 0.26}
    assert PAPER_ORIGINAL == {"r": 0.70, "mse": 0.54}
