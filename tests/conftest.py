"""Shared test setup: the `integration` marker and what happens without data.

Tests marked `integration` read the real Equity Silver data, which is
gitignored and never reaches CI. CI deselects them with
`pytest -m "not integration"` (DEVIATIONS.md D-34).

Run anywhere else without the data, they are skipped with a reason rather
than erroring one by one. Set ACIDITY_REQUIRE_DATA=1 where the data is
supposed to be present, so a missing upload fails loudly instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acidity_lstm.config import load_config  # noqa: E402


def _missing_data() -> list[str]:
    cfg = load_config()
    excel = cfg.resolve("acidity_excel")
    return [] if excel.exists() else [str(excel)]


def pytest_collection_modifyitems(config, items):
    integration = [item for item in items if "integration" in item.keywords]
    if not integration:
        return
    missing = _missing_data()
    if not missing:
        return
    if os.environ.get("ACIDITY_REQUIRE_DATA") == "1":
        raise pytest.UsageError(
            f"ACIDITY_REQUIRE_DATA=1 but the raw data is missing: {missing[0]}"
        )
    skip = pytest.mark.skip(
        reason=f"integration: needs the real data ({missing[0]} not found)"
    )
    for item in integration:
        item.add_marker(skip)
