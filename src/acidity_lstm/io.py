"""Reading the raw Environment Canada weather CSVs and the acidity workbook.

Both readers only standardise names, types and units. All cleaning (calendar
reindexing, duplicate handling, date filtering) lives in `preprocess.py`.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

# Environment Canada column -> our name. Flags are kept alongside.
_EC_COLUMNS = {
    "Date/Time": "date",
    "Total Precip (mm)": "precip_mm",
    "Mean Temp (°C)": "tmean_c",
    "Total Precip Flag": "precip_flag",
    "Mean Temp Flag": "tmean_flag",
    "Station Name": "station_name",
    "Climate ID": "climate_id",
}

# Columns kept only for the Phase 0 audit.
_EC_AUDIT_EXTRA = {
    "Max Temp (°C)": "tmax_c",
    "Min Temp (°C)": "tmin_c",
    "Total Rain (mm)": "rain_mm",
    "Total Snow (cm)": "snow_cm",
    "Snow on Grnd (cm)": "snow_grnd_cm",
}


def weather_files(cfg: Config) -> list[Path]:
    """All yearly weather CSVs matching the configured glob, sorted by name."""
    pattern = str(cfg.resolve("weather_glob"))
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No weather CSVs matched {pattern!r}")
    return [Path(f) for f in files]


def read_weather_raw(cfg: Config, keep_extra: bool = False) -> pd.DataFrame:
    """Concatenate the yearly Environment Canada CSVs into one frame.

    Returns columns `date, precip_mm, tmean_c, precip_flag, tmean_flag,
    station_name, climate_id` (plus the audit extras when `keep_extra`).
    Values are left exactly as recorded: flags are not yet applied.
    """
    wanted = dict(_EC_COLUMNS)
    if keep_extra:
        wanted.update(_EC_AUDIT_EXTRA)

    frames = []
    for path in weather_files(cfg):
        # utf-8-sig strips the BOM that Environment Canada writes.
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
        missing = [c for c in wanted if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")
        df = df[list(wanted)].rename(columns=wanted)
        df["source_file"] = path.name
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], format="%Y-%m-%d")
    for col in ("precip_mm", "tmean_c", *(_EC_AUDIT_EXTRA.values() if keep_extra else ())):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("precip_flag", "tmean_flag"):
        out[col] = out[col].fillna("").str.strip()
    return out.sort_values("date").reset_index(drop=True)


def read_acidity_raw(cfg: Config) -> pd.DataFrame:
    """Read both station sheets of the acidity workbook into long format.

    The workbook has three header rows: a long name, a unit, and the short
    field name actually used (row 3). Data starts on row 4. Dates are stored
    as Excel serial numbers.

    Returns `station, date, acidity_mgL, sheet, excel_row` with `acidity_mgL`
    still as raw text so that non-numeric entries can be audited.
    """
    path = cfg.resolve("acidity_excel")
    sheet_map: dict[str, str] = cfg["data"]["acidity_sheets"]

    frames = []
    for sheet, station in sheet_map.items():
        # header=2 -> the short-name row ("Station Name", "ACIDITY", ...).
        df = pd.read_excel(path, sheet_name=sheet, header=2, dtype=object)
        df.columns = [str(c).strip() for c in df.columns]
        if "ACIDITY" not in df.columns:
            raise ValueError(f"Sheet {sheet!r} has no ACIDITY column; got {list(df.columns)}")

        date_col = _find_date_column(df)
        out = pd.DataFrame(
            {
                "station": station,
                "date": _to_datetime(df[date_col]),
                "acidity_mgL": df["ACIDITY"],
                "sheet": sheet,
                # +4: one-based Excel rows, three header rows skipped.
                "excel_row": np.arange(len(df)) + 4,
            }
        )
        frames.append(out.dropna(subset=["date"], how="all"))

    return pd.concat(frames, ignore_index=True)


def _find_date_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "date" in col.lower():
            return col
    raise ValueError(f"No date-like column among {list(df.columns)}")


def _to_datetime(series: pd.Series) -> pd.Series:
    """Convert a mixed column of Excel serials / datetimes / strings to dates.

    Excel serials are converted on the 1899-12-30 origin (the 1900 system as
    Excel actually implements it, including its leap-year quirk).
    """
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    is_serial = numeric.notna()
    if is_serial.any():
        out.loc[is_serial] = pd.to_datetime(
            numeric[is_serial], unit="D", origin="1899-12-30"
        )
    if (~is_serial).any():
        out.loc[~is_serial] = pd.to_datetime(series[~is_serial], errors="coerce")
    # Measurements carry a collection time; the model works at daily resolution.
    return out.dt.normalize()
