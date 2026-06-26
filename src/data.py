"""src/data.py — Load, validate, clean, and split the raw GPS tracking dataset.

Single source of truth for turning the raw parquet into an analysis/training frame.
Pipeline order mirrors the notebooks:

    load_raw -> validate_raw -> add_gap_flag -> clean_outliers -> time_split

Feature engineering lives in src/features.py; metrics in src/metrics.py.

Key facts baked in (verified in notebooks/01_eda, recorded in DECISIONS.md):
  * no_do = one physical loop (median 19 ordered segments); trip_id = route variant.
  * traveling_time_sec == arrival_time - from_arrival_time_str  (target = timestamp gap),
    so arrival_time is post-hoc -> time features must use departure only (D11).
  * average_time_sec is target leakage -> never used raw (clean baseline in features, D2).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "AI_Engineer_dataset.parquet"

EXPECTED_COLS = [
    "bus_body_no", "segment_id", "route_code", "trip_id", "stop_sequence",
    "traveling_time_sec", "from_arrival_time_str", "arrival_time",
    "average_time_sec", "no_do",
]

# --- thresholds approved at GATE A (see DECISIONS.md D3, D6) ---
OUTLIER_MAX_SEC = 3600           # drop segments slower than 1 hour (physically impossible)
TEST_SPLIT_DATE = "2026-02-22"   # test = last 7 days. Departures span Feb 1-28 (all of
#                                  February, 28d); arrival_time's "Mar 7" was outlier-inflated.
LOOP_KEY = "no_do"


def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw parquet, validate schema, and parse the two timestamps.

    Adds `departure_time` (parsed from from_arrival_time_str = segment START) and
    parses `arrival_time` (segment END). No filtering / feature logic here.
    """
    df = pd.read_parquet(path)
    missing = set(EXPECTED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing expected columns: {sorted(missing)}")
    df = df.copy()
    df["departure_time"] = pd.to_datetime(df["from_arrival_time_str"])
    df["arrival_time"] = pd.to_datetime(df["arrival_time"])
    return df


def validate_raw(df: pd.DataFrame) -> dict:
    """Assert the invariants we rely on + return headline structural facts.

    Catches silent drift if the source data is ever re-dumped differently
    (e.g. if a new dump breaks the "no_do is a loop" or "target = gap" assumption).
    """
    # Invariant 1: target == arrival - departure  -> segment travel is well-defined.
    delta = (df["arrival_time"] - df["departure_time"]).dt.total_seconds()
    max_err = float((delta - df["traveling_time_sec"]).abs().max())
    assert max_err < 2.0, f"target != (arrival - departure); max diff {max_err:.2f}s"
    assert (delta >= 0).all(), "found arrival_time earlier than departure_time"

    return {
        "rows": len(df),
        "n_no_do": int(df[LOOP_KEY].nunique()),
        "n_trip_id": int(df["trip_id"].nunique()),
        "n_segment_id": int(df["segment_id"].nunique()),
        "n_route_code": int(df["route_code"].nunique()),
        "nulls_total": int(df.isna().sum().sum()),
        "time_min": df["departure_time"].min(),
        "time_max": df["departure_time"].max(),
        "target_vs_timestamp_max_diff_sec": max_err,
        "median_segments_per_loop": float(df.groupby(LOOP_KEY).size().median()),
    }


def add_gap_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add `is_gap_suspected` and `loop_n_segments` per loop (Task 4 column; D4).

    `is_gap_suspected` is True for every row of a `no_do` whose ordered
    stop_sequence skips a number (consecutive diff > 1). Computed on the RAW source
    order, BEFORE outlier removal, so it reflects genuine source incompleteness —
    not gaps we ourselves introduce by dropping outlier segments.

    Sorts rows by (no_do, stop_sequence): the natural order for LSTM sequences and
    Loop-MAE summation.
    """
    df = df.sort_values([LOOP_KEY, "stop_sequence"]).copy()
    gap_per_loop = (
        df.groupby(LOOP_KEY)["stop_sequence"]
        .apply(lambda s: bool((s.diff().dropna() > 1).any()))
    )
    df["is_gap_suspected"] = df[LOOP_KEY].map(gap_per_loop).astype(bool)
    df["loop_n_segments"] = df.groupby(LOOP_KEY)[LOOP_KEY].transform("size")
    return df


def clean_outliers(df: pd.DataFrame, max_sec: int = OUTLIER_MAX_SEC,
                   verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Drop physically-impossible segments (> max_sec). Returns (clean_df, report).

    DECISIONS D3: a single BRT inter-stop segment cannot legitimately exceed 1 hour;
    such rows are sensor errors / parked buses. The remaining right-skew (genuine
    congestion <= 1h) is handled by log1p at modeling time, NOT by dropping.
    """
    n0 = len(df)
    mask = df["traveling_time_sec"] <= max_sec
    clean = df[mask].copy()
    report = {
        "rows_before": n0,
        "rows_after": len(clean),
        "rows_dropped": int((~mask).sum()),
        "pct_dropped": round(100 * float((~mask).mean()), 3),
        "max_sec_threshold": max_sec,
    }
    if verbose:
        print(f"[clean_outliers] dropped {report['rows_dropped']} rows "
              f"({report['pct_dropped']}%) with traveling_time_sec > {max_sec}s; "
              f"{report['rows_after']} rows remain.")
    return clean, report


def time_split(df: pd.DataFrame, split_date: str = TEST_SPLIT_DATE,
               time_col: str = "departure_time",
               loop_col: str = LOOP_KEY) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loop-aware time-based train/test split (DECISIONS D6).

    Each loop (`no_do`) is assigned WHOLE to train or test by its EARLIEST departure,
    so a loop never straddles the boundary — which would both leak across the split
    and corrupt that loop's Loop-MAE total. Test = loops that started on/after
    `split_date`; train sees nothing that started in the test window.
    """
    cut = pd.Timestamp(split_date)
    loop_start = df.groupby(loop_col)[time_col].transform("min")
    train = df[loop_start < cut].copy()
    test = df[loop_start >= cut].copy()
    return train, test


def sequence_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Subset usable for LSTM / Loop-MAE: complete loops (no gap) with > 1 segment.

    Per D4 the gap only breaks order-aware uses; this helper carves out that subset
    without touching the tabular frame (which keeps gapped loops + the flag).
    """
    return df[(~df["is_gap_suspected"]) & (df["loop_n_segments"] > 1)].copy()
