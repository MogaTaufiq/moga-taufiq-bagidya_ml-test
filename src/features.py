"""src/features.py — Feature engineering (leakage-aware).

CRITICAL separation, enforced by the column lists at the bottom:
  * FEATURE_COLS     -> model INPUTS. None contains the current row's target.
  * OUTPUT_ONLY_COLS -> deviation_ratio etc. Required as output (Task 4) and useful for
                         anomaly analysis, but NEVER fed to a model: they contain
                         traveling_time_sec (the target) in the numerator -> leakage.
  * average_time_sec / arrival_time -> dropped from features (D2 leakage, D11 post-hoc).

Leakage-free aggregates (baseline_segment_hour, rolling_mean_segment_5) are built on
TIME-ORDERED data with shift(1) so each row sees only its own past. Train-fit stats
(segment_volatility, category codes, the baseline MODEL) are fit on train and mapped to test.

Intended order:  clean_df -> build_base_features -> time_split -> fit_feature_artifacts(train)
                 -> transform_with_artifacts(train/test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "traveling_time_sec"
LOG_TARGET = "log_traveling_time"
LOOP_KEY = "no_do"
TIME_COL = "departure_time"
RUSH_HOURS = {6, 7, 8, 9, 16, 17, 18, 19}   # Jakarta peaks: ~06-09 (AM) & ~16-19 (PM)


# ---------------------------------------------------------------- time features
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """hour / day_of_week / is_weekend / is_rush_hour + cyclic hour encoding.

    All derived from `departure_time` (known at prediction; arrival_time is post-hoc, D11).
    """
    df = df.copy()
    t = df[TIME_COL].dt
    df["hour"] = t.hour
    df["day_of_week"] = t.dayofweek                       # 0=Mon .. 6=Sun
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin(RUSH_HOURS).astype(int)
    # cyclic: hour 23 and 00 are adjacent on a clock but far numerically
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


# ------------------------------------------------ leakage-free past aggregates
def add_leakage_free_baseline(df: pd.DataFrame, target: str = TARGET, k_roll: int = 5) -> pd.DataFrame:
    """baseline_segment_hour (expanding mean per (segment,hour), shifted) and
    rolling_mean_segment_5 (last-k mean per segment, shifted). Both PAST-ONLY.

    Sorts by time so 'expanding/rolling' means 'over the past'. shift(1) guarantees the
    current row is excluded -> no target leakage. Computed on the FULL frame before
    splitting so test rows continue the history accumulated through train.
    """
    df = df.sort_values(TIME_COL).copy()
    g_sh = df.groupby(["segment_id", "hour"])[target]
    df["baseline_segment_hour"] = g_sh.transform(lambda s: s.expanding().mean().shift(1))
    g_s = df.groupby("segment_id")[target]
    df["rolling_mean_segment_5"] = g_s.transform(
        lambda s: s.shift(1).rolling(k_roll, min_periods=1).mean()
    )
    # cold-start (first occurrence in a group) -> fill with past-only global mean, then a constant
    global_exp = df[target].expanding().mean().shift(1)
    for col in ["baseline_segment_hour", "rolling_mean_segment_5"]:
        df[col] = df[col].fillna(global_exp)
        df[col] = df[col].fillna(df[target].median())     # only the very first row overall
    return df


# ----------------------------------------------------------- structural feature
def add_trip_progress(df: pd.DataFrame) -> pd.DataFrame:
    """trip_progress = stop_sequence / max(stop_sequence) within the loop (0..1).

    Position in the loop (start / middle / end) correlates with traffic context. No target.
    """
    df = df.copy()
    max_seq = df.groupby(LOOP_KEY)["stop_sequence"].transform("max")
    df["trip_progress"] = (df["stop_sequence"] / max_seq).clip(0, 1)
    return df


# --------------------------------------------- output-only anomaly columns (D2)
def add_deviation_ratios(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """deviation_ratio (vs brief's average_time_sec — REQUIRED output, Task 4) and
    deviation_ratio_clean (vs our leakage-free baseline).

    OUTPUT ONLY — both have the target in the numerator, so they must NEVER be model
    features. Guards divide-by-zero -> NaN.
    """
    df = df.copy()
    avg = df["average_time_sec"].replace(0, np.nan)
    df["deviation_ratio"] = df[target] / avg
    base = df["baseline_segment_hour"].replace(0, np.nan)
    df["deviation_ratio_clean"] = df[target] / base
    return df


# ----------------------------------------------- per-bus dwell / idle proxy (D12)
def add_dwell_feature(train: pd.DataFrame, test: pd.DataFrame,
                      time_col: str = TIME_COL,
                      arrival_col: str = "arrival_time") -> tuple[pd.DataFrame, pd.DataFrame]:
    """time_since_prev_arrival_sec = current departure - previous row's arrival,
    per bus_body_no (sorted by departure). Leakage-free: only uses past arrivals
    (previous row), which are known before predicting the current segment.

    Computed on the COMBINED train+test frame so the first test row of each bus uses
    the last train arrival as its previous (no cold-start NaN at the split boundary).
    The median for cold-start fill is taken from TRAIN only.
    """
    a = train.assign(_s="tr"); b = test.assign(_s="te")
    both = pd.concat([a, b]).sort_values(["bus_body_no", time_col]).copy()
    prev_arr = both.groupby("bus_body_no")[arrival_col].shift(1)
    both["time_since_prev_arrival_sec"] = (
        (both[time_col] - prev_arr).dt.total_seconds().clip(0, 86400)
    )
    med = float(both.loc[both["_s"] == "tr", "time_since_prev_arrival_sec"].median())
    both["time_since_prev_arrival_sec"] = both["time_since_prev_arrival_sec"].fillna(med)
    tr_out = both[both["_s"] == "tr"].drop(columns="_s").sort_index()
    te_out = both[both["_s"] == "te"].drop(columns="_s").sort_index()
    return tr_out, te_out


# ----------------------------------------------- train-fit stats (fit/transform)
def fit_feature_artifacts(train: pd.DataFrame, target: str = TARGET, smoothing_m: int = 20) -> dict:
    """Fit everything that must come from TRAIN ONLY: segment volatility, category
    vocabularies, the smoothed (segment,hour) baseline MODEL (D9 sparsity), and the
    Bayesian-smoothed bus_body_no target encoding (D13)."""
    g_mean = float(train[target].mean())
    bs = train.groupby("bus_body_no")[target].agg(["mean", "count"])
    bus_encoding = (
        (bs["count"] * bs["mean"] + smoothing_m * g_mean) / (bs["count"] + smoothing_m)
    ).to_dict()
    return {
        "segment_volatility": train.groupby("segment_id")[target].std().to_dict(),
        "global_volatility": float(train[target].std()),
        "segment_categories": sorted(train["segment_id"].unique().tolist()),
        "trip_categories": sorted(train["trip_id"].unique().tolist()),
        "baseline_model": fit_segment_hour_baseline(train, target, smoothing_m),
        "bus_encoding": bus_encoding,
        "global_target_mean": g_mean,
    }


def transform_with_artifacts(df: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Map train-fit stats onto df (train or test). Unseen categories -> code -1;
    unseen buses -> global target mean (no panic, no inf)."""
    df = df.copy()
    df["segment_volatility"] = (
        df["segment_id"].map(art["segment_volatility"]).fillna(art["global_volatility"])
    )
    seg_idx = {c: i for i, c in enumerate(art["segment_categories"])}
    trip_idx = {c: i for i, c in enumerate(art["trip_categories"])}
    df["segment_id_code"] = df["segment_id"].map(seg_idx).fillna(-1).astype(int)
    df["trip_id_code"] = df["trip_id"].map(trip_idx).fillna(-1).astype(int)
    df["bus_encoded"] = df["bus_body_no"].map(art["bus_encoding"]).fillna(art["global_target_mean"])
    return df


# -------------------------------------------------- baseline MODEL (the floor)
def fit_segment_hour_baseline(train: pd.DataFrame, target: str = TARGET, smoothing_m: int = 20) -> dict:
    """Bayesian-smoothed mean traveling_time per (segment,hour) from TRAIN, with a
    hierarchical fallback (segment -> global).

    Smoothing pulls low-count cells toward the global mean so we don't trust a noisy
    average from a handful of observations (D9 sparsity):
        smooth = (n*mean_group + m*global) / (n + m)
    """
    global_mean = float(train[target].mean())
    seg_mean = train.groupby("segment_id")[target].mean().to_dict()
    grp = train.groupby(["segment_id", "hour"])[target].agg(["mean", "count"])
    smooth = (grp["count"] * grp["mean"] + smoothing_m * global_mean) / (grp["count"] + smoothing_m)
    return {"global": global_mean, "seg": seg_mean, "seg_hour": smooth.to_dict()}


def predict_baseline(df: pd.DataFrame, baseline: dict) -> np.ndarray:
    """Vectorised hierarchical lookup: (segment,hour) -> segment -> global."""
    keys = list(zip(df["segment_id"], df["hour"]))
    out = pd.Series(keys, index=df.index).map(baseline["seg_hour"])
    out = out.fillna(df["segment_id"].map(baseline["seg"]))
    out = out.fillna(baseline["global"])
    return out.astype(float).values


# ------------------------------------------------------------------ orchestrate
def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all LEAKAGE-FREE, non-train-dependent features (safe on the full frame
    before splitting), plus the log1p target."""
    df = add_time_features(df)
    df = add_leakage_free_baseline(df)
    df = add_trip_progress(df)
    df = add_deviation_ratios(df)
    df[LOG_TARGET] = np.log1p(df[TARGET])
    return df


# Model inputs (17 features). NOTE: cast is_gap_suspected -> int when building the
# design matrix. The two trailing features (dwell + bus encoding) are added by
# add_dwell_feature() and transform_with_artifacts() respectively — they don't come
# from build_base_features (they depend on the train/test split).
FEATURE_COLS = [
    "hour", "day_of_week", "is_weekend", "is_rush_hour", "hour_sin", "hour_cos",
    "stop_sequence", "trip_progress", "loop_n_segments", "is_gap_suspected",
    "baseline_segment_hour", "rolling_mean_segment_5", "segment_volatility",
    "segment_id_code", "trip_id_code",
    "time_since_prev_arrival_sec", "bus_encoded",
]
OUTPUT_ONLY_COLS = ["deviation_ratio", "deviation_ratio_clean"]   # contain target -> not features
LEAKAGE_DROP = ["average_time_sec", "arrival_time", "from_arrival_time_str"]
