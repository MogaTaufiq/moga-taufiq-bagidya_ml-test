"""src/metrics.py — Evaluation metrics, all computed in ORIGINAL space (seconds).

Always inverse-transform predictions (expm1) before calling these — never evaluate
in log space. Definitions follow METHODOLOGY.md §6.

  * mae          — robust, direct interpretation (seconds).
  * rmse         — penalises large errors.
  * mape_segment — % error per segment, guarded against y≈0 (drops sub-floor rows).
  * smape        — symmetric MAPE, robust to small actuals, drops nothing.
  * loop_mae     — THE business metric: error on the total time of one loop (no_do).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape_segment(y_true, y_pred, eps_floor: float = 5.0) -> float:
    """Mean absolute percentage error (%), guarded against near-zero actuals.

    MAPE explodes when y -> 0. We exclude segments with y < eps_floor seconds
    (too short to carry meaningful percentage error) rather than let them blow up.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    keep = y_true >= eps_floor
    if keep.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[keep] - y_pred[keep]) / y_true[keep]) * 100)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE (%): bounded in [0, 200], robust to small actuals, drops nothing."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2 * np.abs(y_true - y_pred) / denom) * 100)


def loop_mae(df: pd.DataFrame, y_true_col: str, y_pred_col: str,
             loop_col: str = "no_do", complete_only: bool = True,
             gap_col: str = "is_gap_suspected", min_segments: int = 2) -> tuple[float, int]:
    """Loop MAE (MAE putaran): error on the TOTAL time of one loop (no_do).

    Per loop: sum actual & predicted segment times, take |diff|, average over loops.
    This is the metric transit operations care about — a model can look good per
    segment yet accumulate same-direction error across a loop; Loop MAE exposes that.

    complete_only : restrict to loops without suspected gaps (apple-to-apple totals).
    min_segments  : ignore fragment "loops" with fewer than this many segments.

    Returns (loop_mae_seconds, n_loops_used).
    """
    cols = [loop_col, y_true_col, y_pred_col]
    if gap_col in df.columns:
        cols.append(gap_col)
    d = df[cols].copy()
    if complete_only and gap_col in d.columns:
        d = d[~d[gap_col].astype(bool)]
    g = d.groupby(loop_col).agg(
        tot_true=(y_true_col, "sum"),
        tot_pred=(y_pred_col, "sum"),
        n=(y_true_col, "size"),
    )
    g = g[g["n"] >= min_segments]
    if len(g) == 0:
        return float("nan"), 0
    err = (g["tot_true"] - g["tot_pred"]).abs()
    return float(err.mean()), int(len(g))


def all_segment_metrics(y_true, y_pred) -> dict:
    """Convenience: segment-level metrics as a dict (loop_mae needs the frame, called separately)."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_seg": mape_segment(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }
