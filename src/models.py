"""src/models.py — Model training, evaluation, and the fair-comparison harness.

All models are scored on the SAME common test set (complete test loops, where Loop
MAE is well-defined) with the SAME metrics, in ORIGINAL space (predictions expm1'd
back to seconds). Tabular models train on ALL train rows (gapped loops kept, D4);
the LSTM (added during modeling) trains on complete loops only.

Regularisation choices tie back to sparsity (D9): min_child_weight / min_child_samples
keep leaves from forming on a handful of rare (segment,hour) rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from . import metrics as M

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"
SEED = 42


def make_xy(df: pd.DataFrame):
    """Design matrix from FEATURE_COLS (is_gap_suspected -> int). Returns (X, y_log, y_sec)."""
    X = df[F.FEATURE_COLS].copy()
    X["is_gap_suspected"] = X["is_gap_suspected"].astype(int)
    return X, df[F.LOG_TARGET].to_numpy(), df[F.TARGET].to_numpy()


def evaluate(name: str, y_sec, pred_sec, loop_df: pd.DataFrame) -> dict:
    """Standard metric row on (true_sec, pred_sec); Loop MAE uses loop_df grouping."""
    seg = M.all_segment_metrics(y_sec, pred_sec)
    lm, n = M.loop_mae(loop_df.assign(_pred=pred_sec), F.TARGET, "_pred")
    return {"Model": name, "MAE_s": round(seg["MAE"], 2), "RMSE_s": round(seg["RMSE"], 2),
            "MAPE_seg_%": round(seg["MAPE_seg"], 2), "LoopMAE_s": round(lm, 2), "n_loops": n}


# ----------------------------------------------------------------- tabular models
def train_xgb(Xtr, ytr_log, objective: str = "reg:squarederror", **kw):
    """XGBoost regressor on the log target. objective: squared / pseudohuber / absolute."""
    import xgboost as xgb
    params = dict(n_estimators=400, max_depth=7, learning_rate=0.08, subsample=0.9,
                  colsample_bytree=0.9, min_child_weight=5, n_jobs=-1,
                  random_state=SEED, objective=objective)
    params.update(kw)
    m = xgb.XGBRegressor(**params)
    m.fit(Xtr, ytr_log)
    return m


def train_lgbm(Xtr, ytr_log, **kw):
    """LightGBM regressor on the log target (fast contrast model)."""
    import lightgbm as lgb
    params = dict(n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.9,
                  subsample_freq=1, colsample_bytree=0.9, min_child_samples=30,
                  n_jobs=-1, random_state=SEED, verbose=-1)
    params.update(kw)
    m = lgb.LGBMRegressor(**params)
    m.fit(Xtr, ytr_log)
    return m


def predict_sec(model, X, log_clip: float = 11.0) -> np.ndarray:
    """Predict in log space, return seconds. Clip log preds to [0, log_clip] before
    expm1 for numerical safety (expm1(11) ~ 6e4 s, far above any plausible segment)."""
    pred_log = np.clip(model.predict(X), 0.0, log_clip)
    return np.expm1(pred_log)


def compare_table(rows: list[dict]) -> pd.DataFrame:
    """Sort metric rows by the primary business metric (Loop MAE)."""
    return pd.DataFrame(rows).sort_values("LoopMAE_s").reset_index(drop=True)


# --------------------------------------------------------------------- LSTM (PyTorch)
# Sequence model over each loop (no_do), ordered by stop_sequence. Trains on COMPLETE
# loops only (D4). Post-padded; the loss is masked per-timestep via each loop's length
# (padding contributes 0). PyTorch is used instead of TF/Keras: TF 2.16's ml_dtypes is
# not NumPy-2 compatible in this env (fallback pre-declared in DECISIONS D0).
def lstm_max_len(*frames) -> int:
    return int(max(f.groupby("no_do").size().max() for f in frames))


def _scaled_matrix(df, feature_cols, scaler):
    X = df[feature_cols].copy()
    X["is_gap_suspected"] = X["is_gap_suspected"].astype(int)
    return scaler.transform(X.to_numpy(dtype=float))


def build_lstm_arrays(df, feature_cols, max_len, scaler):
    """Per complete loop -> padded X (n,max_len,nfeat), y_log (n,max_len,1), lengths,
    and the ordered frame whose row order matches a flattened prediction."""
    d = df.sort_values(["no_do", "stop_sequence"]).reset_index(drop=True)
    Xs = _scaled_matrix(d, feature_cols, scaler)
    ylog = d[F.LOG_TARGET].to_numpy()
    nfeat = Xs.shape[1]
    order = list(dict.fromkeys(d["no_do"]))       # first-appearance == sorted order
    pos = {nod: i for i, nod in enumerate(order)}
    X = np.zeros((len(order), max_len, nfeat), dtype="float32")
    y = np.zeros((len(order), max_len, 1), dtype="float32")
    lengths = np.zeros(len(order), dtype=int)
    for nod, idx in d.groupby("no_do", sort=False).indices.items():
        i = pos[nod]; L = min(len(idx), max_len); sel = idx[:L]
        X[i, :L, :] = Xs[sel]; y[i, :L, 0] = ylog[sel]; lengths[i] = L
    return X, y, lengths, d


def build_lstm_model(nfeat, units=(64, 32), seed=SEED):
    """Two stacked LSTMs (64 -> 32) + a per-timestep Linear head. Returns an nn.Module."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class LSTMReg(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.LSTM(nfeat, units[0], batch_first=True)
            self.l2 = nn.LSTM(units[0], units[1], batch_first=True)
            self.head = nn.Linear(units[1], 1)

        def forward(self, x):
            o, _ = self.l1(x)
            o, _ = self.l2(o)
            return self.head(o)              # (B, T, 1)

    return LSTMReg()


def train_lstm(model, X, y, lengths, epochs=12, batch_size=64, lr=1e-3,
               val_frac=0.1, seed=SEED, verbose=True):
    """Train with Huber loss masked per-timestep by each loop's length (padding -> 0)."""
    import torch
    import torch.nn as nn
    g = torch.Generator().manual_seed(seed)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    T = Xt.shape[1]
    mask = (torch.arange(T)[None, :] < torch.tensor(lengths)[:, None]).float().unsqueeze(-1)
    idx = torch.randperm(len(Xt), generator=g)
    n_val = int(len(Xt) * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    huber = nn.HuberLoss(reduction="none", delta=1.0)

    def masked(pred, tgt, m):
        return (huber(pred, tgt) * m).sum() / m.sum()

    for ep in range(epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        tot = nb = 0
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = masked(model(Xt[b]), yt[b], mask[b])
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if verbose:
            model.eval()
            with torch.no_grad():
                vl = masked(model(Xt[val_idx]), yt[val_idx], mask[val_idx]).item()
            print(f"  epoch {ep+1:2d}/{epochs}  train={tot/nb:.4f}  val={vl:.4f}")
    return model


def lstm_predict_sec(model, df, feature_cols, max_len, scaler):
    """Return (ordered_df, pred_sec) aligned row-for-row (real steps only, in seconds)."""
    import torch
    X, _, lengths, d = build_lstm_arrays(df, feature_cols, max_len, scaler)
    model.eval()
    with torch.no_grad():
        P = model(torch.tensor(X)).numpy()[..., 0]      # (n, max_len)
    chunks = [np.expm1(np.clip(P[i, :L], 0, 11)) for i, L in enumerate(lengths)]
    return d, np.concatenate(chunks)
