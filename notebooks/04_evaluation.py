# %% [markdown]
# # 04 — Evaluation: final model (XGBoost MAE-log)
#
# Metrik final di test set, lift vs baseline, dan plot diagnostik. Memuat artifact tersimpan
# (`outputs/model_xgb.json`) — tanpa retrain.

# %%
import sys; sys.path.insert(0, "..")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, xgboost as xgb
from src.data import load_raw, add_gap_flag, clean_outliers, time_split, sequence_ready
from src import features as F, models as Mdl, metrics as M

clean, _ = clean_outliers(add_gap_flag(load_raw()), verbose=False)
feat = F.build_base_features(clean); tr, te = time_split(feat)
tr, te = F.add_dwell_feature(tr, te)              # D12 dwell terminal
art = F.fit_feature_artifacts(tr); tr = F.transform_with_artifacts(tr, art); te = F.transform_with_artifacts(te, art)  # D13 bus_encoded
te_c = sequence_ready(te)
model = xgb.XGBRegressor(); model.load_model("../outputs/model_xgb.json")
Xc, _, yc = Mdl.make_xy(te_c)
pred = Mdl.predict_sec(model, Xc)
base = F.predict_baseline(te_c, art["baseline_model"])

# %% [markdown]
# ## Metrik final (common test set, dalam detik)

# %%
final = pd.DataFrame([Mdl.evaluate("Baseline (seg,hour mean)", yc, base, te_c),
                      Mdl.evaluate("XGBoost (final)", yc, pred, te_c)])
final["LoopMAE_lift_%"] = [0.0, round(100 * (final.LoopMAE_s[0] - final.LoopMAE_s[1]) / final.LoopMAE_s[0], 1)]
final

# %% [markdown]
# ## Diagnostik: pred vs actual, distribusi error, MAE per jam

# %%
err = pred - yc
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].scatter(yc, pred, s=3, alpha=0.2); axs[0].plot([0, 1500], [0, 1500], "r--")
axs[0].set(title="pred vs actual (s)", xlim=(0, 1500), ylim=(0, 1500), xlabel="actual", ylabel="pred")
axs[1].hist(np.clip(err, -200, 200), bins=80, color="#16a085")
axs[1].set(title="distribusi error (s)", xlabel="pred - actual")
by_h = te_c.assign(ae=np.abs(err)).groupby("hour")["ae"].mean()
axs[2].bar(by_h.index, by_h.values, color="#d35400"); axs[2].set(title="MAE per jam", xlabel="hour", ylabel="MAE (s)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Feature Importance Model Final
# Hijau = fitur baru iterasi v2 (D12 `time_since_prev_arrival_sec` + D13 `bus_encoded`).

# %%
imp_df = pd.DataFrame({'fitur': F.FEATURE_COLS, 'importance': model.feature_importances_}).sort_values('importance', ascending=True)
clr = ['#27ae60' if f in ('time_since_prev_arrival_sec','bus_encoded') else '#3498db' for f in imp_df['fitur']]
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(imp_df['fitur'], imp_df['importance'], color=clr)
for b, v in zip(bars, imp_df['importance']):
    ax.text(v+0.003, b.get_y()+b.get_height()/2, f"{v:.3f}", va='center', fontsize=9)
ax.set_xlabel("Importance (gain-based)"); ax.set_title("Feature Importance — XGBoost MAE-log (17 fitur)")
fig.tight_layout(); plt.show()

# %% [markdown]
# ## Loop MAE — akurasi satu putaran (metrik bisnis utama)

# %%
lm, n = M.loop_mae(te_c.assign(_p=pred), "traveling_time_sec", "_p")
lmb, _ = M.loop_mae(te_c.assign(_p=base), "traveling_time_sec", "_p")
print(f"Loop MAE  baseline {lmb:.1f}s  ->  XGBoost {lm:.1f}s   (lift {100*(lmb-lm)/lmb:.1f}%, {n} loop)")
print(f"Segment   MAE {M.mae(yc,pred):.1f}s | RMSE {M.rmse(yc,pred):.1f}s | "
      f"MAPE {M.mape_segment(yc,pred):.1f}% | sMAPE {M.smape(yc,pred):.1f}%")
