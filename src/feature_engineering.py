#!/usr/bin/env python3
"""
feature_engineering.py  —  Sliding-Window Feature Construction
================================================================================
IMPROVED VERSION — fixes transformation overcorrection, adds quality tiers,
adds day-of-week feature, and produces cleaner output diagnostics.
"""

# %% [markdown]
# ### Sliding-Window Feature Construction
#
# **Objective**
# Transform the daily-level imputed panel (`data_imputed.csv`) into an
# instance-based modelling dataset where each row is a (patient, day) with:
# - aggregated history features computed over a sliding window of W days,
# - next-day average mood as the prediction target.
#
# This is the dataset that feeds directly into classification and regression.
#
# ---
#
# **Key design decisions (justified by EDA notebooks 01–04)**
#
# 1. **Window size W = 5 days** — ACF of daily mood (the EDA notebooks) shows autocorrelation
#    drops from 0.47 (lag-1) to ~0.30 at lag-3 and plateaus around 0.32–0.34
#    through lag-6. A 5-day window captures the informative range while preserving
#    more training instances than W = 7.
#
# 2. **Feature variables** (5 core predictors) selected by within-patient lag
#    correlations (the EDA notebooks) and rolling improvement analysis.
#
# 3. **Adaptive transformations** — applied only when they actually reduce
#    |skewness| on the imputed daily data (not blindly from raw-level
#    recommendations).
#
# 4. **Four feature types per variable**: lag-1, window mean, window std, window
#    trend. Plus mood_lag2 (justified by lag-2 r = 0.393).
#
# 5. **Day-of-week** as a cyclical feature (no missingness, captures weekly
#    mood patterns).
#
# 6. **Quality tier metadata** so downstream modelling can filter by instance
#    completeness.

# %% [markdown]
# #### 1. Configuration

# %%
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR   = Path("../data")
OUTPUT_DIR = Path("../outputs")
FIG_DIR    = Path("../figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE     = OUTPUT_DIR / "data_imputed.csv"
TRANSFORM_FILE = OUTPUT_DIR / "transformation_recommendations.csv"

# ── Hyperparameters ────────────────────────────────────────────────────
W = 5                    # window size in days
MIN_WINDOW_COVERAGE = 3  # min non-NaN days in window to compute rolling stats

# ── Feature variables ─────────────────────────────────────────────────
FEATURE_VARS = [
    "mood",
    "circumplex.valence",
    "activity",
    "screen",
    "appCat.communication",
]

# Candidate transformations (from b_07 / transformation_recommendations.csv)
# These were computed on raw record-level data. We will validate them on the
# daily-aggregated imputed data and skip any that worsen skewness.
CANDIDATE_TRANSFORMS = {
    "activity":              "sqrt",
    "screen":                "log1p",
    "appCat.communication":  "log1p",
}

print(f"Window size W = {W}")
print(f"Min window coverage = {MIN_WINDOW_COVERAGE}/{W}")
print(f"Feature variables: {FEATURE_VARS}")
print(f"Candidate transforms: {CANDIDATE_TRANSFORMS}")

# %% [markdown]
# #### 2. Load Data

# %%
df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
df = df.sort_values(["id", "date"]).reset_index(drop=True)
print(f"Loaded {len(df)} rows, {df['id'].nunique()} patients, "
      f"date range {df['date'].min().date()} – {df['date'].max().date()}")

# %% [markdown]
# #### 3. Adaptive Transformations
#
# The transformation recommendations from b_07 were computed on raw record-level
# data (`data_cleaned.csv`). However, `data_imputed.csv` is daily-aggregated and
# forward-filled, which already compresses the distribution. A log1p on top of
# that can overcorrect into heavy left skew — which is worse than doing nothing.
#
# **Rule**: apply the transform only if `|transformed_skew| < |raw_skew|`.
# Otherwise, keep the raw values.

# %%
APPLIED_TRANSFORMS = {}   # tracks which transforms were actually applied

for var, transform in CANDIDATE_TRANSFORMS.items():
    raw_skew = df[var].skew()

    # Compute candidate
    if transform == "log1p":
        transformed = np.log1p(df[var])
    elif transform == "sqrt":
        transformed = np.sqrt(df[var])
    else:
        raise ValueError(f"Unknown transform: {transform}")

    new_skew = transformed.skew()

    # Adaptive check: only apply if it actually helps
    if abs(new_skew) < abs(raw_skew):
        col_t = f"{var}_t"
        df[col_t] = transformed
        APPLIED_TRANSFORMS[var] = transform
        status = "✓ APPLIED"
    else:
        status = "✗ SKIPPED (overcorrects)"

    print(f"  {var:30s}  {transform:5s}  "
          f"skew {raw_skew:+.3f} → {new_skew:+.3f}  "
          f"|{abs(raw_skew):.3f}| → |{abs(new_skew):.3f}|  {status}")

# Build the mapping: variable name → column to use for windowing
VAR_COL = {}
for v in FEATURE_VARS:
    if v in APPLIED_TRANSFORMS:
        VAR_COL[v] = f"{v}_t"
    else:
        VAR_COL[v] = v

print(f"\nApplied transforms: {APPLIED_TRANSFORMS}")
print(f"Skipped transforms: "
      f"{set(CANDIDATE_TRANSFORMS) - set(APPLIED_TRANSFORMS)}")
print("\nVariable → column mapping for windowing:")
for v, c in VAR_COL.items():
    print(f"  {v:30s} → {c}")

# %% [markdown]
# #### 4. Sliding-Window Feature Construction
#
# For each patient, sorted by date:
# - `lag_1`    = value at day t (most recent signal, lag-1 relative to target)
# - `w5_mean`  = rolling mean of [t−W+1, ..., t]  (smoothed level)
# - `w5_std`   = rolling std of [t−W+1, ..., t]   (recent variability / emotional inertia)
# - `w5_trend` = value(t) − value(t−W+1)          (directional change)
#
# All features use data up to and including day t.
# The target is mood at day t+1 (next calendar day, with gap check).

# %%
feature_frames = []

for pid, grp in df.groupby("id"):
    grp = grp.sort_values("date").copy()

    for var in FEATURE_VARS:
        col = VAR_COL[var]
        safe = var.replace(".", "_")   # safe column name for output

        # Lag features
        grp[f"{safe}_lag1"] = grp[col].values           # value at day t
        if var == "mood":
            grp[f"{safe}_lag2"] = grp[col].shift(1)      # value at day t−1

        # Rolling window features (window ending at day t inclusive)
        roll = grp[col].rolling(window=W, min_periods=MIN_WINDOW_COVERAGE)
        grp[f"{safe}_w{W}_mean"]  = roll.mean()
        grp[f"{safe}_w{W}_std"]   = roll.std()

        # Trend: value(t) − value(t − W + 1)
        grp[f"{safe}_w{W}_trend"] = grp[col].values - grp[col].shift(W - 1).values

    # ── Day-of-week (cyclical encoding) ───────────────────────────────
    # Captures weekly patterns in mood reporting without introducing a
    # high-cardinality categorical.  No missingness by construction.
    dow = grp["date"].dt.dayofweek  # 0=Mon, 6=Sun
    grp["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    grp["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # ── Target: next-day mood ─────────────────────────────────────────
    grp["target_next_day_mood"] = grp["mood"].shift(-1)

    # Gap validity check: invalidate if next row is not exactly +1 day
    day_gap = grp["date"].shift(-1) - grp["date"]
    grp.loc[day_gap != pd.Timedelta(days=1), "target_next_day_mood"] = np.nan

    # Target date (the day being predicted)
    grp["target_date"] = grp["date"].shift(-1)
    grp.loc[day_gap != pd.Timedelta(days=1), "target_date"] = pd.NaT

    feature_frames.append(grp)

df_feat = pd.concat(feature_frames, ignore_index=True)
print(f"Feature frame: {df_feat.shape}")

# %% [markdown]
# #### 5. Assemble Final Modelling Dataset
#
# Keep only rows that have a valid target. Add quality tier metadata so
# downstream modelling notebooks can filter by instance completeness.

# %%
# Identify feature columns (lag/window features + day-of-week)
feat_cols = [c for c in df_feat.columns
             if (any(c.startswith(v.replace(".", "_"))
                     for v in FEATURE_VARS)
                 and ("_lag" in c or "_w" in c))
             or c in ("dow_sin", "dow_cos")]
feat_cols = sorted(set(feat_cols))

meta_cols = ["id", "date", "target_date", "target_next_day_mood"]

# Drop rows with no valid target
df_model = df_feat[meta_cols + feat_cols].copy()
n_before = len(df_model)
df_model = df_model.dropna(subset=["target_next_day_mood"])
print(f"Dropped {n_before - len(df_model)} rows without valid target")

# ── Quality tier ──────────────────────────────────────────────────────
# Count missing features per instance (excluding dow which is never NaN)
predictor_cols = [c for c in feat_cols if c not in ("dow_sin", "dow_cos")]
df_model["n_feat_missing"] = df_model[predictor_cols].isna().sum(axis=1)
n_predictors = len(predictor_cols)

# Tier A: fully complete (0 missing)
# Tier B: mostly complete (≤50% missing, i.e. at least lag1 for most vars)
# Tier C: sparse (>50% missing)
df_model["quality_tier"] = pd.cut(
    df_model["n_feat_missing"],
    bins=[-1, 0, n_predictors / 2, n_predictors],
    labels=["A_complete", "B_partial", "C_sparse"]
)

print(f"\n{'='*60}")
print(f"Modelling dataset summary")
print(f"{'='*60}")
print(f"Total instances            : {len(df_model)}")
print(f"Patients                   : {df_model['id'].nunique()}")
print(f"Features                   : {len(feat_cols)}")
print(f"\nFeature columns:")
for c in feat_cols:
    pct_miss = df_model[c].isna().mean() * 100
    print(f"  {c:40s}  missing {pct_miss:5.1f}%")

print(f"\nQuality tier distribution:")
tier_counts = df_model["quality_tier"].value_counts().sort_index()
for tier, count in tier_counts.items():
    print(f"  {tier:15s}: {count:5d}  ({count/len(df_model)*100:5.1f}%)")

print(f"\nTarget stats:")
print(df_model["target_next_day_mood"].describe().to_string())
print(f"\nInstances per patient:")
print(df_model.groupby("id").size().describe().to_string())

# %% [markdown]
# #### 6. Instance Quality Analysis
#
# Detailed breakdown of what drives the missingness tiers.

# %%
# Per-tier feature coverage
print("Feature coverage by quality tier:\n")
for tier in ["A_complete", "B_partial", "C_sparse"]:
    sub = df_model[df_model["quality_tier"] == tier]
    if len(sub) == 0:
        continue
    non_null_pct = sub[predictor_cols].notna().mean() * 100
    print(f"--- Tier {tier} ({len(sub)} instances) ---")
    for c in predictor_cols:
        print(f"  {c:40s}  available {non_null_pct[c]:5.1f}%")
    print()

# Which features are the bottleneck?
print("Overall feature availability (non-NaN %):")
avail = df_model[predictor_cols].notna().mean().sort_values() * 100
for c, pct in avail.items():
    print(f"  {c:40s}  {pct:5.1f}%")

# How many Tier A instances per patient?
print(f"\nTier A instances per patient:")
tier_a = df_model[df_model["quality_tier"] == "A_complete"]
tier_a_per_pat = tier_a.groupby("id").size()
print(tier_a_per_pat.describe().to_string())

# %% [markdown]
# #### 7. Export

# %%
OUTPUT_FILE = OUTPUT_DIR / "data_modelling.csv"

# Keep quality_tier in the export — modelling notebooks can filter on it
df_export = df_model.drop(columns=["n_feat_missing"])
df_export.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved modelling dataset to {OUTPUT_FILE}")
print(f"  Shape: {df_export.shape}")
print(f"  Columns: {list(df_export.columns)}")

# Feature metadata table for the report
feat_meta = []
for var in FEATURE_VARS:
    safe = var.replace(".", "_")
    applied_t = APPLIED_TRANSFORMS.get(var, "none")
    candidate_t = CANDIDATE_TRANSFORMS.get(var, "none")
    if var in CANDIDATE_TRANSFORMS and var not in APPLIED_TRANSFORMS:
        transform_note = f"{candidate_t} skipped (overcorrects on imputed data)"
    else:
        transform_note = applied_t

    feat_meta.append({
        "variable": var,
        "transform_applied": transform_note,
        "features": ", ".join([c for c in predictor_cols if c.startswith(safe)]),
        "lag1_r_with_next_mood": {
            "mood": 0.472,
            "circumplex.valence": 0.273,
            "activity": 0.118,
            "screen": 0.034,
            "appCat.communication": 0.057,
        }.get(var, None),
        "rolling_improvement": {
            "mood": -0.052,
            "circumplex.valence": -0.087,
            "activity": -0.040,
            "screen": +0.067,
            "appCat.communication": +0.061,
        }.get(var, None),
        "justification": {
            "mood": "Strongest single predictor; lag-2 also included (r=0.393)",
            "circumplex.valence": "Second strongest within-patient predictor",
            "activity": "Significant within-patient signal; sqrt-transformed",
            "screen": "Weak lag-1 but rolling aggregation improves by +0.067",
            "appCat.communication": "Rolling improves by +0.061; collinearity with screen (r≈0.75) flagged",
        }.get(var, ""),
    })
pd.DataFrame(feat_meta).to_csv(OUTPUT_DIR / "feature_metadata.csv", index=False)
print(f"Saved feature metadata to {OUTPUT_DIR / 'feature_metadata.csv'}")

# %% [markdown]
# #### 8. Sanity Checks

# %%
print(f"\n{'='*60}")
print("Sanity checks")
print(f"{'='*60}")

# Check: no future leakage
assert (df_export["target_date"] > df_export["date"]).all(), \
    "LEAKAGE: target_date must be strictly after feature date"
print("✓ No future leakage: target_date > date for all rows")

# Check: gap validity
gaps = (pd.to_datetime(df_export["target_date"]) -
        pd.to_datetime(df_export["date"])).dt.days
assert (gaps == 1).all(), \
    "GAP ERROR: all target dates must be exactly 1 day after feature date"
print("✓ All target dates are exactly +1 calendar day")

# Check: target range
assert df_export["target_next_day_mood"].between(1, 10).all(), \
    "TARGET ERROR: mood outside [1, 10]"
print("✓ All target values in [1, 10]")

# Check: no patient cross-contamination
for pid in df_export["id"].unique()[:3]:
    sub = df_export[df_export["id"] == pid]
    assert sub["date"].is_monotonic_increasing, \
        f"SORT ERROR: dates not monotonic for {pid}"
print("✓ Dates monotonically increasing within patients (spot-checked)")

# Check: day-of-week has no missing values
assert df_export[["dow_sin", "dow_cos"]].notna().all().all(), \
    "DOW ERROR: day-of-week features should never be NaN"
print("✓ Day-of-week features fully populated")

print(f"\n✅  complete. Modelling dataset ready at {OUTPUT_FILE}")
print(f"   → {len(df_export)} instances, {len(feat_cols)} features")
print(f"   → Tier A (complete): {(df_export['quality_tier'] == 'A_complete').sum()}")
print(f"   → Tier B (partial):  {(df_export['quality_tier'] == 'B_partial').sum()}")
print(f"   → Tier C (sparse):   {(df_export['quality_tier'] == 'C_sparse').sum()}")

# %% [markdown]
# ---
# ### Summary for Report
#
# **Feature engineering approach ()**
#
# We constructed an instance-based modelling dataset by applying a sliding
# window of W=5 days over the daily imputed panel. Each instance corresponds
# to a (patient, day) pair and contains aggregated features from the preceding
# 5 days, with next-day average mood as the prediction target.
#
# **Variable selection** was driven by within-patient lag correlations (the EDA notebooks):
# the five retained predictors all showed statistically significant lag-1
# correlations with next-day mood, or demonstrated meaningful improvement
# under rolling aggregation. Variables with only between-patient signal
# (e.g., appCat.entertainment) or inconsistent significance
# (circumplex.arousal) were excluded.
#
# **Transformations** were validated on the daily-aggregated imputed data rather
# than applied blindly from the raw-level recommendations. Only transformations
# that reduced |skewness| were applied (sqrt for activity). Log1p was skipped
# for screen and appCat.communication because daily aggregation and forward-fill
# imputation had already compressed their distributions, and log1p overcorrected
# into heavy left skew.
#
# **Four feature types** were generated per variable: lag-1 (most recent value),
# window mean (smoothed level), window standard deviation (recent mood
# variability / emotional inertia), and window trend (directional change over
# the 5-day period). An additional lag-2 feature was included for mood,
# justified by its strong lag-2 autocorrelation (r = 0.393). Day-of-week was
# encoded cyclically (sin/cos) to capture weekly mood patterns without
# adding a high-cardinality categorical.
#
# **Target validity**: instances were excluded when the next calendar day was
# not consecutive (gap > 1 day), preventing the model from training on
# temporally disconnected mood observations.
#
# **Instance quality**: each instance carries a quality tier (A/B/C) based on
# feature completeness. Tier A instances (fully complete) form the clean core
# for linear models; tree-based models (XGBoost, LightGBM) can use all tiers
# since they handle NaN natively. This design avoids discarding data while
# maintaining transparency about feature coverage.
#
# **Collinearity**: screen and appCat.communication have inter-feature r ≈ 0.75.
# Both are retained for tree-based models; for linear approaches, one should be
# dropped or PCA applied.
