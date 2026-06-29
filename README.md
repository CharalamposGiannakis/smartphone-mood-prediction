# Smartphone Mood Prediction

Predicting next-day mood from smartphone sensor data for depression monitoring. Built on a real clinical dataset of 27 patients tracked over ~2 months, with 19 sensor variables (mood self-reports, physical activity, screen time, app usage, arousal, valence). The pipeline transforms raw timestamped events into daily features using sliding windows and compares XGBoost (flat features) against LSTM (raw sequences) under walk-forward temporal cross-validation.

| Task | Model | Score | Baseline | vs. baseline |
|------|-------|-------|----------|-------------|
| Classification (3-class macro F1) | XGBoost | **0.439** | Majority: 0.182 | +141% |
| Classification (3-class macro F1) | LSTM | 0.427 | Majority: 0.182 | +134% |
| Regression (MAE) | XGBoost | **0.488** | Naive: 0.547 | +10.8% |
| Regression (MAE) | LSTM | 0.496 | Rolling-5: 0.498 | +0.4% |

## The problem

Smartphone apps for depression monitoring collect continuous sensor data — activity levels, screen usage, communication patterns — and periodically ask users to rate their mood on a 1–10 scale. The goal is to predict tomorrow's average mood from the recent behavioral history, which could enable early intervention when a patient's mood is trending downward.

What makes this hard is the data itself. Twenty-seven patients, roughly 47 mood-days each, 40% of expected mood entries missing entirely. The strongest single predictor is yesterday's mood (lag-1 r = 0.47), which means any model has to demonstrate it captures something beyond simple autocorrelation to justify its complexity.

## Dataset

The data comes from a clinical study of smartphone-based depression monitoring. Each row in the raw data is a timestamped event (a mood rating, an activity reading, a screen-on duration, an app usage session). The pipeline aggregates these to daily level, then constructs sliding-window features for modeling.

| Property | Value |
|----------|-------|
| Patients | 27 |
| Tracking period | 50–101 days per patient |
| Mood-days per patient | 30–68 (mean 47, 40% missing) |
| Sensor variables | 19 (mood, arousal, valence, activity, screen, calls, SMS, 12 app categories) |
| Modeling instances | 1,394 (after windowing) |
| Prediction target | Next-day average mood (continuous 1–10) |

## Approach

### Feature engineering

The pipeline converts the daily panel into a flat feature matrix using a 5-day sliding window. The window size was chosen based on autocorrelation analysis: mood ACF drops from 0.47 at lag-1 to ~0.30 at lag-3, then plateaus through lag-6. Five days captures the informative range while preserving training instances.

Five source variables were selected by within-patient lag correlations with next-day mood: mood itself (r = 0.47), valence (r = 0.27), activity (r = 0.12), screen time (r = 0.03), and communication app usage (r = 0.06). For each variable, four features are computed per window: lag-1 value, window mean, window standard deviation, and window trend (slope of a linear fit). Mood gets an additional lag-2 feature (r = 0.39), and day-of-week is encoded cyclically (sin/cos). Total: 23 features.

Transformations (sqrt, log1p) are applied adaptively — only when they measurably reduce skewness on the imputed data. Several transformations that looked helpful on the raw data actually overcorrected after imputation and were skipped.

### Models

**XGBoost** operates on the flat windowed feature matrix. Hyperparameters were tuned via grid search within each CV fold (learning rate, depth, regularization, subsampling).

**LSTM** receives the raw daily sequences directly — no windowing, no manual feature extraction. The network sees the 5-day history of all variables and learns temporal patterns end-to-end. This comparison tests whether learned temporal representations outperform hand-crafted features on a small clinical dataset.

### Evaluation

Walk-forward temporal cross-validation: each fold trains on all data before a cutoff date and tests on the period after it. This prevents future information from leaking into training — a common mistake when applying standard k-fold CV to time series. The same fold structure is used for both models to ensure comparable results.

For classification, mood is discretized into three classes (low / medium / high) with thresholds at the 33rd and 67th percentiles of each patient's mood distribution, preserving patient-specific baselines.

## Key findings

**Yesterday's mood dominates.** `mood_lag1` is the most important feature by a wide margin (10.5% of total XGBoost importance), followed by `mood_w5_mean` (7.2%) and `mood_lag2` (4.8%). Behavioral variables (activity, screen, communication) contribute individually less but collectively add signal — removing them drops classification F1 from 0.439 to ~0.38.

**XGBoost and LSTM perform similarly.** On this dataset size (1,394 instances for XGBoost, 660 usable sequences for LSTM), the models are roughly tied. LSTM's advantage — learning temporal representations — doesn't materialize with 27 patients. This is consistent with the small-data regime where hand-crafted features informed by domain knowledge match or beat learned representations.

**MSE amplifies a small number of bad predictions.** The top 5% of errors account for 19% of total MAE but 41% of total MSE. This means MSE-optimized models disproportionately focus on the hardest-to-predict days (mood swings, data gaps), which may or may not be the right objective depending on the clinical use case.

**Regression is near the autocorrelation ceiling.** The XGBoost regressor (MAE = 0.488) barely outperforms a 5-day rolling average (MAE = 0.498). With lag-1 autocorrelation of 0.47, most of the variance in next-day mood is explained by recent mood history alone. The models' value lies more in detecting *departures from trend* (classification) than in precise point estimates (regression).

## Project structure

```
smartphone-mood-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_eda_overview.ipynb           # Dataset shape, variable types, core statistics
│   ├── 02_daily_aggregation.ipynb      # Raw events → daily patient panel
│   ├── 03_missingness_analysis.ipynb   # 40% missing mood-days, patterns, implications
│   ├── 04_outlier_detection.ipynb      # Rule-based and statistical outlier removal
│   ├── 05_imputation.ipynb             # Time-series imputation: two methods compared
│   ├── 06_feature_engineering.ipynb    # Sliding-window features (W=5), adaptive transforms
│   ├── 07_classification.ipynb         # XGBoost vs LSTM, walk-forward CV, 3-class mood
│   ├── 08_regression.ipynb             # Same models, continuous target, baselines
│   └── 09_metric_analysis.ipynb        # MSE vs MAE sensitivity, error distribution
├── src/
│   └── feature_engineering.py          # Standalone pipeline script for feature construction
├── figures/                            # EDA and evaluation plots
├── docs/
│   └── approach.md                     # Design decisions and methodology notes
├── data/                               # Raw data (not tracked)
└── outputs/                            # Result CSVs, feature importance, confusion matrices
```

## Quick start

```bash
git clone https://github.com/CharalamposGiannakis/smartphone-mood-prediction.git
cd smartphone-mood-prediction

pip install -r requirements.txt

# Place the raw dataset CSV in data/
# Run notebooks in order: 01 → 09
# Or run the standalone feature engineering script:
python src/feature_engineering.py
```

**Requirements:** Python ≥ 3.10, pandas, numpy, scikit-learn, xgboost, torch, matplotlib, seaborn.

## Limitations

This project reflects the realities of small clinical datasets. Twenty-seven patients is a realistic cohort for an early-stage mHealth study, but it limits what any model can learn. The regression results in particular show that with this sample size, sophisticated models offer marginal gains over simple temporal heuristics. Scaling to hundreds or thousands of patients — as production mHealth platforms can — would likely shift the balance toward learned representations (LSTM or transformer-based) and away from hand-crafted features.

## License

MIT
