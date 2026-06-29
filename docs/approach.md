# Approach and design decisions

## Data pipeline

The raw dataset is a long-format table of timestamped sensor events. Each row records one measurement for one patient: a mood self-report, an activity reading, a screen-on duration, or an app usage session. The pipeline transforms this into a modeling-ready feature matrix in four stages.

### Stage 1: Daily aggregation

Raw events are grouped by (patient, date). Numeric variables are aggregated with type-appropriate functions: mood and subjective scores take the daily mean, activity and screen time take the daily sum, binary indicators (call, SMS) take the daily max. This produces one row per patient-day with 19 columns.

### Stage 2: Cleaning and imputation

**Outlier removal** uses a hybrid approach: domain-based rules for bounded variables (mood outside 1–10, arousal/valence outside −2 to +2) and IQR-based detection for unbounded sensor variables (activity, screen, app usage). Outliers are set to NaN rather than dropped, preserving the time series structure.

**Imputation** compares two time-series-aware methods: linear interpolation (which respects temporal ordering) and forward-fill with exponential decay. Linear interpolation was selected based on lower distortion of distributional properties across patients, particularly for mood — where forward-fill creates artificial plateaus that inflate autocorrelation.

Prolonged gaps (>3 consecutive missing days for a variable) are left as NaN rather than imputed, since interpolation over long gaps amounts to fabrication. These gaps propagate as missing features in the windowed dataset; XGBoost handles them natively, and LSTM sequences containing them are excluded.

### Stage 3: Feature engineering

A 5-day sliding window produces four features per source variable: lag-1 value, window mean, window standard deviation, and window trend (OLS slope). The window size was selected by balancing autocorrelation coverage (mood ACF plateaus by lag-5) against instance count (larger windows discard more early-period data).

The five source variables were chosen by within-patient lag-1 correlation with next-day mood, filtered at p < 0.10:
- mood (r = 0.47) — strongest predictor by far
- circumplex.valence (r = 0.27) — emotional state assessment
- activity (r = 0.12) — physical movement
- appCat.communication (r = 0.06) — social engagement proxy
- screen (r = 0.03) — marginal individually but rolling aggregation improves signal by +0.07

Variables dropped: arousal (r = 0.08 but unstable across patients), call/SMS (too sparse), social apps (negative correlation, confounded by usage patterns), remaining app categories (no significant lag-1 signal).

Additional features: `mood_lag2` (lag-2 r = 0.39 adds independent signal beyond lag-1) and day-of-week encoded as sin/cos (captures weekly mood periodicity without ordinality assumptions).

### Stage 4: Quality metadata

Each instance receives a quality tier (A/B/C) based on the fraction of non-imputed values in its 5-day window. This allows downstream modeling to assess whether performance varies by data completeness, and lets users filter to high-confidence instances.

## Model design

### Why these two models

The comparison tests a specific hypothesis: on a small clinical dataset, do learned temporal representations (LSTM on raw sequences) outperform hand-crafted temporal features (XGBoost on windowed statistics)?

**XGBoost** sees each instance as a flat 23-feature vector. It has no concept of temporal ordering within the window — that information is encoded manually in the lag, mean, std, and trend features. Its advantage: robust to small sample sizes, handles missing values natively, interpretable feature importances.

**LSTM** receives the raw 5-day sequence of daily values (5 timesteps × 5 variables). It learns its own temporal representations — what to remember, what to forget, how to weight recent vs. older observations. Its advantage: can capture nonlinear temporal dynamics that hand-crafted features miss. Its disadvantage: needs more data to learn stable representations.

### Walk-forward temporal cross-validation

Standard k-fold CV is inappropriate for time series because it lets future observations leak into training. Walk-forward CV respects temporal ordering: fold 1 trains on the first 60% of each patient's timeline and tests on the next 10%, fold 2 trains on the first 70% and tests on the next 10%, and so on. This mirrors how the model would be deployed — always predicting forward, never backward.

The same fold boundaries are used for both models so results are directly comparable.

### Classification target

Mood is discretized into three classes using patient-specific percentile thresholds (33rd and 67th percentiles). Per-patient thresholds are critical: a mood of 6 is "low" for a patient whose range is 6–9 but "medium" for a patient ranging 4–8. Using global thresholds would systematically misclassify patients with atypical mood baselines.

## What the results mean

The most useful framing: classification shows whether the models capture signal beyond autocorrelation; regression shows how much precision that signal buys.

For classification, XGBoost's macro F1 of 0.439 is 2.4× the majority baseline. The model correctly identifies mood direction (will tomorrow be a low, medium, or high day?) with meaningful accuracy, even on 27 patients. This is clinically relevant — a monitoring system doesn't need to predict the exact mood score, just whether a patient's mood is likely to dip.

For regression, the story is more humbling. XGBoost's MAE of 0.488 is only marginally better than a 5-day rolling mean (0.498). With mood autocorrelation at 0.47, most of the variance in next-day mood is already captured by recent history. The models squeeze out small additional gains from behavioral covariates, but the ceiling is low with this cohort size.

The near-parity between XGBoost and LSTM confirms the small-data hypothesis: with 27 patients and ~1,400 instances, hand-crafted features informed by domain knowledge match learned representations. The crossover point — where LSTMs or transformers would likely dominate — is at hundreds of patients, where the model has enough variation to learn patient-general temporal patterns.
