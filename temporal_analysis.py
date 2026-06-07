"""
Time Series Analysis Module
Temporal pattern detection and forecasting for SDN monitoring

Dataset assumptions (network_traffic_synthetic_temporal.csv):
  - Timestamp column, 1-minute resolution, already sorted
  - Raw (unscaled) byte/packet values
  - Pre-built lag, rolling, ewm, diff columns
  - IsAnomaly_roll* columns are EXCLUDED from features (label leakage)

Future SDN Application:
  - Real-time traffic prediction
  - Proactive anomaly detection
  - Capacity planning and auto-scaling
  - Dynamic flow routing based on predicted patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

ROLLING_WINDOW = 10
ROLLING_TARGET = 'BytesSent'


LAG_STEPS = [1, 2, 3, 4, 5]


def add_rolling_features(detector, window: int = ROLLING_WINDOW) -> list:
    """
    Compute lag (t-1 to t-5) and rolling (mean, std, trend) features of
    BytesSent and attach them to detector.df before the train/test split.
    Must be called before prepare_data() so features are included in scaling.
    Returns the list of added column names.
    """
    col = ROLLING_TARGET
    if col not in detector.df.columns:
        print(f"  Warning: '{col}' not found — temporal features skipped")
        return []

    series = detector.df[col]
    past   = series.shift(1)   # shift by 1 — no look-ahead into current row

    features = {}

    # Lag features: exact past values at fixed steps
    for lag in LAG_STEPS:
        features[f'{col}_lag{lag}'] = series.shift(lag)

    # Rolling features: statistical summaries over the window
    features[f'{col}_roll{window}_mean']  = past.rolling(window).mean()
    features[f'{col}_roll{window}_std']   = past.rolling(window).std()
    features[f'{col}_roll{window}_trend'] = series.diff()

    added = []
    for name, values in features.items():
        detector.df[name] = values.fillna(0)
        if name not in detector.expected_features:
            detector.expected_features.append(name)
        added.append(name)

    print(f"\n  Temporal features added (lags 1-{max(LAG_STEPS)}, window={window}, target={col}):")
    for name in added:
        print(f"    {name}")

    return added


TRAFFIC_COLS = [
    'BytesSent', 'BytesReceived', 'PacketsSent', 'PacketsReceived',
    'Duration', 'TotalBytes', 'TotalPackets', 'Throughput',
]

CALENDAR_COLS = [
    'Hour_sin', 'Hour_cos', 'DoW_sin', 'DoW_cos',
    'IsWeekend', 'IsNight', 'IsPeakHour',
]

TARGET_PREFERENCE = ['TotalBytes', 'BytesSent', 'BytesReceived', 'PacketsSent']


def _get_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return df with a proper DatetimeIndex.
    Handles: already a DatetimeIndex, Timestamp column, or no timestamp at all.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if 'Timestamp' in df.columns:
        df = df.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.set_index('Timestamp').sort_index()
        return df
    # Fallback: synthesise a 1-minute index
    df = df.copy()
    df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
    return df


def _pick_target(df: pd.DataFrame) -> str:
    """Return the best available forecast target column."""
    for col in TARGET_PREFERENCE:
        if col in df.columns:
            return col
    raise ValueError(
        f"No forecast target found. DataFrame must contain at least one of: "
        f"{TARGET_PREFERENCE}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sliding time windows
# ─────────────────────────────────────────────────────────────────────────────

def create_time_windows(detector) -> pd.DataFrame:
    """
    Create sliding time windows for temporal analysis across five time horizons
    (t1=1 min, t2=2 min, t3=3 min, t4=4 min, t5=5 min).

    Each horizon captures a different granularity of traffic pattern:
      t1 — immediate (last 1 minute)  : catches sudden bursts
      t2 — very short (2 minutes)     : smooths single-point noise
      t3 — short (3 minutes)          : balances reactivity and stability
      t4 — medium-short (4 minutes)   : emerging trend detection
      t5 — short-term (5 minutes)     : stable short-term pattern baseline

    Note: Rolling features are computed on the full dataset for EDA and
    visualisation only. Do not use these as model features without recomputing
    them on the training split alone to avoid look-ahead bias.

    Returns:
        pd.DataFrame: Multi-horizon windowed features + spike flags
    """
    print("\n" + "=" * 80)
    print("TIME SERIES ANALYSIS - Multi-Horizon Temporal Pattern Detection")
    print("=" * 80)
    print("\nSDN Future Application:")
    print("  • Predict traffic spikes before they occur")
    print("  • Enable proactive rerouting and load balancing")
    print("  • Auto-scale network resources based on predictions")
    print("\nTime horizons analysed:")
    print("  t1 = 1 min  → immediate burst detection")
    print("  t2 = 2 min  → noise-smoothed short signal")
    print("  t3 = 3 min  → emerging trend detection")
    print("  t4 = 4 min  → medium-short pattern baseline")
    print("  t5 = 5 min  → stable short-term context")

    df = _get_timestamp_index(detector.df)

    # Select meaningful traffic columns that are actually present
    cols_to_use = [c for c in TRAFFIC_COLS if c in df.columns][:5]
    if not cols_to_use:
        cols_to_use = detector._get_feature_columns()[:5]

    TIME_HORIZONS = [1, 2, 3, 4, 5]   # t1 through t5

    print(f"\n Creating t1-t5 rolling windows over {len(df):,} time points...")
    print(f"  Traffic columns : {cols_to_use}")

    temporal_features = pd.DataFrame(index=df.index)

    for col in cols_to_use:
        for t, w in enumerate(TIME_HORIZONS, start=1):
            # Rolling mean — central tendency at each horizon
            temporal_features[f'{col}_t{t}_mean'] = (
                df[col].rolling(w, min_periods=1).mean()
            )
            # Rolling std — volatility at each horizon
            # min_periods=w ensures std is only computed with a full window;
            # fewer than w points gives NaN which is filled with 0 (no data yet)
            temporal_features[f'{col}_t{t}_std'] = (
                df[col].rolling(w, min_periods=w).std().fillna(0)
            )

        # Rate of change (diff) — same regardless of window
        temporal_features[f'{col}_trend'] = df[col].diff()

    # Spike detection across all five horizons
    # A spike at horizon t_n = change exceeds 4σ of that window's std
    spike_candidates = [c for c in ['BytesSent', 'BytesReceived', 'PacketsSent']
                        if c in df.columns]
    spike_summary = {}
    for col in spike_candidates:
        diff_series = df[col].diff().abs()
        for t, w in enumerate(TIME_HORIZONS, start=1):
            # Compare the change magnitude against the rolling std of the diff
            # series so both sides of the comparison are on the same scale
            rolling_std_diff = diff_series.rolling(w, min_periods=w).std().fillna(
                diff_series.std()
            )
            spike_col = f'{col}_t{t}_spike'
            temporal_features[spike_col] = (
                diff_series > 4 * rolling_std_diff
            ).astype(int)
            spike_summary[spike_col] = int(temporal_features[spike_col].sum())

    # Print spike summary per horizon
    print(f"\n  Spike detection summary (threshold = 4 × rolling σ):")
    print(f"  {'Horizon':<10} {'Column':<20} {'Spikes':>8} {'Rate':>8}")
    print(f"  {'-'*48}")
    for spike_col, count in spike_summary.items():
        parts  = spike_col.rsplit('_', 2)   # e.g. ['BytesSent', 't1', 'spike']
        col    = parts[0]
        horizon= parts[1]
        rate   = count / len(df) * 100
        print(f"  {horizon:<10} {col:<20} {count:>8,} {rate:>7.1f}%")

    print(f"\n  Total features created: {len(temporal_features.columns)}")

    _plot_temporal_patterns(df, temporal_features, cols_to_use)

    return temporal_features


# ─────────────────────────────────────────────────────────────────────────────
# 2. Traffic forecasting
# ─────────────────────────────────────────────────────────────────────────────

def predict_future_traffic(detector, steps_ahead: int = 10) -> dict:
    """
    Predict future network traffic using a Random Forest regressor.

    Key design decisions vs. the original:
      • NO resampling — data is already at 1-min resolution with pre-built
        lag/rolling/ewm features; resampling to 5-min bins destroyed ~80 % of
        rows and misaligned lag semantics.
      • NO lag rebuild — the dataset already contains TotalBytes_lag1/5/15/60,
        roll5/15/60, ewm5/30, and diff columns; we use them directly.
      • IsAnomaly_roll* columns are EXCLUDED — they are derived from the label
        and would constitute data leakage into the feature set.
      • Chronological 70/15/15 train/val/test split — no shuffle.
      • Future extrapolation respects the 1-min cadence for calendar features.

    Args:
        detector    : NetworkAnomalyDetector instance
        steps_ahead : Number of 1-minute steps to forecast beyond the dataset

    Returns:
        dict with keys: target, mae, mse, rmse, directional_accuracy,
                        lead_time, predictions, future_predictions
    """
    print("\n" + "=" * 80)
    print("TRAFFIC FORECASTING - Predictive Analytics")
    print("=" * 80)
    print("\nSDN Future Application:")
    print("  • Anticipate congestion windows before they occur")
    print("  • Pre-configure flow tables for predicted load")
    print("  • Auto-scale SDN controller resources proactively")

    df = _get_timestamp_index(detector.df)
    target_feature = _pick_target(df)

    print(f"\nTarget feature  : '{target_feature}'")
    print(f"Dataset range   : {df.index.min()} → {df.index.max()}")
    print(f"Total rows      : {len(df):,}")

    # ── Build feature set ────────────────────────────────────────────────────
    # Use pre-built lag/rolling/ewm columns if they exist in the dataset.
    # If none are found, build basic lag and rolling features on the fly.
    lag_cols = [c for c in df.columns
                if ('lag' in c or 'diff' in c) and 'IsAnomaly' not in c]
    roll_cols = [c for c in df.columns
                 if ('roll' in c or 'ewm' in c) and 'IsAnomaly' not in c]
    cal_cols  = [c for c in CALENDAR_COLS if c in df.columns]

    feature_cols = lag_cols + roll_cols + cal_cols
    seen = set()
    feature_cols = [c for c in feature_cols
                    if c != target_feature and not (c in seen or seen.add(c))]

    if not feature_cols:
        # No pre-built features — construct lag and rolling features now
        print("\n  No pre-built lag/roll columns found — building features on the fly...")
        series = df[target_feature]
        built  = pd.DataFrame(index=df.index)
        for lag in [1, 2, 3, 5, 10]:
            built[f'{target_feature}_lag{lag}'] = series.shift(lag)
        for w in [5, 10, 20]:
            built[f'{target_feature}_roll{w}_mean'] = series.shift(1).rolling(w).mean()
            built[f'{target_feature}_roll{w}_std']  = series.shift(1).rolling(w).std()
        built[f'{target_feature}_diff1'] = series.diff(1)
        df = pd.concat([df, built], axis=1)
        feature_cols = list(built.columns)
        lag_cols, roll_cols, cal_cols = feature_cols, [], []

    print(f"\nFeature columns : {len(feature_cols)}")
    print(f"  Lag           : {len(lag_cols)}")
    print(f"  Roll/EWM      : {len(roll_cols)}")
    print(f"  Calendar      : {len(cal_cols)}")

    features = df[feature_cols].dropna()
    target   = df[target_feature].loc[features.index]

    # ── Chronological 70 / 15 / 15 split ─────────────────────────────────
    n         = len(features)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train, y_train = features.iloc[:train_end],        target.iloc[:train_end]
    X_val,   y_val   = features.iloc[train_end:val_end], target.iloc[train_end:val_end]
    X_test,  y_test  = features.iloc[val_end:],          target.iloc[val_end:]

    print(f"\nChronological split (no shuffle):")
    print(f"  Train : {len(X_train):,} rows  ({df.index[0]} → "
          f"{features.index[train_end - 1]})")
    print(f"  Val   : {len(X_val):,}  rows")
    print(f"  Test  : {len(X_test):,}  rows  (→ {features.index[-1]})")

    # ── Train ─────────────────────────────────────────────────────────────
    print("\nTraining Random Forest forecaster...")
    forecaster = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    forecaster.fit(X_train, y_train)

    val_preds = forecaster.predict(X_val)
    val_mae   = mean_absolute_error(y_val, val_preds)
    print(f"  Validation MAE : {val_mae:,.2f} bytes")

    # ── Test evaluation ───────────────────────────────────────────────────
    predictions = forecaster.predict(X_test)

    mae  = mean_absolute_error(y_test, predictions)
    mse  = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    # Directional accuracy: does the model predict the correct up/down trend?
    true_dir      = np.sign(np.diff(y_test.values))
    pred_dir      = np.sign(np.diff(predictions))
    directional_acc = float((true_dir == pred_dir).mean() * 100)

    # Anomaly threshold: derived from TRAINING data only — prevents test leakage
    spike_threshold = float(y_train.quantile(0.95))
    actual_spikes   = y_test.values > spike_threshold
    pred_spikes     = predictions   > spike_threshold

    # Forward-looking lead time:
    # When the model predicts a spike at step i, how many steps until the
    # next real spike arrives?  A lead > 0 means the system raised an alert
    # BEFORE the anomaly occurred — that is the proactive detection capability.
    lead_times = []
    for i in range(len(pred_spikes)):
        if pred_spikes[i]:
            for horizon in range(steps_ahead + 1):
                j = i + horizon
                if j < len(actual_spikes) and actual_spikes[j]:
                    lead_times.append(horizon)
                    break   # count only the nearest upcoming anomaly

    avg_lead_time    = float(np.mean(lead_times)) if lead_times else 0.0
    max_lead_time    = int(max(lead_times))        if lead_times else 0
    n_advance_alerts = sum(1 for lt in lead_times if lt > 0)
    n_same_step      = sum(1 for lt in lead_times if lt == 0)

    # ── Future forecast (autoregressive, 1-min steps) ─────────────────────
    # We seed from the last real row and roll forward step by step.
    # For each step we update the lag-1 slot; lag-5/15/60 and roll/ewm
    # columns are held at their last known values (conservative estimate).
    last_row     = features.iloc[[-1]].copy()   # shape (1, n_features)
    future_preds = []

    for step in range(steps_ahead):
        next_val = float(forecaster.predict(last_row)[0])
        future_preds.append(next_val)

        # Advance the feature row
        next_row = last_row.copy()

        # Shift lag columns: lag_n ← old lag_(n-1), lag_1 ← new prediction
        all_lags = sorted([
            int(c.split('_lag')[1])
            for c in next_row.columns
            if f'{target_feature}_lag' in c and c.split('_lag')[1].isdigit()
        ], reverse=True)
        for lag in all_lags:
            col_name = f'{target_feature}_lag{lag}'
            if col_name in next_row.columns:
                prev_col = f'{target_feature}_lag{lag - 1}'
                if lag == 1:
                    next_row[col_name] = next_val
                elif prev_col in last_row.columns:
                    next_row[col_name] = last_row[prev_col].iloc[0]

        # Diff features
        if f'{target_feature}_diff1' in next_row.columns:
            prev = last_row[f'{target_feature}_lag1'].iloc[0] if f'{target_feature}_lag1' in last_row.columns else next_val
            next_row[f'{target_feature}_diff1'] = next_val - prev

        # Calendar features: advance timestamp by (step+1) minutes
        next_ts = features.index[-1] + pd.Timedelta(minutes=step + 1)
        cal_updates = {
            'Hour_sin':   np.sin(2 * np.pi * next_ts.hour / 24),
            'Hour_cos':   np.cos(2 * np.pi * next_ts.hour / 24),
            'DoW_sin':    np.sin(2 * np.pi * next_ts.dayofweek / 7),
            'DoW_cos':    np.cos(2 * np.pi * next_ts.dayofweek / 7),
            'IsWeekend':  int(next_ts.dayofweek >= 5),
            'IsNight':    int(next_ts.hour < 7 or next_ts.hour >= 22),
            'IsPeakHour': int(8 <= next_ts.hour <= 18 and next_ts.dayofweek < 5),
        }
        for col, val in cal_updates.items():
            if col in next_row.columns:
                next_row[col] = val

        last_row = next_row

    # ── Threshold-based anomaly flagging on the future forecast ───────────
    # For each of the 10 forecasted steps, check whether the predicted value
    # exceeds the anomaly threshold. The first step that does gives the
    # proactive lead time the SDN controller would have before the anomaly.
    future_anomaly_steps = [i + 1 for i, v in enumerate(future_preds)
                            if v > spike_threshold]
    earliest_alert = min(future_anomaly_steps) if future_anomaly_steps else None

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n╔{'═' * 56}╗")
    print(f"║   TRAFFIC FORECASTING PERFORMANCE METRICS              ║")
    print(f"╠{'═' * 56}╣")
    print(f"║  Target feature     : {target_feature:<33s}║")
    print(f"║  Test rows          : {len(X_test):<33,}║")
    print(f"║  Directional Acc    : {directional_acc:>6.2f} %                         ║")
    print(f"║  Mean Abs Error     : {mae:>12,.4f} (log1p scale)        ║")
    print(f"║  RMSE               : {rmse:>12,.4f} (log1p scale)        ║")
    print(f"╠{'═' * 56}╣")
    print(f"║  PROACTIVE ANOMALY DETECTION                           ║")
    print(f"╠{'═' * 56}╣")
    print(f"║  Anomaly threshold  : {spike_threshold:>12,.4f} (95th pct, train)   ║")
    print(f"║  Avg lead time      : {avg_lead_time:>6.2f} min-steps                  ║")
    print(f"║  Max lead time      : {max_lead_time:>6d} min-steps                  ║")
    print(f"║  Advance alerts     : {n_advance_alerts:>6d} (predicted before anomaly)  ║")
    print(f"║  Same-step alerts   : {n_same_step:>6d} (detected at anomaly time)  ║")
    if earliest_alert:
        print(f"║  Forecast alert     : Step +{earliest_alert} of +{steps_ahead} exceeds threshold      ║")
        print(f"║  => SDN controller gets {earliest_alert}-min warning from forecast    ║")
    else:
        print(f"║  Forecast alert     : No anomaly predicted in next {steps_ahead} steps   ║")
    print(f"╚{'═' * 56}╝")

    _plot_forecasting_results(
        df, y_test, predictions, future_preds, target_feature, detector,
        spike_threshold=spike_threshold,
        future_anomaly_steps=future_anomaly_steps,
        avg_lead_time=avg_lead_time,
        max_lead_time=max_lead_time,
        n_advance_alerts=n_advance_alerts,
    )

    return {
        'target':               target_feature,
        'mae':                  mae,
        'mse':                  mse,
        'rmse':                 rmse,
        'directional_accuracy': directional_acc,
        'lead_time':            avg_lead_time,
        'max_lead_time':        max_lead_time,
        'n_advance_alerts':     n_advance_alerts,
        'spike_threshold':      spike_threshold,
        'future_anomaly_steps': future_anomaly_steps,
        'predictions':          predictions,
        'future_predictions':   future_preds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pattern recognition
# ─────────────────────────────────────────────────────────────────────────────

def detect_traffic_patterns(detector) -> dict:
    """
    Identify strong feature correlations and traffic burst characteristics.

    Returns:
        dict: correlations, burst_count, burst_threshold
    """
    print("\n" + "=" * 80)
    print("PATTERN RECOGNITION - Traffic Behaviour Analysis")
    print("=" * 80)
    print("\nSDN Future Application:")
    print("  • Automatic traffic classification")
    print("  • Intent-based networking policies")
    print("  • Distinguish anomaly patterns from normal baselines")

    df = _get_timestamp_index(detector.df)

    # Only correlate meaningful traffic columns (skip IP/port/label)
    traffic_cols = [c for c in TRAFFIC_COLS + ['IsAnomaly'] if c in df.columns]

    print(f"\nAnalysing correlations across {len(traffic_cols)} traffic features...")
    correlations = {}
    for i, feat1 in enumerate(traffic_cols):
        for feat2 in traffic_cols[i + 1:]:
            corr = df[feat1].corr(df[feat2])
            if abs(corr) > 0.5:
                correlations[f'{feat1} ↔ {feat2}'] = round(corr, 4)

    print(f"\n✓ Strong correlations found (|r| > 0.5): {len(correlations)}")
    for pair, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:8]:
        bar = '█' * int(abs(corr) * 20)
        print(f"  {pair:<40s}  r={corr:+.3f}  {bar}")

    # Burst detection on TotalBytes (or fallback)
    target_col = next((c for c in ['TotalBytes', 'BytesSent'] if c in df.columns),
                      traffic_cols[0])
    burst_threshold = df[target_col].mean() + 2 * df[target_col].std()
    bursts = int((df[target_col] > burst_threshold).sum())
    burst_pct = bursts / len(df) * 100

    print(f"\nBurst analysis on '{target_col}':")
    print(f"  Burst threshold  : {burst_threshold:>12,.2f} bytes  (mean + 2σ)")
    print(f"  Bursts detected  : {bursts:>6,}  ({burst_pct:.2f} % of flows)")

    # Anomaly-correlated burst overlap
    if 'IsAnomaly' in df.columns and bursts > 0:
        burst_mask   = df[target_col] > burst_threshold
        anomaly_mask = df['IsAnomaly'] == 1
        overlap = int((burst_mask & anomaly_mask).sum())
        print(f"  Burst + anomaly  : {overlap:>6,}  ({overlap / bursts * 100:.1f} % of bursts are anomalous)")

    return {
        'correlations':    correlations,
        'burst_count':     bursts,
        'burst_threshold': burst_threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Flow sequence analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_flow_sequences(detector, sequence_length: int = 5) -> dict:
    """
    Build sliding sequences of consecutive flows for sequential pattern mining.

    Args:
        detector        : NetworkAnomalyDetector instance
        sequence_length : Number of consecutive flows per sequence

    Returns:
        dict: sequences (np.ndarray), labels (np.ndarray | None), sequence_length
    """
    print("\n" + "=" * 80)
    print("FLOW SEQUENCE ANALYSIS - Sequential Pattern Mining")
    print("=" * 80)
    print("\nSDN Future Application:")
    print("  • Predict multi-step attack sequences")
    print("  • Optimise flow-table update scheduling")
    print("  • Chain-based anomaly detection")

    df = _get_timestamp_index(detector.df)

    # Use only base traffic columns (no derived/lag cols — keeps sequences interpretable)
    seq_cols = [c for c in TRAFFIC_COLS if c in df.columns]
    if not seq_cols:
        seq_cols = detector._get_feature_columns()[:7]

    print(f"\nSequence features : {seq_cols}")
    print(f"Sequence length   : {sequence_length} flows")

    arr    = df[seq_cols].values
    labels_raw = df['IsAnomaly'].values if 'IsAnomaly' in df.columns else None

    sequences = np.array([
        arr[i:i + sequence_length].flatten()
        for i in range(len(arr) - sequence_length)
    ])

    labels = None
    if labels_raw is not None:
        # A sequence is anomalous if ANY flow in the window is anomalous
        labels = np.array([
            labels_raw[i:i + sequence_length].max()
            for i in range(len(labels_raw) - sequence_length)
        ])

    print(f"\n✓ Sequences created : {len(sequences):,}")
    print(f"  Shape             : {sequences.shape}")
    if labels is not None:
        anom_seq = int(labels.sum())
        print(f"  Anomalous seqs    : {anom_seq:,}  ({anom_seq / len(labels) * 100:.2f} %)")

    return {
        'sequences':       sequences,
        'labels':          labels,
        'sequence_length': sequence_length,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_temporal_patterns(df: pd.DataFrame,
                             temporal_features: pd.DataFrame,
                             feature_cols: list):
    """
    Six-panel temporal pattern figure showing t1-t5 multi-horizon analysis.

    Panel layout:
      1. Rolling mean across t1-t5 for primary traffic column
      2. Rolling std (volatility) across t1-t5 for primary column
      3. Spike rate per horizon (t1-t5) as stacked comparison
      4. Rate of change (1-min diff) for top 2 traffic columns
      5. Autocorrelation of primary feature
      6. Summary card with t1-t5 spike counts
    """
    TIME_HORIZONS = [1, 2, 3, 4, 5]
    HORIZON_COLORS = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']
    # red=t1, orange=t2, green=t3, blue=t4, purple=t5

    fig = plt.figure(figsize=(22, 14))

    primary_col = feature_cols[0] if feature_cols else None

    # ── 1. Rolling mean across t1–t5 for primary column ──────────────────
    ax1 = plt.subplot(3, 2, 1)
    if primary_col:
        for t, (w, color) in enumerate(zip(TIME_HORIZONS, HORIZON_COLORS), start=1):
            mean_col = f'{primary_col}_t{t}_mean'
            if mean_col in temporal_features.columns:
                ax1.plot(temporal_features.index,
                         temporal_features[mean_col],
                         label=f't{t} ({w} min)', color=color,
                         alpha=0.75, linewidth=1.2)
    ax1.set_title(f'Rolling Mean t1–t5: {primary_col}',
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Bytes')
    ax1.legend(fontsize=9, ncol=5)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=30)

    # ── 2. Rolling std across t1–t5 for primary column ───────────────────
    ax2 = plt.subplot(3, 2, 2)
    if primary_col:
        for t, (w, color) in enumerate(zip(TIME_HORIZONS, HORIZON_COLORS), start=1):
            std_col = f'{primary_col}_t{t}_std'
            if std_col in temporal_features.columns:
                ax2.plot(temporal_features.index,
                         temporal_features[std_col],
                         label=f't{t} ({w} min)', color=color,
                         alpha=0.75, linewidth=1.2)
    ax2.set_title(f'Volatility t1–t5: {primary_col}',
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Standard Deviation (bytes)')
    ax2.legend(fontsize=9, ncol=5)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=30)

    # ── 3. Spike rate per horizon as bar chart ────────────────────────────
    ax3 = plt.subplot(3, 2, 3)
    spike_data = {}   # horizon label → total spike count across all cols
    for t in TIME_HORIZONS:
        t_spikes = [c for c in temporal_features.columns
                    if f'_t{t}_spike' in c]
        if t_spikes:
            spike_data[f't{t}\n({t} min)'] = int(
                temporal_features[t_spikes].values.sum()
            )

    if spike_data:
        bars = ax3.bar(spike_data.keys(), spike_data.values(),
                       color=HORIZON_COLORS[:len(spike_data)],
                       edgecolor='black', alpha=0.8)
        # Annotate count on each bar
        for bar, (label, count) in zip(bars, spike_data.items()):
            rate = count / len(temporal_features) * 100
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(spike_data.values()) * 0.01,
                     f'{count:,}\n({rate:.1f}%)',
                     ha='center', va='bottom', fontsize=9)
    ax3.set_title('Spike Count per Time Horizon (t1–t5)',
                  fontsize=13, fontweight='bold')
    ax3.set_xlabel('Time Horizon')
    ax3.set_ylabel('Total Spikes Detected')
    ax3.grid(True, alpha=0.3, axis='y')

    # ── 4. Rate of change (1-min diff) ───────────────────────────────────
    ax4 = plt.subplot(3, 2, 4)
    for col in feature_cols[:2]:
        trend_col = f'{col}_trend'
        if trend_col in temporal_features.columns:
            ax4.plot(temporal_features.index,
                     temporal_features[trend_col],
                     label=col, alpha=0.6, linewidth=0.9)
    ax4.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax4.set_title('Rate of Change (1-min Diff)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Timestamp')
    ax4.set_ylabel('Δ bytes per Minute')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=30)

    # ── 5. Autocorrelation ────────────────────────────────────────────────
    ax5 = plt.subplot(3, 2, 5)
    if primary_col:
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(df[primary_col], ax=ax5)
        ax5.set_title(f'Autocorrelation: {primary_col}',
                      fontsize=13, fontweight='bold')
        ax5.set_xlim(0, min(200, len(df) // 2))

    # ── 6. Summary card ───────────────────────────────────────────────────
    ax6 = plt.subplot(3, 2, 6)
    spike_lines = "\n".join(
        [f"    {label.replace(chr(10),' ')}: {count:,} spikes ({count/len(temporal_features)*100:.1f}%)"
         for label, count in spike_data.items()]
    ) if spike_data else "    No spike data"

    summary = (
        "TEMPORAL ANALYSIS SUMMARY\n\n"
        f"  Time range  : {df.index.min().date()} → {df.index.max().date()}\n"
        f"  Data points : {len(df):,}\n"
        f"  Resolution  : 1-minute flows\n"
        f"  Horizons    : t1 (1 min) → t5 (5 min)\n\n"
        "Spike Detection (4σ threshold):\n"
        f"{spike_lines}\n\n"
        "SDN Applications\n"
        "  • t1: Instant burst response\n"
        "  • t2-t3: Short-term load prediction\n"
        "  • t4-t5: Proactive flow pre-configuration"
    )
    ax6.text(0.5, 0.5, summary, ha='center', va='center',
             fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax6.axis('off')
    ax6.set_title('Analysis Summary', fontsize=13, fontweight='bold')

    plt.suptitle('SDN Multi-Horizon Temporal Pattern Analysis (t1–t5)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    try:
        output_path = 'temporal_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Temporal analysis plot saved to '{output_path}'")
    except Exception as e:
        print(f"  (Plot save skipped: {e})")
    plt.close()


def _plot_forecasting_results(df: pd.DataFrame,
                               y_test: pd.Series,
                               predictions: np.ndarray,
                               future_pred: list,
                               feature_name: str,
                               detector=None,
                               spike_threshold=None,
                               future_anomaly_steps=None,
                               avg_lead_time=None,
                               max_lead_time=None,
                               n_advance_alerts=None):
    """Four-panel forecasting results figure."""
    mae  = mean_absolute_error(y_test, predictions)
    mse  = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    true_dir = np.sign(np.diff(y_test.values))
    pred_dir = np.sign(np.diff(predictions))
    dir_acc  = float((true_dir == pred_dir).mean() * 100)
    future_anomaly_steps = future_anomaly_steps or []

    fig = plt.figure(figsize=(20, 10))

    # 1. Actual vs Predicted (first 2000 test points)
    ax1 = plt.subplot(2, 2, 1)
    plot_n = min(2000, len(y_test))
    idx = range(plot_n)
    ax1.plot(idx, y_test.values[:plot_n],
             label='Actual', linewidth=1, alpha=0.8, color='steelblue')
    ax1.plot(idx, predictions[:plot_n],
             label='Predicted', linewidth=1, alpha=0.8, color='tomato', linestyle='--')
    ax1.set_title(f'Actual vs Predicted: {feature_name}', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Test Index')
    ax1.set_ylabel('Bytes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Prediction error distribution
    ax2 = plt.subplot(2, 2, 2)
    errors = y_test.values - predictions
    ax2.hist(errors, bins=60, edgecolor='black', alpha=0.75, color='darkorange')
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.8, label='Zero error')
    ax2.axvline(errors.mean(), color='navy', linestyle=':', linewidth=1.5, label=f'Mean={errors.mean():,.0f}')
    ax2.set_title('Prediction Error Distribution', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Prediction Error (bytes)')
    ax2.set_ylabel('Frequency')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.annotate(f'Mean : {errors.mean():>10,.1f}\nStd  : {errors.std():>10,.1f}',
                 xy=(0.97, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # 3. Future forecast bridge with anomaly threshold
    ax3 = plt.subplot(2, 2, 3)
    last_n = min(60, len(y_test))
    hist_idx   = range(-last_n, 0)
    future_idx = range(len(future_pred))
    ax3.plot(hist_idx,   y_test.values[-last_n:],
             label='Historical (test)', linewidth=2, color='steelblue')
    ax3.plot(future_idx, future_pred,
             label=f'Forecast (+{len(future_pred)} min)', linewidth=2,
             color='seagreen', linestyle='--', marker='o', markersize=4)
    ax3.axvline(0, color='red', linestyle='--', alpha=0.6, linewidth=1.2, label='Now')

    # Draw anomaly threshold and highlight predicted anomaly steps
    if spike_threshold is not None:
        ax3.axhline(spike_threshold, color='crimson', linestyle=':',
                    linewidth=1.5, label=f'Anomaly threshold')
        if future_anomaly_steps:
            alert_x = [s - 1 for s in future_anomaly_steps]
            alert_y = [future_pred[s - 1] for s in future_anomaly_steps]
            ax3.scatter(alert_x, alert_y, color='crimson', zorder=6,
                        s=80, marker='X', label='Predicted anomaly')
            earliest = min(future_anomaly_steps)
            ax3.annotate(
                f'Alert at +{earliest} min',
                xy=(earliest - 1, future_pred[earliest - 1]),
                xytext=(earliest + 0.5, future_pred[earliest - 1] * 1.05),
                fontsize=9, color='crimson', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='crimson'),
            )

    ax3.set_title(f'{feature_name} — {len(future_pred)}-step Forecast with Anomaly Detection',
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Minutes relative to end of dataset')
    ax3.set_ylabel(f'{feature_name} (log1p scale)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. SDN application summary with proactive lead time metrics
    ax4 = plt.subplot(2, 2, 4)
    lead_line = (
        f"  Avg lead time : {avg_lead_time:.2f} min-steps\n"
        f"  Max lead time : {max_lead_time} min-steps\n"
        f"  Advance alerts: {n_advance_alerts} predictions ahead of anomaly"
    ) if avg_lead_time is not None else "  Lead time: N/A"

    forecast_alert_line = (
        f"  Forecast flags anomaly at step +{min(future_anomaly_steps)}\n"
        f"  SDN gets {min(future_anomaly_steps)}-min proactive warning"
        if future_anomaly_steps
        else "  No anomaly predicted in next forecast window"
    )

    sdn_text = (
        "PREDICTIVE SDN APPLICATIONS\n\n"
        "1. Proactive Resource Allocation\n"
        "     Scale bandwidth before demand spikes\n"
        "     Pre-configure SDN flow tables\n\n"
        "2. Anomaly Prevention\n"
        "     Flag predicted spikes before they arrive\n"
        "     Trigger mitigation ahead of impact\n\n"
        f"──────── Proactive Detection Results ────────\n"
        f"{forecast_alert_line}\n\n"
        f"──────────── Model Performance ──────────────\n"
        f"  Dir. Acc  : {dir_acc:>9.2f} %\n"
        f"  MAE       : {mae:>9.4f} (log1p)\n"
        f"  RMSE      : {rmse:>9.4f} (log1p)"
    )
    ax4.text(0.5, 0.5, sdn_text, ha='center', va='center',
             fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('SDN Benefits & Proactive Detection Metrics', fontsize=13, fontweight='bold')

    plt.suptitle('SDN Traffic Forecasting Results', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    try:
        output_path = 'traffic_forecasting.png'
        if detector is not None and hasattr(detector, '_get_output_path'):
            output_path = detector._get_output_path('traffic_forecasting.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Forecasting plot saved to '{output_path}'")
    except Exception as e:
        print(f"  (Plot save skipped: {e})")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_temporal_analysis(detector) -> dict:
    """
    Execute the full temporal analysis suite in order:
      1. Sliding time windows + spike detection
      2. Random Forest traffic forecasting
      3. Feature correlation + burst detection
      4. Flow sequence mining

    Args:
        detector : NetworkAnomalyDetector instance

    Returns:
        dict with keys: temporal_features, forecast, patterns, sequences
    """
    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS SUITE")
    print("=" * 80)

    temporal_features = create_time_windows(detector)
    forecast_results  = predict_future_traffic(detector, steps_ahead=10)
    patterns          = detect_traffic_patterns(detector)
    sequences         = analyze_flow_sequences(detector, sequence_length=5)

    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n  Directional accuracy : {forecast_results['directional_accuracy']:.2f} %")
    print(f"  Forecast MAE         : {forecast_results['mae']:,.2f} bytes")
    print(f"  Anomaly lead time    : {forecast_results['lead_time']:.2f} min-steps")
    print(f"  Strong correlations  : {len(patterns['correlations'])}")
    print(f"  Traffic bursts       : {patterns['burst_count']:,}")
    print(f"  Flow sequences built : {len(sequences['sequences']):,}")

    return {
        'temporal_features': temporal_features,
        'forecast':          forecast_results,
        'patterns':          patterns,
        'sequences':         sequences,
    }