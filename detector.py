import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config
import warnings
warnings.filterwarnings('ignore')


class NetworkAnomalyDetector:

    def __init__(self, data_path, output_dir='results'):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.supervised_model = None
        self.unsupervised_model = None
        self.scaler = StandardScaler()

        self.expected_features    = list(config.EXPECTED_FEATURES)
        self.rolling_feature_cols = []

    def _get_output_path(self, filename):
        return os.path.join(self.output_dir, filename)

    def _get_feature_columns(self):
        return [col for col in self.expected_features if col in self.df.columns]

    def load_data(self):
        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)

        self.df = pd.read_csv(self.data_path)

        if len(self.df) > 50000:
            print(f"\nDataset has {len(self.df):,} rows - sampling to 50,000 for analysis")
            self.df = self.df.sample(n=50000, random_state=42)

        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        print(f"\nData Types:")
        print(self.df.dtypes)
        print(f"\nMissing Values:")
        print(self.df.isnull().sum())
        print(f"\nBasic Statistics:")
        print(self.df.describe())

        if 'IsAnomaly' in self.df.columns:
            print(f"\nClass Distribution:")
            print(self.df['IsAnomaly'].value_counts())
            print(f"\nAnomaly Percentage: {self.df['IsAnomaly'].mean() * 100:.2f}%")

        return self

    def _engineer_features(self):
        """Add ratio features that capture anomalous byte/packet behaviour."""
        if 'BytesSent' in self.df.columns and 'PacketsSent' in self.df.columns:
            self.df['BytesPerPacketSent'] = (
                self.df['BytesSent'] / (self.df['PacketsSent'] + 1)
            )
        if 'BytesReceived' in self.df.columns and 'PacketsReceived' in self.df.columns:
            self.df['BytesPerPacketReceived'] = (
                self.df['BytesReceived'] / (self.df['PacketsReceived'] + 1)
            )

    def _log_transform_skewed(self):
        """Log1p-transform right-skewed features to reduce outlier influence."""
        for col in config.SKEWED_FEATURES:
            if col in self.df.columns:
                self.df[col] = np.log1p(self.df[col])

    def prepare_data(self, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
        print("\n" + "=" * 80)
        print("Data Preparation")
        print("=" * 80)

        # Feature engineering then log-transform (on raw values, before scaling)
        self._engineer_features()
        self._log_transform_skewed()
        print("  Applied BytesPerPacket feature engineering")
        print("  Applied log1p transform to skewed features")

        feature_cols = self._get_feature_columns()
        X = self.df[feature_cols].values

        if 'IsAnomaly' in self.df.columns:
            y = self.df['IsAnomaly'].values

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # Fit scaler on train only, transform both splits
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            print("  Applied StandardScaler (fit on train, transform on test)")

            print(f"\nTraining set : {self.X_train.shape[0]:,} samples")
            print(f"Test set     : {self.X_test.shape[0]:,} samples")
            print(f"Features     : {self.X_train.shape[1]}")
            print(f"\nTraining anomaly rate : {self.y_train.mean() * 100:.2f}%")
            print(f"Test anomaly rate     : {self.y_test.mean() * 100:.2f}%")
        else:
            self.X_train = self.scaler.fit_transform(X)
            self.X_test = None
            self.y_train = None
            self.y_test = None
            print(f"\nNo labels found. Using {X.shape[0]:,} samples for unsupervised learning")

        return self

    def run_complete_pipeline(self, supervised=True, unsupervised=True,
                              temporal=True, behavioral=True, deep_learning=True):
        from eda import perform_eda
        from supervised import train_supervised_model
        from unsupervised import train_unsupervised_model
        from temporal_analysis import run_temporal_analysis
        from behavior_analysis import run_behavior_analysis
        from deep_learning import run_deep_learning_analysis
        from export import export_results

        print("Network Traffic Anomaly Detection Pipeline")
        print("=" * 50)
        print(f"Output directory: {os.path.abspath(self.output_dir)}")

        timings = {}

        t0 = time.time(); self.load_data();    timings['Data Loading']    = time.time() - t0
        t0 = time.time(); perform_eda(self);   timings['EDA']             = time.time() - t0

        if temporal:
            from temporal_analysis import add_rolling_features
            t0 = time.time()
            self.rolling_feature_cols = add_rolling_features(self, window=10)
            timings['Rolling Features'] = time.time() - t0

        t0 = time.time(); self.prepare_data(); timings['Data Preparation'] = time.time() - t0

        if supervised and self.y_train is not None:
            t0 = time.time(); train_supervised_model(self); timings['Random Forest'] = time.time() - t0

        if unsupervised:
            t0 = time.time(); train_unsupervised_model(self); timings['Isolation Forest'] = time.time() - t0

        if temporal:
            t0 = time.time(); run_temporal_analysis(self); timings['Temporal Analysis'] = time.time() - t0

        if behavioral:
            t0 = time.time(); run_behavior_analysis(self); timings['Behavioural Analysis'] = time.time() - t0

        if deep_learning:
            t0 = time.time(); run_deep_learning_analysis(self); timings['Deep Learning + Ensemble'] = time.time() - t0

        t0 = time.time(); export_results(self); timings['Export'] = time.time() - t0

        print("\n" + "=" * 50)
        print("PIPELINE TRAINING TIMES")
        print("=" * 50)
        total = sum(timings.values())
        for stage, secs in timings.items():
            print(f"  {stage:<30} : {secs:6.1f}s  ({secs/total*100:.1f}%)")
        print(f"  {'TOTAL':<30} : {total:6.1f}s")
        print("=" * 50)

        print("Pipeline complete")
        print("=" * 50)
        print(f"\nAll results saved to: {os.path.abspath(self.output_dir)}")
        print("\nGenerated files:")
        print("  - eda_analysis.png              - Exploratory data analysis")
        if supervised and self.y_train is not None:
            print("  - supervised_results.png        - Random Forest results")
            print("  - supervised_predictions.csv")
        if unsupervised:
            print("  - unsupervised_results.png      - Isolation Forest results")
        if temporal:
            print("  - temporal_analysis.png         - Rolling window analysis")
            print("  - traffic_forecasting.png       - Traffic forecasting")
        if behavioral:
            print("  - behavioral_profiling.png      - K-Means traffic profiles")
            print("  - density_clustering.png        - DBSCAN outlier detection")
            print("  - flow_analysis.png             - Network flow statistics")
            print("  - severity_scoring.png          - Anomaly severity levels")
        if deep_learning:
            print("  - autoencoder_results.png       - Neural network autoencoder")
        print("  - unsupervised_predictions.csv")

        return self
