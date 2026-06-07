# This handles the anomaly detection in the supervised and unsupervised learning modules. It trains the models, evaluates their performance, and generates results for both approaches.

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           f1_score, accuracy_score)
import warnings
warnings.filterwarnings('ignore')


class NetworkAnomalyDetector:

    def __init__(self, data_path, output_dir='results'):
        """Initialize the detector with dataset path and output directory"""
        self.data_path = data_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.supervised_model = None
        self.unsupervised_model = None
        self.scaler = StandardScaler()
        
        # Expected feature columns for the dataset
        self.expected_features = ['SourceIP', 'DestinationIP', 'SourcePort', 
                                 'DestinationPort', 'Protocol', 'BytesSent', 
                                 'BytesReceived', 'PacketsSent', 'PacketsReceived', 
                                 'Duration']
        
    def _get_output_path(self, filename):
        """Get the full path for an output file"""
        return os.path.join(self.output_dir, filename)
    
    def _get_feature_columns(self):
        """Get feature columns that exist in the dataframe"""
        return [col for col in self.expected_features if col in self.df.columns]
        
    def load_data(self):
        """Load and perform initial exploration of the dataset"""
        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)

        if len(self.df) > 50000:
            print(f"\n Dataset has {len(self.df):,} rows - sampling to 50,000 for analysis")
            self.df = self.df.sample(n=50000, random_state=42)
    
            print(f"\nDataset Shape: {self.df.shape}")

        
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
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for modeling"""
        print("\n" + "=" * 80)
        print("DATA PREPARATION")
        print("=" * 80)
        
        # Separate features and target
        feature_cols = self._get_feature_columns()
        X = self.df[feature_cols].values
        
        if 'IsAnomaly' in self.df.columns:
            y = self.df['IsAnomaly'].values
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            print(f"\nTraining set: {self.X_train.shape[0]} samples")
            print(f"Test set: {self.X_test.shape[0]} samples")
            print(f"\nTraining set anomaly rate: {self.y_train.mean() * 100:.2f}%")
            print(f"Test set anomaly rate: {self.y_test.mean() * 100:.2f}%")
        else:
            # No labels - use all data for unsupervised learning
            self.X_train = X
            self.X_test = None
            self.y_train = None
            self.y_test = None
            print(f"\nNo labels found. Using {X.shape[0]} samples for unsupervised learning")
        
        return self
    
    def run_complete_pipeline(self, supervised=True, unsupervised=True):
        """Run the complete anomaly detection pipeline"""
        from eda import perform_eda
        from supervised import train_supervised_model
        from unsupervised import train_unsupervised_model
        from export import export_results
        
        print("\n" + "=" * 80)
        print("NETWORK TRAFFIC ANOMALY DETECTION PIPELINE")
        print("=" * 80)
        print(f"Output directory: {os.path.abspath(self.output_dir)}")
        
        self.load_data()
        perform_eda(self)
        self.prepare_data()
        
        if supervised and self.y_train is not None:
            train_supervised_model(self)
        
        if unsupervised:
            train_unsupervised_model(self)
        
        export_results(self)
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nAll results saved to: {os.path.abspath(self.output_dir)}")
        print("\nGenerated files:")
        print("  • eda_analysis.png - Exploratory data analysis")
        if supervised and self.y_train is not None:
            print("  • supervised_results.png - Random Forest results")
            print("  • supervised_predictions.csv - Supervised predictions")
        if unsupervised:
            print("  • unsupervised_results.png - Isolation Forest results")
            print("  • unsupervised_predictions.csv - Unsupervised predictions")
        
        return self
