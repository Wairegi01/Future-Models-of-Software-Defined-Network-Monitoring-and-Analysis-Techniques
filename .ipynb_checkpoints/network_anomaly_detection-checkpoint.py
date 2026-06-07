"""
Network Traffic Anomaly Detection
Dataset: Kaggle Synthetic Network Traffic (vidhikishorwaghela)
Features: Normalized numerical values for network attributes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           f1_score, accuracy_score)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class NetworkAnomalyDetector:
    """
    Comprehensive Network Traffic Anomaly Detection System
    Supports both supervised and unsupervised learning approaches
    """
    
    def __init__(self, data_path):
        """Initialize the detector with dataset path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.supervised_model = None
        self.unsupervised_model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and perform initial exploration of the dataset"""
        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        
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
    
    def exploratory_analysis(self):
        """Perform exploratory data analysis with visualizations"""
        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        # Define expected feature columns
        expected_features = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort',
                           'Protocol', 'BytesSent', 'BytesReceived', 'PacketsSent', 
                           'PacketsReceived', 'Duration']
        
        # Create figure for multiple plots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Class distribution (if labels exist)
        if 'IsAnomaly' in self.df.columns:
            ax1 = plt.subplot(2, 3, 1)
            self.df['IsAnomaly'].value_counts().plot(kind='bar', ax=ax1)
            ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('IsAnomaly (0=Normal, 1=Anomaly)')
            ax1.set_ylabel('Count')
            
        # 2. Correlation heatmap
        ax2 = plt.subplot(2, 3, 2)
        feature_cols = [col for col in expected_features if col in self.df.columns]
        correlation_matrix = self.df[feature_cols].corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', ax=ax2, 
                   cbar_kws={'shrink': 0.8})
        ax2.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        
        # 3. Feature distributions
        ax3 = plt.subplot(2, 3, 3)
        self.df[feature_cols].boxplot(ax=ax3, vert=False)
        ax3.set_title('Feature Distributions', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Normalized Values')
        
        # 4. PCA visualization (if labels exist)
        if 'IsAnomaly' in self.df.columns:
            ax4 = plt.subplot(2, 3, 4)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(self.df[feature_cols])
            
            scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1], 
                                c=self.df['IsAnomaly'], 
                                cmap='coolwarm', alpha=0.6, s=10)
            ax4.set_title('PCA Visualization (2D)', fontsize=14, fontweight='bold')
            ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.colorbar(scatter, ax=ax4, label='IsAnomaly')
            
        # 5. Feature importance preview (sample)
        ax5 = plt.subplot(2, 3, 5)
        feature_std = self.df[feature_cols].std().sort_values(ascending=False)
        feature_std.head(10).plot(kind='barh', ax=ax5)
        ax5.set_title('Top 10 Features by Std Deviation', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Standard Deviation')
        
        # 6. Sample size distribution
        ax6 = plt.subplot(2, 3, 6)
        ax6.text(0.5, 0.5, f'Total Samples: {len(self.df):,}\n\n' + 
                          f'Features: {len(feature_cols)}\n\n' +
                          (f'Normal: {(self.df["IsAnomaly"]==0).sum():,}\n' +
                           f'Anomaly: {(self.df["IsAnomaly"]==1).sum():,}' 
                           if 'IsAnomaly' in self.df.columns else ''),
                ha='center', va='center', fontsize=16, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax6.axis('off')
        ax6.set_title('Dataset Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/eda_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ EDA visualization saved to 'eda_analysis.png'")
        plt.close()
        
        return self
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for modeling"""
        print("\n" + "=" * 80)
        print("DATA PREPARATION")
        print("=" * 80)
        
        # Define expected feature columns
        expected_features = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort',
                           'Protocol', 'BytesSent', 'BytesReceived', 'PacketsSent', 
                           'PacketsReceived', 'Duration']
        
        # Separate features and target
        feature_cols = [col for col in expected_features if col in self.df.columns]
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
    
    def train_supervised_model(self, n_estimators=100):
        """Train supervised Random Forest model"""
        if self.y_train is None:
            print("\n⚠ No labels available for supervised learning")
            return self
            
        print("\n" + "=" * 80)
        print("SUPERVISED LEARNING - Random Forest")
        print("=" * 80)
        
        self.supervised_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        print("\nTraining Random Forest model...")
        self.supervised_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = self.supervised_model.predict(self.X_train)
        y_test_pred = self.supervised_model.predict(self.X_test)
        y_test_proba = self.supervised_model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        print("\n--- Training Set Performance ---")
        print(f"Accuracy: {accuracy_score(self.y_train, y_train_pred):.4f}")
        print(f"F1-Score: {f1_score(self.y_train, y_train_pred):.4f}")
        
        print("\n--- Test Set Performance ---")
        print(f"Accuracy: {accuracy_score(self.y_test, y_test_pred):.4f}")
        print(f"F1-Score: {f1_score(self.y_test, y_test_pred):.4f}")
        print(f"ROC-AUC: {roc_auc_score(self.y_test, y_test_proba):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_test_pred, 
                                   target_names=['Normal', 'Anomaly']))
        
        # Feature importance
        expected_features = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort',
                           'Protocol', 'BytesSent', 'BytesReceived', 'PacketsSent', 
                           'PacketsReceived', 'Duration']
        feature_cols = [col for col in expected_features if col in self.df.columns]
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.supervised_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Visualization
        self._plot_supervised_results(y_test_pred, y_test_proba, feature_importance)
        
        return self
    
    def train_unsupervised_model(self, contamination=0.1):
        """Train unsupervised Isolation Forest model"""
        print("\n" + "=" * 80)
        print("UNSUPERVISED LEARNING - Isolation Forest")
        print("=" * 80)
        
        self.unsupervised_model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        
        print(f"\nTraining Isolation Forest (contamination={contamination})...")
        self.unsupervised_model.fit(self.X_train)
        
        # Predictions (-1 for anomaly, 1 for normal)
        train_predictions = self.unsupervised_model.predict(self.X_train)
        train_scores = self.unsupervised_model.score_samples(self.X_train)
        
        # Convert to binary (1 for anomaly, 0 for normal)
        train_anomalies = (train_predictions == -1).astype(int)
        
        print(f"\nDetected anomalies in training set: {train_anomalies.sum()} " +
              f"({train_anomalies.mean() * 100:.2f}%)")
        
        if self.y_train is not None:
            # If we have labels, evaluate performance
            accuracy = accuracy_score(self.y_train, train_anomalies)
            f1 = f1_score(self.y_train, train_anomalies)
            print(f"\nPerformance on labeled data:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            if self.X_test is not None:
                test_predictions = self.unsupervised_model.predict(self.X_test)
                test_anomalies = (test_predictions == -1).astype(int)
                test_accuracy = accuracy_score(self.y_test, test_anomalies)
                test_f1 = f1_score(self.y_test, test_anomalies)
                print(f"\nTest Set Performance:")
                print(f"Accuracy: {test_accuracy:.4f}")
                print(f"F1-Score: {test_f1:.4f}")
        
        # Visualization
        self._plot_unsupervised_results(train_predictions, train_scores)
        
        return self
    
    def _plot_supervised_results(self, y_pred, y_proba, feature_importance):
        """Plot supervised learning results"""
        fig = plt.figure(figsize=(20, 10))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. ROC Curve
        ax2 = plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        auc_score = roc_auc_score(self.y_test, y_proba)
        ax2.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})', linewidth=2)
        ax2.plot([0, 1], [0, 1], 'k--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        ax3 = plt.subplot(2, 3, 3)
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
        ax3.plot(recall, precision, linewidth=2)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature Importance
        ax4 = plt.subplot(2, 3, 4)
        top_features = feature_importance.head(15)
        ax4.barh(range(len(top_features)), top_features['importance'])
        ax4.set_yticks(range(len(top_features)))
        ax4.set_yticklabels(top_features['feature'])
        ax4.set_xlabel('Importance')
        ax4.set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
        ax4.invert_yaxis()
        
        # 5. Prediction Distribution
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(y_proba[self.y_test == 0], bins=50, alpha=0.6, label='Normal', density=True)
        ax5.hist(y_proba[self.y_test == 1], bins=50, alpha=0.6, label='Anomaly', density=True)
        ax5.set_xlabel('Predicted Probability')
        ax5.set_ylabel('Density')
        ax5.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance Metrics Summary
        ax6 = plt.subplot(2, 3, 6)
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'F1-Score': f1_score(self.y_test, y_pred),
            'ROC-AUC': auc_score
        }
        ax6.bar(metrics.keys(), metrics.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax6.set_ylim([0, 1])
        ax6.set_title('Performance Metrics Summary', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Score')
        for i, (k, v) in enumerate(metrics.items()):
            ax6.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/supervised_results.png', dpi=300, bbox_inches='tight')
        print("\n✓ Supervised learning results saved to 'supervised_results.png'")
        plt.close()
    
    def _plot_unsupervised_results(self, predictions, scores):
        """Plot unsupervised learning results"""
        fig = plt.figure(figsize=(20, 10))
        
        # Convert predictions to binary
        anomalies = (predictions == -1).astype(int)
        
        # 1. Anomaly Score Distribution
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(scores, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(scores[predictions == -1].max(), color='red', 
                   linestyle='--', label='Anomaly Threshold')
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. PCA with Anomalies Highlighted
        ax2 = plt.subplot(2, 3, 2)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_train)
        
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=anomalies, cmap='coolwarm', alpha=0.6, s=10)
        ax2.set_title('PCA Visualization with Detected Anomalies', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, ax=ax2, label='Anomaly')
        
        # 3. Score vs Index
        ax3 = plt.subplot(2, 3, 3)
        indices = np.arange(len(scores))
        ax3.scatter(indices, scores, c=anomalies, cmap='coolwarm', 
                   alpha=0.5, s=1)
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Anomaly Score')
        ax3.set_title('Anomaly Scores Across Samples', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. If labels exist, show confusion matrix
        if self.y_train is not None:
            ax4 = plt.subplot(2, 3, 4)
            cm = confusion_matrix(self.y_train, anomalies)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax4)
            ax4.set_title('Confusion Matrix (vs True Labels)', 
                         fontsize=14, fontweight='bold')
            ax4.set_ylabel('True Label')
            ax4.set_xlabel('Predicted Label')
        
        # 5. Top Anomalies
        ax5 = plt.subplot(2, 3, 5)
        top_anomalies = np.argsort(scores)[:100]
        ax5.scatter(range(len(top_anomalies)), scores[top_anomalies], 
                   color='red', alpha=0.6)
        ax5.set_xlabel('Rank')
        ax5.set_ylabel('Anomaly Score')
        ax5.set_title('Top 100 Anomalies by Score', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary Statistics
        ax6 = plt.subplot(2, 3, 6)
        summary_text = f"""
        Total Samples: {len(predictions):,}
        
        Detected Anomalies: {anomalies.sum():,}
        Detection Rate: {anomalies.mean() * 100:.2f}%
        
        Score Range: [{scores.min():.4f}, {scores.max():.4f}]
        Mean Score: {scores.mean():.4f}
        Std Score: {scores.std():.4f}
        """
        
        if self.y_train is not None:
            accuracy = accuracy_score(self.y_train, anomalies)
            f1 = f1_score(self.y_train, anomalies)
            summary_text += f"\n\nAccuracy: {accuracy:.4f}\nF1-Score: {f1:.4f}"
        
        ax6.text(0.5, 0.5, summary_text, ha='center', va='center', 
                fontsize=12, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax6.axis('off')
        ax6.set_title('Detection Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/unsupervised_results.png', 
                   dpi=300, bbox_inches='tight')
        print("\n✓ Unsupervised learning results saved to 'unsupervised_results.png'")
        plt.close()
    
    def export_results(self):
        """Export predictions and analysis results"""
        print("\n" + "=" * 80)
        print("EXPORTING RESULTS")
        print("=" * 80)
        
        results = {}
        
        if self.supervised_model is not None and self.y_test is not None:
            # Supervised predictions
            y_pred = self.supervised_model.predict(self.X_test)
            y_proba = self.supervised_model.predict_proba(self.X_test)[:, 1]
            
            supervised_results = pd.DataFrame({
                'True_Label': self.y_test,
                'Predicted_Label': y_pred,
                'Anomaly_Probability': y_proba,
                'Correct': self.y_test == y_pred
            })
            
            supervised_results.to_csv('/mnt/user-data/outputs/supervised_predictions.csv', 
                                     index=False)
            print("✓ Supervised predictions saved to 'supervised_predictions.csv'")
        
        if self.unsupervised_model is not None:
            # Unsupervised predictions
            predictions = self.unsupervised_model.predict(self.X_train)
            scores = self.unsupervised_model.score_samples(self.X_train)
            anomalies = (predictions == -1).astype(int)
            
            unsupervised_results = pd.DataFrame({
                'Predicted_Anomaly': anomalies,
                'Anomaly_Score': scores
            })
            
            if self.y_train is not None:
                unsupervised_results['True_Label'] = self.y_train
                unsupervised_results['Correct'] = self.y_train == anomalies
            
            unsupervised_results.to_csv('/mnt/user-data/outputs/unsupervised_predictions.csv', 
                                       index=False)
            print("✓ Unsupervised predictions saved to 'unsupervised_predictions.csv'")
        
        return self
    
    def run_complete_pipeline(self, supervised=True, unsupervised=True):
        """Run the complete anomaly detection pipeline"""
        print("\n" + "=" * 80)
        print("NETWORK TRAFFIC ANOMALY DETECTION PIPELINE")
        print("=" * 80)
        
        self.load_data()
        self.exploratory_analysis()
        self.prepare_data()
        
        if supervised and self.y_train is not None:
            self.train_supervised_model()
        
        if unsupervised:
            self.train_unsupervised_model()
        
        self.export_results()
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nGenerated files:")
        print("  • eda_analysis.png - Exploratory data analysis")
        if supervised and self.y_train is not None:
            print("  • supervised_results.png - Random Forest results")
            print("  • supervised_predictions.csv - Supervised predictions")
        if unsupervised:
            print("  • unsupervised_results.png - Isolation Forest results")
            print("  • unsupervised_predictions.csv - Unsupervised predictions")
        
        return self


# Main execution
if __name__ == "__main__":
    # Initialize detector with your dataset path
    DATA_PATH = "C:\\Users\\iamwa\\source\\repos\\Thesis\\Network Analysis\\Synthetic Network Traffic\\archive\\synthetic_network_traffic.csv"  # Update this to your actual file path
    
    detector = NetworkAnomalyDetector(DATA_PATH)
    
    # Run complete pipeline
    detector.run_complete_pipeline(
        supervised=True,      # Train Random Forest classifier
        unsupervised=True     # Train Isolation Forest
    )
    
    print("\n✓ All analyses complete!")
