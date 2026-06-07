# Handles the supervised Random Forest model training and evaluation to detect network anomalies.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           f1_score, accuracy_score)


def train_supervised_model(detector, n_estimators=100):
    """
    Train supervised Random Forest model
    
    Args:
        detector: NetworkAnomalyDetector instance
        n_estimators: Number of trees in the forest
    """
    if detector.y_train is None:
        print("\n⚠ No labels available for supervised learning")
        return detector
        
    print("\n" + "=" * 80)
    print("SUPERVISED LEARNING - Random Forest")
    print("=" * 80)
    
    detector.supervised_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    print("\nTraining Random Forest model...")
    detector.supervised_model.fit(detector.X_train, detector.y_train)
    
    # Predictions
    y_train_pred = detector.supervised_model.predict(detector.X_train)
    y_test_pred = detector.supervised_model.predict(detector.X_test)
    y_test_proba = detector.supervised_model.predict_proba(detector.X_test)[:, 1]
    
    # Metrics
    print("\n--- Training Set Performance ---")
    print(f"Accuracy: {accuracy_score(detector.y_train, y_train_pred):.4f}")
    print(f"F1-Score: {f1_score(detector.y_train, y_train_pred):.4f}")
    
    print("\n--- Test Set Performance ---")
    print(f"Accuracy: {accuracy_score(detector.y_test, y_test_pred):.4f}")
    print(f"F1-Score: {f1_score(detector.y_test, y_test_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(detector.y_test, y_test_proba):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(detector.y_test, y_test_pred, 
                               target_names=['Normal', 'Anomaly']))
    
    # Feature importance
    feature_cols = detector._get_feature_columns()
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': detector.supervised_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Visualization
    _plot_supervised_results(detector, y_test_pred, y_test_proba, feature_importance)
    
    return detector


def _plot_supervised_results(detector, y_pred, y_proba, feature_importance):
    """Plot supervised learning results"""
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(detector.y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. ROC Curve
    ax2 = plt.subplot(2, 3, 2)
    fpr, tpr, _ = roc_curve(detector.y_test, y_proba)
    auc_score = roc_auc_score(detector.y_test, y_proba)
    ax2.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    ax3 = plt.subplot(2, 3, 3)
    precision, recall, _ = precision_recall_curve(detector.y_test, y_proba)
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
    ax5.hist(y_proba[detector.y_test == 0], bins=50, alpha=0.6, label='Normal', density=True)
    ax5.hist(y_proba[detector.y_test == 1], bins=50, alpha=0.6, label='Anomaly', density=True)
    ax5.set_xlabel('Predicted Probability')
    ax5.set_ylabel('Density')
    ax5.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance Metrics Summary
    ax6 = plt.subplot(2, 3, 6)
    metrics = {
        'Accuracy': accuracy_score(detector.y_test, y_pred),
        'F1-Score': f1_score(detector.y_test, y_pred),
        'ROC-AUC': auc_score
    }
    ax6.bar(metrics.keys(), metrics.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax6.set_ylim([0, 1])
    ax6.set_title('Performance Metrics Summary', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Score')
    for i, (k, v) in enumerate(metrics.items()):
        ax6.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    output_path = detector._get_output_path('supervised_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Supervised learning results saved to '{output_path}'")
    plt.close()
