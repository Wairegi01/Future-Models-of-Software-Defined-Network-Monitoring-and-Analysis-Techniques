# Handles the supervised Random Forest model training and evaluation to detect network anomalies.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                           roc_auc_score, roc_curve, precision_recall_curve,
                           f1_score, accuracy_score)


def _find_optimal_threshold(y_true, y_proba):
    """Find the probability threshold that maximises F1 on the given set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve appends a sentinel at the end; thresholds is one shorter
    f1_scores = np.where(
        (precision[:-1] + recall[:-1]) == 0,
        0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1])
    )
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def train_supervised_model(detector, n_estimators=None):
    if detector.y_train is None:
        print("No labels available for supervised learning")
        return detector

    n_estimators = n_estimators or config.RF_N_ESTIMATORS

    # ── Rolling window feature comparison ────────────────────────────────────
    rolling_cols = getattr(detector, 'rolling_feature_cols', [])
    if rolling_cols:
        feature_cols = detector._get_feature_columns()
        base_idx = [i for i, c in enumerate(feature_cols) if c not in rolling_cols]
        roll_idx = [i for i, c in enumerate(feature_cols) if c in rolling_cols]

        print("\nRolling Window Feature Comparison")
        print("-" * 50)
        print(f"  Base features    : {len(base_idx)}")
        print(f"  Rolling features : {len(roll_idx)}  {rolling_cols}")

        def _run_rf_f1(X_tr, y_tr, X_te, y_te):
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=config.RF_MAX_DEPTH,
                min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
                min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
                class_weight=config.RF_CLASS_WEIGHT,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )
            rf.fit(X_tr, y_tr)
            train_proba = rf.predict_proba(X_tr)[:, 1]
            test_proba  = rf.predict_proba(X_te)[:, 1]
            thr, _  = _find_optimal_threshold(y_tr, train_proba)
            y_pred  = (test_proba >= thr).astype(int)
            return f1_score(y_te, y_pred, zero_division=0)

        f1_base = _run_rf_f1(
            detector.X_train[:, base_idx], detector.y_train,
            detector.X_test[:, base_idx],  detector.y_test,
        )
        f1_roll = _run_rf_f1(
            detector.X_train, detector.y_train,
            detector.X_test,  detector.y_test,
        )

        print(f"\n  F1 without rolling features : {f1_base:.4f}  ({f1_base * 100:.1f}%)")
        print(f"  F1 with rolling features    : {f1_roll:.4f}  ({f1_roll * 100:.1f}%)")
        print(f"  Improvement                 : {(f1_roll - f1_base) * 100:+.2f} percentage points")

    print("\nRandom Forest (full model)")
    print("-" * 50)

    detector.supervised_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        class_weight=config.RF_CLASS_WEIGHT,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    print(f" Training Random Forest ({n_estimators} trees, "
          f"class_weight={config.RF_CLASS_WEIGHT})...")
    detector.supervised_model.fit(detector.X_train, detector.y_train)
    print("Training complete")

    # Probabilities for threshold tuning
    y_train_proba = detector.supervised_model.predict_proba(detector.X_train)[:, 1]
    y_test_proba  = detector.supervised_model.predict_proba(detector.X_test)[:, 1]

    # Default (0.5) predictions
    y_train_pred_default = detector.supervised_model.predict(detector.X_train)
    y_test_pred_default  = detector.supervised_model.predict(detector.X_test)

    # Optimal threshold — tuned on the training set, applied to the test set
    optimal_threshold, train_best_f1 = _find_optimal_threshold(
        detector.y_train, y_train_proba
    )
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

    print(f"\nOptimal threshold (tuned on train): {optimal_threshold:.4f}  "
          f"(train F1 @ threshold: {train_best_f1:.4f})")

    # --- Training set performance (default threshold) ---
    print("\n--- Training Set Performance (threshold = 0.50) ---")
    print(f"Accuracy: {accuracy_score(detector.y_train, y_train_pred_default):.4f}")
    print(f"F1-Score: {f1_score(detector.y_train, y_train_pred_default):.4f}")

    # --- Test set performance: default vs optimised ---
    print("\n--- Test Set Performance (threshold = 0.50) ---")
    _print_metrics(detector.y_test, y_test_pred_default, y_test_proba)

    print(f"\n--- Test Set Performance (threshold = {optimal_threshold:.4f}, optimised) ---")
    _print_metrics(detector.y_test, y_test_pred, y_test_proba)

    print("\nClassification Report (optimised threshold):")
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
    _plot_supervised_results(
        detector, y_test_pred, y_test_proba,
        feature_importance, optimal_threshold
    )

    return detector


def _print_metrics(y_true, y_pred, y_proba):
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")


def _plot_supervised_results(detector, y_pred, y_proba, feature_importance, threshold):
    """Plot supervised learning results"""
    fig = plt.figure(figsize=(20, 10))

    # 1. Confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(detector.y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'Confusion Matrix (thr={threshold:.2f})', fontsize=14, fontweight='bold')
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

    # 3. Precision-Recall Curve (with optimal threshold marked)
    ax3 = plt.subplot(2, 3, 3)
    precision, recall, thresholds = precision_recall_curve(detector.y_test, y_proba)
    ax3.plot(recall, precision, linewidth=2)
    # Mark the chosen threshold
    idx = np.searchsorted(thresholds, threshold)
    if idx < len(thresholds):
        ax3.scatter(recall[idx], precision[idx], color='red', zorder=5,
                    label=f'thr={threshold:.2f}')
        ax3.legend()
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

    # 5. Prediction Probability Distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(y_proba[detector.y_test == 0], bins=50, alpha=0.6, label='Normal', density=True)
    ax5.hist(y_proba[detector.y_test == 1], bins=50, alpha=0.6, label='Anomaly', density=True)
    ax5.axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold:.2f}')
    ax5.set_xlabel('Predicted Probability')
    ax5.set_ylabel('Density')
    ax5.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Performance Metrics Summary
    ax6 = plt.subplot(2, 3, 6)
    metrics = {
        'Accuracy':  accuracy_score(detector.y_test, y_pred),
        'F1-Score':  f1_score(detector.y_test, y_pred, zero_division=0),
        'ROC-AUC':   auc_score,
        'Precision': precision_score(detector.y_test, y_pred, zero_division=0),
        'Recall':    recall_score(detector.y_test, y_pred, zero_division=0),
    }
    bars = ax6.bar(metrics.keys(), metrics.values(),
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax6.set_ylim([0, 1.1])
    ax6.set_title('Performance Metrics Summary', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Score')
    for bar, v in zip(bars, metrics.values()):
        ax6.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                 f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    output_path = detector._get_output_path('supervised_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Supervised learning results saved to '{output_path}'")
    plt.close()
