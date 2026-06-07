import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, precision_recall_curve,
    average_precision_score,
)
import config


def train_unsupervised_model(detector):
    print("\n" + "=" * 80)
    print("Unsupervised Anomaly Detection  (Isolation Forest)")
    print("=" * 80)

    if_results = _run_isolation_forest(detector)
    _plot_unsupervised_results(detector, if_results)

    return detector


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

def _run_isolation_forest(detector):
    print("\n--- Isolation Forest ---")

    model = IsolationForest(
        contamination=config.ISO_CONTAMINATION,
        n_estimators=config.ISO_N_ESTIMATORS,
        max_samples=min(config.ISO_MAX_SAMPLES, len(detector.X_train)),
        max_features=config.ISO_MAX_FEATURES,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(detector.X_train)
    detector.unsupervised_model = model

    train_scores = model.score_samples(detector.X_train)
    train_preds  = model.predict(detector.X_train)
    train_binary = (train_preds == -1).astype(int)

    print(f"  Detected anomalies (train): {train_binary.sum():,} "
          f"({train_binary.mean() * 100:.2f}%)")

    results = {
        'train_scores':   train_scores,
        'train_binary':   train_binary,
        'test_scores':    None,
        'test_binary':    None,
        'best_threshold': None,
    }

    if detector.y_train is not None:
        best_threshold, tuned_binary = _tune_threshold(
            detector.y_train, train_scores
        )
        results['best_threshold'] = best_threshold
        results['train_binary']   = tuned_binary

        _print_metrics("Train (threshold-tuned)", detector.y_train, tuned_binary, train_scores)

        if detector.X_test is not None:
            test_scores = model.score_samples(detector.X_test)
            test_binary = (test_scores <= best_threshold).astype(int)
            results['test_scores'] = test_scores
            results['test_binary'] = test_binary
            _print_metrics("Test ", detector.y_test, test_binary, test_scores)

    return results


def _tune_threshold(y_true, scores):
    """Return the score threshold and binary predictions that maximise F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, -scores)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / denom, 0)
    best_idx       = f1.argmax()
    best_threshold = -thresholds[best_idx]
    binary         = (scores <= best_threshold).astype(int)
    print(f"  Best F1 threshold: {best_threshold:.4f}  (F1={f1[best_idx]:.4f})")
    return best_threshold, binary


def _print_metrics(label, y_true, y_pred, scores):
    print(f"\n  [{label}]")
    print(f"    Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"    Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"    Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"    F1-Score  : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"    ROC-AUC   : {roc_auc_score(y_true, -scores):.4f}")
    print(f"    Avg Prec  : {average_precision_score(y_true, -scores):.4f}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_unsupervised_results(detector, if_res):
    has_labels = detector.y_test is not None

    if_binary      = if_res['test_binary']  if if_res['test_binary']  is not None else if_res['train_binary']
    if_scores_plot = if_res['test_scores']  if if_res['test_scores']  is not None else if_res['train_scores']
    y_ref          = detector.y_test        if has_labels              else detector.y_train
    split_label    = 'Test'                 if has_labels              else 'Train'

    plt.figure(figsize=(20, 10))

    # 1. Score distribution
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(if_res['train_scores'], bins=60, edgecolor='black', alpha=0.7, color='steelblue')
    if if_res.get('best_threshold') is not None:
        ax1.axvline(if_res['best_threshold'], color='red', linestyle='--', label='Tuned threshold')
        ax1.legend()
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('IF: Score Distribution (Train)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. PCA — detections
    ax2 = plt.subplot(2, 3, 2)
    pca   = PCA(n_components=2, random_state=42)
    X_ref = detector.X_test if detector.X_test is not None else detector.X_train
    X_pca = pca.fit_transform(X_ref)
    sc    = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=if_binary,
                        cmap='coolwarm', alpha=0.5, s=8)
    ax2.set_title(f'IF: PCA Detections ({split_label})', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.colorbar(sc, ax=ax2, label='Anomaly')

    if has_labels:
        # 3. ROC curve
        ax3 = plt.subplot(2, 3, 3)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_ref, -if_scores_plot)
        auc = roc_auc_score(y_ref, -if_scores_plot)
        ax3.plot(fpr, tpr, color='steelblue', linewidth=2, label=f'ROC (AUC={auc:.3f})')
        ax3.plot([0, 1], [0, 1], 'k--', label='Random')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title(f'ROC Curve ({split_label})', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Confusion matrix
        ax4 = plt.subplot(2, 3, 4)
        sns.heatmap(confusion_matrix(y_ref, if_binary), annot=True, fmt='d',
                    cmap='Blues', ax=ax4,
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        ax4.set_title(f'IF: Confusion Matrix ({split_label})', fontsize=14, fontweight='bold')
        ax4.set_ylabel('True Label')
        ax4.set_xlabel('Predicted Label')

        # 5. Precision-Recall curve
        ax5 = plt.subplot(2, 3, 5)
        prec, rec, _ = precision_recall_curve(y_ref, -if_scores_plot)
        ap = average_precision_score(y_ref, -if_scores_plot)
        ax5.plot(rec, prec, color='steelblue', linewidth=2, label=f'AP={ap:.3f}')
        ax5.set_xlabel('Recall')
        ax5.set_ylabel('Precision')
        ax5.set_title(f'Precision-Recall Curve ({split_label})', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Performance metrics summary
        ax6 = plt.subplot(2, 3, 6)
        metrics = {
            'Accuracy':  accuracy_score(y_ref, if_binary),
            'Precision': precision_score(y_ref, if_binary, zero_division=0),
            'Recall':    recall_score(y_ref, if_binary, zero_division=0),
            'F1-Score':  f1_score(y_ref, if_binary, zero_division=0),
            'ROC-AUC':   auc,
        }
        bars = ax6.bar(metrics.keys(), metrics.values(),
                       color=['#1f77b4', '#d62728', '#9467bd', '#ff7f0e', '#2ca02c'])
        ax6.set_ylim(0, 1.1)
        ax6.set_title(f'Performance Metrics ({split_label})', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Score')
        for bar, v in zip(bars, metrics.values()):
            ax6.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                     f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

    else:
        # No labels — score-vs-index scatter and summary
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(range(len(if_scores_plot)), if_scores_plot,
                    c=if_binary, cmap='coolwarm', alpha=0.4, s=2)
        ax3.set_title('IF: Scores Across Samples', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Anomaly Score')

        ax4 = plt.subplot(2, 3, 4)
        summary = (
            f"Samples analysed : {len(if_scores_plot):,}\n\n"
            f"Anomalies detected: {if_binary.sum():,}\n"
            f"({if_binary.mean()*100:.1f}%)"
        )
        ax4.text(0.5, 0.5, summary, ha='center', va='center', fontsize=14,
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax4.axis('off')
        ax4.set_title('Detection Summary', fontsize=14, fontweight='bold')

    plt.suptitle('Isolation Forest — Anomaly Detection Results',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    output_path = detector._get_output_path('unsupervised_results.png')
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"\n  Unsupervised results saved to '{output_path}'")
    plt.close()
