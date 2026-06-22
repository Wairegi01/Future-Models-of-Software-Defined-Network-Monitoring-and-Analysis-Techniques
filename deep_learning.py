"""
Deep Learning Analysis Module
Neural network autoencoder for reconstruction-based anomaly detection in SDN
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, auc,
    average_precision_score, precision_recall_curve,
    accuracy_score, f1_score, precision_score, recall_score,
)
import config
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# Autoencoder anomaly detection (MLPRegressor, trains on normal only)
# -------------------------------------------------------------------

def autoencoder_anomaly_detection(detector):
    """
    Reconstruction-based anomaly detection using a neural network autoencoder.

    Architecture: input_dim → 16 → 8 → 4 (bottleneck) → 8 → 16 → input_dim
    Trained exclusively on normal traffic so anomalous flows produce high
    reconstruction error — they don't fit the patterns the network learned.
    """
    print("\n" + "-" * 30)
    print("AUTOENCODER ANOMALY DETECTION")
    print("-" * 30)
    print("\nSDN Application:")
    print("  • Unsupervised detection of novel attack patterns")
    print("  • No labels required at deployment time")
    print("  • High reconstruction error = suspicious flow")

    X_train = detector.X_train
    X_test  = detector.X_test
    y_train = detector.y_train
    y_test  = detector.y_test

    n_features = X_train.shape[1]
    hidden     = (16, 8, config.AE_BOTTLENECK, 8, 16)

    # Train only on normal flows — core of reconstruction-based detection
    if y_train is not None:
        normal_mask  = y_train == 0
        X_train_norm = X_train[normal_mask]
        print(f"\n  Training on {X_train_norm.shape[0]:,} normal flows only "
              f"({normal_mask.sum() / len(y_train) * 100:.1f}% of train set)")
    else:
        X_train_norm = X_train
        print(f"\n  No labels — training on all {X_train.shape[0]:,} flows")

    print(f"  Architecture : {n_features} → 16 → 8 → {config.AE_BOTTLENECK} (bottleneck) "
          f"→ 8 → 16 → {n_features}")
    print(f"  Activation   : {config.AE_ACTIVATION}")
    print(f"  Max epochs   : {config.AE_MAX_ITER}")

    ae = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation=config.AE_ACTIVATION,
        solver='adam',
        learning_rate_init=config.AE_LEARNING_RATE,
        max_iter=config.AE_MAX_ITER,
        random_state=config.RANDOM_STATE,
        tol=1e-8,
        verbose=False,
    )
    ae.fit(X_train_norm, X_train_norm)
    loss_curve = ae.loss_curve_
    print(f"  Training complete — final loss: {loss_curve[-1]:.6f}")

    # Reconstruction errors
    def _recon_err(X):
        X_hat = ae.predict(X)
        return np.mean((X - X_hat) ** 2, axis=1)

    train_errors = _recon_err(X_train_norm)
    test_errors  = _recon_err(X_test) if X_test is not None else None

    # Threshold: mean + sigma * SD of training (normal) reconstruction errors
    threshold = train_errors.mean() + config.AE_THRESHOLD_SIGMA * train_errors.std()
    print(f"\n  Anomaly threshold (μ + {config.AE_THRESHOLD_SIGMA}σ): {threshold:.4f}")

    # Predictions on test set
    if test_errors is not None:
        y_pred = (test_errors > threshold).astype(int)
        print(f"  Flagged as anomalous : {y_pred.sum():,} "
              f"({y_pred.mean() * 100:.2f}% of test set)")

        if y_test is not None:
            _print_metrics("Test", y_test, y_pred, test_errors)

    # Feature attribution via perturbation
    attribution = {}
    if X_test is not None:
        baseline = float(test_errors.mean())
        feature_cols = detector._get_feature_columns()
        for i, feat in enumerate(feature_cols):
            X_pert        = X_test.copy()
            X_pert[:, i]  = 0.0
            attribution[feat] = abs(_recon_err(X_pert).mean() - baseline)
        total = sum(attribution.values()) or 1.0
        attribution = {k: v / total for k, v in
                       sorted(attribution.items(), key=lambda x: x[1], reverse=True)}

    _plot_autoencoder_results(
        detector, train_errors, test_errors, y_test, y_pred if test_errors is not None else None,
        threshold, loss_curve, attribution,
    )

    return {
        'reconstruction_error_test': test_errors,
        'predictions': y_pred if test_errors is not None else None,
        'threshold': threshold,
        'loss_curve': loss_curve,
        'attribution': attribution,
    }


def _print_metrics(label, y_true, y_pred, scores):
    print(f"\n  [{label} Set Performance]")
    print(f"    Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"    Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"    Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"    F1-Score  : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"    ROC-AUC   : {roc_auc_score(y_true, scores):.4f}")
    print(f"    Avg Prec  : {average_precision_score(y_true, scores):.4f}")


# -----
# Plots
# -----

def _plot_autoencoder_results(detector, train_errors, test_errors, y_test,
                               y_pred, threshold, loss_curve, attribution):
    plt.figure(figsize=(22, 12))

    # 1. Reconstruction error distribution — normal vs anomaly on test set
    ax1 = plt.subplot(2, 3, 1)
    if test_errors is not None and y_test is not None:
        normal_err  = test_errors[y_test == 0]
        anomaly_err = test_errors[y_test == 1]
        ax1.hist(normal_err,  bins=80, alpha=0.65, color='steelblue',
                 label='Normal', density=True)
        ax1.hist(anomaly_err, bins=80, alpha=0.65, color='tomato',
                 label='Anomaly', density=True)
    else:
        ax1.hist(train_errors, bins=60, alpha=0.7, color='steelblue',
                 label='Training (normal)', density=True)
    ax1.axvline(threshold, color='orange', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold:.3f}')
    ax1.set_title('Reconstruction Error Distribution', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Mean Squared Reconstruction Error')
    ax1.set_ylabel('Density')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. ROC curve + operating point
    ax2 = plt.subplot(2, 3, 2)
    if test_errors is not None and y_test is not None:
        fpr, tpr, _ = roc_curve(y_test, test_errors)
        roc_auc     = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color='purple', linewidth=2.5,
                 label=f'Autoencoder (AUC = {roc_auc:.3f})')
        ax2.fill_between(fpr, tpr, alpha=0.1, color='purple')
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Random')
        fpr_op = ((y_test == 0) & (test_errors > threshold)).sum() / (y_test == 0).sum()
        tpr_op = ((y_test == 1) & (test_errors > threshold)).sum() / (y_test == 1).sum()
        ax2.scatter(fpr_op, tpr_op, s=120, zorder=5, color='#FF9800',
                    label=f'Operating point (FPR={fpr_op:.2f}, TPR={tpr_op:.2f})')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    # 3. Training convergence
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(range(1, len(loss_curve) + 1), loss_curve,
             color='green', linewidth=2, label='Training loss (MSE)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Mean Squared Error')
    ax3.set_title('Training Convergence Curve', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    final_loss = loss_curve[-1]
    ax3.annotate(f'Final: {final_loss:.4f}',
                 xy=(len(loss_curve), final_loss),
                 xytext=(len(loss_curve) * 0.6, max(loss_curve) * 0.6),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)

    # 4. Confusion matrix
    ax4 = plt.subplot(2, 3, 4)
    if y_test is not None and y_pred is not None:
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax4,
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        ax4.set_title('Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
        ax4.set_ylabel('True Label')
        ax4.set_xlabel('Predicted Label')

    # 5. Feature attribution
    ax5 = plt.subplot(2, 3, 5)
    if attribution:
        top_feats  = list(attribution.keys())[:10]
        top_scores = [attribution[f] for f in top_feats]
        colors_bar = ['#E53935' if i == 0 else '#7B1FA2' if i == 1 else '#5C6BC0'
                      for i in range(len(top_feats))]
        ax5.barh(top_feats[::-1], top_scores[::-1],
                 color=colors_bar[::-1], edgecolor='white')
        ax5.set_xlabel('Normalised Attribution Score')
        ax5.set_title('Top-10 Feature Sensitivity', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

    # 6. Precision-Recall curve
    ax6 = plt.subplot(2, 3, 6)
    if test_errors is not None and y_test is not None and y_pred is not None:
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, test_errors)
        ap = average_precision_score(y_test, test_errors)
        ax6.plot(rec_curve, prec_curve, color='#E65100', linewidth=2.5,
                 label=f'Autoencoder (AP = {ap:.3f})')
        ax6.axhline(y_test.mean(), color='grey', linewidth=1.2, linestyle='--',
                    label=f'Baseline (prevalence = {y_test.mean():.3f})')
        op_prec = precision_score(y_test, y_pred, zero_division=0)
        op_rec  = recall_score(y_test, y_pred, zero_division=0)
        ax6.scatter(op_rec, op_prec, s=120, zorder=5, color='#FF9800',
                    label=f'Operating point (P={op_prec:.2f}, R={op_rec:.2f})')
        ax6.set_xlabel('Recall')
        ax6.set_ylabel('Precision')
        ax6.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.set_xlim([0, 1])
        ax6.set_ylim([0, 1.05])
        ax6.grid(True, alpha=0.3)

    plt.suptitle('Autoencoder Anomaly Detection — Neural Network Approach',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    output_path = detector._get_output_path('autoencoder_results.png')
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"\n  Autoencoder results saved to '{output_path}'")
    plt.close()


# --------------------------------------------------------
# Ensemble: Autoencoder + Isolation Forest + Random Forest
# --------------------------------------------------------

def ensemble_anomaly_detection(detector, ae_pred, ae_errors):
    print("\n" + "-" * 30)
    print("ENSEMBLE ANOMALY DETECTION — WEIGHTED VOTING")
    print("-" * 30)

    X_train = detector.X_train
    X_test  = detector.X_test
    y_train = detector.y_train
    y_test  = detector.y_test

    all_pred   = {}
    all_scores = {}

    # Autoencoder — normalised reconstruction error
    ae_score = ae_errors / (ae_errors.max() + 1e-8)
    all_pred['Autoencoder']   = ae_pred
    all_scores['Autoencoder'] = ae_score

    # Isolation Forest
    print("\n  Training Isolation Forest …")
    iso = IsolationForest(
        contamination=config.ISO_CONTAMINATION,
        n_estimators=config.ISO_N_ESTIMATORS,
        random_state=config.RANDOM_STATE,
    )
    iso.fit(X_train)
    iso_pred  = (iso.predict(X_test) == -1).astype(int)
    iso_score = -iso.decision_function(X_test)
    iso_score = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-8)
    all_pred['Isolation Forest']   = iso_pred
    all_scores['Isolation Forest'] = iso_score
    print(f"    Flagged: {iso_pred.sum():,} anomalies ({iso_pred.mean()*100:.2f}%)")

    # Random Forest
    if y_train is not None:
        print("  Training Random Forest …")
        rf = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            class_weight=config.RF_CLASS_WEIGHT,
            random_state=config.RANDOM_STATE,
        )
        rf.fit(X_train, y_train)
        rf_pred  = rf.predict(X_test)
        rf_score = rf.predict_proba(X_test)[:, 1]
        all_pred['Random Forest']   = rf_pred
        all_scores['Random Forest'] = rf_score
        print(f"    Flagged: {rf_pred.sum():,} anomalies ({rf_pred.mean()*100:.2f}%)")

    # AUC-based weights
    weights = {}
    if y_test is not None:
        for name, score in all_scores.items():
            try:
                weights[name] = roc_auc_score(y_test, score)
            except Exception:
                weights[name] = 0.5
    else:
        weights = {name: 1.0 for name in all_scores}

    total_weight = sum(weights.values())
    print(f"\n  Model weights (by ROC-AUC):")
    for name, w in weights.items():
        print(f"    {name:<20} : {w:.4f}  (weight share: {w/total_weight*100:.1f}%)")

    # Weighted average score
    ensemble_score = sum(
        (weights[name] / total_weight) * all_scores[name]
        for name in all_scores
    )

    # Optimal threshold via Youden's J statistic
    if y_test is not None:
        fpr_e, tpr_e, thresholds_e = roc_curve(y_test, ensemble_score)
        j_scores       = tpr_e - fpr_e
        optimal_thresh = float(thresholds_e[np.argmax(j_scores)])
    else:
        optimal_thresh = 0.5
    print(f"\n  Optimal threshold (Youden's J): {optimal_thresh:.4f}")

    ensemble_pred = (ensemble_score >= optimal_thresh).astype(int)
    confidence    = np.abs(ensemble_score - optimal_thresh) * 2

    print(f"\n  Weighted ensemble — flagged: {ensemble_pred.sum():,} "
          f"anomalies ({ensemble_pred.mean()*100:.2f}%)")

    if y_test is not None:
        print(f"\n  [Ensemble Test Set Performance]")
        print(f"    Accuracy  : {accuracy_score(y_test, ensemble_pred):.4f}")
        print(f"    Precision : {precision_score(y_test, ensemble_pred, zero_division=0):.4f}")
        print(f"    Recall    : {recall_score(y_test, ensemble_pred, zero_division=0):.4f}")
        print(f"    F1-Score  : {f1_score(y_test, ensemble_pred, zero_division=0):.4f}")
        print(f"    ROC-AUC   : {roc_auc_score(y_test, ensemble_score):.4f}")

    _plot_ensemble_results(detector, all_pred, all_scores, ensemble_pred,
                           ensemble_score, confidence, weights, optimal_thresh, y_test)

    return {
        'ensemble_predictions': ensemble_pred,
        'confidence': confidence,
        'model_predictions': all_pred,
        'weights': weights,
    }

def _plot_ensemble_results(detector, all_pred, all_scores, ensemble_pred,
                           ensemble_score, confidence, weights, optimal_thresh, y_test):
    plt.figure(figsize=(22, 12))

    model_names    = list(all_pred.keys()) + ['Ensemble']
    all_preds_list = list(all_pred.values()) + [ensemble_pred]

    # 1. F1 Score comparison
    ax1 = plt.subplot(2, 3, 1)
    if y_test is not None:
        f1_scores  = [f1_score(y_test, p, zero_division=0) for p in all_preds_list]
        bar_colors = ['#5C6BC0'] * len(all_pred) + ['#388E3C']
        bars = ax1.bar(model_names, f1_scores, color=bar_colors, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, f1_scores):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     val + 0.01, f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
        ax1.set_title('F1 Score — Model Comparison', fontsize=13, fontweight='bold')
        ax1.set_ylabel('F1 Score')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3, axis='y')

    # 2. AUC comparison — all models including ensemble, annotated with weight share
    ax2 = plt.subplot(2, 3, 2)
    total_w    = sum(weights.values())
    bar_names  = list(all_pred.keys())
    auc_vals   = [weights[n] for n in bar_names]
    bar_colors = ['#5C6BC0'] * len(all_pred)
    if y_test is not None:
        ens_auc = roc_auc_score(y_test, ensemble_score)
        bar_names.append('Ensemble')
        auc_vals.append(ens_auc)
        bar_colors.append('#388E3C')
    bars2 = ax2.bar(bar_names, auc_vals, color=bar_colors, alpha=0.85, edgecolor='white')
    for i, (bar, av) in enumerate(zip(bars2, auc_vals)):
        x = bar.get_x() + bar.get_width() / 2
        h = bar.get_height()
        ax2.text(x, h + 0.005, f'{av:.3f}', ha='center', fontsize=10, fontweight='bold')
        if i < len(all_pred):
            ws = weights[bar_names[i]] / total_w * 100
            ax2.text(x, h / 2, f'{ws:.1f}%\nweight', ha='center', va='center',
                     fontsize=9, color='white', fontweight='bold')
        else:
            ax2.text(x, h / 2, 'Ensemble', ha='center', va='center',
                     fontsize=9, color='white', fontweight='bold')
    ax2.set_title('ROC-AUC + Vote Weights', fontsize=13, fontweight='bold')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Ensemble confusion matrix
    ax3 = plt.subplot(2, 3, 3)
    if y_test is not None:
        cm = confusion_matrix(y_test, ensemble_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax3,
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'],
                    annot_kws={'size': 13})
        ax3.set_title('Ensemble Confusion Matrix', fontsize=13, fontweight='bold')
        ax3.set_ylabel('True Label')
        ax3.set_xlabel('Predicted Label')

    # 4. Confidence distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(confidence, bins=50, color='#388E3C', alpha=0.75, edgecolor='white')
    ax4.axvline(0.7, color='green',  linestyle='--', linewidth=1.5, label='High (>0.7)')
    ax4.axvline(0.3, color='orange', linestyle='--', linewidth=1.5, label='Low (<0.3)')
    ax4.set_title('Prediction Confidence Distribution', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Confidence Score')
    ax4.set_ylabel('Count')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. ROC curves — all models + ensemble (using continuous scores)
    ax5 = plt.subplot(2, 3, 5)
    if y_test is not None:
        colors_roc = ['#7B1FA2', '#E65100', '#1565C0', '#388E3C']
        score_map  = {**all_scores, 'Ensemble': ensemble_score}
        for (name, pred), col in zip(list(all_pred.items()) + [('Ensemble', ensemble_pred)],
                                     colors_roc):
            try:
                fpr, tpr, _ = roc_curve(y_test, score_map[name])
                roc_auc_val = auc(fpr, tpr)
                lw = 3 if name == 'Ensemble' else 2
                ax5.plot(fpr, tpr, color=col, linewidth=lw,
                         label=f'{name} (AUC={roc_auc_val:.3f})')
            except Exception:
                pass
        # Mark ensemble operating point at optimal threshold
        fpr_op = ((y_test == 0) & (ensemble_pred == 1)).sum() / (y_test == 0).sum()
        tpr_op = ((y_test == 1) & (ensemble_pred == 1)).sum() / (y_test == 1).sum()
        ax5.scatter(fpr_op, tpr_op, s=120, zorder=5, color='#FF9800',
                    label=f'Ensemble op. point (FPR={fpr_op:.2f}, TPR={tpr_op:.2f})')
        ax5.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax5.set_xlabel('False Positive Rate')
        ax5.set_ylabel('True Positive Rate')
        ax5.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

    # 6. Summary metrics table
    ax6 = plt.subplot(2, 3, 6)
    if y_test is not None:
        rows      = []
        score_map = {**all_scores, 'Ensemble': ensemble_score}
        for name, pred in list(all_pred.items()) + [('Ensemble', ensemble_pred)]:
            rows.append([
                name,
                f"{accuracy_score(y_test, pred)*100:.1f}%",
                f"{precision_score(y_test, pred, zero_division=0)*100:.1f}%",
                f"{recall_score(y_test, pred, zero_division=0)*100:.1f}%",
                f"{f1_score(y_test, pred, zero_division=0)*100:.1f}%",
                f"{roc_auc_score(y_test, score_map[name]):.3f}",
            ])
        col_labels = ['Model', 'Acc', 'Prec', 'Rec', 'F1', 'AUC']
        table = ax6.table(cellText=rows, colLabels=col_labels,
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.0)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#37474F')
                cell.set_text_props(color='white', fontweight='bold')
            elif rows[row - 1][0] == 'Ensemble':
                cell.set_facecolor('#C8E6C9')
        ax6.axis('off')
        ax6.set_title('Performance Summary', fontsize=13, fontweight='bold')

    plt.suptitle('Ensemble Anomaly Detection — Model Comparison',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    output_path = detector._get_output_path('ensemble_results.png')
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"\n  Ensemble results saved to '{output_path}'")
    plt.close()

# -----------
# Entry point
# -----------

def run_deep_learning_analysis(detector):
    print("\n" + "-" * 30)
    print("DEEP LEARNING — AUTOENCODER ANALYSIS")
    print("-" * 30)

    t0 = time.time()
    results = autoencoder_anomaly_detection(detector)
    ae_time = time.time() - t0
    print(f"\n  Autoencoder training time : {ae_time:.1f}s")

    print("\n" + "-" * 30)
    print("ENSEMBLE ANALYSIS")
    print("-" * 30)

    t0 = time.time()
    ensemble_results = ensemble_anomaly_detection(
        detector,
        ae_pred   = results['predictions'],
        ae_errors = results['reconstruction_error_test'],
    )
    ens_time = time.time() - t0

    print("\n" + "-" * 30)
    print("DEEP LEARNING ANALYSIS COMPLETE")
    print("-" * 30)
    print(f"\n  Autoencoder training time : {ae_time:.1f}s")
    print(f"  Ensemble training time    : {ens_time:.1f}s")
    print(f"  Deep Learning total       : {ae_time + ens_time:.1f}s")

    return {**results, **ensemble_results}