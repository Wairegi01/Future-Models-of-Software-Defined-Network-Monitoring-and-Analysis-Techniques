"""
Deep Learning Analysis Module
Advanced neural network approaches for SDN monitoring

Future SDN Application:
- Complex pattern recognition beyond traditional ML
- Real-time deep packet inspection alternatives
- Automated feature learning
- Transfer learning across network domains
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def prepare_deep_learning_data(detector, sequence_length=10):
    """
    Prepare data for deep learning models (sequence-based)
    
    Args:
        detector: NetworkAnomalyDetector instance
        sequence_length: Length of sequences for LSTM/RNN
        
    Returns:
        dict: Prepared sequences and labels
    """
    print("\n" + "=" * 80)
    print("DEEP LEARNING DATA PREPARATION")
    print("=" * 80)
    print("\nSDN Future Application:")
    print("  • Learn complex temporal dependencies")
    print("  • Detect sophisticated multi-stage attacks")
    print("  • Automated feature engineering")
    
    feature_cols = detector._get_feature_columns()
    
    # Create sequences for LSTM-style processing
    print(f"\nCreating sequences of length {sequence_length}...")
    
    sequences = []
    labels = []
    
    for i in range(len(detector.df) - sequence_length):
        # Get sequence of features
        seq = detector.df[feature_cols].iloc[i:i+sequence_length].values
        sequences.append(seq)
        
        # Get label for next time step (if available)
        if 'IsAnomaly' in detector.df.columns:
            label = detector.df['IsAnomaly'].iloc[i+sequence_length]
            labels.append(label)
    
    sequences = np.array(sequences)
    labels = np.array(labels) if labels else None
    
    print(f"✓ Created {len(sequences)} sequences")
    print(f"  Shape: {sequences.shape}")
    if labels is not None:
        print(f"  Anomaly rate: {labels.mean()*100:.2f}%")
    
    return {
        'sequences': sequences,
        'labels': labels,
        'feature_cols': feature_cols,
        'sequence_length': sequence_length
    }


def autoencoder_anomaly_detection(detector, encoding_dim=5):
    """
    Use autoencoder for unsupervised anomaly detection
    
    Args:
        detector: NetworkAnomalyDetector instance
        encoding_dim: Dimension of encoded representation
        
    Returns:
        dict: Reconstruction errors and anomaly scores
    """
    print("\n" + "=" * 80)
    print("AUTOENCODER ANOMALY DETECTION")
    print("=" * 80)
    print("\nSDN Future Application:")
    print("  • Learn normal traffic patterns automatically")
    print("  • Detect novel/zero-day attacks")
    print("  • Compressed representation for efficient storage")
    
    # Note: This is a simplified version using dimensionality reduction
    # In production, you'd use actual neural network autoencoders
    from sklearn.decomposition import PCA
    
    feature_cols = detector._get_feature_columns()
    X = detector.df[feature_cols].values
    
    # Use PCA as a simple autoencoder approximation
    print(f"\nTraining autoencoder (encoding dim={encoding_dim})...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # "Encode" - reduce dimensionality
    encoder = PCA(n_components=encoding_dim)
    encoded = encoder.fit_transform(X_scaled)
    
    # "Decode" - reconstruct
    decoded = encoder.inverse_transform(encoded)
    
    # Calculate reconstruction error
    reconstruction_error = np.mean((X_scaled - decoded) ** 2, axis=1)
    
    # Determine threshold for anomalies
    threshold = np.percentile(reconstruction_error, 95)
    predictions = (reconstruction_error > threshold).astype(int)
    
    print(f"✓ Autoencoder trained")
    print(f"  Variance explained: {encoder.explained_variance_ratio_.sum():.2%}")
    print(f"  Anomaly threshold: {threshold:.4f}")
    print(f"  Detected anomalies: {predictions.sum()} ({predictions.mean()*100:.2f}%)")
    
    # Evaluate if labels available
    if 'IsAnomaly' in detector.df.columns:
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(detector.df['IsAnomaly'], predictions)
        f1 = f1_score(detector.df['IsAnomaly'], predictions)
        print(f"\nPerformance vs true labels:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    # Visualize
    _plot_autoencoder_results(detector, reconstruction_error, predictions, threshold)
    
    return {
        'reconstruction_error': reconstruction_error,
        'predictions': predictions,
        'threshold': threshold,
        'encoding': encoded
    }


def attention_based_analysis(detector):
    """
    Simulate attention mechanism to identify important features
    
    Args:
        detector: NetworkAnomalyDetector instance
        
    Returns:
        dict: Attention weights and important features
    """
    print("\n" + "=" * 80)
    print("ATTENTION-BASED FEATURE ANALYSIS")
    print("=" * 80)
    print("\nSDN Future Application:")
    print("  • Identify which features matter most for each decision")
    print("  • Explainable AI for network security")
    print("  • Dynamic feature selection")
    
    feature_cols = detector._get_feature_columns()
    
    # Simulate attention weights using feature importance
    if detector.supervised_model is not None:
        # Use Random Forest feature importance as attention proxy
        importance = detector.supervised_model.feature_importances_
        attention_weights = importance / importance.sum()
    else:
        # Use variance as attention proxy
        variances = detector.df[feature_cols].var()
        attention_weights = variances / variances.sum()
    
    # Create attention dataframe
    attention_df = pd.DataFrame({
        'feature': feature_cols,
        'attention_weight': attention_weights
    }).sort_values('attention_weight', ascending=False)
    
    print("\n✓ Attention weights calculated")
    print("\nTop 5 Features by Attention:")
    print(attention_df.head().to_string(index=False))
    
    # Visualize
    _plot_attention_weights(detector, attention_df)
    
    return {
        'attention_weights': attention_df,
        'top_features': attention_df.head(5)['feature'].tolist()
    }


def ensemble_deep_analysis(detector):
    """
    Combine multiple deep learning insights
    
    Args:
        detector: NetworkAnomalyDetector instance
        
    Returns:
        dict: Ensemble predictions and confidence scores
    """
    print("\n" + "=" * 80)
    print("ENSEMBLE DEEP LEARNING ANALYSIS")
    print("=" * 80)
    print("\nSDN Future Application:")
    print("  • Combine multiple AI models for robust detection")
    print("  • Higher accuracy through model diversity")
    print("  • Confidence-based automated responses")
    
    # Get predictions from multiple approaches
    predictions_list = []
    
    # 1. Autoencoder predictions
    if detector.unsupervised_model is not None:
        pred_unsup = (detector.unsupervised_model.predict(detector.X_train) == -1).astype(int)
        predictions_list.append(pred_unsup)
        print("✓ Unsupervised model predictions")
    
    # 2. Supervised predictions
    if detector.supervised_model is not None and detector.y_train is not None:
        pred_sup = detector.supervised_model.predict(detector.X_train)
        predictions_list.append(pred_sup)
        print("✓ Supervised model predictions")
    
    if len(predictions_list) >= 2:
        # Ensemble via voting
        ensemble_pred = np.mean(predictions_list, axis=0)
        ensemble_binary = (ensemble_pred >= 0.5).astype(int)
        
        # Confidence score
        confidence = np.abs(ensemble_pred - 0.5) * 2  # 0 to 1 scale
        
        print(f"\n✓ Ensemble predictions created")
        print(f"  High confidence predictions: {(confidence > 0.7).sum()}")
        print(f"  Low confidence predictions: {(confidence < 0.3).sum()}")
        
        if detector.y_train is not None:
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(detector.y_train, ensemble_binary)
            f1 = f1_score(detector.y_train, ensemble_binary)
            print(f"\nEnsemble Performance:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1-Score: {f1:.4f}")
        
        # Visualize
        _plot_ensemble_results(detector, ensemble_binary, confidence)
        
        return {
            'predictions': ensemble_binary,
            'confidence': confidence,
            'individual_predictions': predictions_list
        }
    else:
        print("⚠ Need at least 2 models for ensemble")
        return None


def _plot_autoencoder_results(detector, reconstruction_error, predictions, threshold):
    """Plot autoencoder results"""
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Reconstruction error distribution
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(reconstruction_error, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold: {threshold:.4f}')
    ax1.set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Reconstruction Error')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error over time
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(reconstruction_error, alpha=0.7, linewidth=1)
    ax2.axhline(threshold, color='red', linestyle='--', alpha=0.5)
    ax2.fill_between(range(len(reconstruction_error)), 
                     reconstruction_error, threshold,
                     where=reconstruction_error > threshold,
                     alpha=0.3, color='red', label='Anomalies')
    ax2.set_title('Reconstruction Error Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion matrix (if labels available)
    if 'IsAnomaly' in detector.df.columns:
        ax3 = plt.subplot(2, 3, 3)
        cm = confusion_matrix(detector.df['IsAnomaly'], predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax3)
        ax3.set_title('Autoencoder Detection Results', fontsize=14, fontweight='bold')
        ax3.set_ylabel('True Label')
        ax3.set_xlabel('Predicted Label')
    
    # 4. Top anomalies
    ax4 = plt.subplot(2, 3, 4)
    top_anomalies = np.argsort(reconstruction_error)[-50:]
    ax4.scatter(top_anomalies, reconstruction_error[top_anomalies],
               c='red', s=50, alpha=0.6)
    ax4.set_title('Top 50 Anomalies by Reconstruction Error', 
                 fontsize=14, fontweight='bold')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Reconstruction Error')
    ax4.grid(True, alpha=0.3)
    
    # 5. SDN Applications
    ax5 = plt.subplot(2, 3, 5)
    sdn_text = """
    AUTOENCODER SDN BENEFITS
    
    ✓ Zero-Day Attack Detection
      • Detect unknown attack patterns
      • No labeled training data needed
    
    ✓ Network Compression
      • Efficient traffic representation
      • Reduced storage requirements
    
    ✓ Anomaly Explanation
      • Identify which features deviate
      • Root cause analysis
    
    ✓ Transfer Learning
      • Pre-train on one network
      • Fine-tune for another
    """
    ax5.text(0.5, 0.5, sdn_text, ha='center', va='center', 
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
    ax5.axis('off')
    ax5.set_title('Deep Learning Benefits', fontsize=14, fontweight='bold')
    
    # 6. Summary
    ax6 = plt.subplot(2, 3, 6)
    summary = f"""
    AUTOENCODER SUMMARY
    
    Total Samples: {len(reconstruction_error):,}
    Detected Anomalies: {predictions.sum():,}
    Detection Rate: {predictions.mean()*100:.2f}%
    
    Threshold: {threshold:.4f}
    Mean Error: {reconstruction_error.mean():.4f}
    Max Error: {reconstruction_error.max():.4f}
    """
    ax6.text(0.5, 0.5, summary, ha='center', va='center', 
            fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax6.axis('off')
    ax6.set_title('Detection Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = detector._get_output_path('deep_learning_autoencoder.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Autoencoder results saved to '{output_path}'")
    plt.close()


def _plot_attention_weights(detector, attention_df):
    """Plot attention mechanism visualization"""
    fig = plt.figure(figsize=(16, 8))
    
    # 1. Attention weights bar chart
    ax1 = plt.subplot(1, 2, 1)
    colors = plt.cm.viridis(attention_df['attention_weight'] / attention_df['attention_weight'].max())
    ax1.barh(range(len(attention_df)), attention_df['attention_weight'], color=colors)
    ax1.set_yticks(range(len(attention_df)))
    ax1.set_yticklabels(attention_df['feature'])
    ax1.set_xlabel('Attention Weight', fontsize=12)
    ax1.set_title('Feature Attention Weights', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. SDN Application
    ax2 = plt.subplot(1, 2, 2)
    sdn_text = """
    ATTENTION MECHANISM IN SDN
    
    ✓ Explainable AI
      • Understand why decisions were made
      • Trust and transparency
    
    ✓ Dynamic Feature Selection
      • Focus on important features
      • Reduce computation overhead
    
    ✓ Adaptive Monitoring
      • Monitor high-attention features closely
      • Optimize resource allocation
    
    ✓ Attack Attribution
      • Identify attack characteristics
      • Targeted defense strategies
    
    Top 3 Features:
    """
    for i, row in attention_df.head(3).iterrows():
        sdn_text += f"\n    {i+1}. {row['feature']}: {row['attention_weight']:.3f}"
    
    ax2.text(0.5, 0.5, sdn_text, ha='center', va='center', 
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    ax2.axis('off')
    ax2.set_title('SDN Applications', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = detector._get_output_path('deep_learning_attention.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Attention analysis saved to '{output_path}'")
    plt.close()


def _plot_ensemble_results(detector, predictions, confidence):
    """Plot ensemble results"""
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Confidence distribution
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(confidence, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax1.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # 2. Confidence over time
    ax2 = plt.subplot(2, 3, 2)
    colors = ['red' if p == 1 else 'blue' for p in predictions]
    ax2.scatter(range(len(confidence)), confidence, c=colors, alpha=0.5, s=10)
    ax2.axhline(0.7, color='green', linestyle='--', label='High Confidence')
    ax2.axhline(0.3, color='orange', linestyle='--', label='Low Confidence')
    ax2.set_title('Prediction Confidence Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Confidence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion matrix
    if detector.y_train is not None:
        ax3 = plt.subplot(2, 3, 3)
        cm = confusion_matrix(detector.y_train, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax3)
        ax3.set_title('Ensemble Model Performance', fontsize=14, fontweight='bold')
        ax3.set_ylabel('True Label')
        ax3.set_xlabel('Predicted Label')
    
    # 4. Confidence categories
    ax4 = plt.subplot(2, 3, 4)
    high_conf = (confidence > 0.7).sum()
    med_conf = ((confidence >= 0.3) & (confidence <= 0.7)).sum()
    low_conf = (confidence < 0.3).sum()
    
    categories = ['High\nConfidence', 'Medium\nConfidence', 'Low\nConfidence']
    counts = [high_conf, med_conf, low_conf]
    colors_cat = ['green', 'yellow', 'red']
    
    ax4.bar(categories, counts, color=colors_cat, alpha=0.7, edgecolor='black')
    ax4.set_title('Confidence Categories', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Count')
    for i, (cat, count) in enumerate(zip(categories, counts)):
        ax4.text(i, count + max(counts)*0.02, f'{count:,}', 
                ha='center', fontweight='bold')
    
    # 5. SDN Applications
    ax5 = plt.subplot(2, 3, 5)
    sdn_text = """
    ENSEMBLE SDN BENEFITS
    
    ✓ Robust Detection
      • Multiple models reduce errors
      • Higher overall accuracy
    
    ✓ Confidence Scores
      • Automated response levels
      • High confidence → Immediate action
      • Low confidence → Human review
    
    ✓ Fail-Safe Operation
      • If one model fails, others continue
      • Redundancy and reliability
    """
    ax5.text(0.5, 0.5, sdn_text, ha='center', va='center', 
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax5.axis('off')
    ax5.set_title('Ensemble Benefits', fontsize=14, fontweight='bold')
    
    # 6. Summary
    ax6 = plt.subplot(2, 3, 6)
    summary = f"""
    ENSEMBLE SUMMARY
    
    Total Predictions: {len(predictions):,}
    Detected Anomalies: {predictions.sum():,}
    
    High Confidence: {high_conf:,} ({high_conf/len(predictions)*100:.1f}%)
    Medium Confidence: {med_conf:,} ({med_conf/len(predictions)*100:.1f}%)
    Low Confidence: {low_conf:,} ({low_conf/len(predictions)*100:.1f}%)
    
    Avg Confidence: {confidence.mean():.3f}
    """
    ax6.text(0.5, 0.5, summary, ha='center', va='center', 
            fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax6.axis('off')
    ax6.set_title('Ensemble Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = detector._get_output_path('deep_learning_ensemble.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Ensemble results saved to '{output_path}'")
    plt.close()


def run_deep_learning_analysis(detector):
    """
    Run complete deep learning analysis suite
    
    Args:
        detector: NetworkAnomalyDetector instance
    """
    print("\n" + "=" * 80)
    print("DEEP LEARNING ANALYSIS SUITE")
    print("=" * 80)
    
    # 1. Prepare sequential data
    dl_data = prepare_deep_learning_data(detector, sequence_length=10)
    
    # 2. Autoencoder anomaly detection
    autoencoder_results = autoencoder_anomaly_detection(detector, encoding_dim=5)
    
    # 3. Attention-based feature analysis
    attention_results = attention_based_analysis(detector)
    
    # 4. Ensemble analysis
    ensemble_results = ensemble_deep_analysis(detector)
    
    print("\n" + "=" * 80)
    print("DEEP LEARNING ANALYSIS COMPLETE")
    print("=" * 80)
    
    return {
        'sequential_data': dl_data,
        'autoencoder': autoencoder_results,
        'attention': attention_results,
        'ensemble': ensemble_results
    }
