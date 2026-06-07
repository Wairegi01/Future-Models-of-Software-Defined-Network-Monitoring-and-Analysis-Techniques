"""
Advanced Network Behavior Analysis
Graph-based and behavioral analytics for SDN

Future SDN Application:
- Network topology-aware anomaly detection
- Behavioral profiling and baseline learning
- Intent-based networking insights
- Automated policy generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def behavioral_profiling(detector, n_profiles=5):
    """
    Create behavioral profiles of network traffic
    
    Args:
        detector: NetworkAnomalyDetector instance
        n_profiles: Number of behavior profiles to create
        
    Returns:
        dict: Behavior profiles and classifications
    """
    print("\n" + "=" * 80)
    print("BEHAVIORAL PROFILING - Traffic Pattern Classification")
    print("=" * 80)
    print("\nSDN Future Application:")
    print("  • Automatic traffic classification")
    print("  • Intent-based policy recommendations")
    print("  • Behavioral baseline for anomaly detection")
    
    feature_cols = detector._get_feature_columns()
    X = detector.df[feature_cols].values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster into behavioral profiles
    print(f"\nCreating {n_profiles} behavioral profiles using K-Means...")
    
    kmeans = KMeans(n_clusters=n_profiles, random_state=42, n_init=10)
    profiles = kmeans.fit_predict(X_scaled)
    
    # Analyze each profile
    profile_summary = []
    for i in range(n_profiles):
        mask = profiles == i
        count = mask.sum()
        pct = count / len(profiles) * 100
        
        # Get representative features for this profile
        profile_data = detector.df[mask][feature_cols]
        means = profile_data.mean()
        
        profile_summary.append({
            'profile_id': i,
            'count': count,
            'percentage': pct,
            'characteristics': means.to_dict()
        })
        
        print(f"\nProfile {i}: {count:,} samples ({pct:.1f}%)")
        print(f"  Top characteristics:")
        top_features = means.abs().sort_values(ascending=False).head(3)
        for feat, val in top_features.items():
            print(f"    {feat}: {val:.3f}")
    
    # Detect anomalous profiles
    if 'IsAnomaly' in detector.df.columns:
        print("\n" + "-" * 60)
        print("Profile-Anomaly Relationship:")
        for i in range(n_profiles):
            mask = profiles == i
            anomaly_rate = detector.df[mask]['IsAnomaly'].mean()
            print(f"  Profile {i}: {anomaly_rate*100:.1f}% anomalies")
    
    # Visualize
    _plot_behavioral_profiles(detector, profiles, profile_summary, n_profiles)
    
    return {
        'profiles': profiles,
        'profile_summary': profile_summary,
        'n_profiles': n_profiles
    }


def density_based_clustering(detector, eps=0.5, min_samples=5):
    """
    DBSCAN clustering for outlier detection
    
    Args:
        detector: NetworkAnomalyDetector instance
        eps: DBSCAN epsilon parameter
        min_samples: Minimum samples for core point
        
    Returns:
        dict: Clusters and outliers
    """
    print("\n" + "=" * 80)
    print("DENSITY-BASED CLUSTERING - Outlier Detection")
    print("=" * 80)
    print("\nSDN Future Application:")
    print("  • Identify rare traffic patterns")
    print("  • Detect network scanning and probing")
    print("  • Flexible anomaly boundaries")
    
    feature_cols = detector._get_feature_columns()
    X = detector.df[feature_cols].values
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # DBSCAN clustering
    print(f"\nRunning DBSCAN (eps={eps}, min_samples={min_samples})...")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    clusters = dbscan.fit_predict(X_scaled)
    
    # Analyze results
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_outliers = (clusters == -1).sum()
    outlier_pct = n_outliers / len(clusters) * 100
    
    print(f"\n✓ Clustering complete")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Outliers detected: {n_outliers:,} ({outlier_pct:.2f}%)")
    
    # Cluster sizes
    cluster_counts = Counter(clusters)
    print(f"\n  Cluster sizes:")
    for cluster_id, count in sorted(cluster_counts.items()):
        if cluster_id == -1:
            print(f"    Outliers: {count:,}")
        else:
            print(f"    Cluster {cluster_id}: {count:,}")
    
    # Evaluate against true labels
    if 'IsAnomaly' in detector.df.columns:
        outlier_labels = (clusters == -1).astype(int)
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        acc = accuracy_score(detector.df['IsAnomaly'], outlier_labels)
        prec = precision_score(detector.df['IsAnomaly'], outlier_labels, zero_division=0)
        rec = recall_score(detector.df['IsAnomaly'], outlier_labels)
        
        print(f"\n  Performance vs true anomalies:")
        print(f"    Accuracy: {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall: {rec:.4f}")
    
    # Visualize
    _plot_density_clustering(detector, clusters, X_scaled)
    
    return {
        'clusters': clusters,
        'n_clusters': n_clusters,
        'n_outliers': n_outliers,
        'outlier_indices': np.where(clusters == -1)[0]
    }


def network_flow_analysis(detector):
    """
    Analyze network flow characteristics
    
    Args:
        detector: NetworkAnomalyDetector instance
        
    Returns:
        dict: Flow statistics and patterns
    """
    print("\n" + "=" * 80)
    print("NETWORK FLOW ANALYSIS")
    print("=" * 80)
    print("\nSDN Future Application:")
    print("  • Flow table optimization")
    print("  • Predictive flow installation")
    print("  • Traffic engineering policies")
    
    feature_cols = detector._get_feature_columns()
    
    # Analyze key flow metrics
    flow_stats = {}
    
    # 1. Packet size distribution
    if 'BytesSent' in feature_cols and 'PacketsSent' in feature_cols:
        bytes_per_packet = detector.df['BytesSent'] / (detector.df['PacketsSent'] + 1e-6)
        flow_stats['avg_packet_size'] = bytes_per_packet.mean()
        flow_stats['packet_size_std'] = bytes_per_packet.std()
        print(f"\nAverage packet size: {flow_stats['avg_packet_size']:.4f}")
    
    # 2. Flow duration patterns
    if 'Duration' in feature_cols:
        flow_stats['avg_duration'] = detector.df['Duration'].mean()
        flow_stats['long_flows'] = (detector.df['Duration'] > detector.df['Duration'].quantile(0.95)).sum()
        flow_stats['short_flows'] = (detector.df['Duration'] < detector.df['Duration'].quantile(0.05)).sum()
        print(f"Average flow duration: {flow_stats['avg_duration']:.4f}")
        print(f"Long flows (>95th percentile): {flow_stats['long_flows']:,}")
        print(f"Short flows (<5th percentile): {flow_stats['short_flows']:,}")
    
    # 3. Traffic symmetry
    if 'BytesSent' in feature_cols and 'BytesReceived' in feature_cols:
        symmetry = detector.df['BytesSent'] / (detector.df['BytesReceived'] + 1e-6)
        flow_stats['traffic_symmetry'] = symmetry.mean()
        flow_stats['asymmetric_flows'] = ((symmetry > 10) | (symmetry < 0.1)).sum()
        print(f"\nTraffic symmetry ratio: {flow_stats['traffic_symmetry']:.4f}")
        print(f"Highly asymmetric flows: {flow_stats['asymmetric_flows']:,}")
    
    # 4. Protocol distribution (if available)
    if 'Protocol' in feature_cols:
        protocol_dist = detector.df['Protocol'].value_counts()
        flow_stats['protocol_distribution'] = protocol_dist.to_dict()
        print(f"\nProtocol distribution (top 5):")
        for proto, count in protocol_dist.head().items():
            print(f"  Protocol {proto}: {count:,} ({count/len(detector.df)*100:.1f}%)")
    
    # Visualize
    _plot_flow_analysis(detector, flow_stats, feature_cols)
    
    return flow_stats


def anomaly_severity_scoring(detector):
    """
    Calculate severity scores for detected anomalies
    
    Args:
        detector: NetworkAnomalyDetector instance
        
    Returns:
        dict: Severity scores and categorization
    """
    print("\n" + "=" * 80)
    print("ANOMALY SEVERITY SCORING")
    print("=" * 80)
    print("\nSDN Future Application:")
    print("  • Prioritize security responses")
    print("  • Automated threat response levels")
    print("  • Resource allocation for investigation")
    
    # Calculate anomaly scores from multiple models
    scores = []
    
    # 1. Isolation Forest score (if available)
    if detector.unsupervised_model is not None:
        iso_scores = detector.unsupervised_model.score_samples(detector.X_train)
        # Normalize to 0-1 (higher = more anomalous)
        iso_normalized = 1 / (1 + np.exp(iso_scores))  # Sigmoid
        scores.append(iso_normalized)
    
    # 2. Random Forest probability (if available)
    if detector.supervised_model is not None and detector.X_train is not None:
        try:
            rf_probs = detector.supervised_model.predict_proba(detector.X_train)[:, 1]
            scores.append(rf_probs)
        except:
            pass
    
    if scores:
        # Combine scores
        combined_score = np.mean(scores, axis=0)
        
        # Categorize severity
        severity = np.zeros(len(combined_score), dtype=int)
        severity[combined_score >= 0.8] = 3  # Critical
        severity[(combined_score >= 0.6) & (combined_score < 0.8)] = 2  # High
        severity[(combined_score >= 0.4) & (combined_score < 0.6)] = 1  # Medium
        # Low is 0 (default)
        
        severity_counts = Counter(severity)
        
        print(f"\n✓ Severity scoring complete")
        print(f"\nSeverity Distribution:")
        print(f"  Critical (≥0.8): {severity_counts.get(3, 0):,}")
        print(f"  High (0.6-0.8): {severity_counts.get(2, 0):,}")
        print(f"  Medium (0.4-0.6): {severity_counts.get(1, 0):,}")
        print(f"  Low (<0.4): {severity_counts.get(0, 0):,}")
        
        # Identify top threats
        top_threats = np.argsort(combined_score)[-10:][::-1]
        print(f"\nTop 10 Most Severe Anomalies:")
        for rank, idx in enumerate(top_threats, 1):
            print(f"  #{rank}: Sample {idx}, Score: {combined_score[idx]:.4f}")
        
        # Visualize
        _plot_severity_analysis(detector, combined_score, severity)
        
        return {
            'severity_scores': combined_score,
            'severity_levels': severity,
            'top_threats': top_threats,
            'distribution': severity_counts
        }
    else:
        print("⚠ No models available for severity scoring")
        return None


def _plot_behavioral_profiles(detector, profiles, profile_summary, n_profiles):
    """Plot behavioral profiling results"""
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Profile distribution
    ax1 = plt.subplot(2, 3, 1)
    profile_counts = Counter(profiles)
    colors = plt.cm.Set3(range(n_profiles))
    ax1.bar(range(n_profiles), [profile_counts[i] for i in range(n_profiles)], 
           color=colors, edgecolor='black')
    ax1.set_title('Behavioral Profile Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Profile ID')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. PCA visualization of profiles
    ax2 = plt.subplot(2, 3, 2)
    from sklearn.decomposition import PCA
    feature_cols = detector._get_feature_columns()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(detector.df[feature_cols])
    
    for i in range(n_profiles):
        mask = profiles == i
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   label=f'Profile {i}', alpha=0.6, s=10, color=colors[i])
    ax2.set_title('Profiles in PCA Space', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.legend()
    
    # 3. Profile characteristics heatmap
    ax3 = plt.subplot(2, 3, 3)
    char_matrix = []
    for summary in profile_summary:
        char_matrix.append(list(summary['characteristics'].values())[:8])
    char_matrix = np.array(char_matrix)
    
    sns.heatmap(char_matrix, annot=False, cmap='coolwarm', 
               xticklabels=list(profile_summary[0]['characteristics'].keys())[:8],
               yticklabels=[f'P{i}' for i in range(n_profiles)],
               ax=ax3, cbar_kws={'label': 'Value'})
    ax3.set_title('Profile Characteristics', fontsize=14, fontweight='bold')
    
    # 4. Anomaly rate by profile
    if 'IsAnomaly' in detector.df.columns:
        ax4 = plt.subplot(2, 3, 4)
        anomaly_rates = []
        for i in range(n_profiles):
            mask = profiles == i
            rate = detector.df[mask]['IsAnomaly'].mean()
            anomaly_rates.append(rate)
        
        ax4.bar(range(n_profiles), anomaly_rates, color=colors, edgecolor='black')
        ax4.set_title('Anomaly Rate by Profile', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Profile ID')
        ax4.set_ylabel('Anomaly Rate')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for i, rate in enumerate(anomaly_rates):
            ax4.text(i, rate + 0.01, f'{rate*100:.1f}%', 
                    ha='center', fontweight='bold')
    
    # 5. SDN Applications
    ax5 = plt.subplot(2, 3, 5)
    sdn_text = """
    BEHAVIORAL PROFILING SDN USES
    
    ✓ Intent-Based Networking
      • Auto-classify traffic types
      • Generate QoS policies
    
    ✓ Baseline Learning
      • Normal behavior per profile
      • Detect deviations
    
    ✓ Traffic Engineering
      • Route optimization per profile
      • Resource allocation
    
    ✓ Security Policies
      • Profile-specific rules
      • Anomaly thresholds
    """
    ax5.text(0.5, 0.5, sdn_text, ha='center', va='center', 
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax5.axis('off')
    ax5.set_title('SDN Applications', fontsize=14, fontweight='bold')
    
    # 6. Summary
    ax6 = plt.subplot(2, 3, 6)
    summary = f"""
    PROFILING SUMMARY
    
    Total Samples: {len(profiles):,}
    Behavior Profiles: {n_profiles}
    
    Largest Profile: {max(profile_counts.values()):,}
    Smallest Profile: {min(profile_counts.values()):,}
    """
    ax6.text(0.5, 0.5, summary, ha='center', va='center', 
            fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax6.axis('off')
    ax6.set_title('Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = detector._get_output_path('behavioral_profiling.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Behavioral profiling saved to '{output_path}'")
    plt.close()


def _plot_density_clustering(detector, clusters, X_scaled):
    """Plot DBSCAN clustering results"""
    fig = plt.figure(figsize=(20, 10))
    
    # Use PCA for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 1. Clusters in PCA space
    ax1 = plt.subplot(2, 3, 1)
    unique_clusters = set(clusters)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster_id in enumerate(sorted(unique_clusters)):
        mask = clusters == cluster_id
        if cluster_id == -1:
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c='red', s=20, alpha=0.8, label='Outliers', marker='x')
        else:
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[i]], s=10, alpha=0.6, label=f'Cluster {cluster_id}')
    
    ax1.set_title('DBSCAN Clustering Results', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.legend()
    
    # 2. Cluster sizes
    ax2 = plt.subplot(2, 3, 2)
    cluster_counts = Counter(clusters)
    cluster_ids = sorted([c for c in cluster_counts.keys() if c != -1])
    counts = [cluster_counts[c] for c in cluster_ids]
    
    ax2.bar(range(len(cluster_ids)), counts, color='steelblue', edgecolor='black')
    if -1 in cluster_counts:
        ax2.bar([len(cluster_ids)], [cluster_counts[-1]], 
               color='red', edgecolor='black', label='Outliers')
    ax2.set_xticks(range(len(cluster_ids) + (1 if -1 in cluster_counts else 0)))
    ax2.set_xticklabels(cluster_ids + ([-1] if -1 in cluster_counts else []))
    ax2.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Cluster ID (-1 = Outliers)')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Outlier analysis
    if 'IsAnomaly' in detector.df.columns:
        ax3 = plt.subplot(2, 3, 3)
        outlier_mask = clusters == -1
        cm = [[0, 0], [0, 0]]
        
        # True negatives
        cm[0][0] = ((detector.df['IsAnomaly'] == 0) & (~outlier_mask)).sum()
        # False positives
        cm[0][1] = ((detector.df['IsAnomaly'] == 0) & (outlier_mask)).sum()
        # False negatives
        cm[1][0] = ((detector.df['IsAnomaly'] == 1) & (~outlier_mask)).sum()
        # True positives
        cm[1][1] = ((detector.df['IsAnomaly'] == 1) & (outlier_mask)).sum()
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax3,
                   xticklabels=['Not Outlier', 'Outlier'],
                   yticklabels=['Normal', 'Anomaly'])
        ax3.set_title('Outlier Detection Performance', fontsize=14, fontweight='bold')
        ax3.set_ylabel('True Label')
        ax3.set_xlabel('DBSCAN Prediction')
    
    # 4. Distance to nearest cluster
    ax4 = plt.subplot(2, 3, 4)
    outlier_mask = clusters == -1
    if outlier_mask.sum() > 0:
        outlier_points = X_pca[outlier_mask]
        cluster_points = X_pca[~outlier_mask]
        
        # Calculate distances
        from scipy.spatial.distance import cdist
        if len(cluster_points) > 0:
            distances = cdist(outlier_points, cluster_points).min(axis=1)
            ax4.hist(distances, bins=30, edgecolor='black', alpha=0.7, color='red')
            ax4.set_title('Outlier Isolation', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Distance to Nearest Cluster')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
    
    # 5. SDN Applications
    ax5 = plt.subplot(2, 3, 5)
    sdn_text = """
    DENSITY-BASED SDN USES
    
    ✓ Rare Event Detection
      • Port scanning
      • Network probing
      • DDoS precursors
    
    ✓ Flexible Boundaries
      • No fixed anomaly threshold
      • Adapts to data shape
    
    ✓ Noise Filtering
      • Separate true anomalies
      • From random fluctuations
    """
    ax5.text(0.5, 0.5, sdn_text, ha='center', va='center', 
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    ax5.axis('off')
    ax5.set_title('SDN Benefits', fontsize=14, fontweight='bold')
    
    # 6. Summary
    ax6 = plt.subplot(2, 3, 6)
    n_clusters = len([c for c in unique_clusters if c != -1])
    n_outliers = (clusters == -1).sum()
    summary = f"""
    DBSCAN SUMMARY
    
    Total Samples: {len(clusters):,}
    Clusters Found: {n_clusters}
    Outliers: {n_outliers:,} ({n_outliers/len(clusters)*100:.1f}%)
    
    Outliers are potential:
    • Attack traffic
    • Misconfigurations
    • Rare normal events
    """
    ax6.text(0.5, 0.5, summary, ha='center', va='center', 
            fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax6.axis('off')
    ax6.set_title('Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = detector._get_output_path('density_clustering.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Density clustering saved to '{output_path}'")
    plt.close()


def _plot_flow_analysis(detector, flow_stats, feature_cols):
    """Plot network flow analysis"""
    fig = plt.figure(figsize=(16, 10))
    
    # Create multiple subplots based on available metrics
    n_plots = 0
    plots_config = []
    
    if 'BytesSent' in feature_cols and 'PacketsSent' in feature_cols:
        n_plots += 1
        plots_config.append(('packet_size', 'Packet Size Distribution'))
    
    if 'Duration' in feature_cols:
        n_plots += 1
        plots_config.append(('duration', 'Flow Duration Distribution'))
    
    if 'BytesSent' in feature_cols and 'BytesReceived' in feature_cols:
        n_plots += 1
        plots_config.append(('symmetry', 'Traffic Symmetry'))
    
    # Add SDN application plot
    n_plots += 1
    
    rows = (n_plots + 1) // 2
    cols = 2
    
    plot_idx = 1
    
    # Plot each configured analysis
    for plot_type, title in plots_config:
        ax = plt.subplot(rows, cols, plot_idx)
        plot_idx += 1
        
        if plot_type == 'packet_size':
            packet_sizes = detector.df['BytesSent'] / (detector.df['PacketsSent'] + 1e-6)
            ax.hist(packet_sizes, bins=50, edgecolor='black', alpha=0.7, color='blue')
            ax.axvline(flow_stats['avg_packet_size'], color='red', 
                      linestyle='--', linewidth=2, label=f'Mean: {flow_stats["avg_packet_size"]:.2f}')
            ax.set_xlabel('Bytes per Packet')
            ax.set_ylabel('Frequency')
            ax.legend()
        
        elif plot_type == 'duration':
            ax.hist(detector.df['Duration'], bins=50, edgecolor='black', alpha=0.7, color='green')
            ax.axvline(flow_stats['avg_duration'], color='red', 
                      linestyle='--', linewidth=2, label=f'Mean: {flow_stats["avg_duration"]:.2f}')
            ax.set_xlabel('Duration')
            ax.set_ylabel('Frequency')
            ax.legend()
        
        elif plot_type == 'symmetry':
            symmetry = detector.df['BytesSent'] / (detector.df['BytesReceived'] + 1e-6)
            ax.hist(np.log10(symmetry + 1), bins=50, edgecolor='black', alpha=0.7, color='orange')
            ax.set_xlabel('Log10(Sent/Received Ratio)')
            ax.set_ylabel('Frequency')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # SDN Applications
    ax_sdn = plt.subplot(rows, cols, plot_idx)
    sdn_text = """
    FLOW ANALYSIS SDN USES
    
    ✓ Flow Table Optimization
      • Predict flow duration
      • Efficient timeout values
    
    ✓ QoS Classification
      • Packet size profiling
      • Priority assignment
    
    ✓ Attack Detection
      • Unusual flow patterns
      • Asymmetric traffic
    
    ✓ Capacity Planning
      • Bandwidth requirements
      • Resource allocation
    """
    ax_sdn.text(0.5, 0.5, sdn_text, ha='center', va='center', 
               fontsize=11, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax_sdn.axis('off')
    ax_sdn.set_title('SDN Applications', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = detector._get_output_path('flow_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Flow analysis saved to '{output_path}'")
    plt.close()


def _plot_severity_analysis(detector, severity_scores, severity_levels):
    """Plot anomaly severity scoring results"""
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Severity score distribution
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(severity_scores, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax1.axvline(0.4, color='yellow', linestyle='--', label='Medium')
    ax1.axvline(0.6, color='orange', linestyle='--', label='High')
    ax1.axvline(0.8, color='red', linestyle='--', label='Critical')
    ax1.set_title('Severity Score Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Severity Score')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Severity levels pie chart
    ax2 = plt.subplot(2, 3, 2)
    severity_counts = Counter(severity_levels)
    labels = ['Low', 'Medium', 'High', 'Critical']
    colors_sev = ['green', 'yellow', 'orange', 'red']
    sizes = [severity_counts.get(i, 0) for i in range(4)]
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_sev,
                                        autopct='%1.1f%%', startangle=90)
    ax2.set_title('Severity Level Distribution', fontsize=14, fontweight='bold')
    
    # 3. Severity over time
    ax3 = plt.subplot(2, 3, 3)
    colors_time = ['green' if s == 0 else 'yellow' if s == 1 
                   else 'orange' if s == 2 else 'red' for s in severity_levels]
    ax3.scatter(range(len(severity_scores)), severity_scores, 
               c=colors_time, alpha=0.6, s=10)
    ax3.set_title('Severity Scores Over Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Severity Score')
    ax3.grid(True, alpha=0.3)
    
    # 4. Top threats table
    ax4 = plt.subplot(2, 3, 4)
    top_threats = np.argsort(severity_scores)[-10:][::-1]
    
    table_data = []
    for rank, idx in enumerate(top_threats, 1):
        level = ['Low', 'Medium', 'High', 'Critical'][severity_levels[idx]]
        table_data.append([f'#{rank}', idx, f'{severity_scores[idx]:.4f}', level])
    
    table = ax4.table(cellText=table_data, 
                     colLabels=['Rank', 'Index', 'Score', 'Level'],
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.3, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax4.axis('off')
    ax4.set_title('Top 10 Threats', fontsize=14, fontweight='bold')
    
    # 5. SDN Response Actions
    ax5 = plt.subplot(2, 3, 5)
    sdn_text = """
    SEVERITY-BASED SDN ACTIONS
    
    🔴 CRITICAL (≥0.8)
      • Immediate isolation
      • Alert security team
      • Block traffic
    
    🟠 HIGH (0.6-0.8)
      • Rate limiting
      • Enhanced monitoring
      • Threat investigation
    
    🟡 MEDIUM (0.4-0.6)
      • Logging and tracking
      • Pattern analysis
      • Watchlist addition
    
    🟢 LOW (<0.4)
      • Normal processing
      • Routine logging
    """
    ax5.text(0.5, 0.5, sdn_text, ha='center', va='center', 
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
    ax5.axis('off')
    ax5.set_title('Automated Response Matrix', fontsize=14, fontweight='bold')
    
    # 6. Summary
    ax6 = plt.subplot(2, 3, 6)
    summary = f"""
    SEVERITY SCORING SUMMARY
    
    Total Samples: {len(severity_scores):,}
    
    Critical: {severity_counts.get(3, 0):,}
    High: {severity_counts.get(2, 0):,}
    Medium: {severity_counts.get(1, 0):,}
    Low: {severity_counts.get(0, 0):,}
    
    Highest Score: {severity_scores.max():.4f}
    Average Score: {severity_scores.mean():.4f}
    """
    ax6.text(0.5, 0.5, summary, ha='center', va='center', 
            fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax6.axis('off')
    ax6.set_title('Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = detector._get_output_path('severity_scoring.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Severity analysis saved to '{output_path}'")
    plt.close()


def run_behavior_analysis(detector):
    """
    Run complete behavioral analysis suite
    
    Args:
        detector: NetworkAnomalyDetector instance
    """
    print("\n" + "=" * 80)
    print("BEHAVIORAL ANALYSIS SUITE")
    print("=" * 80)
    
    # 1. Behavioral profiling
    profiles = behavioral_profiling(detector, n_profiles=5)
    
    # 2. Density-based clustering
    density_results = density_based_clustering(detector, eps=0.5, min_samples=5)
    
    # 3. Network flow analysis
    flow_stats = network_flow_analysis(detector)
    
    # 4. Severity scoring
    severity_results = anomaly_severity_scoring(detector)
    
    print("\n" + "=" * 80)
    print("BEHAVIORAL ANALYSIS COMPLETE")
    print("=" * 80)
    
    return {
        'profiles': profiles,
        'density_clustering': density_results,
        'flow_analysis': flow_stats,
        'severity_scoring': severity_results
    }
