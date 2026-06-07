"""
Advanced Network Behavior Analysis
Graph-based and behavioural analytics for SDN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)
from collections import Counter
import config
import warnings
warnings.filterwarnings('ignore')


def _get_scaled_X(detector):
    """Return the full dataset scaled with the pipeline scaler.

    After prepare_data() the df columns have been log-transformed but not
    scaled.  Applying the already-fitted scaler gives results consistent
    with detector.X_train / detector.X_test.
    """
    feature_cols = detector._get_feature_columns()
    return detector.scaler.transform(detector.df[feature_cols].values), feature_cols


# ---------------------------------------------------------------------------
# Behavioural profiling
# ---------------------------------------------------------------------------

def behavioral_profiling(detector, n_profiles=5):
    print("\n" + "=" * 80)
    print("BEHAVIOURAL PROFILING — Traffic Pattern Classification")
    print("=" * 80)

    X_scaled, feature_cols = _get_scaled_X(detector)

    print(f"  Creating {n_profiles} behavioural profiles using K-Means…")
    kmeans   = KMeans(n_clusters=n_profiles, random_state=config.RANDOM_STATE, n_init=10)
    profiles = kmeans.fit_predict(X_scaled)

    profile_summary = []
    for i in range(n_profiles):
        mask  = profiles == i
        count = int(mask.sum())
        pct   = count / len(profiles) * 100
        means = detector.df[mask][feature_cols].mean()
        profile_summary.append({
            'profile_id': i,
            'count': count,
            'percentage': pct,
            'characteristics': means.to_dict(),
        })
        print(f"\n  Profile {i}: {count:,} samples ({pct:.1f}%)")
        top = means.abs().sort_values(ascending=False).head(3)
        for feat, val in top.items():
            print(f"    {feat}: {val:.3f}")

    if 'IsAnomaly' in detector.df.columns:
        print("\n  Profile — Anomaly Rates:")
        for i in range(n_profiles):
            mask = profiles == i
            rate = detector.df[mask]['IsAnomaly'].mean()
            print(f"    Profile {i}: {rate*100:.1f}% anomalies")

    _plot_behavioral_profiles(detector, profiles, profile_summary, n_profiles, X_scaled)

    return {'profiles': profiles, 'profile_summary': profile_summary, 'n_profiles': n_profiles}


# ---------------------------------------------------------------------------
# Density-based clustering (DBSCAN)
# ---------------------------------------------------------------------------

def _estimate_eps(X_scaled, min_samples):
    """
    Estimate eps using the 90th percentile of distances to the min_samples-th
    nearest neighbour. This is robust for high-dimensional scaled data and
    naturally produces a small number of meaningful clusters.
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_dists = np.sort(distances[:, -1])
    eps = float(np.percentile(k_dists, 90))
    return eps, k_dists


def density_based_clustering(detector, min_samples=None):
    print("\n" + "=" * 80)
    print("DENSITY-BASED CLUSTERING — Outlier Detection")
    print("=" * 80)

    X_scaled, _ = _get_scaled_X(detector)

    if min_samples is None:
        min_samples = config.BEHAVIOR_DBSCAN_MIN_SAMPLES

    # Auto-estimate eps from k-distance graph (90th percentile)
    eps, k_dists = _estimate_eps(X_scaled, min_samples)
    print(f"  Auto-estimated eps  : {eps:.4f}  (90th pct k-distance, k={min_samples})")
    print(f"  Running DBSCAN (eps={eps:.4f}, min_samples={min_samples})…")

    dbscan   = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    clusters = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_outliers = int((clusters == -1).sum())
    print(f"  Clusters found : {n_clusters}")
    print(f"  Outliers       : {n_outliers:,} ({n_outliers/len(clusters)*100:.2f}%)")

    for cid, cnt in sorted(Counter(clusters).items()):
        label = "Outliers" if cid == -1 else f"Cluster {cid}"
        print(f"    {label}: {cnt:,}")

    if 'IsAnomaly' in detector.df.columns:
        outlier_labels = (clusters == -1).astype(int)
        print(f"\n  Performance vs true anomalies:")
        print(f"    Accuracy  : {accuracy_score(detector.df['IsAnomaly'], outlier_labels):.4f}")
        print(f"    Precision : {precision_score(detector.df['IsAnomaly'], outlier_labels, zero_division=0):.4f}")
        print(f"    Recall    : {recall_score(detector.df['IsAnomaly'], outlier_labels, zero_division=0):.4f}")
        print(f"    F1-Score  : {f1_score(detector.df['IsAnomaly'], outlier_labels, zero_division=0):.4f}")

    _plot_density_clustering(detector, clusters, X_scaled, k_dists=k_dists, eps=eps)

    return {
        'clusters': clusters,
        'n_clusters': n_clusters,
        'n_outliers': n_outliers,
        'outlier_indices': np.where(clusters == -1)[0],
    }


# ---------------------------------------------------------------------------
# Network flow analysis
# ---------------------------------------------------------------------------

def network_flow_analysis(detector):
    """Analyse flow characteristics using engineered ratio features where
    available, since BytesSent/PacketsSent etc. have been log-transformed
    in detector.df and cannot be used for raw ratio arithmetic."""
    print("\n" + "=" * 80)
    print("NETWORK FLOW ANALYSIS")
    print("=" * 80)

    feature_cols = detector._get_feature_columns()
    flow_stats   = {}

    # Bytes-per-packet: use the pre-computed ratio columns (raw values, not log-transformed)
    if 'BytesPerPacketSent' in feature_cols:
        bpp = detector.df['BytesPerPacketSent']
        flow_stats['avg_bytes_per_pkt_sent'] = bpp.mean()
        flow_stats['std_bytes_per_pkt_sent'] = bpp.std()
        print(f"  Avg BytesPerPacketSent : {flow_stats['avg_bytes_per_pkt_sent']:.4f}")

    if 'BytesPerPacketReceived' in feature_cols:
        bpp_r = detector.df['BytesPerPacketReceived']
        flow_stats['avg_bytes_per_pkt_recv'] = bpp_r.mean()
        print(f"  Avg BytesPerPacketRecv : {flow_stats['avg_bytes_per_pkt_recv']:.4f}")

    # Flow duration (log-transformed; relative comparisons still valid)
    if 'Duration' in feature_cols:
        dur = detector.df['Duration']
        flow_stats['avg_duration']  = dur.mean()
        flow_stats['long_flows']    = int((dur > dur.quantile(0.95)).sum())
        flow_stats['short_flows']   = int((dur < dur.quantile(0.05)).sum())
        print(f"  Avg Duration (log)     : {flow_stats['avg_duration']:.4f}")
        print(f"  Long flows (>95th pct) : {flow_stats['long_flows']:,}")
        print(f"  Short flows (<5th pct) : {flow_stats['short_flows']:,}")

    # Traffic symmetry: use ratio of log-transformed sent/received
    # (log(a) - log(b) = log(a/b), so this is still meaningful as asymmetry)
    if 'BytesSent' in feature_cols and 'BytesReceived' in feature_cols:
        symmetry = detector.df['BytesSent'] - detector.df['BytesReceived']  # log-space difference
        flow_stats['traffic_symmetry']   = symmetry.mean()
        flow_stats['asymmetric_flows']   = int(((symmetry.abs()) > 1).sum())
        print(f"  Traffic log-symmetry   : {flow_stats['traffic_symmetry']:.4f}")
        print(f"  Asymmetric flows       : {flow_stats['asymmetric_flows']:,}")

    if 'Protocol' in feature_cols:
        proto_dist = detector.df['Protocol'].value_counts()
        flow_stats['protocol_distribution'] = proto_dist.to_dict()
        print(f"\n  Protocol distribution (top 5):")
        for proto, cnt in proto_dist.head().items():
            print(f"    Protocol {proto}: {cnt:,} ({cnt/len(detector.df)*100:.1f}%)")

    _plot_flow_analysis(detector, flow_stats, feature_cols)
    return flow_stats


# ---------------------------------------------------------------------------
# Anomaly severity scoring
# ---------------------------------------------------------------------------

def anomaly_severity_scoring(detector):
    print("\n" + "=" * 80)
    print("ANOMALY SEVERITY SCORING")
    print("=" * 80)

    X_all, _ = _get_scaled_X(detector)   # full dataset, consistent with other analyses
    scores    = []

    if detector.unsupervised_model is not None:
        iso_scores = detector.unsupervised_model.score_samples(X_all)
        iso_norm   = 1 / (1 + np.exp(iso_scores))   # sigmoid: higher = more anomalous
        scores.append(iso_norm)

    if detector.supervised_model is not None:
        try:
            rf_probs = detector.supervised_model.predict_proba(X_all)[:, 1]
            scores.append(rf_probs)
        except Exception:
            pass

    if not scores:
        print("  No trained models available for severity scoring")
        return None

    combined = np.mean(scores, axis=0)

    severity = np.zeros(len(combined), dtype=int)
    severity[combined >= 0.8] = 3
    severity[(combined >= 0.6) & (combined < 0.8)] = 2
    severity[(combined >= 0.4) & (combined < 0.6)] = 1

    sev_counts = Counter(severity)
    print(f"  Critical  (>=0.8): {sev_counts.get(3, 0):,}")
    print(f"  High   (0.6-0.8) : {sev_counts.get(2, 0):,}")
    print(f"  Medium (0.4-0.6) : {sev_counts.get(1, 0):,}")
    print(f"  Low      (<0.4)  : {sev_counts.get(0, 0):,}")

    top_threats = np.argsort(combined)[-10:][::-1]
    print("\n  Top 10 Most Severe Anomalies:")
    for rank, idx in enumerate(top_threats, 1):
        print(f"    #{rank}: sample {idx}, score {combined[idx]:.4f}")

    _plot_severity_analysis(detector, combined, severity)

    return {
        'severity_scores': combined,
        'severity_levels': severity,
        'top_threats': top_threats,
        'distribution': sev_counts,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_behavioral_profiles(detector, profiles, profile_summary, n_profiles, X_scaled):
    plt.figure(figsize=(20, 12))

    profile_counts = Counter(profiles)
    colors = plt.cm.Set3(np.linspace(0, 1, n_profiles))

    # 1. Profile size bar chart
    ax1 = plt.subplot(2, 3, 1)
    ax1.bar(range(n_profiles), [profile_counts[i] for i in range(n_profiles)],
            color=colors, edgecolor='black')
    ax1.set_title('Behavioural Profile Distribution', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Profile ID')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. PCA (uses pre-scaled X, same as clustering)
    ax2 = plt.subplot(2, 3, 2)
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    for i in range(n_profiles):
        mask = profiles == i
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=f'Profile {i}', alpha=0.5, s=8, color=colors[i])
    ax2.set_title('Profiles in PCA Space', fontsize=13, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.legend(markerscale=2)

    # 3. Profile characteristics heatmap
    ax3 = plt.subplot(2, 3, 3)
    feat_keys = list(profile_summary[0]['characteristics'].keys())[:8]
    char_matrix = np.array([
        [s['characteristics'][k] for k in feat_keys]
        for s in profile_summary
    ])
    sns.heatmap(char_matrix, annot=False, cmap='coolwarm',
                xticklabels=feat_keys,
                yticklabels=[f'P{i}' for i in range(n_profiles)],
                ax=ax3, cbar_kws={'label': 'Value (log-transformed)'})
    ax3.set_title('Profile Characteristics', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)

    # 4. Anomaly rate by profile
    if 'IsAnomaly' in detector.df.columns:
        ax4 = plt.subplot(2, 3, 4)
        anomaly_rates = [
            detector.df[profiles == i]['IsAnomaly'].mean()
            for i in range(n_profiles)
        ]
        ax4.bar(range(n_profiles), anomaly_rates, color=colors, edgecolor='black')
        ax4.set_title('Anomaly Rate by Profile', fontsize=13, fontweight='bold')
        ax4.set_xlabel('Profile ID')
        ax4.set_ylabel('Anomaly Rate')
        ax4.grid(True, alpha=0.3, axis='y')
        for i, rate in enumerate(anomaly_rates):
            ax4.text(i, rate + 0.005, f'{rate*100:.1f}%', ha='center', fontweight='bold')

    # 5–6. Summary panels
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    ax5.set_title('SDN Applications', fontsize=13, fontweight='bold')
    ax5.text(0.5, 0.5,
             "Behavioural Profiling SDN Uses\n\n"
             "Intent-Based Networking\n"
             "  Auto-classify traffic types\n\n"
             "Baseline Learning\n"
             "  Normal behaviour per profile\n\n"
             "Traffic Engineering\n"
             "  Route optimisation per profile",
             ha='center', va='center', fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    ax6.set_title('Summary', fontsize=13, fontweight='bold')
    ax6.text(0.5, 0.5,
             f"Total Samples : {len(profiles):,}\n"
             f"Profiles      : {n_profiles}\n\n"
             f"Largest       : {max(profile_counts.values()):,}\n"
             f"Smallest      : {min(profile_counts.values()):,}",
             ha='center', va='center', fontsize=13, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.suptitle('Behavioural Profiling Results', fontsize=15, fontweight='bold')
    plt.tight_layout()
    output_path = detector._get_output_path('behavioral_profiling.png')
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"\n  Behavioural profiling saved to '{output_path}'")
    plt.close()


def _plot_density_clustering(detector, clusters, X_scaled, k_dists=None, eps=None):
    plt.figure(figsize=(20, 10))

    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    unique_clusters = sorted(set(clusters))
    palette = plt.cm.Spectral(np.linspace(0, 1, len(unique_clusters)))

    # 1. PCA coloured by cluster
    ax1 = plt.subplot(2, 3, 1)
    for i, cid in enumerate(unique_clusters):
        mask = clusters == cid
        if cid == -1:
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c='red', s=20, alpha=0.8, label='Outliers', marker='x')
        else:
            # Only add a legend entry for the first few clusters to avoid overflow
            label = f'Cluster {cid}' if cid < 10 else None
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=[palette[i]], s=8, alpha=0.5, label=label)
    ax1.set_title('DBSCAN Clustering Results', fontsize=13, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.legend(markerscale=2, fontsize=8, ncol=2)

    # 2. Cluster size bar chart — show top-10 clusters + outliers only
    ax2 = plt.subplot(2, 3, 2)
    cluster_counts = Counter(clusters)
    all_ids  = sorted([c for c in cluster_counts if c != -1],
                      key=lambda c: cluster_counts[c], reverse=True)
    top_ids  = all_ids[:10]          # at most 10 bars to keep the chart readable
    top_cnts = [cluster_counts[c] for c in top_ids]
    bar_colors = plt.cm.tab10(np.linspace(0, 1, max(len(top_ids), 1)))
    ax2.bar(range(len(top_ids)), top_cnts, color=bar_colors, edgecolor='black')
    if -1 in cluster_counts:
        ax2.bar([len(top_ids)], [cluster_counts[-1]], color='red',
                edgecolor='black', label='Outliers')
        ax2.legend(fontsize=9)
    tick_labels = [f'C{c}' for c in top_ids] + (['Out'] if -1 in cluster_counts else [])
    ax2.set_xticks(range(len(top_ids) + (1 if -1 in cluster_counts else 0)))
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    n_total = len(all_ids)
    title_suffix = f' (top 10 of {n_total})' if n_total > 10 else ''
    ax2.set_title(f'Cluster Size Distribution{title_suffix}', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Confusion matrix vs true labels
    if 'IsAnomaly' in detector.df.columns:
        ax3 = plt.subplot(2, 3, 3)
        outlier_labels = (clusters == -1).astype(int)
        cm = confusion_matrix(detector.df['IsAnomaly'], outlier_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax3,
                    xticklabels=['Not Outlier', 'Outlier'],
                    yticklabels=['Normal', 'Anomaly'])
        ax3.set_title('Outlier Detection Performance', fontsize=13, fontweight='bold')
        ax3.set_ylabel('True Label')
        ax3.set_xlabel('DBSCAN Prediction')

    # 4. k-distance graph — shows how eps was selected
    ax4 = plt.subplot(2, 3, 4)
    if k_dists is not None:
        ax4.plot(k_dists, linewidth=1.2, color='steelblue')
        if eps is not None:
            ax4.axhline(eps, color='red', linestyle='--', linewidth=1.5,
                        label=f'Selected eps = {eps:.3f}')
            ax4.legend(fontsize=9)
        ax4.set_title('k-Distance Graph (eps Selection)', fontsize=13, fontweight='bold')
        ax4.set_xlabel('Points sorted by distance')
        ax4.set_ylabel(f'Distance to {config.BEHAVIOR_DBSCAN_MIN_SAMPLES}-th neighbour')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.axis('off')

    # 5–6. Text panels
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    ax5.set_title('SDN Benefits', fontsize=13, fontweight='bold')
    ax5.text(0.5, 0.5,
             "Density-Based SDN Uses\n\n"
             "Rare Event Detection\n"
             "  Port scanning, probing\n\n"
             "Flexible Boundaries\n"
             "  No fixed anomaly threshold\n\n"
             "Noise Filtering\n"
             "  Separate true anomalies",
             ha='center', va='center', fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    n_cl  = len([c for c in unique_clusters if c != -1])
    n_out = int((clusters == -1).sum())
    ax6.text(0.5, 0.5,
             f"Total Samples : {len(clusters):,}\n"
             f"Clusters      : {n_cl}\n"
             f"Outliers      : {n_out:,} ({n_out/len(clusters)*100:.1f}%)",
             ha='center', va='center', fontsize=13, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax6.set_title('Summary', fontsize=13, fontweight='bold')

    plt.suptitle('DBSCAN Density-Based Clustering', fontsize=15, fontweight='bold')
    plt.tight_layout()
    output_path = detector._get_output_path('density_clustering.png')
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"\n  Density clustering saved to '{output_path}'")
    plt.close()


def _plot_flow_analysis(detector, flow_stats, feature_cols):
    plt.figure(figsize=(16, 10))
    plot_idx = 1
    plots = []

    if 'BytesPerPacketSent' in feature_cols:
        plots.append(('bpp_sent', 'Bytes-Per-Packet Sent Distribution'))
    if 'Duration' in feature_cols:
        plots.append(('duration', 'Flow Duration Distribution (log-scale)'))
    if 'BytesSent' in feature_cols and 'BytesReceived' in feature_cols:
        plots.append(('symmetry', 'Log-Space Traffic Symmetry'))
    plots.append(('sdn', 'SDN Applications'))

    rows = (len(plots) + 1) // 2

    for ptype, title in plots:
        ax = plt.subplot(rows, 2, plot_idx)
        plot_idx += 1

        if ptype == 'bpp_sent':
            ax.hist(detector.df['BytesPerPacketSent'], bins=50,
                    edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(flow_stats['avg_bytes_per_pkt_sent'], color='red',
                       linestyle='--', label=f"Mean: {flow_stats['avg_bytes_per_pkt_sent']:.2f}")
            ax.set_xlabel('Bytes per Packet (Sent)')
            ax.set_ylabel('Frequency')
            ax.legend()

        elif ptype == 'duration':
            ax.hist(detector.df['Duration'], bins=50,
                    edgecolor='black', alpha=0.7, color='green')
            ax.axvline(flow_stats['avg_duration'], color='red', linestyle='--',
                       label=f"Mean: {flow_stats['avg_duration']:.2f}")
            ax.set_xlabel('Duration (log1p-transformed)')
            ax.set_ylabel('Frequency')
            ax.legend()

        elif ptype == 'symmetry':
            diff = detector.df['BytesSent'] - detector.df['BytesReceived']
            ax.hist(diff, bins=50, edgecolor='black', alpha=0.7, color='orange')
            ax.set_xlabel('log(BytesSent) - log(BytesReceived)')
            ax.set_ylabel('Frequency')

        elif ptype == 'sdn':
            ax.text(0.5, 0.5,
                    "Flow Analysis SDN Uses\n\n"
                    "Flow Table Optimisation\n"
                    "  Predict flow duration\n\n"
                    "QoS Classification\n"
                    "  Packet-size profiling\n\n"
                    "Attack Detection\n"
                    "  Asymmetric traffic patterns",
                    ha='center', va='center', fontsize=11, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            ax.axis('off')

        ax.set_title(title, fontsize=12, fontweight='bold')
        if ptype != 'sdn':
            ax.grid(True, alpha=0.3)

    plt.suptitle('Network Flow Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    output_path = detector._get_output_path('flow_analysis.png')
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"\n  Flow analysis saved to '{output_path}'")
    plt.close()


def _plot_severity_analysis(detector, severity_scores, severity_levels):
    plt.figure(figsize=(20, 10))

    sev_counts = Counter(severity_levels)

    # 1. Score distribution
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(severity_scores, bins=60, edgecolor='black', alpha=0.7, color='purple')
    for thr, lbl, col in [(0.4, 'Medium', 'gold'), (0.6, 'High', 'orange'), (0.8, 'Critical', 'red')]:
        ax1.axvline(thr, color=col, linestyle='--', label=lbl)
    ax1.set_title('Severity Score Distribution', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Severity Score')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Pie chart
    ax2 = plt.subplot(2, 3, 2)
    sizes  = [sev_counts.get(i, 0) for i in range(4)]
    labels = ['Low', 'Medium', 'High', 'Critical']
    colors = ['green', 'gold', 'orange', 'red']
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Severity Level Distribution', fontsize=13, fontweight='bold')

    # 3. Scores over samples
    ax3 = plt.subplot(2, 3, 3)
    pt_colors = ['green' if s == 0 else 'gold' if s == 1
                 else 'orange' if s == 2 else 'red'
                 for s in severity_levels]
    ax3.scatter(range(len(severity_scores)), severity_scores,
                c=pt_colors, alpha=0.5, s=5)
    ax3.set_title('Severity Scores Over Samples', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Severity Score')
    ax3.grid(True, alpha=0.3)

    # 4. Response matrix
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    ax4.set_title('Automated Response Matrix', fontsize=13, fontweight='bold')
    ax4.text(0.5, 0.5,
             "CRITICAL (>=0.8)\n"
             "  Immediate isolation, block traffic\n\n"
             "HIGH (0.6-0.8)\n"
             "  Rate limiting, enhanced monitoring\n\n"
             "MEDIUM (0.4-0.6)\n"
             "  Logging, pattern analysis\n\n"
             "LOW (<0.4)\n"
             "  Normal processing",
             ha='center', va='center', fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))

    # 5. Summary
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    ax5.set_title('Summary', fontsize=13, fontweight='bold')
    ax5.text(0.5, 0.5,
             f"Total Samples : {len(severity_scores):,}\n\n"
             f"Critical : {sev_counts.get(3,0):,}\n"
             f"High     : {sev_counts.get(2,0):,}\n"
             f"Medium   : {sev_counts.get(1,0):,}\n"
             f"Low      : {sev_counts.get(0,0):,}\n\n"
             f"Max Score : {severity_scores.max():.4f}\n"
             f"Avg Score : {severity_scores.mean():.4f}",
             ha='center', va='center', fontsize=13, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.suptitle('Anomaly Severity Scoring', fontsize=15, fontweight='bold')
    plt.tight_layout()
    output_path = detector._get_output_path('severity_scoring.png')
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"\n  Severity analysis saved to '{output_path}'")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_behavior_analysis(detector):
    print("\n" + "=" * 80)
    print("BEHAVIOURAL ANALYSIS SUITE")
    print("=" * 80)

    profiles         = behavioral_profiling(detector, n_profiles=config.BEHAVIOR_N_PROFILES)
    density_results  = density_based_clustering(detector)
    flow_stats       = network_flow_analysis(detector)
    severity_results = anomaly_severity_scoring(detector)

    print("\n" + "=" * 80)
    print("BEHAVIOURAL ANALYSIS COMPLETE")
    print("=" * 80)

    return {
        'profiles': profiles,
        'density_clustering': density_results,
        'flow_analysis': flow_stats,
        'severity_scoring': severity_results,
    }
