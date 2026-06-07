"""
Exploratory Data Analysis Module
Handles all EDA visualizations and analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def perform_eda(detector):
    print("Exploratory Data Analysis (EDA)")
    print("=" * 50)

    # ── Sample 50,000 rows (stratified by IsAnomaly if labels exist) ──────────
    SAMPLE_SIZE = 50_000
    df = detector.df

    if len(df) > SAMPLE_SIZE:
        if 'IsAnomaly' in df.columns:
            # Stratified sample: preserves the normal/anomaly ratio
            df = df.groupby('IsAnomaly', group_keys=False).apply(
                lambda g: g.sample(
                    n=min(len(g), int(SAMPLE_SIZE * len(g) / len(detector.df))),
                    random_state=42
                )
            ).sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
        else:
            df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

        print(f"  Sampled {len(df):,} records from {len(detector.df):,} total")
    else:
        print(f"  Using all {len(df):,} records")

    # ── Create figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    feature_cols = detector._get_feature_columns()

    # 1. Class distribution
    if 'IsAnomaly' in df.columns:
        ax1 = plt.subplot(2, 3, 1)
        df['IsAnomaly'].value_counts().plot(kind='bar', ax=ax1)
        ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('IsAnomaly (0=Normal, 1=Anomaly)')
        ax1.set_ylabel('Count')

    # 2. Correlation heatmap
    ax2 = plt.subplot(2, 3, 2)
    correlation_matrix = df[feature_cols].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', ax=ax2,
                cbar_kws={'shrink': 0.8})
    ax2.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

    # 3. Feature distributions
    ax3 = plt.subplot(2, 3, 3)
    df[feature_cols].boxplot(ax=ax3, vert=False)
    ax3.set_title('Feature Distributions', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Normalized Values')

    # 4. PCA visualization
    if 'IsAnomaly' in df.columns:
        ax4 = plt.subplot(2, 3, 4)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(df[feature_cols])

        scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1],
                              c=df['IsAnomaly'],
                              cmap='coolwarm', alpha=0.6, s=10)
        ax4.set_title('PCA Visualization (2D)', fontsize=14, fontweight='bold')
        ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, ax=ax4, label='IsAnomaly')

    # 5. Feature importance preview
    ax5 = plt.subplot(2, 3, 5)
    feature_std = df[feature_cols].std().sort_values(ascending=False)
    feature_std.head(10).plot(kind='barh', ax=ax5)
    ax5.set_title('Top 10 Features by Std Deviation', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Standard Deviation')

    # 6. Dataset summary
    ax6 = plt.subplot(2, 3, 6)
    summary = (
        f'Sample Size: {len(df):,}\n'
        f'Features: {len(feature_cols)}\n\n'
    )
    if 'IsAnomaly' in df.columns:
        summary += (
            f'Normal:  {(df["IsAnomaly"] == 0).sum():,}\n'
            f'Anomaly: {(df["IsAnomaly"] == 1).sum():,}'
        )
    ax6.text(0.5, 0.5, summary,
             ha='center', va='center', fontsize=16,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.axis('off')
    ax6.set_title('Dataset Summary', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = detector._get_output_path('eda_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n EDA visualization saved to '{output_path}'")
    plt.close()