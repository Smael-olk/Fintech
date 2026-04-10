import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_target_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, title in zip(
        axes,
        ['IncomeInvestment', 'AccumulationInvestment'],
        ['Income Investment', 'Accumulation Investment']
    ):
        counts = df[col].value_counts()
        bars = ax.bar(counts.index.astype(str), counts.values,
                      color=['#4C72B0', '#DD8452'], edgecolor='none', width=0.4)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f'{val:,}', ha='center', va='bottom', fontsize=10)
        pct = counts / counts.sum() * 100
        ax.set_title(f'{title}\n(0: {pct[0]:.1f}%  |  1: {pct[1]:.1f}%)', pad=8)
        ax.set_xlabel('Class (1 = Yes, 0 = No)')
        ax.set_ylabel('Count')

    fig.suptitle('Target Variables — Class Distribution', fontsize=13, y=1.03)
    plt.tight_layout()
    plt.show()


def plot_skew_comparison(transformed_df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    pairs = [
        (axes[0, 0], transformed_df['Wealth'],     'Wealth (Original)', 'Wealth'),
        (axes[0, 1], transformed_df['Wealth_log'], 'Wealth (log1p)',    'log(Wealth + 1)'),
        (axes[1, 0], transformed_df['Income'],     'Income (Original)', 'Income'),
        (axes[1, 1], transformed_df['Income_log'], 'Income (log1p)',    'log(Income + 1)'),
    ]
    for ax, data, title, xlabel in pairs:
        ax.hist(data, bins=40, color='#4C72B0', edgecolor='none', alpha=0.85)
        ax.set_title(f'{title}  (skew={data.skew():.2f})', pad=6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')

    fig.suptitle('Wealth & Income — Before / After log1p', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(feature_df, numerical_features):
    n, ncols = len(numerical_features), 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.2))
    axes = axes.flatten()

    for i, col in enumerate(numerical_features):
        axes[i].hist(feature_df[col], bins=35, color='#6890C8', edgecolor='none', alpha=0.85)
        axes[i].set_title(f'{col}  (skew={feature_df[col].skew():.2f})', pad=5)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Feature Distributions (Post-Engineering)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(feature_df, numerical_features):
    corr = feature_df[numerical_features].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                linewidths=0.4, linecolor='#111', ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Matrix — Numerical Features', pad=12)
    plt.tight_layout()
    plt.show()

    high_corr = (
        corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
        .stack().reset_index()
    )
    high_corr.columns = ['Feature A', 'Feature B', 'Correlation']
    high_corr = high_corr[high_corr['Correlation'].abs() > 0.7].sort_values(
        'Correlation', ascending=False)
    print('High-correlation pairs (|r| > 0.7):')
    print(high_corr.to_string(index=False))


def plot_feature_vs_target(plot_df, numerical_features):
    fig, axes = plt.subplots(len(numerical_features), 2,
                             figsize=(12, len(numerical_features) * 3))

    for i, col in enumerate(numerical_features):
        for j, target in enumerate(['IncomeInvestment', 'AccumulationInvestment']):
            ax = axes[i, j]
            no  = plot_df[plot_df[target] == 0][col]
            yes = plot_df[plot_df[target] == 1][col]
            ax.boxplot([no, yes], vert=False, tick_labels=['No', 'Yes'])
            ax.set_title(f'{col} vs {target}', pad=5)
            ax.set_xlabel(col)

    fig.suptitle('Feature Distributions by Target Class', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_scree(explained_var, cumulative_var):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n_show = len(explained_var)

    ax1.bar(range(1, n_show+1), explained_var, color='#4C72B0', edgecolor='none', alpha=0.85)
    ax1.plot(range(1, n_show+1), explained_var, 'o-', color='#DD8452', linewidth=1.5, markersize=5)
    ax1.axhline(100/n_show, color='#55BF3B', linestyle='--', linewidth=1, label='Kaiser threshold')
    ax1.set(xlabel='Component', ylabel='Explained Variance (%)', title='Scree Plot')
    ax1.legend(fontsize=9)

    ax2.plot(range(1, n_show+1), cumulative_var, 's-', color='#4C72B0', linewidth=2, markersize=6)
    for thr in [70, 80, 90]:
        k_thr = int(np.searchsorted(cumulative_var, thr)) + 1
        ax2.axhline(thr, color='#DD8452', linestyle=':', linewidth=1)
        ax2.annotate(f'{thr}% @ k={k_thr}', xy=(k_thr, thr), xytext=(k_thr+0.2, thr-4),
                     fontsize=8, color='#DD8452')
    ax2.set(xlabel='Components', ylabel='Cumulative Variance (%)',
            title='Cumulative Explained Variance', ylim=(0, 100))

    fig.suptitle('FAMD — Explained Variance', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_factor_distributions(factors_df, factor_cols):
    n = len(factor_cols)
    fig, axes = plt.subplots(1, n, figsize=(max(12, n * 3), 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, factor_cols):
        ax.hist(factors_df[col], bins=40, color='#6890C8', edgecolor='none', alpha=0.85)
        ax.set_title(f'{col}\nskew={factors_df[col].skew():.2f}', pad=5)
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')

    fig.suptitle('FAMD Factor Score Distributions', fontsize=13, y=1.04)
    plt.tight_layout()
    plt.show()


def plot_gmm_selection(k_list, bic_scores, aic_scores, sil_scores):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, scores, label, color in zip(
        axes,
        [bic_scores, aic_scores, sil_scores],
        ['BIC — lower is better', 'AIC — lower is better', 'Silhouette — higher is better'],
        ['#4C72B0', '#DD8452', '#55BF3B']
    ):
        ax.plot(k_list, scores, 'o-', color=color, linewidth=2, markersize=6)
        ax.set_xlabel('k')
        ax.set_title(label)
        ax.set_xticks(k_list)

    fig.suptitle('GMM Model Selection', fontsize=13, y=1.03)
    plt.tight_layout()
    plt.show()
