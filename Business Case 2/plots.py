import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


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
    """
    Plots the distributions for original, log1p, and power-transformed
    Wealth and Income, along with their Q-Q plots.
    """
    for feature in ['Wealth', 'Income']:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        orig = transformed_df[feature]
        log = transformed_df[f'{feature}_log']
        pow_val = transformed_df[f'{feature}_power']
            
        data_configs = [
            (orig, f"Original {feature}"),
            (log, f"Log-transformed {feature}"),
            (pow_val, f"Power-transformed {feature} (0.1)")
        ]
        
        for i, (data, title) in enumerate(data_configs):
            data_clean = data.dropna()
            
            # Histograms
            ax_hist = axes[0, i]
            sns.histplot(data_clean, bins=50, kde=True, color='mediumpurple', ax=ax_hist, edgecolor='white')
            
            skew = data_clean.skew()
            kurt = data_clean.kurtosis()
            ax_hist.set_title(f"{title}\n(Skew: {skew:.2f}, Kurt: {kurt:.2f})")
            ax_hist.set_xlabel("Value")
            ax_hist.set_ylabel("Count")
            
            # Q-Q Plots
            ax_qq = axes[1, i]
            stats.probplot(data_clean, dist="norm", plot=ax_qq)
            ax_qq.set_title(f"Q-Q Plot: {title}")
            
            lines = ax_qq.get_lines()
            if len(lines) >= 2:
                lines[0].set_marker('o')
                lines[0].set_markerfacecolor('#4C72B0')
                lines[0].set_markeredgecolor('#4C72B0')
                lines[0].set_markersize(5)
                lines[0].set_linestyle('None')
                lines[1].set_color('#D62728')
                
            ax_qq.set_xlabel("Theoretical Quantiles")
            ax_qq.set_ylabel("Ordered Values")
            
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


def plot_correlation_matrix2(corr_matrix, features):
    subset_corr = corr_matrix.loc[features, features]
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(subset_corr, dtype=bool))
    show_annot = len(features) <= 15
    
    sns.heatmap(subset_corr, mask=mask, annot=show_annot, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                linewidths=0.2, linecolor='#111', ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Matrix — Selected Features', fontsize=16, fontweight='bold', pad=12)
    plt.xticks(rotation=75, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()


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


def plot_confusion_matrix(y_true, y_pred, model_name, feature_type, ax=None):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix\n{model_name} — {feature_type}")


def plot_roc_curve(y_true, y_proba, model_name, feature_type, ax=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve\n{model_name} — {feature_type}")
    ax.legend(loc="lower right")


def plot_model_diagnostics(y_true, y_pred, y_proba, model_name, feature_type):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"{model_name} — {feature_type}", fontsize=13, fontweight="bold")
    plot_confusion_matrix(y_true, y_pred, model_name, feature_type, ax=ax1)
    plot_roc_curve(y_true, y_proba, model_name, feature_type, ax=ax2)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, title):
    model = model.trained_model
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importances, x='importance', y='feature')
    plt.title(title)
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.show()

def plot_shap_values(model, X, title):
    import shap
    if hasattr(model, 'trained_model'):
        model = model.trained_model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap.summary_plot(shap_values, X, plot_type='bar', show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f'{title} - Feature Impacts')
    plt.tight_layout()
    plt.show()

def plot_product_distribution(df, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    order = df['Product_Name'].value_counts().index

    sns.countplot(
        data=df,
        y='Product_Name',
        order=order,
        ax=ax
    )

    ax.set_title("Next Best Action: Product Recommendations", fontweight='bold')
    ax.set_xlabel("Number of Clients")
    ax.set_ylabel("")

    # Add labels
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 1, p.get_y() + p.get_height()/2, int(width), va='center')

    return ax

def plot_strategy_breakdown(df, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    action_counts = df['recommendation'].replace({
        "ADVISOR_REVIEW": "Flagged for Advisor",
        "INCOME": "Income Need",
        "ACCUMULATION": "Accumulation Need",
        "NO_ACTION": "No Action"
    }).value_counts()

    ax.pie(
        action_counts.values,
        labels=action_counts.index,
        autopct='%1.1f%%',
        startangle=90
    )

    ax.set_title("High-Level Campaign Breakdown", fontweight='bold')

    return ax

def plot_risk_distribution(df, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    product_only_df = df[
        ~df['Product_Name'].isin(['Flagged for Human Advisor', 'No Immediate Need'])
    ]

    order = (
        product_only_df.groupby('Product_Name')['RiskPropensity']
        .median()
        .sort_values()
        .index
    )

    sns.boxplot(
        data=product_only_df,
        x='RiskPropensity',
        y='Product_Name',
        order=order,
        ax=ax
    )

    ax.set_title("Risk Propensity per Product (Sanity Check)", fontweight='bold')
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("")

    return ax

def plot_full_dashboard(df):
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(3, 1, figsize=(14, 20))
    plt.subplots_adjust(hspace=0.4)

    plot_product_distribution(df, ax=axes[0])
    plot_strategy_breakdown(df, ax=axes[1])
    plot_risk_distribution(df, ax=axes[2])

    plt.show()