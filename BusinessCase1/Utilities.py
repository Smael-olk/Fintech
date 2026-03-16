import fastcluster
import fastcluster
import gower
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.core.display_functions import display
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from skopt import gp_minimize
from skopt.space import Real


def compute_distance_block(df, cont_cols, ordinal_cols, cat_cols):
    D_cont = D_ord = D_cat = None

    if cont_cols:
        arr    = df[cont_cols].values.astype(float)
        D_cont = squareform(pdist(arr, metric='cityblock') / len(cont_cols))
        D_cont /= (D_cont.max() + 1e-9)

    if ordinal_cols:
        D_ord = np.mean([ordinal_distance_matrix(df[col])
                         for col in ordinal_cols], axis=0)

    if cat_cols:
        D_cat = hamming_distance_matrix(df[cat_cols])

    return D_cont, D_ord, D_cat


def combine_distance_blocks(D_cont, D_ord, D_cat, alpha, beta, gamma):
    """Combine distance blocks using given weights."""
    D = np.zeros_like(next(d for d in [D_cont, D_ord, D_cat] if d is not None))
    if D_cont is not None: D += alpha * D_cont
    if D_ord is not None: D += beta * D_ord
    if D_cat is not None: D += gamma * D_cat
    return D


def cluster_distance_matrix(dist_mat, n_clusters):
    """Run fastcluster on a precomputed distance matrix."""
    linkage = fastcluster.linkage(squareform(dist_mat), method='average')
    labels = fcluster(linkage, t=n_clusters, criterion='maxclust') - 1
    score = silhouette_score(dist_mat, labels, metric='precomputed')
    return labels, score


def find_best_weights_and_k(df_sample, cont_cols, ordinal_cols, cat_cols, cluster_options):
    """Bayesian search for best alpha/beta and n_clusters on a sample."""
    D_cont, D_ord, D_cat = compute_distance_block(
        df_sample, cont_cols, ordinal_cols, cat_cols
    )

    def objective(params):
        alpha, beta, n_clusters = params[0], params[1], round(params[2])
        gamma = 1.0 - alpha - beta
        if gamma < 0:
            return 1.0
        g_mat = combine_distance_blocks(D_cont, D_ord, D_cat, alpha, beta, gamma)
        labels, score_combined = cluster_distance_matrix(g_mat, n_clusters)
        _, score_cont = cluster_distance_matrix(D_cont, n_clusters) if D_cont is not None else (None, score_combined)
        # Harmonic mean — penalizes solutions where one block dominates
        return -(2 * score_combined * score_cont / (score_combined + score_cont + 1e-9))

    space = [
        Real(0.2, 0.7, name='alpha'),
        Real(0.1, 0.5, name='beta'),
        Real(min(cluster_options), max(cluster_options), name='n_clusters')
    ]
    result = gp_minimize(objective, space, n_calls=30, random_state=42)
    best_alpha = result.x[0]
    best_beta = result.x[1]
    best_gamma = 1.0 - best_alpha - best_beta
    best_n = round(result.x[2])
    return best_alpha, best_beta, best_gamma, best_n


def build_lens_distance(
        df,
        cont_cols,
        ordinal_cols=[],
        cat_cols=[],
        cluster_options=[2, 3, 4],
        sample_size=500
):
    """
    Main function. Works for all lenses:
    - Pure continuous (lens1, lens2): no optimization, alpha=1.0 fixed
    - Mixed (lens3): Bayesian optimization on sample, full matrix on all data

    Returns: best_n, (alpha, beta, gamma), silhouette_score, distance_matrix
    """
    is_mixed = len(ordinal_cols) > 0 or len(cat_cols) > 0

    # Step 1 — find best weights (only needed for mixed lenses)
    if is_mixed:
        df_sample = df.sample(n=sample_size, random_state=42)
        alpha, beta, gamma, best_n = find_best_weights_and_k(
            df_sample, cont_cols, ordinal_cols, cat_cols, cluster_options
        )
    else:
        alpha, beta, gamma = 1.0, 0.0, 0.0
        best_n = None  # determined in step 3

    # Step 2 — build full distance matrix on ALL data
    D_cont, D_ord, D_cat = compute_distance_block(df, cont_cols, ordinal_cols, cat_cols)
    dist_mat = combine_distance_blocks(D_cont, D_ord, D_cat, alpha, beta, gamma)

    # Step 3 — cluster on full matrix
    if best_n is None:  # pure continuous — find best k now
        best_n = max(cluster_options,
                     key=lambda k: cluster_distance_matrix(dist_mat, k)[1])

    labels, score = cluster_distance_matrix(dist_mat, best_n)
    return best_n, (alpha, beta, gamma), score, dist_mat



def manhattan_distance_matrix(df_cont):
    """L1 distance for continuous block."""
    arr = df_cont.values.astype(float)
    return cdist(arr, arr, metric='cityblock') / arr.shape[1]  # normalize by n_features

def ordinal_distance_matrix(series):
    """Normalized ordinal distance — respects ordering."""
    values = series.values.astype(float)
    min_val, max_val = values.min(), values.max()
    normalized = (values - min_val) / (max_val - min_val)
    return np.abs(normalized[:, None] - normalized[None, :])

def hamming_distance_matrix(df_cat):
    """Hamming distance for categorical block — fraction of mismatches."""
    arr = df_cat.values.astype(str)
    n = len(arr)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.mean(arr[i] != arr[j])
            dist[i, j] = dist[j, i] = d
    return dist


from sklearn.preprocessing import MinMaxScaler


def build_personas(df_demo, data, labels_lens1, labels_lens2, labels_lens3, output_file="df_final.xlsx"):
    df_final = df_demo.copy()

    # Add lens labels
    df_final["Lens1_Capacity"] = labels_lens1
    df_final["Lens2_Engagement"] = labels_lens2
    df_final["Lens3_Behavior"] = labels_lens3

    # Build personas
    persona_counts = (
        df_final
        .value_counts(["Lens1_Capacity", "Lens2_Engagement", "Lens3_Behavior"])
        .reset_index(name="Client_Count")
    )
    persona_counts["Percentage"] = persona_counts["Client_Count"] / len(df_final) * 100
    persona_counts["Persona_ID"] = range(1, len(persona_counts) + 1)

    df_final = df_final.merge(
        persona_counts[["Lens1_Capacity", "Lens2_Engagement", "Lens3_Behavior", "Persona_ID"]],
        on=["Lens1_Capacity", "Lens2_Engagement", "Lens3_Behavior"]
    )

    # Drop lenses and add pillars
    df_final.drop(columns=["Lens1_Capacity", "Lens2_Engagement", "Lens3_Behavior"], inplace=True)
    scaler = MinMaxScaler()

    df_final["DebtCycleStress"] = scaler.fit_transform(
        (data['Debt'] - (data['Income'] + data['Wealth']) / 2).values.reshape(-1, 1)
    ).flatten()

    df_final["FinancialLiteracy"] = data[['FinEdu', 'Digital', 'BankFriend']].mean(axis=1)

    df_final["ManagementApproach"] = scaler.fit_transform(
        ((data['Saving'] + data['Investments'] + data['ESG']) - data['Luxury']).values.reshape(-1, 1)
    ).flatten()

    pillars = ["DebtCycleStress", "FinancialLiteracy", "ManagementApproach"]
    summary_table = (
        df_final.groupby("Persona_ID")[pillars]
        .mean()
        .round(3)
    )
    summary_table.insert(0, "Client_Count", df_final["Persona_ID"].value_counts())
    summary_table = summary_table.sort_values("Client_Count", ascending=False)

    print("\n--- TOP DOMINANT PERSONAS ---")
    print(persona_counts.head())

    print("\n--- PERSONA PILLAR SUMMARY ---")
    display(summary_table)

    df_final.to_excel(output_file, index=False)
    return df_final, persona_counts,summary_table



def plot_lens_distributions(df, continuous_cols, categorical_col):
    """
    Plots histograms for continuous variables and a count plot for a categorical variable.
    """
    # Calculate how many subplots we need
    total_plots = len(continuous_cols) + 1

    # We want 3 columns of plots. Calculate how many rows are needed.
    cols = 3
    rows = (total_plots // cols) + (1 if total_plots % cols != 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten to make it easy to loop through

    # 1. Plot continuous variables
    for i, col in enumerate(continuous_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='teal', bins=20)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_ylabel('Number of Clients')

    # 2. Plot categorical variable
    cat_idx = len(continuous_cols)
    sns.countplot(x=df[categorical_col], ax=axes[cat_idx], palette='viridis')
    axes[cat_idx].set_title(f'Count of {categorical_col} Types')
    axes[cat_idx].set_ylabel('Number of Clients')

    # 3. Hide any empty subplots at the end of the grid
    for j in range(cat_idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def find_optimal_clusters(distance_matrix, k_range=range(2, 8), linkage='average'):
    """
    Finds the optimal number of clusters using silhouette score
    for a precomputed distance matrix (e.g., Gower).
    """

    scores = {}

    for k in k_range:
        model = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage=linkage
        )

        labels = model.fit_predict(distance_matrix)

        score = silhouette_score(
            distance_matrix,
            labels,
            metric='precomputed'
        )

        scores[k] = score

    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]

    return best_k, best_score, scores


def normalize_df(df, scaler):
    """
    Normalize only the numerical columns of a DataFrame,
    leaving categorical/object columns untouched so Gower can read them.
    """
    df_norm = df.copy()
    num_cols = df_norm.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df_norm[num_cols] = scaler.fit_transform(df_norm[num_cols])
    return df_norm

def find_outliers_selective(df, columns_to_check):
    """
    df: Your pandas DataFrame
    columns_to_check: List of strings (names of continuous numerical columns)
    """
    rows_to_drop = set()

    for col in columns_to_check:
        data = df[col].values
        mean, std = data.mean(), data.std()

        # 3-sigma rule
        lower, upper = mean - 3* std, mean + 3 * std

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        if not outliers.empty:
            print(f"Feature '{col}' has {len(outliers)} outliers.")
            rows_to_drop.update(outliers.index.tolist())

    # Drop rows by index
    df_cleaned = df.drop(index=list(rows_to_drop))
    print(f"\nDropped {len(rows_to_drop)} total rows.")
    return df_cleaned



# unused codes:

# import seaborn as sns
#
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#
# # Plot 1: Colored by your Model's Clusters (What you already saw)
# sns.scatterplot(x=tsne_lens3[:, 0], y=tsne_lens3[:, 1], hue=labels_lens3, palette='Set3', ax=axes[0])
# axes[0].set_title('Lens 3: Colored by Cluster Label')
#
# # Plot 2: Colored by the RAW Investment Variable
# sns.scatterplot(x=tsne_lens3[:, 0], y=tsne_lens3[:, 1], hue=df_lens3['Investments'], palette='Set1', ax=axes[1])
# axes[1].set_title('Lens 3: Colored by Investment Type')
#
# plt.tight_layout()
# plt.show()

#
# # Step 1 — cluster on BEHAVIOR only (no Investment)
# best_k3, weights3, score3, dist_mat_lens3 = Utilities.build_lens_distance(
#     df_lens3_norm,
#     cont_cols=['LifeStyle', 'Luxury', 'Saving', 'ESG']
#     # Investment deliberately excluded
# )
#
# # Step 2 — after clustering, analyze how Investment distributes across clusters
# df_lens3_norm['cluster']    = labels_lens3
# df_lens3_norm['Investment'] = df_lens3['Investments'].values
#
# print(df_lens3_norm.groupby('cluster')['Investment'].value_counts(normalize=True))


# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.feature_selection import mutual_info_classif
#
# cols = ['LifeStyle', 'Luxury', 'Saving', 'ESG']
# X    = df_lens3_norm[cols].values
# y    = df_lens3['Investments'].values.astype(int)
#
# # Compute metrics
# corr_vals     = [df_lens3_norm[c].corr(df_lens3['Investments']) for c in cols]
# individual_r2 = [r2_score(y, LinearRegression().fit(df_lens3_norm[[c]], y).predict(df_lens3_norm[[c]])) for c in cols]
# joint_r2      = r2_score(y, LinearRegression().fit(X, y).predict(X))
# mi_vals       = mutual_info_classif(X, y, random_state=42)
#
# # Plot
# fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# fig.suptitle('Individual vs Joint Predictive Power on Investment', fontsize=13)
#
# for ax, vals, title, ylabel in zip(
#     axes,
#     [corr_vals, mi_vals, individual_r2 + [joint_r2]],
#     ['Pairwise Correlation\n(individually weak)',
#      'Mutual Information\n(individually weak)',
#      'R² Individual vs Joint\n(jointly strong)'],
#     ['Pearson r', 'Mutual Information', 'R²']
# ):
#     labels  = cols if ax != axes[2] else cols + ['ALL 4\nJOINT']
#     colors  = ['steelblue'] * 4 if ax != axes[2] else ['steelblue'] * 4 + ['tomato']
#     ax.bar(labels, vals, color=colors, edgecolor='white')
#     ax.set_title(title)
#     ax.set_ylabel(ylabel)
#     ax.set_ylim(0, max(vals) * 1.3)
#     for i, v in enumerate(vals):
#         ax.text(i, v + max(vals) * 0.02, f'{v:.3f}', ha='center', fontsize=9)
#
# plt.tight_layout()
# plt.show()
#
# # Print summary
# print("=" * 50)
# print("JOINT vs INDIVIDUAL PREDICTIVE POWER")
# print("=" * 50)
# for c, r, r2, mi in zip(cols, corr_vals, individual_r2, mi_vals):
#     print(f"  {c:<12}  r={r:.3f}  R²={r2:.3f}  MI={mi:.3f}")
# print(f"\n  Joint R² (all 4): {joint_r2:.4f}")
# print("=" * 50)

#
#
# from sklearn.manifold import TSNE
#
# tsne      = TSNE(n_components=2, perplexity=30, random_state=42)
# embedding = tsne.fit_transform(df_lens3_norm[['LifeStyle', 'Luxury', 'Saving', 'ESG']])
#
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# fig.suptitle('Lens 3 — t-SNE on Behavioral Features Only (No Investment)', fontsize=13)
#
# # Plot 1 — colored by cluster
# sc0 = axes[0].scatter(embedding[:, 0], embedding[:, 1],
#                        c=labels_lens3, cmap='tab10', alpha=0.6, s=10)
# axes[0].set_title('Colored by Behavioral Cluster')
# axes[0].legend(*sc0.legend_elements(), title='Cluster')
#
# # Plot 2 — colored by Investment
# sc1 = axes[1].scatter(embedding[:, 0], embedding[:, 1],
#                        c=df_lens3['Investments'].values, cmap='tab10', alpha=0.6, s=10)
# axes[1].set_title('Colored by Investment Value')
# axes[1].legend(*sc1.legend_elements(), title='Investment')
#
# plt.tight_layout()
# plt.show()