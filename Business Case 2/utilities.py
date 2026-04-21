from itertools import combinations

import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
from tabulate import tabulate
from itertools import combinations
from metrics import *
from plots import plot_model_diagnostics

def _run_fold(model, X_train, y_train, train_idx, val_idx):
    fold_model = clone(model)
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    fold_model.fit(X_tr, y_tr)
    y_pred = fold_model.predict(X_val)
    return compute_metrics(y_val, y_pred)


# ============================================================
# CROSS-VALIDATION + TEST EVALUATION
# ============================================================

def train_cross_validate_and_evaluate(X_train, y_train, X_test, y_test, model, k_folds=5):
    """
    Runs stratified k-fold CV on train, then fits a fresh model on the full
    train set and evaluates once on the held-out test set.

    Data-leakage notes
    ------------------
    - The scaler / any upstream transformer must be fit on X_train ONLY before
      calling this function.  This function does not touch the scaler.
    - CV folds only see their own fold's training rows — no val leakage.
    - The final model is fit on X_train; X_test is touched exactly once.
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    X_tr_np = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
    y_tr_np = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train
    X_te_np = X_test.to_numpy()  if hasattr(X_test,  "to_numpy") else X_test
    y_te_np = y_test.to_numpy()  if hasattr(y_test,  "to_numpy") else y_test

    fold_results = Parallel(n_jobs=-1)(
        delayed(_run_fold)(model, X_tr_np, y_tr_np, train_idx, val_idx)
        for train_idx, val_idx in skf.split(X_tr_np, y_tr_np)
    )

   # cv_scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    cv_scores = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "safety": [],
        "balanced": [],
    }
    for acc, prec, rec, f1, safety, balanced in fold_results:
        cv_scores["accuracy"].append(acc)
        cv_scores["precision"].append(prec)
        cv_scores["recall"].append(rec)
        cv_scores["f1"].append(f1)
        cv_scores["safety"].append(safety)
        cv_scores["balanced"].append(balanced)

    # Final model — fit once on all of X_train, evaluate on X_test
    final_model = clone(model)
    final_model.fit(X_tr_np, y_tr_np)
    y_test_pred = final_model.predict(X_te_np)

    acc, prec, rec, f1, safety, balanced = compute_metrics(y_te_np, y_test_pred)

    test_metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "safety": safety,
        "balanced": balanced,
    }
    cv_metrics = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in cv_scores.items()
    }

    return {
        "cv_metrics":   cv_metrics,
        "test_metrics": test_metrics,
        "final_model":  final_model,
    }


# ============================================================
# DISPLAY
# ============================================================

def display_results_table(results, model_name, feature_type):
    """
    Displays CV + test metrics including custom business metrics.
    Handles per-class metrics cleanly.
    """

    def _format_metric(val):
        # Handle scalar
        if isinstance(val, (int, float)):
            return round(val, 3)
        # Handle array (per-class metrics)
        if isinstance(val, (list, tuple, np.ndarray)):
            return " / ".join([f"{v:.3f}" for v in val])
        return val

    metrics_order = ["accuracy", "precision", "recall", "f1", "safety", "balanced"]
    metric_names  = ["Accuracy", "Precision (0/1)", "Recall (0/1)", "F1 (0/1)", "Safety", "Balanced"]

    rows = {
        "Metric": metric_names,
        "CV Mean": [
            _format_metric(results["cv_metrics"][m]["mean"]) for m in metrics_order
        ],
        "CV Std": [
            _format_metric(results["cv_metrics"][m]["std"]) for m in metrics_order
        ],
        "Test Set": [
            _format_metric(results["test_metrics"][m]) for m in metrics_order
        ],
    }

    df = pd.DataFrame(rows)

    print(f"\n{model_name} — {feature_type}")
    print("=" * 70)
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))



def display_tuning_results(tuning_results, model_name=None):
    if tuning_results is None:
        print("No tuning results available.")
        return

    best_params = tuning_results.get("best_params", {})
    best_params_str = (
        "\n".join(f"{k}: {v}" for k, v in best_params.items())
        if isinstance(best_params, dict)
        else str(best_params)
    )

    rows = [
        ["Model",         model_name],
        ["Best CV Score", round(tuning_results.get("best_score", 0), 4)],
        ["Best Params",   best_params_str],
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])

    print("\n" + "=" * 60)
    print("TUNING SUMMARY")
    print("=" * 60)
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))



# ============================================================
# ENTROPY UTILITIES
# ============================================================

def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy in bits. p: (n_samples,) probabilities of class 1."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def split_by_entropy(X, y, y_prob, top_percent=0.05):
    """
    Split samples into low/high-entropy sets based on predicted probabilities.

    Parameters
    ----------
    X          : DataFrame of features (must have a pandas index)
    y          : Series of labels (aligned with X)
    y_prob     : array (n_samples,) — class-1 probabilities from a trained model
    top_percent: fraction of samples to put in the high-entropy (review) set

    Returns
    -------
    clean_df   : low-entropy samples with target, entropy, review_flag columns
    review_df  : high-entropy samples with the same extra columns
    threshold  : entropy quantile used as the cut-off
    """


    entropy = binary_entropy(np.asarray(y_prob))

    n_total = len(entropy)
    k = int(np.ceil(top_percent * n_total))

    # rank indices by entropy (descending)
    ranked_idx = np.argsort(entropy)[::-1]

    high_idx = ranked_idx[:k]
    low_idx  = ranked_idx[k:]

    def _build(idx, flag):
        df = X.iloc[idx].copy()
        df["target"] = y.iloc[idx].values
        df["entropy"] = entropy[idx]
        df["review_flag"] = flag
        return df

    return _build(low_idx, False), _build(high_idx, True), entropy[high_idx[-1]]


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(model_key, feature_type,
                   X_train, y_train, X_test, y_test,
                   tune=False):
    from models import ModelFactory

    model = ModelFactory.create(model_key)

    # 1. Tuning — X_train only
    if tune:
        tuning_results = model.tune(X_train, y_train)
        display_tuning_results(tuning_results, model.name)

    # 2. CV + test evaluation
    results = model.evaluate(X_train, y_train, X_test, y_test)
    display_results_table(results, model.name, feature_type)

    # 3. Final fit on full training data
    model.train(X_train, y_train)

    # ============================================================
    # 4. PREDICTIONS (IMPORTANT FIX)
    # ============================================================

    # TRAIN predictions (for ensemble learning / weights)
    y_prob_train = (
        model.predict_proba(X_train)[:, 1]
        if hasattr(model.trained_model, "predict_proba")
        else None
    )

    # TEST predictions (for final evaluation)
    y_prob_test = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model.trained_model, "predict_proba")
        else None
    )

    y_pred_test = model.predict(X_test)

    # 5. Diagnostics (TEST ONLY)
    plot_model_diagnostics(y_test, y_pred_test, y_prob_test,
                           model.name, feature_type)

    return model, results, y_prob_train, y_prob_test, y_pred_test



# def run_experiment(model_key, feature_type,
#                    X_train, y_train, X_test, y_test,
#                    tune=False):
#     """
#     Full pipeline for one model / feature-set combination:
#       1. (Optional) Bayesian hyperparameter tuning on X_train only
#       2. Stratified k-fold CV + final test evaluation
#       3. Fit trained_model on all of X_train
#       4. Predict on X_test
#       5. Plot diagnostics
#
#     Data-leakage notes
#     ------------------
#     - Tuning (BayesSearchCV) is run on X_train only.
#     - model.evaluate() internally clones the (tuned) model, runs CV on
#       X_train, fits a final clone on X_train, and scores on X_test once.
#     - model.train() afterwards stores the final fitted model for inference;
#       it does NOT re-expose X_test.
#     """
#     from models import ModelFactory
#
#     model = ModelFactory.create(model_key)
#
#     # 1. Tuning — X_train only
#     if tune:
#         tuning_results = model.tune(X_train, y_train)
#         display_tuning_results(tuning_results, model.name)
#
#     # 2. CV + test evaluation
#     results = model.evaluate(X_train, y_train, X_test, y_test)
#     display_results_table(results, model.name, feature_type)
#
#     # 3. Final fit for inference
#     model.train(X_train, y_train)
#
#     # 4. Predictions from the fitted wrapper
#     y_pred = model.predict(X_test)
#     y_prob = (
#         model.predict_proba(X_test)[:, 1]
#         if hasattr(model.trained_model, "predict_proba")
#         else None
#     )
#
#     # 5. Diagnostics
#     plot_model_diagnostics(y_test, y_pred, y_prob, model.name, feature_type)
#
#     return model, results, y_prob, y_pred


# ============================================================
# ENTROPY SPLIT EVALUATION  (fixed — no data leakage)
# ============================================================
def evaluate_entropy_splits(model, X_test, y_test, y_pred, y_prob,
                            model_name, feature_type, alpha=0.05):
    """
    Evaluate model performance separately on low- and high-entropy test samples.
    Includes business metrics (safety + balanced).
    """

    entropy   = binary_entropy(np.asarray(y_prob))
    threshold = np.quantile(entropy, 1 - alpha)

    low_mask  = entropy < threshold
    high_mask = entropy >= threshold

    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = np.asarray(y_prob)

    for mask, label in [(low_mask, "Low Entropy"), (high_mask, "High Entropy")]:
        idx        = X_test.index[mask]
        y_split    = y_test.loc[idx]
        pred_split = y_pred_arr[mask]
        prob_split = y_prob_arr[mask]

        # ✅ Add business metrics here
        safety   = safety_constrained_precision(y_split, pred_split)
        balanced = balanced_business_score(y_split, pred_split)

        test_metrics = {
            "accuracy":  float(accuracy_score(y_split, pred_split)),
            "precision": float(precision_score(y_split, pred_split, zero_division=0, average="weighted")),
            "recall":    float(recall_score(y_split, pred_split, zero_division=0, average="weighted")),
            "f1":        float(f1_score(y_split, pred_split, zero_division=0, average="weighted")),
            "safety":    float(safety),
            "balanced":  float(balanced),
        }

        # CV not meaningful here → keep NaN
        nan_cv = {"mean": float("nan"), "std": float("nan")}
        results = {
            "cv_metrics": {
                m: nan_cv for m in ("accuracy", "precision", "recall", "f1", "safety", "balanced")
            },
            "test_metrics": test_metrics,
        }

        plot_model_diagnostics(
            y_split, pred_split, prob_split,
            f"{model_name} ({label})", feature_type,
        )

        display_results_table(results, f"{model_name} ({label})", feature_type)

        # Optional but very useful warning
        if safety < 0:
            print(f"⚠️ WARNING: {label} set violates safety constraint!")

    return {"threshold": threshold}


def generate_recommendations(df, inc_col='income_prob', acc_col='accum_prob', 
                           inc_threshold=0.45, acc_threshold=0.55, 
                           entropy_quantile=0.95):
    """
    Business logic for product assignment after modelling.
    Combines probability thresholds with an entropy guardrail.
    """
    # 1. Calculate Entropy for both targets using existing utility
    # binary_entropy is already defined in this file at line 107
    df['inc_h'] = df[inc_col].apply(binary_entropy)
    df['acc_h'] = df[acc_col].apply(binary_entropy)
    
    # 2. Rank entropy to identify top X% most uncertain cases
    df['inc_h_rank'] = df['inc_h'].rank(pct=True)
    df['acc_h_rank'] = df['acc_h'].rank(pct=True)
    
    def apply_rules(row):
        # RULE 1: Entropy Guardrail (Regulatory/Safety Check)
        # If uncertainty is too high (top 5%), flag for human advisor
        if row['inc_h_rank'] >= entropy_quantile or row['acc_h_rank'] >= entropy_quantile:
            return "ADVISOR_REVIEW", True
            
        # RULE 2: Probability Thresholds (Need Identification)
        is_inc = row[inc_col] >= inc_threshold
        is_acc = row[acc_col] >= acc_threshold
        
        # Conflict Resolution & Assignment
        if is_inc and is_acc:
            # If both products fit, pick the one with highest probability
            return "INCOME" if row[inc_col] > row[acc_col] else "ACCUMULATION", False
        elif is_inc:
            return "INCOME", False
        elif is_acc:
            return "ACCUMULATION", False
        else:
            return "NO_ACTION", False

    # Apply rules and expand results into two columns
    results = df.apply(apply_rules, axis=1)
    df[['recommendation', 'review_flag']] = pd.DataFrame(results.tolist(), index=df.index)
    
    return df


def compute_oof_matrix(model_wrappers, X_train, y_train, n_splits=5):
    """
    Generates Out-Of-Fold predictions for a list of BaseModel wrappers.
    Clones each wrapper's trained_model directly — no key lookup needed.
    Returns an OOF matrix of shape (n_samples, n_models).
    """
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X_np = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
    y_np = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train

    oof_matrix = np.zeros((len(X_np), len(model_wrappers)))

    for j, wrapper in enumerate(model_wrappers):
        oof_preds = np.zeros(len(X_np))

        for train_idx, val_idx in skf.split(X_np, y_np):
            # Clone directly from wrapper.trained_model — avoids any key mismatch
            base_est = clone(wrapper.trained_model)
            base_est.fit(X_np[train_idx], y_np[train_idx])
            oof_preds[val_idx] = base_est.predict_proba(X_np[val_idx])[:, 1]

        oof_matrix[:, j] = oof_preds
#        print(f"  {wrapper.name:20s} — OOF mean: {oof_preds.mean():.3f}  std: {oof_preds.std():.3f}")

    return oof_matrix


def build_test_matrix(model_wrappers, X_test):
    """
    Stacks each trained BaseModel's test probabilities into a (n_test, n_models) matrix.
    """
    return np.column_stack([
        wrapper.predict_proba(X_test)[:, 1]
        for wrapper in model_wrappers
    ])

# MARTHEMATICAl FEATURE ENGINEERING AND EVALUATION

def engineer_and_evaluate(df, base_columns, targets):
    df_ext = df.copy()
    
    # 1. Generate Interactions (Multiplications & Safe Divisions)
    for c1, c2 in combinations(base_columns, 2):
        df_ext[f"{c1}_X_{c2}"] = df[c1] * df[c2]
        
    denominators = ['Age', 'FamilyMembers']
    for num in base_columns:
        for den in denominators:
            if num != den: # we check that they are not the same feature
                df_ext[f"{num}_DIV_{den}"] = df[num] / (df[den] + 1e-5) #we create the list of denominators and numerators and check they are denom is not equal to 0

    # 2. Evaluate Features (Predictive Power vs. Multicollinearity)
    corr = df_ext.corr()
    new_features = [col for col in df_ext.columns if col not in df.columns]
    results = []
    
    for feat in new_features:
        max_signal = corr.loc[feat, targets].abs().max() # Target correlation
        other_feats = [col for col in df_ext.columns if col not in targets + [feat]]
        max_collin = corr.loc[feat, other_feats].abs().max() # Other feature correlation
        
        results.append({'Feature': feat, 'Max_Signal': max_signal, 'Max_Multicollinearity': max_collin})
    
    # Rank FIRST by minimum collinearity (ascending=True), THEN by predictive power (ascending=False)
    results_df = pd.DataFrame(results).sort_values(
        by=['Max_Multicollinearity', 'Max_Signal'], ascending=[True, False]
    )
    
    return df_ext, results_df, corr


# DEFINE RISK BUCKET HELPER (Vectorized)
# Instead of slow loops, we calculate the bin edges to use with pd.cut()
def get_risk_bins_and_labels(prod_df, global_min_risk, global_max_risk):
    risks = prod_df['Risk'].values

    # Calculate midpoints between adjacent product risks
    midpoints = [(risks[i] + risks[i+1]) / 2.0 for i in range(len(risks)-1)]

    # Create bin edges: [min, mid1, mid2... max]
    # We subtract/add 0.001 to the min/max edges to ensure the lowest/highest clients aren't excluded
    bins = [global_min_risk - 0.001] + midpoints + [global_max_risk + 0.001]

    # The labels are the product IDs that fall within these bins
    labels = prod_df['IDProduct'].values
    return bins, labels

# ASSIGN THE FINAL NEXT BEST ACTION
def finalize_nba(row):
    # Guardrail: If they are flagged for review or uncertainty is too high, DO NOT automate a product
    if row.get('review_flag', False) == True or row.get('recommendation') == 'ADVISOR_REVIEW':
        return "HUMAN_ADVISOR"

    # Map to the specific product based on the need we predicted
    elif row['recommendation'] == 'INCOME':
        return row['Income_ProductID']
    elif row['recommendation'] == 'ACCUMULATION':
        return row['Accum_ProductID']
    else:
        return "NO_ACTION"
