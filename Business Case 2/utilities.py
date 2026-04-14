import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)


# ============================================================
# METRICS
# ============================================================

def _compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average="weighted",
        zero_division=0,
    )
    return acc, prec, rec, f1


def _run_fold(model, X_train, y_train, train_idx, val_idx):
    fold_model = clone(model)
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    fold_model.fit(X_tr, y_tr)
    y_pred = fold_model.predict(X_val)
    return _compute_metrics(y_val, y_pred)


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

    cv_scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for acc, prec, rec, f1 in fold_results:
        cv_scores["accuracy"].append(acc)
        cv_scores["precision"].append(prec)
        cv_scores["recall"].append(rec)
        cv_scores["f1"].append(f1)

    # Final model — fit once on all of X_train, evaluate on X_test
    final_model = clone(model)
    final_model.fit(X_tr_np, y_tr_np)
    y_test_pred = final_model.predict(X_te_np)

    test_metrics = dict(zip(
        ["accuracy", "precision", "recall", "f1"],
        _compute_metrics(y_te_np, y_test_pred),
    ))

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
    rows = {
        "Metric":   ["Accuracy", "Precision", "Recall", "F1"],
        "CV Mean":  [results["cv_metrics"][m]["mean"] for m in ("accuracy", "precision", "recall", "f1")],
        "CV Std":   [results["cv_metrics"][m]["std"]  for m in ("accuracy", "precision", "recall", "f1")],
        "Test Set": [results["test_metrics"][m]       for m in ("accuracy", "precision", "recall", "f1")],
    }
    df = pd.DataFrame(rows).round(3)
    print(f"\n{model_name} — {feature_type}")
    print("=" * 60)
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
# PLOTTING
# ============================================================

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

# def split_by_entropy(X, y, y_prob, top_percent=0.05):
#     entropy   = binary_entropy(np.asarray(y_prob))
#     threshold = np.quantile(entropy, 1 - top_percent)
#     low_mask  = entropy <  threshold
#     high_mask = entropy >= threshold
#
#     def _build(mask, flag):
#         df = X.loc[mask].copy()
#         df["target"]      = y.loc[mask].values
#         df["entropy"]     = entropy[mask]
#         df["review_flag"] = flag
#         return df
#
#     return _build(low_mask, False), _build(high_mask, True), threshold


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(model_key, feature_type,
                   X_train, y_train, X_test, y_test,
                   tune=False):
    """
    Full pipeline for one model / feature-set combination:
      1. (Optional) Bayesian hyperparameter tuning on X_train only
      2. Stratified k-fold CV + final test evaluation
      3. Fit trained_model on all of X_train
      4. Predict on X_test
      5. Plot diagnostics

    Data-leakage notes
    ------------------
    - Tuning (BayesSearchCV) is run on X_train only.
    - model.evaluate() internally clones the (tuned) model, runs CV on
      X_train, fits a final clone on X_train, and scores on X_test once.
    - model.train() afterwards stores the final fitted model for inference;
      it does NOT re-expose X_test.
    """
    from models import ModelFactory

    model = ModelFactory.create(model_key)

    # 1. Tuning — X_train only
    if tune:
        tuning_results = model.tune(X_train, y_train)
        display_tuning_results(tuning_results, model.name)

    # 2. CV + test evaluation
    results = model.evaluate(X_train, y_train, X_test, y_test)
    display_results_table(results, model.name, feature_type)

    # 3. Final fit for inference
    model.train(X_train, y_train)

    # 4. Predictions from the fitted wrapper
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model.trained_model, "predict_proba")
        else None
    )

    # 5. Diagnostics
    plot_model_diagnostics(y_test, y_pred, y_prob, model.name, feature_type)

    return model, results, y_prob, y_pred


# ============================================================
# ENTROPY SPLIT EVALUATION  (fixed — no data leakage)
# ============================================================

def evaluate_entropy_splits(model, X_test, y_test, y_pred, y_prob,
                            model_name, feature_type, alpha=0.05):
    """
    Evaluate model performance separately on low- and high-entropy test samples.
    ----------------------------------

    """
    entropy   = binary_entropy(np.asarray(y_prob))
    threshold = np.quantile(entropy, 1 - alpha)
    low_mask  = entropy <  threshold
    high_mask = entropy >= threshold

    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = np.asarray(y_prob)

    for mask, label in [(low_mask, "Low Entropy"), (high_mask, "High Entropy")]:
        idx        = X_test.index[mask]
        y_split    = y_test.loc[idx]
        pred_split = y_pred_arr[mask]
        prob_split = y_prob_arr[mask]

        # Test metrics from existing predictions — NO refit, NO leakage
        test_metrics = {
            "accuracy":  float(accuracy_score(y_split, pred_split)),
            "precision": float(precision_score(y_split, pred_split, zero_division=0, average="weighted")),
            "recall":    float(recall_score(y_split, pred_split, zero_division=0, average="weighted")),
            "f1":        float(f1_score(y_split, pred_split, zero_division=0, average="weighted")),
        }

        # CV columns are not meaningful for entropy-filtered slices,
        # so we fill with NaN to keep display_results_table working
        nan_cv = {"mean": float("nan"), "std": float("nan")}
        results = {
            "cv_metrics":   {m: nan_cv for m in ("accuracy", "precision", "recall", "f1")},
            "test_metrics": test_metrics,
        }

        plot_model_diagnostics(
            y_split, pred_split, prob_split,
            f"{model_name} ({label})", feature_type,
        )
        display_results_table(results, f"{model_name} ({label})", feature_type)

    return {"threshold": threshold}
