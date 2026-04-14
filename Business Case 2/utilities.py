import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def _compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )
    return acc, prec, rec, f1


def _run_fold(model, X_train, y_train, train_idx, val_idx):
    fold_model = clone(model)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    fold_model.fit(X_tr, y_tr)
    y_pred = fold_model.predict(X_val)

    return _compute_metrics(y_val, y_pred)


# ------------------------------------------------------------
# MAIN CV + TEST EVALUATION
# ------------------------------------------------------------

def train_cross_validate_and_evaluate(X_train, y_train, X_test, y_test, model, k_folds=5):

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test  = X_test.to_numpy()
    y_test  = y_test.to_numpy()

    # IMPORTANT FIX: use tuned estimator
    base_model = model

    results = Parallel(n_jobs=-1)(
        delayed(_run_fold)(
            base_model,
            X_train, y_train,
            train_idx, val_idx
        )
        for train_idx, val_idx in skf.split(X_train, y_train)
    )

    cv_scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for acc, prec, rec, f1 in results:
        cv_scores["accuracy"].append(acc)
        cv_scores["precision"].append(prec)
        cv_scores["recall"].append(rec)
        cv_scores["f1"].append(f1)

    # --------------------------------------------------------
    # FINAL MODEL (IMPORTANT FIX HERE)
    # --------------------------------------------------------
    final_model = clone(model)
    final_model.fit(X_train, y_train)

    y_test_pred = final_model.predict(X_test)

    test_metrics = dict(zip(
        ["accuracy", "precision", "recall", "f1"],
        _compute_metrics(y_test, y_test_pred)
    ))

    cv_metrics = {
        k: {"mean": np.mean(v), "std": np.std(v)}
        for k, v in cv_scores.items()
    }

    return {
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "final_model": final_model
    }


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# PLOTTING (UNCHANGED)
# ------------------------------------------------------------

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



import pandas as pd
from tabulate import tabulate

import pandas as pd
from tabulate import tabulate

def display_tuning_results(tuning_results, model_name=None):
    """
    Clean display of hyperparameter tuning results.
    """

    if tuning_results is None:
        print("No tuning results available.")
        return

    # -------------------------
    # format best params nicely
    # -------------------------
    best_params = tuning_results.get("best_params", {})

    if isinstance(best_params, dict):
        best_params_str = "\n".join(
            f"{k}: {v}" for k, v in best_params.items()
        )
    else:
        best_params_str = str(best_params)

    # -------------------------
    # build table
    # -------------------------
    rows = [
        ["Model", model_name],
        ["Best CV Score", round(tuning_results.get("best_score", 0), 4)],
        ["Best Params", best_params_str],
    ]

    df = pd.DataFrame(rows, columns=["Metric", "Value"])

    print("\n" + "=" * 60)
    print("TUNING SUMMARY")
    print("=" * 60)
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))