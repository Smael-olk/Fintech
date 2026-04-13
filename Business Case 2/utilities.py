import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)


def train_cross_validate_and_evaluate(X_train, y_train, X_test, y_test, model, k_folds=5):
    """
    Stratified k-fold CV on the training set, then final evaluation on the test set.

    Returns
    -------
    dict with keys:
      'cv_metrics'   — mean / std per metric across folds
      'test_metrics' — metrics on the held-out test set
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    cv_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr,  X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr,  y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)
        y_pred = fold_model.predict(X_val)

        cv_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
        cv_metrics["precision"].append(
            precision_score(y_val, y_pred, average="weighted", zero_division=0)
        )
        cv_metrics["recall"].append(
            recall_score(y_val, y_pred, average="weighted", zero_division=0)
        )
        cv_metrics["f1"].append(
            f1_score(y_val, y_pred, average="weighted", zero_division=0)
        )

    # Retrain on the full training set and evaluate on the test set
    final_model = clone(model)
    final_model.fit(X_train, y_train)
    y_test_pred = final_model.predict(X_test)

    return {
        "cv_metrics": {
            metric: {"mean": np.mean(scores), "std": np.std(scores)}
            for metric, scores in cv_metrics.items()
        },
        "test_metrics": {
            "accuracy":  accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
            "recall":    recall_score(y_test, y_test_pred, average="weighted", zero_division=0),
            "f1":        f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
        },
    }


def display_results_table(results, model_name, feature_type):
    """Print a formatted table of CV and test metrics."""
    rows = {
        "Metric":   ["Accuracy", "Precision", "Recall", "F1"],
        "CV Mean":  [results["cv_metrics"][m]["mean"]  for m in ("accuracy", "precision", "recall", "f1")],
        "CV Std":   [results["cv_metrics"][m]["std"]   for m in ("accuracy", "precision", "recall", "f1")],
        "Test Set": [results["test_metrics"][m]        for m in ("accuracy", "precision", "recall", "f1")],
    }
    df = pd.DataFrame(rows).round(3)
    print(f"\n{model_name} — {feature_type}")
    print("=" * 60)
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))


def plot_confusion_matrix(y_true, y_pred, model_name, feature_type, ax=None):
    """Normalised confusion matrix. Pass ax to embed in a larger figure."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix\n{model_name} — {feature_type}")


def plot_roc_curve(y_true, y_proba, model_name, feature_type, ax=None):
    """ROC curve with AUC. y_proba should be positive-class probabilities."""
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
    """Side-by-side confusion matrix and ROC curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"{model_name} — {feature_type}", fontsize=13, fontweight="bold")
    plot_confusion_matrix(y_true, y_pred, model_name, feature_type, ax=ax1)
    plot_roc_curve(y_true, y_proba, model_name, feature_type, ax=ax2)
    plt.tight_layout()
    plt.show()
