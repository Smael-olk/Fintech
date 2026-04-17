import numpy as np
from sklearn.metrics import recall_score, precision_score, make_scorer, accuracy_score, precision_recall_fscore_support


# ============================================================

#  FILE CONTAINING ALL CUSTOMIZED METRIC UTILITIES

# ============================================================
# 1. ENTROPY METRIC
# ============================================================

def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy in bits. p: (n_samples,) probabilities of class 1."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))



# ============================================================
# 2. SAFETY CONSTRAINED PRECISION (Hard Constraint)
# ============================================================

def safety_constrained_precision(y_true, y_pred, recall_floor: float = 0.90):
    """
    Hard risk-control metric.

    Business meaning:
    - First ensures we correctly identify bad clients (class 0)
    - If this fails → model is rejected (penalty = -1)
    - If safe → reward precision of good clients (class 1)
    """

    recall_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    if recall_0 < recall_floor:
        return -1.0  # unsafe model → reject

    # Among safe models: reward approval quality
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0)


safety_scorer = make_scorer(
    safety_constrained_precision,
    greater_is_better=True
)


# ============================================================
# 3. BALANCED BUSINESS SCORE (Soft Objective)
# ============================================================

def balanced_business_score(y_true, y_pred, alpha: float = 0.7):
    """
    Soft business trade-off metric.

    Business meaning:
    - alpha → prioritize capturing good clients (recall_1)
    - (1 - alpha) → prioritize approval quality (precision_1)

    No direct risk constraint (that is handled by safety scorer).
    """

    rec1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    prec1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)

    return alpha * rec1 + (1 - alpha) * prec1


balanced_scorer = make_scorer(
    balanced_business_score,
    greater_is_better=True
)

# ============================================================
# 4. COMPUTE METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average=None,
        zero_division=0,
    )

    safety  = safety_constrained_precision(y_true, y_pred)
    balanced = balanced_business_score(y_true, y_pred)

    return acc, prec, rec, f1, safety, balanced
