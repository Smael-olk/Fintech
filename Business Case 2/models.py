import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from xgboost import XGBClassifier

from utilities import train_cross_validate_and_evaluate


# ============================================================
# BASE CLASS
# ============================================================

class BaseModel:
    def __init__(self, name, model):
        self.name          = name
        self.model         = model       # prototype
        self.trained_model = None        # fitted instance (set by train())
        self.tuning_results = None

    def train(self, X_train, y_train):
        """Fit a fresh clone of the (tuned) model on the full training set."""
        self.trained_model = clone(self.model)
        self.trained_model.fit(X_train, y_train)

    def predict(self, X):
        if self.trained_model is None:
            raise ValueError("Call train() before predict().")
        return self.trained_model.predict(X)

    def predict_proba(self, X):
        if self.trained_model is None:
            raise ValueError("Call train() before predict_proba().")
        if not hasattr(self.trained_model, "predict_proba"):
            raise AttributeError(f"{self.name} does not support predict_proba.")
        return self.trained_model.predict_proba(X)

    def evaluate(self, X_train, y_train, X_test, y_test, k_folds=5):
        """
        CV on X_train + one final evaluation on X_test.
        Always clones self.model so tuned params are used but the fitted
        state of self.trained_model is not disturbed.

        Data-leakage note: X_test is only touched by the final clone inside
        train_cross_validate_and_evaluate; it is never seen during CV.
        """
        return train_cross_validate_and_evaluate(
            X_train, y_train, X_test, y_test,
            clone(self.model), k_folds,
        )

    def tune(self, X_train, y_train):
        raise NotImplementedError(
            f"{self.name}.tune() is not implemented. "
            "Either implement it or call run_experiment with tune=False."
        )

    # ------------------------------------------------------------------
    # Internal helper — used by every tune() implementation
    # ------------------------------------------------------------------
    def _store_tuning(self, best_params, best_score):
        """Update self.model with best params and cache results."""
        self.model.set_params(**best_params)
        self.tuning_results = {
            "best_params": dict(best_params),
            "best_score":  float(best_score),
        }
        return self.tuning_results


# ============================================================
# TREE-BASED MODELS
# ============================================================

class RandomForestModel(BaseModel):
    def __init__(self, random_state=42):
        super().__init__("RandomForest", RandomForestClassifier(random_state=random_state))

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            estimator=clone(self.model),
            search_spaces={
                "n_estimators":     Integer(100, 600),
                "max_depth":        Categorical([None, 5, 10, 20]),
                "min_samples_split": Integer(2, 20),
                "min_samples_leaf":  Integer(1, 10),
            },
            n_iter=20, cv=3, scoring="f1", n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


class ExtraTreesModel(BaseModel):
    def __init__(self, random_state=42):
        super().__init__("ExtraTrees", ExtraTreesClassifier(random_state=random_state))

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            estimator=clone(self.model),
            search_spaces={
                "n_estimators":      Integer(100, 500),
                "max_depth":         Categorical([None, 5, 10, 20]),
                "min_samples_split": Integer(2, 20),
                "min_samples_leaf":  Integer(1, 10),
            },
            n_iter=20, cv=3, scoring="f1", n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


class GradientBoostingModel(BaseModel):
    def __init__(self):
        super().__init__("GradientBoosting", GradientBoostingClassifier())

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            estimator=clone(self.model),
            search_spaces={
                "n_estimators":  Integer(100, 500),
                "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
                "max_depth":     Integer(2, 6),
                "min_samples_split": Integer(2, 20),
                "min_samples_leaf":  Integer(1, 10),
                "subsample":     Real(0.5, 1.0),
                "max_features":  Real(0.5, 1.0),
            },
            n_iter=30, cv=3, scoring="f1", n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


class HistGradientBoostingModel(BaseModel):
    def __init__(self):
        super().__init__("HistGradientBoosting", HistGradientBoostingClassifier())

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            estimator=clone(self.model),
            search_spaces={
                "max_iter":      Integer(100, 500),
                "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
                "max_depth":     Categorical([None, 5, 10, 20]),
                "min_samples_leaf": Integer(10, 100),
                "l2_regularization": Real(0.0, 10.0),
            },
            n_iter=20, cv=3, scoring="f1", n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


# ============================================================
# BOOSTING LIBRARIES
# ============================================================

class XGBoostModel(BaseModel):
    def __init__(self, random_state=42):
        super().__init__(
            "XGBoost",
            XGBClassifier(random_state=random_state, eval_metric="logloss"),
        )

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            estimator=clone(self.model),
            search_spaces={
                "n_estimators":    Integer(100, 500),
                "learning_rate":   Real(0.01, 0.2, prior="log-uniform"),
                "max_depth":       Integer(2, 6),
                "min_child_weight": Integer(1, 10),
                "subsample":       Real(0.5, 1.0),
                "colsample_bytree": Real(0.5, 1.0),
                "gamma":           Real(0, 5),
                "reg_alpha":       Real(0, 5),
                "reg_lambda":      Real(0.1, 10, prior="log-uniform"),
            },
            n_iter=20, cv=5, scoring="f1", n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


class LightGBMModel(BaseModel):
    def __init__(self):
        super().__init__("LightGBM", LGBMClassifier())

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            estimator=clone(self.model),
            search_spaces={
                "n_estimators":  Integer(100, 500),
                "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
                "max_depth":     Integer(2, 6),
                "num_leaves":    Integer(20, 100),
                "min_child_samples": Integer(5, 50),
                "subsample":     Real(0.5, 1.0),
                "colsample_bytree": Real(0.5, 1.0),
                "reg_alpha":     Real(0, 5),
                "reg_lambda":    Real(0, 5),
            },
            n_iter=20, cv=3, scoring="f1", n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


class CatBoostModel(BaseModel):
    def __init__(self):
        super().__init__("CatBoost", CatBoostClassifier(verbose=0))

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            estimator=clone(self.model),
            search_spaces={
                "iterations":    Integer(100, 500),
                "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
                "depth":         Integer(2, 8),
                "l2_leaf_reg":   Real(1, 10, prior="log-uniform"),
            },
            n_iter=20, cv=3, scoring="f1", n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


# ============================================================
# LINEAR MODELS
# ============================================================

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__("LogisticRegression", LogisticRegression(max_iter=1000))

    def tune(self, X_train, y_train):
        grid = GridSearchCV(
            estimator=clone(self.model),
            param_grid={
                "C":       [0.01, 0.1, 1, 10],
                "penalty": ["l1", "l2"],
                "solver":  ["liblinear"],
            },
            cv=5, scoring="f1", n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return self._store_tuning(grid.best_params_, grid.best_score_)


class SGDModel(BaseModel):
    def __init__(self):
        super().__init__("SGDClassifier", SGDClassifier(loss="log_loss"))

    def tune(self, X_train, y_train):
        grid = GridSearchCV(
            estimator=clone(self.model),
            param_grid={
                "alpha":   [1e-4, 1e-3, 1e-2, 1e-1],
                "penalty": ["l1", "l2", "elasticnet"],
            },
            cv=5, scoring="f1", n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return self._store_tuning(grid.best_params_, grid.best_score_)


# ============================================================
# KNN
# ============================================================

class KNNModel(BaseModel):
    def __init__(self):
        super().__init__("KNN", KNeighborsClassifier())

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            estimator=clone(self.model),
            search_spaces={
                "n_neighbors": Integer(3, 50),
                "weights":     Categorical(["uniform", "distance"]),
                "metric":      Categorical(["euclidean", "manhattan"]),
            },
            n_iter=30, cv=5, scoring="f1", n_jobs=-1, random_state=42, verbose=0,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


# ============================================================
# SVM
# ============================================================

class SVMModel(BaseModel):
    def __init__(self):
        # probability=True required for predict_proba (uses Platt scaling)
        super().__init__("SVM", SVC(probability=True))

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            estimator=clone(self.model),
            search_spaces={
                "C":     Real(0.01, 100, prior="log-uniform"),
                "gamma": Categorical(["scale", "auto"]),
                "kernel": Categorical(["rbf", "poly"]),
            },
            n_iter=20, cv=3, scoring="f1", n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


# ============================================================
# NAIVE BAYES
# ============================================================

class GaussianNBModel(BaseModel):
    def __init__(self):
        super().__init__("GaussianNB", GaussianNB())

    def tune(self, X_train, y_train):
        grid = GridSearchCV(
            estimator=clone(self.model),
            param_grid={"var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]},
            cv=5, scoring="f1", n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return self._store_tuning(grid.best_params_, grid.best_score_)


class BernoulliNBModel(BaseModel):
    def __init__(self):
        super().__init__("BernoulliNB", BernoulliNB())

    def tune(self, X_train, y_train):
        grid = GridSearchCV(
            estimator=clone(self.model),
            param_grid={"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]},
            cv=5, scoring="f1", n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return self._store_tuning(grid.best_params_, grid.best_score_)


# ============================================================
# FACTORY
# ============================================================

class ModelFactory:
    MODELS = {
        "rf":          RandomForestModel,
        "extra_trees": ExtraTreesModel,
        "gb":          GradientBoostingModel,
        "hist_gb":     HistGradientBoostingModel,
        "xgb":         XGBoostModel,
        "lgbm":        LightGBMModel,
        "catboost":    CatBoostModel,
        "logistic":    LogisticRegressionModel,
        "sgd":         SGDModel,
        "knn":         KNNModel,
        "svm":         SVMModel,
        "gnb":         GaussianNBModel,
        "bnb":         BernoulliNBModel,
    }

    @staticmethod
    def create(model_name: str) -> BaseModel:
        if model_name not in ModelFactory.MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(ModelFactory.MODELS)}"
            )
        return ModelFactory.MODELS[model_name]()
