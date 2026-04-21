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

from utilities import *


# ============================================================
# BASE CLASS
# ============================================================

class BaseModel:
    def __init__(self, name, model,
        #         scoring=balanced_scorer   use this for regulatory purposes
                  scoring = "f1"
                 ):
        self.name           = name
        self.model          = model
        self.scoring        = scoring
        self.trained_model  = None
        self.tuning_results = None
        self.best_params    = None

    def train(self, X_train, y_train):
        self.trained_model = clone(self.model)

        if self.best_params is not None:
            self.trained_model.set_params(**self.best_params)

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
        return train_cross_validate_and_evaluate(
            X_train, y_train, X_test, y_test,
            clone(self.model), k_folds,
        )

    def tune(self, X_train, y_train):
        raise NotImplementedError

    def _store_tuning(self, best_params, best_score):
        self.best_params = dict(best_params)
        self.model.set_params(**best_params)
        self.tuning_results = {
            "best_params": self.best_params,
            "best_score":  float(best_score),
        }
        return self.tuning_results


# ============================================================
# TREE-BASED MODELS
# ============================================================

class RandomForestModel(BaseModel):
    def __init__(self, random_state=42, scoring=balanced_scorer):
        super().__init__("RandomForest",
                         RandomForestClassifier(random_state=random_state),
                         scoring)

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            clone(self.model),
            {
                "n_estimators": Integer(100, 600),
                "max_depth": Categorical([None, 5, 10, 20]),
                "min_samples_split": Integer(2, 20),
                "min_samples_leaf": Integer(1, 10),
            },
            n_iter=20, cv=3, scoring=self.scoring, n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


class ExtraTreesModel(BaseModel):
    def __init__(self, random_state=42, scoring=balanced_scorer):
        super().__init__("ExtraTrees",
                         ExtraTreesClassifier(random_state=random_state),
                         scoring)

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            clone(self.model),
            {
                "n_estimators": Integer(100, 500),
                "max_depth": Categorical([None, 5, 10, 20]),
                "min_samples_split": Integer(2, 20),
                "min_samples_leaf": Integer(1, 10),
            },
            n_iter=20, cv=3, scoring=self.scoring, n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


class GradientBoostingModel(BaseModel):
    def __init__(self, scoring=balanced_scorer):
        super().__init__("GradientBoosting", GradientBoostingClassifier(), scoring)

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            clone(self.model),
            {
                "n_estimators": Integer(100, 500),
                "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
                "max_depth": Integer(2, 6),
                "min_samples_split": Integer(2, 20),
                "min_samples_leaf": Integer(1, 10),
                "subsample": Real(0.5, 1.0),
                "max_features": Real(0.5, 1.0),
            },
            n_iter=30, cv=3, scoring=self.scoring, n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


class HistGradientBoostingModel(BaseModel):
    def __init__(self, scoring=balanced_scorer):
        super().__init__("HistGradientBoosting", HistGradientBoostingClassifier(), scoring)

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            clone(self.model),
            {
                "max_iter": Integer(100, 500),
                "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
                "max_depth": Categorical([None, 5, 10, 20]),
                "min_samples_leaf": Integer(10, 100),
                "l2_regularization": Real(0.0, 10.0),
            },
            n_iter=20, cv=3, scoring=self.scoring, n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


# ============================================================
# BOOSTING LIBRARIES
# ============================================================

class XGBoostModel(BaseModel):
    def __init__(self, random_state=42, scoring=balanced_scorer):
        super().__init__(
            "XGBoost",
            XGBClassifier(random_state=random_state, eval_metric="logloss", base_score=0.5),
            scoring
        )

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            clone(self.model),
            {
                "n_estimators": Integer(100, 500),
                "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
                "max_depth": Integer(2, 6),
                "min_child_weight": Integer(1, 10),
                "subsample": Real(0.5, 1.0),
                "colsample_bytree": Real(0.5, 1.0),
                "gamma": Real(0, 5),
                "reg_alpha": Real(0, 5),
                "reg_lambda": Real(0.1, 10, prior="log-uniform"),
            },
            n_iter=20, cv=5, scoring=self.scoring, n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


class LightGBMModel(BaseModel):
    def __init__(self, scoring=balanced_scorer):
        super().__init__("LightGBM", LGBMClassifier(), scoring)

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            clone(self.model),
            {
                "n_estimators": Integer(100, 500),
                "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
                "max_depth": Integer(2, 6),
                "num_leaves": Integer(20, 100),
                "min_child_samples": Integer(5, 50),
                "subsample": Real(0.5, 1.0),
                "colsample_bytree": Real(0.5, 1.0),
                "reg_alpha": Real(0, 5),
                "reg_lambda": Real(0, 5),
            },
            n_iter=20, cv=3, scoring=self.scoring, n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


class CatBoostModel(BaseModel):
    def __init__(self, scoring=balanced_scorer):
        super().__init__("CatBoost", CatBoostClassifier(verbose=0), scoring)

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            clone(self.model),
            {
                "iterations": Integer(100, 500),
                "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
                "depth": Integer(2, 8),
                "l2_leaf_reg": Real(1, 10, prior="log-uniform"),
            },
            n_iter=20, cv=3, scoring=self.scoring, n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


# ============================================================
# LINEAR + OTHERS
# ============================================================

class LogisticRegressionModel(BaseModel):
    def __init__(self, scoring=balanced_scorer):
        super().__init__("LogisticRegression",
                         LogisticRegression(max_iter=1000),
                         scoring)

    def tune(self, X_train, y_train):
        grid = GridSearchCV(
            clone(self.model),
            {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"],
            },
            cv=5, scoring=self.scoring, n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return self._store_tuning(grid.best_params_, grid.best_score_)


class SGDModel(BaseModel):
    def __init__(self, scoring=balanced_scorer):
        super().__init__("SGDClassifier",
                         SGDClassifier(loss="log_loss"),
                         scoring)

    def tune(self, X_train, y_train):
        grid = GridSearchCV(
            clone(self.model),
            {
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
                "penalty": ["l1", "l2", "elasticnet"],
            },
            cv=5, scoring=self.scoring, n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return self._store_tuning(grid.best_params_, grid.best_score_)


class KNNModel(BaseModel):
    def __init__(self, scoring=balanced_scorer):
        super().__init__("KNN", KNeighborsClassifier(), scoring)

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            clone(self.model),
            {
                "n_neighbors": Integer(3, 50),
                "weights": Categorical(["uniform", "distance"]),
                "metric": Categorical(["euclidean", "manhattan"]),
            },
            n_iter=30, cv=5, scoring=self.scoring, n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


class SVMModel(BaseModel):
    def __init__(self, scoring=balanced_scorer):
        super().__init__("SVM", SVC(probability=True), scoring)

    def tune(self, X_train, y_train):
        opt = BayesSearchCV(
            clone(self.model),
            {
                "C": Real(0.01, 100, prior="log-uniform"),
                "gamma": Categorical(["scale", "auto"]),
                "kernel": Categorical(["rbf", "poly"]),
            },
            n_iter=20, cv=3, scoring=self.scoring, n_jobs=-1, random_state=42,
        )
        opt.fit(X_train, y_train)
        return self._store_tuning(opt.best_params_, opt.best_score_)


class GaussianNBModel(BaseModel):
    def __init__(self, scoring=balanced_scorer):
        super().__init__("GaussianNB", GaussianNB(), scoring)

    def tune(self, X_train, y_train):
        grid = GridSearchCV(
            clone(self.model),
            {"var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]},
            cv=5, scoring=self.scoring, n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return self._store_tuning(grid.best_params_, grid.best_score_)


class BernoulliNBModel(BaseModel):
    def __init__(self, scoring=balanced_scorer):
        super().__init__("BernoulliNB", BernoulliNB(), scoring)

    def tune(self, X_train, y_train):
        grid = GridSearchCV(
            clone(self.model),
            {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]},
            cv=5, scoring=self.scoring, n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return self._store_tuning(grid.best_params_, grid.best_score_)


# ============================================================
# FACTORY
# ============================================================

class ModelFactory:
    MODELS = {
        "rf": RandomForestModel,
        "extra_trees": ExtraTreesModel,
        "gb": GradientBoostingModel,
        "hist_gb": HistGradientBoostingModel,
        "xgb": XGBoostModel,
        "lgbm": LightGBMModel,
        "catboost": CatBoostModel,
        "logistic": LogisticRegressionModel,
        "sgd": SGDModel,
        "knn": KNNModel,
        "svm": SVMModel,
        "gnb": GaussianNBModel,
        "bnb": BernoulliNBModel,
    }

    @staticmethod
    def create(model_name: str, scoring=balanced_scorer) -> BaseModel:
        if model_name not in ModelFactory.MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(ModelFactory.MODELS)}"
            )
        return ModelFactory.MODELS[model_name](scoring=scoring)