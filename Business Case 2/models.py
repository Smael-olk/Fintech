
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
from skopt.space import Categorical
from skopt.space import Real, Integer
from xgboost import XGBClassifier

from utilities import train_cross_validate_and_evaluate


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseModel:
    def __init__(self, name, model):
        self.name = name
        self.model = model          # base estimator (may be replaced after tuning)
        self.trained_model = None   # fitted estimator after calling train()

    def train(self, X_train, y_train):
        # clone ensures we don't mutate the prototype after tuning
        self.trained_model = clone(self.model)
        self.trained_model.fit(X_train, y_train)

    def predict(self, X):
        if self.trained_model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.trained_model.predict(X)

    def predict_proba(self, X):
        if self.trained_model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        if not hasattr(self.trained_model, "predict_proba"):
            raise NotImplementedError(f"{self.name} does not support predict_proba.")
        return self.trained_model.predict_proba(X)

    def evaluate(self, X_train, y_train, X_test, y_test, k_folds=5):
        """
        Stratified CV on the training set, then final evaluation on the test set.
        Uses self.model (not self.trained_model), so it is independent of train().
        """
        return train_cross_validate_and_evaluate(
            X_train, y_train, X_test, y_test, self.model, k_folds
        )

    def tune(self, X_train, y_train):
        """Subclasses must implement their own tuning strategy."""
        raise NotImplementedError("Each model must implement its own tuning strategy.")


# ---------------------------------------------------------------------------
# Tree-based models
# ---------------------------------------------------------------------------

class RandomForestModel(BaseModel):
    def __init__(self, random_state=42):
        super().__init__("RandomForest", RandomForestClassifier(random_state=random_state))

    def tune(self, X_train, y_train):
        # Bayesian search over a continuous hyperparameter space
        search_space = {
            "n_estimators": Integer(100, 600),
            "max_depth": Categorical([None, 5, 10, 20]),
            "min_samples_split": Integer(2, 20),
            "min_samples_leaf": Integer(1, 10),
        }
        opt = BayesSearchCV(
            estimator=self.model,
            search_spaces=search_space,
            n_iter=20,
            cv=3,    # for computational efficiency
            scoring="f1",
            n_jobs=-1,
            random_state=42,
        )
        opt.fit(X_train, y_train)
        self.model = opt.best_estimator_
        return opt.best_params_


class ExtraTreesModel(BaseModel):
    def __init__(self, random_state=42):
        super().__init__("ExtraTrees", ExtraTreesClassifier(random_state=random_state))

    def tune(self, X_train, y_train):
        pass


class GradientBoostingModel(BaseModel):
    def __init__(self):
        super().__init__("GradientBoosting", GradientBoostingClassifier())

    def tune(self, X_train, y_train):
        search_space = {
            'n_estimators': Integer(100, 500),
            'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
            'max_depth': Integer(2, 6),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'subsample': Real(0.5, 1.0),
            'max_features': Real(0.5, 1.0)
        }
        opt = BayesSearchCV(
            estimator=self.model,
            search_spaces=search_space,
            n_iter=30,
            cv=3, # heavy computational cost
            scoring="f1",   # adjust it later
            n_jobs=-1,
            random_state=42,
        )
        opt.fit(X_train, y_train)
        self.model = opt.best_estimator_
        return opt.best_params_




class HistGradientBoostingModel(BaseModel):
    def __init__(self):
        super().__init__("HistGradientBoosting", HistGradientBoostingClassifier())

    def tune(self, X_train, y_train):
        pass


# ---------------------------------------------------------------------------
# Boosting libraries
# ---------------------------------------------------------------------------

class XGBoostModel(BaseModel):
    def __init__(self, random_state=42):
        super().__init__(
            "XGBoost",
            XGBClassifier(random_state=random_state, eval_metric="logloss"),
        )

    def tune(self, X_train, y_train):
        search_space = {
            # Boosting
            'n_estimators': Integer(100, 500),
            'learning_rate': Real(0.01, 0.2, prior='log-uniform'),

            # Tree complexity
            'max_depth': Integer(2, 6),
            'min_child_weight': Integer(1, 10),

            # Sampling (VERY important)
            'subsample': Real(0.5, 1.0),
            'colsample_bytree': Real(0.5, 1.0),

            # Regularization (this is where XGBoost shines)
            'gamma': Real(0, 5),  # minimum loss reduction
            'reg_alpha': Real(0, 5),  # L1 regularization
            'reg_lambda': Real(0.1, 10, prior='log-uniform')  # L2 regularization
        }
        opt = BayesSearchCV(
            estimator=self.model,
            search_spaces=search_space,
            n_iter=20,
            cv=5, # heavy computational cost
            scoring="f1",  # adjust it later
            n_jobs=-1,
            random_state=42,
        )
        opt.fit(X_train, y_train)
        self.model = opt.best_estimator_
        return opt.best_params_


class LightGBMModel(BaseModel):
    def __init__(self):
        super().__init__("LightGBM", LGBMClassifier())

    def tune(self, X_train, y_train):
        pass


class CatBoostModel(BaseModel):
    def __init__(self):
        super().__init__("CatBoost", CatBoostClassifier(verbose=0))

    def tune(self, X_train, y_train):
        pass


# ---------------------------------------------------------------------------
# Linear models
# ---------------------------------------------------------------------------

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__("LogisticRegression", LogisticRegression(max_iter=1000))

    def tune(self, X_train, y_train):
        param_grid = {
            "C": [0.01, 0.1, 1, 10],   # inverse regularisation strength
            "penalty": ["l1", "l2"],    # regularisation type
            "solver": ["liblinear"],    # supports both l1 and l2
        }
        grid = GridSearchCV(self.model, param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        return grid.best_params_


class SGDModel(BaseModel):
    def __init__(self):
        super().__init__("SGDClassifier", SGDClassifier(loss="log_loss"))

    def tune(self, X_train, y_train):
        pass


# ---------------------------------------------------------------------------
# Distance-based
# ---------------------------------------------------------------------------

class KNNModel(BaseModel):
    def __init__(self):
        super().__init__("KNN", KNeighborsClassifier())

    def tune(self, X_train, y_train):
        pass


# ---------------------------------------------------------------------------
# SVM
# ---------------------------------------------------------------------------

class SVMModel(BaseModel):
    def __init__(self):
        # probability=True required for predict_proba
        super().__init__("SVM", SVC(probability=True))

    def tune(self, X_train, y_train):
        pass


# ---------------------------------------------------------------------------
# Naive Bayes
# ---------------------------------------------------------------------------

class GaussianNBModel(BaseModel):
    def __init__(self):
        super().__init__("GaussianNB", GaussianNB())

    def tune(self, X_train, y_train):
        pass


class BernoulliNBModel(BaseModel):
    def __init__(self):
        super().__init__("BernoulliNB", BernoulliNB())

    def tune(self, X_train, y_train):
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class ModelFactory:
    MODELS = {
        "rf":         RandomForestModel,
        "extra_trees": ExtraTreesModel,
        "gb":         GradientBoostingModel,
        "hist_gb":    HistGradientBoostingModel,
        "xgb":        XGBoostModel,
        "lgbm":       LightGBMModel,
        "catboost":   CatBoostModel,
        "logistic":   LogisticRegressionModel,
        "sgd":        SGDModel,
        "knn":        KNNModel,
        "svm":        SVMModel,
        "gnb":        GaussianNBModel,
        "bnb":        BernoulliNBModel,
    }

    @staticmethod
    def create(model_name):
        if model_name not in ModelFactory.MODELS:
            raise ValueError(f"Model '{model_name}' not supported.")
        return ModelFactory.MODELS[model_name]()
