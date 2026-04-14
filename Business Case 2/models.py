
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

from sklearn.base import clone

class BaseModel:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.trained_model = None
        self.tuning_results = None

    def train(self, X_train, y_train):
        self.trained_model = clone(self.model)
        self.trained_model.fit(X_train, y_train)

    def predict(self, X):
        if self.trained_model is None:
            raise ValueError("Model not trained.")
        return self.trained_model.predict(X)

    def predict_proba(self, X):
        if self.trained_model is None:
            raise ValueError("Model not trained.")
        if not hasattr(self.trained_model, "predict_proba"):
            return None
        return self.trained_model.predict_proba(X)

    def evaluate(self, X_train, y_train, X_test, y_test, k_folds=5):
        # IMPORTANT: use current (possibly tuned) model
        model = clone(self.model)
        return train_cross_validate_and_evaluate(
            X_train, y_train, X_test, y_test, model, k_folds
        )

    def tune(self, X_train, y_train):
        raise NotImplementedError

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

        # best model
        self.model = opt.best_estimator_

        # store everything you care about
        self.tuning_results = {
            "best_params": opt.best_params_,
            "best_score": opt.best_score_,
            "cv_results": opt.cv_results_
        }
        return self.tuning_results




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
        # best model
        self.model.set_params(**opt.best_params_)
       # self.model = opt.best_estimator_

        # store everything you care about
        self.tuning_results = {
            "best_params": opt.best_params_,
            "best_score": opt.best_score_,
            "cv_results": opt.cv_results_
        }

        return self.tuning_results


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

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__(
            "LogisticRegression",
            LogisticRegression(max_iter=1000)
        )

    def tune(self, X_train, y_train):

        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"]
        }

        grid = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        # store BEST model
        self.model = grid.best_estimator_

        # CLEAN results (same format as KNN, XGB, etc.)
        self.tuning_results = {
            "best_params": grid.best_params_,
            "best_score": float(grid.best_score_)
        }

        return self.tuning_results


class SGDModel(BaseModel):
    def __init__(self):
        super().__init__("SGDClassifier", SGDClassifier(loss="log_loss"))

    def tune(self, X_train, y_train):
        pass


from sklearn.neighbors import KNeighborsClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
import warnings
warnings.filterwarnings("ignore")

from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

class KNNModel(BaseModel):
    def __init__(self):
        super().__init__("KNN", KNeighborsClassifier())

    def tune(self, X_train, y_train):

        search_space = {
            "n_neighbors": Integer(3, 50),
            "weights": Categorical(["uniform", "distance"]),
            "metric": Categorical(["euclidean", "manhattan"])
        }

        opt = BayesSearchCV(
            estimator=self.model,
            search_spaces=search_space,
            n_iter=30,
            cv=5,
            scoring="f1",
            n_jobs=-1,
            random_state=42,
            verbose=0
        )

        opt.fit(X_train, y_train)

        # set best model
        self.model = opt.best_estimator_

        # store results
        self.tuning_results = {
            "best_params": opt.best_params_,
            "best_score": float(opt.best_score_)
        }

        return self.tuning_results

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
