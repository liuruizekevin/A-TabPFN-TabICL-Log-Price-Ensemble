"""Model specifications fixed before the first evaluation run."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from .preprocessing import ALL_GROUPS, FeatureBuilder, make_preprocessor


def experiment_specs():
    specs = [
        dict(id="E01_raw_ridge", family="Ridge", target="raw", semantic=False, groups=(), skew=False, params={"alpha": 20.0}, stage="baseline"),
        dict(id="E02_log_ridge", family="Ridge", target="log", semantic=False, groups=(), skew=False, params={"alpha": 20.0}, stage="pathway"),
        dict(id="E03_semantic", family="Ridge", target="log", semantic=True, groups=(), skew=False, params={"alpha": 20.0}, stage="pathway"),
        dict(id="E04_domain", family="Ridge", target="log", semantic=True, groups=ALL_GROUPS, skew=False, params={"alpha": 20.0}, stage="pathway"),
        dict(id="E05_skew", family="Ridge", target="log", semantic=True, groups=ALL_GROUPS, skew=True, params={"alpha": 20.0}, stage="pathway"),
    ]
    for alpha in [5.0, 50.0, 100.0]:
        specs.append(dict(id=f"E06_ridge_alpha{int(alpha)}", family="Ridge", target="log", semantic=True, groups=ALL_GROUPS, skew=True, params={"alpha": alpha}, stage="tuning"))
    for alpha, l1 in [(0.0005, 0.5), (0.001, 0.5)]:
        specs.append(dict(id=f"E07_elastic_{alpha}", family="ElasticNet", target="log", semantic=True, groups=ALL_GROUPS, skew=True, params={"alpha": alpha, "l1_ratio": l1, "max_iter": 30000, "tol": 1e-5}, stage="model_comparison"))
    specs.extend([
        dict(id="E08_decision_tree", family="DecisionTree", target="log", semantic=True, groups=ALL_GROUPS, skew=False, params={"max_depth": 8, "min_samples_leaf": 5, "random_state": 42}, stage="model_comparison"),
        dict(id="E08_random_forest", family="RandomForest", target="log", semantic=True, groups=ALL_GROUPS, skew=False, params={"n_estimators": 350, "min_samples_leaf": 2, "max_features": 0.8, "random_state": 42, "n_jobs": 4}, stage="model_comparison"),
        dict(id="E09_gradient_boosting", family="GradientBoosting", target="log", semantic=True, groups=ALL_GROUPS, skew=False, params={"n_estimators": 650, "learning_rate": 0.035, "max_depth": 3, "min_samples_leaf": 5, "loss": "huber", "random_state": 42}, stage="model_comparison"),
        dict(id="E10_catboost_depth4", family="CatBoost", target="log", semantic=True, groups=ALL_GROUPS, skew=False, params={"iterations": 1100, "depth": 4, "learning_rate": 0.035, "l2_leaf_reg": 5.0, "loss_function": "RMSE", "random_seed": 42, "thread_count": 4, "verbose": False, "allow_writing_files": False}, stage="model_comparison"),
        dict(id="E11_catboost_depth5", family="CatBoost", target="log", semantic=True, groups=ALL_GROUPS, skew=False, params={"iterations": 1100, "depth": 5, "learning_rate": 0.035, "l2_leaf_reg": 5.0, "loss_function": "RMSE", "random_seed": 42, "thread_count": 4, "verbose": False, "allow_writing_files": False}, stage="tuning"),
    ])
    # Paired group removals keep every other E05 setting identical.
    for group in ALL_GROUPS:
        specs.append(dict(id=f"A_without_{group}", family="Ridge", target="log", semantic=True, groups=tuple(g for g in ALL_GROUPS if g != group), skew=True, params={"alpha": 20.0}, stage="group_ablation"))
    return specs


BLEND_COMPONENTS = ("E06_ridge_alpha50", "E07_elastic_0.0005", "E10_catboost_depth4")
BLEND_ID = "E12_equal_log_blend"


def make_pipeline(spec):
    classes = {"Ridge": Ridge, "ElasticNet": ElasticNet, "DecisionTree": DecisionTreeRegressor, "RandomForest": RandomForestRegressor, "GradientBoosting": GradientBoostingRegressor}
    if spec["family"] == "CatBoost":
        from catboost import CatBoostRegressor
        estimator = CatBoostRegressor(**spec["params"])
    else:
        estimator = classes[spec["family"]](**spec["params"])
    return Pipeline([
        ("features", FeatureBuilder(semantic=spec["semantic"], groups=spec["groups"])),
        ("preprocessor", make_preprocessor(spec["skew"])),
        ("model", estimator),
    ])


class PricePredictor(RegressorMixin, BaseEstimator):
    """Saved estimator accepts original/augmented feature frames; predicts dollars."""
    def __init__(self, components, name):
        self.components = components
        self.name = name

    def predict_log(self, X):
        logs = []
        for pipeline, target in self.components:
            prediction = pipeline.predict(X)
            logs.append(prediction if target == "log" else np.log(np.maximum(prediction, 1.0)))
        return np.mean(logs, axis=0)

    def predict(self, X):
        return np.exp(self.predict_log(X))
