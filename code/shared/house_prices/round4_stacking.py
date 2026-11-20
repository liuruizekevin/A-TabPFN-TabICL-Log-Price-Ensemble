"""Nested evaluation of two fixed stacking rules over four historical models.

Base families were selected using prior development experiments. Nested fitting
keeps each current outer validation target out of meta fitting, but does not undo
the adaptive historical choice of base estimators.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from .experiment import file_sha256
from .followup import followup_pipeline, followup_specs
from .models import PricePredictor, experiment_specs, make_pipeline
from .optimization import make_optimization_pipeline, optimization_specs
from .preprocessing import read_data


BASE_IDS = (
    "O09_native_mae", "F03_robust_c20_g0003", "O22_xgboost_d3",
    "E07_elastic_0.0005",
)
META_IDS = ("S01_nested_convex", "S02_nested_ridge10")


class ConvexCombiner(RegressorMixin, BaseEstimator):
    """Least-squares log blend constrained to nonnegative weights summing to one."""

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        n = X.shape[1]
        result = minimize(
            lambda w: np.mean((X @ w - y) ** 2),
            np.full(n, 1 / n),
            jac=lambda w: 2 * X.T @ (X @ w - y) / len(y),
            bounds=[(0., 1.)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                         "jac": lambda w: np.ones_like(w)},
            method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success:
            raise RuntimeError(f"Convex meta optimization failed: {result.message}")
        self.coef_ = result.x
        self.intercept_ = 0.
        assert np.min(self.coef_) >= -1e-10
        assert abs(self.coef_.sum() - 1) < 1e-8
        return self

    def predict(self, X):
        return np.asarray(X) @ self.coef_


class StackedLogPredictor(PricePredictor):
    def __init__(self, components, meta_model, name):
        self.components = components
        self.meta_model = meta_model
        self.name = name

    def predict_log(self, X):
        base_predictions = np.column_stack([model.predict(X) for model in self.components])
        return self.meta_model.predict(base_predictions)


def make_base(base_id):
    if base_id.startswith("O"):
        spec = next(s for s in optimization_specs() if s["id"] == base_id)
        return make_optimization_pipeline(spec)
    if base_id.startswith("F"):
        spec = next(s for s in followup_specs() if s["id"] == base_id)
        return followup_pipeline(spec)
    spec = next(s for s in experiment_specs() if s["id"] == base_id)
    return make_pipeline(spec)


def _fit_audited(base_id, X, logs, train_indices, forbidden_indices):
    estimator = make_base(base_id).fit(X.iloc[train_indices], logs[train_indices])
    features = estimator.named_steps["features"]
    fitted_ids = getattr(features, "base_", features).fit_row_ids_
    assert set(fitted_ids) == set(X.iloc[train_indices].Id)
    assert set(fitted_ids).isdisjoint(X.iloc[forbidden_indices].Id)
    return estimator


def _make_metas():
    return {META_IDS[0]: ConvexCombiner(), META_IDS[1]: Ridge(alpha=10.)}


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def run_stacking(data_dir, output_dir):
    start = time.perf_counter()
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    for subdir in ("results", "candidate_models", "submissions", "cache"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    train, test = read_data(data_dir / "train.csv"), read_data(data_dir / "test.csv")
    X = train.drop(columns="SalePrice")
    logs = np.log(train.SalePrice.to_numpy(float))
    assert len(X) == 1460 and len(test) == 1459
    plan = {
        "round": 4,
        "base_experiments": BASE_IDS,
        "outer_cv": {"folds": 5, "shuffle": True, "seed": 42},
        "inner_cv": {"folds": 3, "shuffle": True, "seed": 2026},
        "final_meta_training_cv": {"folds": 3, "shuffle": True, "seed": 2026},
        "meta_rules": {META_IDS[0]: "Nonnegative weights summing to one; minimize inner OOF log MSE",
                       META_IDS[1]: "Ridge alpha=10 with intercept, trained on inner OOF log predictions"},
        "all_1460_outer_validation_rows_retained": True,
        "preprocessing": "Every base model is a full pipeline fitted inside its own training subset.",
        "limitations": "Outer-fold targets are never used to fit their meta model. Base families and hyperparameters were chosen adaptively in earlier development on this same dataset, so this is not an untouched generalization test.",
        "source_sha256": file_sha256(Path(__file__)),
        "train_sha256": file_sha256(data_dir / "train.csv"),
        "test_sha256": file_sha256(data_dir / "test.csv"),
    }
    _json(output_dir / "results/plan.json", plan)
    outer_splits = list(KFold(5, shuffle=True, random_state=42).split(X))
    oof = {name: np.full(len(X), np.nan) for name in META_IDS}
    fold_ids = np.zeros(len(X), dtype=int)
    coefficient_records, score_records, audit = [], [], []
    with threadpool_limits(limits=4):
        for fold, (outer_train, outer_valid) in enumerate(outer_splits, 1):
            t0 = time.perf_counter()
            fold_ids[outer_valid] = fold
            inner_splits = list(KFold(3, shuffle=True, random_state=2026).split(outer_train))
            inner_oof = np.full((len(outer_train), len(BASE_IDS)), np.nan)
            outer_predictions = np.zeros((len(outer_valid), len(BASE_IDS)))
            outer_models = []
            for j, base_id in enumerate(BASE_IDS):
                for inner_fold, (inner_tr, inner_va) in enumerate(inner_splits, 1):
                    tr, va = outer_train[inner_tr], outer_train[inner_va]
                    forbidden = np.concatenate([va, outer_valid])
                    estimator = _fit_audited(base_id, X, logs, tr, forbidden)
                    inner_oof[inner_va, j] = estimator.predict(X.iloc[va])
                    audit.append({"stage": "outer_inner", "outer_fold": fold,
                                  "inner_fold": inner_fold, "base_id": base_id,
                                  "n_train": len(tr), "n_inner_valid": len(va),
                                  "n_outer_valid": len(outer_valid),
                                  "fitted_rows_disjoint_from_both_validation_sets": True})
                estimator = _fit_audited(base_id, X, logs, outer_train, outer_valid)
                outer_predictions[:, j] = estimator.predict(X.iloc[outer_valid])
                outer_models.append(estimator)
                print(f"Outer {fold}/5: {base_id} inner+outer fitted", flush=True)
            assert np.isfinite(inner_oof).all()
            metas = _make_metas()
            for name, meta in metas.items():
                meta.fit(inner_oof, logs[outer_train])
                values = meta.predict(outer_predictions)
                oof[name][outer_valid] = values
                score = float(np.sqrt(np.mean((values - logs[outer_valid]) ** 2)))
                coefficient_records.append({"stage": "outer", "fold": fold, "experiment_id": name,
                                            "weights": meta.coef_.tolist(),
                                            "intercept": float(meta.intercept_),
                                            "meta_training_rows": len(outer_train)})
                score_records.append({"experiment_id": name, "fold": fold,
                                      "log_rmse": score, "n_train": len(outer_train),
                                      "n_valid": len(outer_valid)})
                print(f"{name} fold {fold}: {score:.6f}; weights={meta.coef_.round(4)}", flush=True)
            pd.DataFrame({"Id": X.iloc[outer_train].Id, "log_SalePrice": logs[outer_train],
                          **{f"{name}__inner_log_prediction": inner_oof[:, j]
                             for j, name in enumerate(BASE_IDS)}}).to_csv(
                output_dir / f"cache/outer_{fold}_inner_oof.csv", index=False)
            joblib.dump({"base_ids": BASE_IDS, "models": outer_models, "metas": metas,
                         "train_indices": outer_train, "validation_indices": outer_valid},
                        output_dir / f"cache/outer_{fold}_models.joblib", compress=3)
            pd.DataFrame(score_records).to_csv(output_dir / "results/fold_scores.csv", index=False)
            _json(output_dir / "results/meta_coefficients.json", coefficient_records)
            print(f"Outer {fold} completed in {time.perf_counter()-t0:.1f}s", flush=True)

        for values in oof.values():
            assert np.isfinite(values).all()
        assert (fold_ids > 0).all()
        pd.DataFrame({"Id": train.Id, "fold": fold_ids, "SalePrice": train.SalePrice,
                      "log_SalePrice": logs,
                      **{name + "__log_prediction": values for name, values in oof.items()}}).to_csv(
            output_dir / "results/oof_predictions.csv", index=False)
        pd.DataFrame({"Id": train.Id, "fold": fold_ids}).to_csv(
            output_dir / "results/fold_assignments.csv", index=False)

        # Fit final meta rules on newly computed three-fold predictions. These
        # are fitting data, not the score reported for the complete stack.
        final_inner = np.full((len(X), len(BASE_IDS)), np.nan)
        final_splits = list(KFold(3, shuffle=True, random_state=2026).split(X))
        full_models = []
        for j, base_id in enumerate(BASE_IDS):
            for inner_fold, (tr, va) in enumerate(final_splits, 1):
                estimator = _fit_audited(base_id, X, logs, tr, va)
                final_inner[va, j] = estimator.predict(X.iloc[va])
                audit.append({"stage": "final_inner", "inner_fold": inner_fold,
                              "base_id": base_id, "n_train": len(tr), "n_valid": len(va),
                              "train_validation_disjoint": True})
            full_models.append(make_base(base_id).fit(X, logs))
            print(f"Final stack: {base_id} inner+full fitted", flush=True)
        assert np.isfinite(final_inner).all()
        pd.DataFrame({"Id": train.Id, "log_SalePrice": logs,
                      **{f"{name}__log_prediction": final_inner[:, j]
                         for j, name in enumerate(BASE_IDS)}}).to_csv(
            output_dir / "results/final_meta_training_predictions.csv", index=False)
        rows, reload_checks = [], {}
        for name, meta in _make_metas().items():
            meta.fit(final_inner, logs)
            predictor = StackedLogPredictor(full_models, meta, name)
            predictions = predictor.predict(test)
            assert np.isfinite(predictions).all() and (predictions > 0).all()
            model_path = output_dir / "candidate_models" / (name + ".joblib")
            joblib.dump(predictor, model_path, compress=3)
            difference = float(np.max(np.abs(joblib.load(model_path).predict(test) - predictions)))
            assert difference < 1e-6
            reload_checks[name] = difference
            pd.DataFrame({"Id": test.Id, "SalePrice": predictions}).to_csv(
                output_dir / "submissions" / (name + ".csv"), index=False)
            scores = [record["log_rmse"] for record in score_records if record["experiment_id"] == name]
            coefficient_records.append({"stage": "final", "experiment_id": name,
                                        "weights": meta.coef_.tolist(),
                                        "intercept": float(meta.intercept_), "meta_training_rows": len(X)})
            rows.append({"experiment_id": name, "model": "Nested stacking",
                         "stage": "round4_nested_stacking", "target": "log",
                         "cv_mean": float(np.mean(scores)), "cv_std": float(np.std(scores, ddof=1)),
                         "pooled_oof_log_rmse": float(np.sqrt(np.mean((logs-oof[name])**2))),
                         "params_json": json.dumps({"base_ids": BASE_IDS,
                             "meta_rule": plan["meta_rules"][name], "final_weights": meta.coef_.tolist(),
                             "final_intercept": float(meta.intercept_)})})
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results/experiment_results.csv", index=False)
    _json(output_dir / "results/meta_coefficients.json", coefficient_records)
    _json(output_dir / "results/preprocessing_audit.json", audit)
    best = results.loc[results.cv_mean.idxmin()]
    summary = {"experiment_count": len(results), "best_experiment": best.experiment_id,
               "best_cv_mean": float(best.cv_mean), "best_pooled_oof_log_rmse": float(best.pooled_oof_log_rmse),
               "runtime_seconds": time.perf_counter() - start, "reload_prediction_checks": reload_checks,
               "outer_validation_rows": len(X), "promoted": False,
               "validation_scope": plan["limitations"]}
    _json(output_dir / "results/summary.json", summary)
    return summary
