"""Shared split experiment runner, honest metrics, artifacts, and audit records."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from .models import BLEND_COMPONENTS, BLEND_ID, PricePredictor, experiment_specs, make_pipeline
from .preprocessing import ALL_GROUPS, deterministic_features, read_data


def log_rmse(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if np.any(y <= 0) or np.any(prediction <= 0):
        raise ValueError("Log-RMSE requires strictly positive targets and predictions.")
    return float(np.sqrt(np.mean((np.log(y) - np.log(prediction)) ** 2)))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_experiments(data_dir, output_dir, feature_output_dir=None):
    started = time.perf_counter()
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    results_dir, artifacts_dir, submissions_dir = [output_dir / d for d in ["results", "artifacts", "submissions"]]
    for directory in [results_dir, artifacts_dir, submissions_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    raw_train, raw_test = read_data(data_dir / "train.csv"), read_data(data_dir / "test.csv")
    if len(raw_train) != 1460 or len(raw_test) != 1459:
        raise ValueError("Expected the canonical 1460 training and 1459 competition-test rows.")
    if raw_train.drop(columns="SalePrice").columns.tolist() != raw_test.columns.tolist():
        raise ValueError("Raw train/test feature columns do not match.")
    if "SalePrice" in raw_test or not raw_train.Id.is_unique or not raw_test.Id.is_unique:
        raise ValueError("Invalid target boundary or row identifiers.")
    y = raw_train.SalePrice.to_numpy(dtype=float)
    y_log = np.log(y)
    # A handoff is deterministic and may be exported before folds exist.
    X = deterministic_features(raw_train.drop(columns="SalePrice"), semantic=False, groups=ALL_GROUPS)
    X_test = deterministic_features(raw_test, semantic=False, groups=ALL_GROUPS)
    if feature_output_dir is not None:
        feature_output_dir = Path(feature_output_dir)
        feature_output_dir.mkdir(parents=True, exist_ok=True)
        X.assign(SalePrice=y).to_csv(feature_output_dir / "train_processed.csv", index=False)
        X_test.to_csv(feature_output_dir / "test_processed.csv", index=False)
        # Train using the exported feature handoff, not a parallel variant.
        handed_train = read_data(feature_output_dir / "train_processed.csv")
        X = handed_train.drop(columns="SalePrice")
        X_test = read_data(feature_output_dir / "test_processed.csv")
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(splitter.split(X))
    fold_ids = np.zeros(len(X), dtype=int)
    for fold, (_, valid) in enumerate(splits, 1):
        fold_ids[valid] = fold
    pd.DataFrame({"Id": raw_train.Id, "fold": fold_ids}).to_csv(results_dir / "fold_assignments.csv", index=False)
    specs = experiment_specs()
    (results_dir / "experiment_plan.json").write_text(json.dumps({"experiments": specs, "blend": {"id": BLEND_ID, "components": BLEND_COMPONENTS, "weights": [1/3]*3}, "fixed_before_first_run": True}, indent=2))
    records, fold_records, audit = [], [], []
    oof_logs = {}
    with threadpool_limits(limits=4):
        for spec in specs:
            exp_start = time.perf_counter()
            pred_logs = np.empty(len(X))
            scores = []
            clipped_count = 0
            for fold, (train_idx, valid_idx) in enumerate(splits, 1):
                pipeline = make_pipeline(spec)
                pipeline.fit(X.iloc[train_idx], y_log[train_idx] if spec["target"] == "log" else y[train_idx])
                predictions = pipeline.predict(X.iloc[valid_idx])
                if spec["target"] == "raw":
                    clipped_count += int(np.sum(predictions < 1.0))
                    predictions = np.log(np.maximum(predictions, 1.0))
                pred_logs[valid_idx] = predictions
                score = float(np.sqrt(np.mean((y_log[valid_idx] - predictions)**2)))
                scores.append(score)
                prep = pipeline.named_steps["preprocessor"]
                fitted_ids = pipeline.named_steps["features"].fit_row_ids_
                assert set(fitted_ids).isdisjoint(X.iloc[valid_idx].Id.tolist())
                fold_records.append({"experiment_id": spec["id"], "fold": fold, "log_rmse": score, "n_train": len(train_idx), "n_valid": len(valid_idx)})
                audit.append({"experiment_id": spec["id"], "fold": fold, "train_id_sha256": hashlib.sha256(json.dumps(fitted_ids).encode()).hexdigest(), "training_rows": len(fitted_ids), "validation_rows": len(valid_idx), "train_validation_disjoint": True, "numeric_features": len(prep.named_transformers_["num"].named_steps["imputer"].statistics_), "encoded_features": int(prep.transform(X.iloc[train_idx].pipe(pipeline.named_steps["features"].transform)).shape[1]), "skew_columns": prep.named_transformers_["num"].named_steps["skew"].selected_columns_})
            oof_logs[spec["id"]] = pred_logs
            record = {"experiment_id": spec["id"], "model": spec["family"], "stage": spec["stage"], "target": spec["target"], "semantic_missing": spec["semantic"], "feature_groups": ",".join(spec["groups"]), "skew_transform": spec["skew"], "params_json": json.dumps(spec["params"], sort_keys=True), "cv_mean": float(np.mean(scores)), "cv_std": float(np.std(scores, ddof=1)), "pooled_oof_log_rmse": float(np.sqrt(np.mean((pred_logs-y_log)**2))), "runtime_seconds": time.perf_counter() - exp_start, "raw_predictions_clipped": clipped_count}
            records.append(record)
            pd.DataFrame(records).to_csv(results_dir / "experiment_results.csv", index=False)
            pd.DataFrame(fold_records).to_csv(results_dir / "fold_scores.csv", index=False)
            print(f"{spec['id']}: {record['cv_mean']:.6f} +/- {record['cv_std']:.6f} ({record['runtime_seconds']:.1f}s)", flush=True)

    blend_logs = np.mean([oof_logs[c] for c in BLEND_COMPONENTS], axis=0)
    blend_scores = [float(np.sqrt(np.mean((blend_logs[v]-y_log[v])**2))) for _,v in splits]
    oof_logs[BLEND_ID] = blend_logs
    for fold, score in enumerate(blend_scores, 1):
        fold_records.append({"experiment_id": BLEND_ID, "fold": fold, "log_rmse": score, "n_train": 1168, "n_valid": 292})
    records.append({"experiment_id": BLEND_ID, "model": "Equal log-space blend", "stage": "ensemble", "target": "log", "semantic_missing": True, "feature_groups": ",".join(ALL_GROUPS), "skew_transform": "component-specific", "params_json": json.dumps({"components": BLEND_COMPONENTS, "weights": [1/3]*3}), "cv_mean": float(np.mean(blend_scores)), "cv_std": float(np.std(blend_scores, ddof=1)), "pooled_oof_log_rmse": float(np.sqrt(np.mean((blend_logs-y_log)**2))), "runtime_seconds": 0.0, "raw_predictions_clipped": 0})
    results = pd.DataFrame(records)
    baseline = results.loc[results.experiment_id == "E01_raw_ridge"].iloc[0]
    results["baseline_absolute_improvement"] = baseline.cv_mean - results.cv_mean
    results["baseline_relative_improvement_pct"] = 100 * results.baseline_absolute_improvement / baseline.cv_mean
    results.to_csv(results_dir / "experiment_results.csv", index=False)
    results.to_csv(output_dir / "model_results.csv", index=False)
    pd.DataFrame(fold_records).to_csv(results_dir / "fold_scores.csv", index=False)
    pd.DataFrame({"Id": raw_train.Id, "fold": fold_ids, "SalePrice": y, "log_SalePrice": y_log, **{k+"__log_prediction": v for k,v in oof_logs.items()}}).to_csv(results_dir / "oof_predictions.csv", index=False)
    (results_dir / "preprocessing_audit.json").write_text(json.dumps(audit, indent=2))
    full = results.loc[results.experiment_id == "E05_skew"].iloc[0]
    ablations = results[results.stage == "group_ablation"].copy()
    ablations["removed_group"] = ablations.experiment_id.str.replace("A_without_", "", regex=False)
    ablations["full_model_cv_mean"] = full.cv_mean
    ablations["removal_penalty"] = ablations.cv_mean - full.cv_mean
    for row in ablations.itertuples():
        scores_full = np.array([r["log_rmse"] for r in fold_records if r["experiment_id"] == "E05_skew"])
        scores_without = np.array([r["log_rmse"] for r in fold_records if r["experiment_id"] == row.experiment_id])
        ablations.loc[row.Index, "paired_penalty_std"] = np.std(scores_without - scores_full, ddof=1)
        ablations.loc[row.Index, "folds_helped"] = np.sum(scores_without > scores_full)
    ablations.to_csv(results_dir / "ablation_results.csv", index=False)
    pathway = results.iloc[:5].copy()
    pathway["previous_cv_mean"] = pathway.cv_mean.shift(1)
    pathway["improvement_over_previous"] = pathway.previous_cv_mean - pathway.cv_mean
    pathway.to_csv(results_dir / "pathway_improvements.csv", index=False)
    # Feature ablations diagnose groups; they are deliberately not final-model candidates.
    eligible = results[results.stage != "group_ablation"]
    best = eligible.loc[eligible.cv_mean.idxmin()]
    best_id = best.experiment_id
    spec_by_id = {s["id"]: s for s in specs}
    component_ids = list(BLEND_COMPONENTS) if best_id == BLEND_ID else [best_id]
    full_fitted = {}
    with threadpool_limits(limits=4):
        for exp_id in sorted(set(component_ids + ["E01_raw_ridge"])):
            spec = spec_by_id[exp_id]
            model = make_pipeline(spec)
            model.fit(X, y_log if spec["target"] == "log" else y)
            full_fitted[exp_id] = (model, spec["target"])
    predictor = PricePredictor([full_fitted[k] for k in component_ids], best_id)
    baseline_predictor = PricePredictor([full_fitted["E01_raw_ridge"]], "E01_raw_ridge")
    joblib.dump(predictor, artifacts_dir / "best_model.joblib", compress=3)
    joblib.dump(baseline_predictor, artifacts_dir / "baseline_model.joblib", compress=3)
    final_predictions = predictor.predict(X_test)
    for filename, values in [("final_submission.csv", final_predictions), ("baseline_submission.csv", baseline_predictor.predict(X_test))]:
        submission = pd.DataFrame({"Id": raw_test.Id, "SalePrice": values})
        assert submission.shape == (1459, 2) and np.isfinite(values).all() and (values>0).all()
        submission.to_csv(submissions_dir / filename, index=False)
        submission.to_csv(output_dir / filename, index=False)
    restored = joblib.load(artifacts_dir / "best_model.joblib")
    serialization_max_abs_diff = float(np.max(np.abs(restored.predict(raw_test) - final_predictions)))
    assert serialization_max_abs_diff < 1e-6
    largest_step = pathway.iloc[1:].loc[pathway.iloc[1:].improvement_over_previous.idxmax()]
    helpful = ablations.loc[ablations.removal_penalty.idxmax()]
    summary = {
        "metric": "sqrt(mean((log(y_true) - log(y_pred))**2)); natural logarithm, not log1p",
        "validation": {"type": "KFold", "n_splits": 5, "shuffle": True, "random_state": 42, "train_rows_per_fold": 1168, "validation_rows_per_fold": 292, "reported_cv_std_ddof": 1, "all_training_rows_retained": True, "selection_limitation": "The same five development folds select features/hyperparameters/model. Reported CV is a development estimate and may be optimistic after selection; no independent held-out test or Kaggle score has been measured."},
        "data": {"training_rows": len(X), "competition_test_rows": len(X_test), "raw_predictors_excluding_id": len(raw_test.columns)-1, "train_sha256": file_sha256(data_dir / "train.csv"), "test_sha256": file_sha256(data_dir / "test.csv"), "read_csv": {"keep_default_na": False, "na_values": ["NA", ""]}},
        "experiment_count": len(results),
        "best_experiment": best_id,
        "best_model": best.model,
        "best_cv_mean": float(best.cv_mean),
        "best_cv_std": float(best.cv_std),
        "best_pooled_oof_log_rmse": float(best.pooled_oof_log_rmse),
        "best_components": component_ids,
        "best_parameters": json.loads(best.params_json),
        "baseline": {"experiment_id": "E01_raw_ridge", "cv_mean": float(baseline.cv_mean), "cv_std": float(baseline.cv_std), "pooled_oof_log_rmse": float(baseline.pooled_oof_log_rmse), "raw_predictions_clipped": int(baseline.raw_predictions_clipped)},
        "absolute_improvement_over_baseline": float(baseline.cv_mean-best.cv_mean),
        "relative_improvement_over_baseline_pct": float(100*(baseline.cv_mean-best.cv_mean)/baseline.cv_mean),
        "largest_pathway_improvement": {"to_experiment": largest_step.experiment_id, "from_cv_mean": float(largest_step.previous_cv_mean), "to_cv_mean": float(largest_step.cv_mean), "absolute_improvement": float(largest_step.improvement_over_previous), "scope": "Largest consecutive local CV decrease within E01-E05; not a Kaggle submission improvement."},
        "most_helpful_group": {"group": helpful.removed_group, "full_cv_mean": float(full.cv_mean), "without_group_cv_mean": float(helpful.cv_mean), "removal_penalty": float(helpful.removal_penalty), "folds_helped": int(helpful.folds_helped), "scope": "Largest positive paired removal penalty among five predefined groups with E05 Ridge alpha20; model-specific development evidence, not causal or statistical proof."},
        "serialization_max_absolute_prediction_difference": serialization_max_abs_diff,
        "test_prediction_summary": {"min": float(final_predictions.min()), "median": float(np.median(final_predictions)), "max": float(final_predictions.max())},
        "kaggle": {"submitted": False, "team_name": "", "baseline_score": None, "final_score": None, "rank": None},
        "runtime_seconds": time.perf_counter()-started,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib"]}},
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"best_experiment": best_id, "cv_mean": best.cv_mean, "runtime_seconds": summary["runtime_seconds"]}, indent=2), flush=True)
    return summary
