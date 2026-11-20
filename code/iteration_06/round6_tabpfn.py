"""Four fixed no-Y-transform TabPFN ablations; all-row fold-isolated validation.

Use build/tabpfn-env/bin/python with OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and
MKL_NUM_THREADS unset. --plan-only writes the complete plan before any model fit.
Only official training labels are used. Built with PriorLabs-TabPFN.
"""

import argparse
from collections import Counter
from dataclasses import asdict, fields
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import tabpfn
import torch
from sklearn.model_selection import KFold
from tabpfn.inference_config import InferenceConfig

import round5_tabpfn as base

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "models/round6_tabpfn"
sys.path.insert(0, str(ROOT / "src"))
from house_prices.preprocessing import ALL_GROUPS, deterministic_features

CANDIDATES = [
    {"id": "R6TP_noY_nofingerprint_s42", "feature_recipe": "raw",
     "inference_config": {"REGRESSION_Y_PREPROCESS_TRANSFORMS": [None], "FINGERPRINT_FEATURE": False},
     "hypothesis": "On log targets with no internal Y transform, remove row fingerprints."},
    {"id": "R6TP_noY_subsample090_s42", "feature_recipe": "raw",
     "inference_config": {"REGRESSION_Y_PREPROCESS_TRANSFORMS": [None], "SUBSAMPLE_SAMPLES": 0.9},
     "hypothesis": "Use 90 percent context-row subsampling per inference estimator; retain all validation rows."},
    {"id": "R6TP_noY_domain_s42", "feature_recipe": "domain",
     "inference_config": {"REGRESSION_Y_PREPROCESS_TRANSFORMS": [None]},
     "hypothesis": "Combine no internal Y transform with the existing deterministic domain recipe."},
    {"id": "R6TP_noY_poly50_s42", "feature_recipe": "raw",
     "inference_config": {"REGRESSION_Y_PREPROCESS_TRANSFORMS": [None], "POLYNOMIAL_FEATURES": 50},
     "hypothesis": "Add at most 50 two-factor polynomial features inside fold-fitted preprocessing."},
]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def frozen_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing plan/provenance differs: {path}; use a separate output directory.")
    else:
        write_json(path, value)


def dependency_manifest():
    package = Path(tabpfn.__file__).resolve().parent
    return {
        "project_sources": {str(p.relative_to(ROOT)): base.sha256(p) for p in [
            Path(__file__).resolve(), ROOT / "scripts/round5_tabpfn.py",
            ROOT / "src/house_prices/preprocessing.py", ROOT / "data/source_manifest.json",
            ROOT / "references/tabpfn/remote_file_manifest.json"]},
        "tabpfn_python_sources": {str(p.relative_to(package)): base.sha256(p)
                                  for p in sorted(package.rglob("*.py"))},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("tabpfn", "torch", "numpy", "pandas", "scikit-learn", "scipy", "pydantic")},
    }


def validate_config(candidate):
    # Type/constraint checking only. Model defaults are loaded from checkpoints,
    # never from this empty PREPROCESS_TRANSFORMS validation object.
    override = dict(candidate["inference_config"])
    override["REGRESSION_Y_PREPROCESS_TRANSFORMS"] = tuple(override["REGRESSION_Y_PREPROCESS_TRANSFORMS"])
    checked = InferenceConfig(PREPROCESS_TRANSFORMS=[]).override_with_user_input_and_resolve_auto(override)
    annotations = {f.name: str(f.type) for f in fields(checked)}
    assert checked.REGRESSION_Y_PREPROCESS_TRANSFORMS == (None,)
    if "SUBSAMPLE_SAMPLES" in override:
        assert type(checked.SUBSAMPLE_SAMPLES) is float and 0 < checked.SUBSAMPLE_SAMPLES <= 1
    if "POLYNOMIAL_FEATURES" in override:
        assert type(checked.POLYNOMIAL_FEATURES) is int and checked.POLYNOMIAL_FEATURES > 0
    if "FINGERPRINT_FEATURE" in override:
        assert type(checked.FINGERPRINT_FEATURE) is bool
    return {key: {"annotation": annotations[key], "validated_value": value} for key, value in candidate["inference_config"].items()}


def load_verified_data(input_path):
    source = json.loads((ROOT / "data/source_manifest.json").read_text())
    for filename, path in (("train.csv", ROOT / "data/raw/train.csv"), ("test.csv", input_path)):
        expected = source["files"][filename]
        if base.sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Official input checksum/size mismatch: {path}")
    train, test, raw, raw_test = base.load_data(input_path)
    assert train.shape == (1460, 81) and test.shape == (1459, 80)
    assert train.Id.is_unique and test.Id.is_unique and not set(train.Id) & set(test.Id)
    assert "SalePrice" not in test and np.isfinite(train.SalePrice).all() and (train.SalePrice > 0).all()
    assert raw.shape == (1460, 79) and raw_test.shape == (1459, 79)
    return train, test, raw, raw_test


def prepare_features(raw, raw_test, recipe):
    if recipe == "domain":
        X = deterministic_features(raw, semantic=True, groups=ALL_GROUPS)
        Xt = deterministic_features(raw_test, semantic=True, groups=ALL_GROUPS)
    elif recipe == "raw":
        X, Xt = raw.copy(), raw_test.copy()
    else:
        raise ValueError(f"Unrecognized feature recipe: {recipe}")
    for col in X.select_dtypes(exclude=np.number).columns:
        X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
        Xt[col] = Xt[col].astype(object).where(Xt[col].notna(), np.nan)
    assert len(X) == 1460 and len(Xt) == 1459
    assert X.shape[1] == (98 if recipe == "domain" else 79)
    assert list(X.columns) == list(Xt.columns) and not {"Id", "SalePrice"} & set(X.columns)
    return X, Xt


def make_spec(candidate, checkpoints, input_path, dependencies, feature_columns):
    spec = base.make_spec(candidate, checkpoints, input_path)
    spec.update({
        "target": "log", "random_state": 42, "features": len(feature_columns),
        "source_sha256": base.sha256(__file__), "dependency_manifest": dependencies,
        "feature_columns": list(feature_columns), "feature_groups": list(ALL_GROUPS) if candidate["feature_recipe"] == "domain" else [],
        "semantic_missingness": candidate["feature_recipe"] == "domain",
        "configuration_type_validation": validate_config(candidate),
        "training_rows_removed": 0, "validation_rows": 1460, "id_is_predictor": False,
        "test_labels_used": False, "folds_exhaustive_and_disjoint": True,
        "predictive_summary": "mean", "secondary_summary": "No secondary predictive summary is selected in this experiment.",
        "target_summary": "Natural log of positive SalePrice; internal REGRESSION_Y_PREPROCESS_TRANSFORMS=(None,).",
        "selection_limitation": "Four fixed follow-up candidates evaluated on reused seed42 development folds; not an unbiased generalization estimate. Public score must be verified separately.",
    })
    return spec


def audit_fitted(model, spec, X, label):
    effective = json.loads(json.dumps(asdict(model.inference_config_)))
    for key, value in spec["inference_config"].items():
        if effective[key] != value:
            raise ValueError(f"Requested override was not applied: {key}")
    assert len(model.models_) == 3 and model.n_estimators_ == 16
    assert model.n_features_in_ == spec["features"] == X.shape[1]
    members = model.ensemble_configs_
    assert len(members) == 16 and all(c.target_transform is None for c in members)
    counts = dict(Counter(str(c._model_index) for c in members))
    assert sum(counts.values()) == 16 and len(counts) == 3
    return {"fit": label, "training_rows": len(X), "input_features": X.shape[1],
            "checkpoint_count": len(model.models_), "estimator_count": model.n_estimators_,
            "estimators_per_checkpoint_index": counts, "all_target_transforms_none": True,
            "effective_inference_config": effective}


def release(model):
    del model
    gc.collect()
    torch.mps.empty_cache()


def run_candidate(spec, out, train, test, X, Xt, folds):
    previous = base.validated_existing(spec, out)
    if previous is not None:
        print(f"SKIP verified completed candidate: {spec['id']}", flush=True)
        return previous
    name = spec["id"]
    frozen_json(out / "results" / f"{name}_plan.json", spec)
    logs = np.log(train.SalePrice.to_numpy(float))
    oof, fold_ids = np.full(len(X), np.nan), np.zeros(len(X), int)
    fold_scores, audits = [], []
    start = time.perf_counter()
    for fold, (tr, va) in enumerate(folds, 1):
        began = time.perf_counter()
        assert not set(tr) & set(va) and len(tr) + len(va) == 1460
        model = base.make_model(spec)
        model.fit(X.iloc[tr], logs[tr])
        audits.append(audit_fitted(model, spec, X.iloc[tr], f"fold{fold}"))
        oof[va] = np.asarray(model.predict(X.iloc[va]), float)
        fold_ids[va] = fold
        score = float(np.sqrt(np.mean((oof[va] - logs[va]) ** 2)))
        fold_scores.append({"fold": fold, "log_rmse": score, "n_train": len(tr), "n_valid": len(va),
                            "train_validation_disjoint": True, "seconds": time.perf_counter() - began})
        pd.DataFrame(fold_scores).to_csv(out / "results" / f"{name}_fold_scores.csv", index=False)
        write_json(out / "results" / f"{name}_fitted_config_audit.json", audits)
        print(f"{name} fold {fold}: {score:.8f}; {time.perf_counter()-began:.1f}s", flush=True)
        del model
        gc.collect()
        torch.mps.empty_cache()
    assert np.isfinite(oof).all() and (fold_ids > 0).all()
    cv_mean = float(np.mean([r["log_rmse"] for r in fold_scores]))
    pooled = float(np.sqrt(np.mean((oof - logs) ** 2)))
    print(f"{name}: CV mean {cv_mean:.8f}; pooled {pooled:.8f}; final training-data fit", flush=True)
    pd.DataFrame({"Id": train.Id, "fold": fold_ids, "log_SalePrice": logs, "log_prediction": oof}).to_csv(
        out / "results" / f"{name}_oof.csv", index=False)
    model = base.make_model(spec)
    model.fit(X, logs)
    audits.append(audit_fitted(model, spec, X, "final_training_fit"))
    test_logs = np.asarray(model.predict(Xt), float)
    prices = np.exp(test_logs)
    assert len(prices) == len(test) and np.isfinite(prices).all() and (prices > 0).all()
    pd.DataFrame({"Id": test.Id, "log_prediction": test_logs}).to_csv(out / "results" / f"{name}_test_logs.csv", index=False)
    submission = out / "submissions" / f"{name}.csv"
    pd.DataFrame({"Id": test.Id, "SalePrice": prices}).to_csv(submission, index=False)
    audit_path = out / "results" / f"{name}_fitted_config_audit.json"
    write_json(audit_path, audits)
    artifacts = [out / "results" / f"{name}_{suffix}.csv" for suffix in ("fold_scores", "oof", "test_logs")] + [submission, audit_path]
    result = spec | {"status": "complete", "fold_scores": fold_scores, "cv_mean": cv_mean,
        "cv_std": float(np.std([r["log_rmse"] for r in fold_scores], ddof=1)), "pooled_oof_log_rmse": pooled,
        "runtime_seconds": time.perf_counter() - start,
        "submission_sha256": base.sha256(submission),
        "artifact_sha256": {str(p.relative_to(out)): base.sha256(p) for p in artifacts}}
    write_json(out / "results" / f"{name}_summary.json", result)
    print(f"COMPLETE {name}: mean={cv_mean:.8f}, pooled={pooled:.8f}, {result['runtime_seconds']:.1f}s", flush=True)
    del model
    gc.collect()
    torch.mps.empty_cache()
    return result


def update_comparison(out, specs):
    results = [base.validated_existing(spec, out) for spec in specs]
    results = [result for result in results if result is not None]
    if results:
        keys = ("id", "target", "random_state", "feature_recipe", "cv_mean", "cv_std", "pooled_oof_log_rmse", "runtime_seconds")
        pd.DataFrame([{key: row[key] for key in keys} for row in results]).to_csv(out / "comparison.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", choices=[r["id"] for r in CANDIDATES])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/test.csv")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    base.configure_runtime()
    out = args.output_dir
    for folder in (out, out / "results", out / "submissions"):
        folder.mkdir(parents=True, exist_ok=True)
    train, test, raw, raw_test = load_verified_data(args.input)
    features = {recipe: prepare_features(raw, raw_test, recipe) for recipe in ("raw", "domain")}
    checkpoints, dependencies = base.checkpoint_manifest(), dependency_manifest()
    specs = [make_spec(candidate, checkpoints, args.input, dependencies, features[candidate["feature_recipe"]][0].columns) for candidate in CANDIDATES]
    folds = list(KFold(5, shuffle=True, random_state=42).split(raw))
    assert sorted(np.concatenate([va for _, va in folds]).tolist()) == list(range(1460))
    fold_manifest = [{"fold": i, "train_ids": train.Id.iloc[tr].tolist(), "validation_ids": train.Id.iloc[va].tolist()}
                     for i, (tr, va) in enumerate(folds, 1)]
    frozen_json(out / "preregistered_plan.json", specs)
    frozen_json(out / "fold_manifest.json", fold_manifest)
    frozen_json(out / "dependency_manifest.json", dependencies)
    frozen_json(out / "plan_registration.json", {"plan_sha256": base.sha256(out / "preregistered_plan.json"),
        "fold_manifest_sha256": base.sha256(out / "fold_manifest.json"),
        "dependency_manifest_sha256": base.sha256(out / "dependency_manifest.json"),
        "candidate_ids": [s["id"] for s in specs], "candidate_count": 4, "fold_seed": 42,
        "all_1460_rows_validated_once": True, "no_test_labels": True, "no_id_predictor": True})
    if not (out / "plan_registration_record.json").exists():
        write_json(out / "plan_registration_record.json", {"status": "planned_before_fit"})
    if args.plan_only:
        print(json.dumps({"status": "planned_not_executed", "candidate_count": len(specs), "output": str(out)}), flush=True)
        return
    for spec in specs:
        if args.candidate and spec["id"] not in args.candidate:
            continue
        X, Xt = features[spec["feature_recipe"]]
        try:
            run_candidate(spec, out, train, test, X, Xt, folds)
        except Exception as exc:
            write_json(out / "results" / f"{spec['id']}_failure.json", {"id": spec["id"],
                "error_type": type(exc).__name__, "error": str(exc), "source_sha256": base.sha256(__file__)})
            raise
        update_comparison(out, specs)


if __name__ == "__main__":
    main()
