"""Controlled TabPFN-v2 inference ablations on the frozen Ames folds.

Run with the recorded environment, for example::

    env -u OMP_NUM_THREADS -u OPENBLAS_NUM_THREADS -u MKL_NUM_THREADS \
      build/tabpfn-env/bin/python scripts/round5_tabpfn.py

Only official training labels are used. The eight configurations are declared
before training. Model preprocessing is fitted within each training fold.
Built with PriorLabs-TabPFN.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "models/round5_tabpfn"
FILES = ["tabpfn-v2-regressor.ckpt", "tabpfn-v2-regressor-09gpqh39.ckpt",
         "tabpfn-v2-regressor-2noar4o2.ckpt"]
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
CANDIDATES = [
    {"id": "R5TP_log_s17", "target": "log", "random_state": 17,
     "hypothesis": "Changing the inference ensemble seed measures seed sensitivity and supplies an independent ensemble member."},
    {"id": "R5TP_log_s2026", "target": "log", "random_state": 2026,
     "hypothesis": "A third predetermined inference seed supports a fixed equal-weight seed ensemble."},
    {"id": "R5TP_raw_s42", "target": "raw", "random_state": 42,
     "hypothesis": "Raw-scale targets supply a controlled comparison to the metric-aligned log target."},
    {"id": "R5TP_boxcox015_s42", "target": "boxcox", "target_power": 0.15,
     "random_state": 42,
     "hypothesis": "A mild fixed Box-Cox power interpolates log and raw target scales without learning a transform from validation labels."},
    {"id": "R5TP_boxcox050_s42", "target": "boxcox", "target_power": 0.5,
     "random_state": 42,
     "hypothesis": "The square-root member tests a larger target-scale change in the same controlled family."},
    {"id": "R5TP_log_nofingerprint_s42", "target": "log", "random_state": 42,
     "inference_config": {"FINGERPRINT_FEATURE": False},
     "hypothesis": "Disable the synthetic row fingerprint to test whether prediction depends on an uninformative identifier-like feature."},
    {"id": "R5TP_log_noYtransform_s42", "target": "log", "random_state": 42,
     "inference_config": {"REGRESSION_Y_PREPROCESS_TRANSFORMS": [None]},
     "hypothesis": "On already logged prices, remove the additional internal target power transform to isolate its contribution."},
    {"id": "R5TP_log_safepower_s42", "target": "log", "random_state": 42,
     "inference_config": {"REGRESSION_Y_PREPROCESS_TRANSFORMS": ["safepower"]},
     "hypothesis": "Complement the no-transform ablation with the internal safe-power transform alone."},
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def configure_runtime():
    active = {name: os.environ[name] for name in THREAD_VARIABLES if name in os.environ}
    if active:
        raise RuntimeError(f"Unset these variables to match the recorded baseline: {active}")
    torch.set_num_threads(12)
    if not torch.backends.mps.is_available():
        raise RuntimeError("The preregistered run requires the recorded Apple MPS backend.")


def checkpoint_manifest():
    official = json.loads((ROOT / "references/tabpfn/remote_file_manifest.json").read_text())
    result = []
    for filename in FILES:
        entry = next(row for row in official if row["path"] == filename)
        path = ROOT / "references/tabpfn" / filename
        digest = sha256(path)
        assert digest == entry["lfs"]["oid"] and path.stat().st_size == entry["size"]
        result.append({"filename": filename, "bytes": path.stat().st_size,
                       "sha256": digest,
                       "url": "https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/" + filename})
    return result


def load_data(input_path):
    train = pd.read_csv(ROOT / "data/raw/train.csv", keep_default_na=False, na_values=["NA", ""])
    test = pd.read_csv(input_path, keep_default_na=False, na_values=["NA", ""])
    X = train.drop(columns=["Id", "SalePrice"])
    Xt = test.drop(columns="Id")
    assert list(X.columns) == list(Xt.columns)
    for col in X.select_dtypes(exclude=np.number).columns:
        X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
        Xt[col] = Xt[col].astype(object).where(Xt[col].notna(), np.nan)
    return train, test, X, Xt


def transform_y(prices, spec):
    prices = np.asarray(prices, float)
    if spec["target"] == "log":
        return np.log(prices)
    if spec["target"] == "raw":
        return prices
    # A fixed $100,000 scale makes the power numerically well conditioned. It
    # is not estimated from either training or validation target statistics.
    lam = spec["target_power"]
    return np.expm1(lam * np.log(prices / spec["target_scale"])) / lam


def to_log_prices(prediction, spec):
    values = np.asarray(prediction, float)
    if spec["target"] == "log":
        return values
    if spec["target"] == "raw":
        if not (values > 0).all():
            raise ValueError("Nonpositive raw-scale predictions cannot be scored on the log metric.")
        return np.log(values)
    base = 1 + spec["target_power"] * values
    if not (base > 0).all():
        raise ValueError("Box-Cox predictions fall outside the inverse transform domain.")
    return np.log(base) / spec["target_power"] + np.log(spec["target_scale"])


def make_model(spec):
    paths = []
    for entry in spec["checkpoint_manifest"]:
        path = ROOT / "references/tabpfn" / entry["filename"]
        assert sha256(path) == entry["sha256"]
        paths.append(path)
    inference_config = dict(spec.get("inference_config", {}))
    if "REGRESSION_Y_PREPROCESS_TRANSFORMS" in inference_config:
        inference_config["REGRESSION_Y_PREPROCESS_TRANSFORMS"] = tuple(
            inference_config["REGRESSION_Y_PREPROCESS_TRANSFORMS"])
    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, model_path=paths, n_estimators=spec["n_estimators"],
        device=spec["device"], random_state=spec["random_state"],
        n_preprocessing_jobs=spec["n_preprocessing_jobs"],
        inference_config=inference_config or None)


def make_spec(candidate, checkpoints, input_path):
    return {
        "model": "TabPFN-v2 three-checkpoint ensemble", "n_estimators": 16,
        "device": "mps", "n_preprocessing_jobs": 2, "torch_num_threads": 12,
        "fold_seed": 42, "n_splits": 5, "rows": 1460, "features": 79,
        "feature_recipe": "raw", "filter_train": False, "target_scale": 100000.0,
        "inference_config": {}, "checkpoint_manifest": checkpoints,
        "library_version": importlib.metadata.version("tabpfn"),
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "platform": platform.platform(), "source_sha256": sha256(__file__),
        "train_sha256": sha256(ROOT / "data/raw/train.csv"),
        "test_sha256": sha256(input_path),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "target_summary": "predict default mean in the chosen transformed target space",
        "predictive_summary": "mean",
        "secondary_summary": "Median and 50/50 mean-median log predictions are saved as diagnostic comparisons from the same forward pass.",
        "preprocessing": "A fresh TabPFN model fits internal preprocessing on each training fold only.",
        "selection_limitation": "Frozen development folds reused after round4; candidate selection is not an unbiased generalization estimate.",
        "source": "https://github.com/PriorLabs/TabPFN",
        "attribution": "Built with PriorLabs-TabPFN.",
    } | candidate


def validated_existing(spec, out):
    summary_path = out / "results" / f"{spec['id']}_summary.json"
    if not summary_path.exists():
        return None
    previous = json.loads(summary_path.read_text())
    if previous.get("status") != "complete":
        return None
    for key, value in spec.items():
        if previous.get(key) != value:
            raise ValueError(f"Existing {spec['id']} differs at {key}; choose a separate output directory.")
    for relative, digest in previous["artifact_sha256"].items():
        if sha256(out / relative) != digest:
            raise ValueError(f"Existing artifact hash mismatch: {relative}")
    return previous


def run_candidate(spec, out, train, test, X, Xt, folds):
    previous = validated_existing(spec, out)
    if previous is not None:
        print(f"SKIP verified completed candidate: {spec['id']}", flush=True)
        return previous
    name = spec["id"]
    write_json(out / "results" / f"{name}_plan.json", spec)
    logs = np.log(train.SalePrice.to_numpy(float))
    target = transform_y(train.SalePrice.to_numpy(float), spec)
    oof = np.full(len(X), np.nan)
    median_oof = np.full(len(X), np.nan)
    fold_ids = np.zeros(len(X), int)
    fold_scores = []
    start = time.perf_counter()
    for fold, (tr, va) in enumerate(folds, 1):
        began = time.perf_counter()
        assert not set(tr) & set(va)
        model = make_model(spec)
        model.fit(X.iloc[tr], target[tr])
        outputs = model.predict(X.iloc[va], output_type="main")
        oof[va] = to_log_prices(outputs["mean"], spec)
        median_oof[va] = to_log_prices(outputs["median"], spec)
        fold_ids[va] = fold
        score = float(np.sqrt(np.mean((oof[va] - logs[va]) ** 2)))
        fold_scores.append({"fold": fold, "log_rmse": score, "n_train": len(tr),
                            "n_valid": len(va), "train_validation_disjoint": True,
                            "seconds": time.perf_counter() - began})
        pd.DataFrame(fold_scores).to_csv(out / "results" / f"{name}_fold_scores.csv", index=False)
        print(f"{name} fold {fold}: {score:.8f}; {time.perf_counter()-began:.1f}s", flush=True)
        del model
        gc.collect()
        torch.mps.empty_cache()
    assert np.isfinite(oof).all() and (fold_ids > 0).all()
    cv_mean = float(np.mean([row["log_rmse"] for row in fold_scores]))
    pooled = float(np.sqrt(np.mean((oof - logs) ** 2)))
    print(f"{name}: CV mean {cv_mean:.8f}; pooled {pooled:.8f}; full-data fit", flush=True)
    pd.DataFrame({"Id": train.Id, "fold": fold_ids, "log_SalePrice": logs,
                  "log_prediction": oof, "log_prediction_median": median_oof,
                  "log_prediction_mean_median": 0.5 * (oof + median_oof)}).to_csv(
        out / "results" / f"{name}_oof.csv", index=False)
    model = make_model(spec)
    model.fit(X, target)
    test_outputs = model.predict(Xt, output_type="main")
    test_logs = to_log_prices(test_outputs["mean"], spec)
    median_test_logs = to_log_prices(test_outputs["median"], spec)
    prices = np.exp(test_logs)
    assert len(prices) == len(test) and np.isfinite(prices).all() and (prices > 0).all()
    pd.DataFrame({"Id": test.Id, "log_prediction": test_logs,
                  "log_prediction_median": median_test_logs,
                  "log_prediction_mean_median": 0.5 * (test_logs + median_test_logs)}).to_csv(
        out / "results" / f"{name}_test_logs.csv", index=False)
    submission = out / "submissions" / f"{name}.csv"
    pd.DataFrame({"Id": test.Id, "SalePrice": prices}).to_csv(submission, index=False)
    artifacts = [out / "results" / f"{name}_{suffix}.csv"
                 for suffix in ("fold_scores", "oof", "test_logs")] + [submission]
    result = spec | {"status": "complete", "fold_scores": fold_scores, "cv_mean": cv_mean,
                     "cv_std": float(np.std([row["log_rmse"] for row in fold_scores], ddof=1)),
                     "pooled_oof_log_rmse": pooled,
                     "runtime_seconds": time.perf_counter() - start,
                     "submission_sha256": sha256(submission),
                     "artifact_sha256": {str(path.relative_to(out)): sha256(path) for path in artifacts}}
    result["secondary_metrics"] = {}
    for label, values in (("median", median_oof), ("mean_median", 0.5 * (oof + median_oof))):
        scores = [float(np.sqrt(np.mean((values[va] - logs[va]) ** 2))) for _, va in folds]
        result["secondary_metrics"][label] = {"cv_mean": float(np.mean(scores)),
            "cv_std": float(np.std(scores, ddof=1)),
            "pooled_oof_log_rmse": float(np.sqrt(np.mean((values - logs) ** 2)))}
    write_json(out / "results" / f"{name}_summary.json", result)
    print(f"COMPLETE {name}: mean={cv_mean:.8f}, pooled={pooled:.8f}, {result['runtime_seconds']:.1f}s", flush=True)
    del model
    gc.collect()
    torch.mps.empty_cache()
    return result




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", choices=[row["id"] for row in CANDIDATES])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/test.csv")
    args = parser.parse_args()
    configure_runtime()
    out = args.output_dir
    for directory in (out, out / "results", out / "submissions"):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoints = checkpoint_manifest()
    specifications = [make_spec(row, checkpoints, args.input) for row in CANDIDATES]
    plan_path = out / "preregistered_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != specifications:
        raise ValueError("The existing preregistered plan differs; choose a separate output directory.")
    write_json(plan_path, specifications)
    train, test, X, Xt = load_data(args.input)
    assert X.shape == (1460, 79) and Xt.shape == (1459, 79)
    folds = list(KFold(5, shuffle=True, random_state=42).split(X))
    fold_manifest = [{"fold": i, "train_ids": train.Id.iloc[tr].tolist(),
                      "validation_ids": train.Id.iloc[va].tolist()}
                     for i, (tr, va) in enumerate(folds, 1)]
    write_json(out / "fold_manifest.json", fold_manifest)
    results = []
    for spec in specifications:
        if args.candidate and spec["id"] not in args.candidate:
            continue
        results.append(run_candidate(spec, out, train, test, X, Xt, folds))
        pd.DataFrame([{key: row[key] for key in ("id", "target", "random_state", "cv_mean",
                         "cv_std", "pooled_oof_log_rmse", "runtime_seconds")}
                      for row in results]).to_csv(out / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
