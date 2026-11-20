"""Fixed TabICLv2 comparison with original five-fold validation and test predictions.

Dependencies are installed in the active Python environment.
No external sale-price labels are used.
--plan-only reports the configuration; --check-api inspects the estimator API.
"""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import psutil
import torch
import tabicl
from sklearn.model_selection import KFold
from tabicl import TabICLRegressor

BASE = ROOT / "models/round6_tabicl"
CHECKPOINT_NAME = "tabicl-regressor-v2-20260212.ckpt"
CHECKPOINT = ROOT / "build/tabicl-checkpoints" / CHECKPOINT_NAME
CHECKPOINT_SHA256 = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"
CHECKPOINT_BYTES = 114324594
CHECKPOINT_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
ANCHOR = "R5E08_noY_domain20"
MODEL_ID = "R6ICL_v2_raw79_log_n8_s42"
BLEND_ID = "R6ICL_anchor80_tabicl20"
DEPENDENCIES = ("tabicl", "psutil", "torch", "scikit-learn", "numpy", "scipy",
                "einops", "tqdm", "huggingface-hub", "pandas")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def local_path(path):
    return str(Path(path).resolve().relative_to(ROOT))


def verify_dependencies():
    versions = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
    if versions["tabicl"] != "2.2.0" or versions["psutil"] != "7.2.2":
        raise RuntimeError("This experiment requires tabicl==2.2.0 and psutil==7.2.2.")
    return versions


def verify_checkpoint():
    if CHECKPOINT.stat().st_size != CHECKPOINT_BYTES or sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("The local checkpoint does not match the official pinned file.")
    return {
        "local_path": local_path(CHECKPOINT), "filename": CHECKPOINT_NAME,
        "repository": "jingang/TabICL", "revision": CHECKPOINT_REVISION,
        "url": f"https://huggingface.co/jingang/TabICL/resolve/{CHECKPOINT_REVISION}/{CHECKPOINT_NAME}",
        "sha256": CHECKPOINT_SHA256, "bytes": CHECKPOINT_BYTES,
        "license": "BSD-3-Clause", "license_source": "https://huggingface.co/jingang/TabICL",
    }


def load_data(input_path):
    train = pd.read_csv(ROOT / "data/raw/train.csv", keep_default_na=False, na_values=["NA", ""])
    test = pd.read_csv(input_path, keep_default_na=False, na_values=["NA", ""])
    X, Xt = train.drop(columns=["Id", "SalePrice"]), test.drop(columns="Id")
    if X.shape != (1460, 79) or list(X.columns) != list(Xt.columns):
        raise ValueError("The fixed experiment requires the original 1,460 rows and 79 predictors.")
    if train.Id.duplicated().any() or test.Id.duplicated().any():
        raise ValueError("Identifiers must be unique.")
    for column in X.select_dtypes(exclude=np.number).columns:
        X[column] = X[column].astype(object).where(X[column].notna(), np.nan)
        Xt[column] = Xt[column].astype(object).where(Xt[column].notna(), np.nan)
    y = np.log(train.SalePrice.to_numpy(float))
    if not np.isfinite(y).all():
        raise ValueError("Training log prices must be finite.")
    return train, test, X, Xt, y


def model_parameters():
    return {
        "n_estimators": 8, "norm_methods": ["none", "power"],
        "feat_shuffle_method": "latin", "outlier_threshold": 4.0,
        "batch_size": 8, "kv_cache": False, "model_path": str(CHECKPOINT),
        "allow_auto_download": False, "checkpoint_version": CHECKPOINT_NAME,
        "device": "mps", "use_amp": False, "use_fa3": False,
        "offload_mode": "auto", "random_state": 42, "n_jobs": 12,
        "verbose": False,
    }


def make_model():
    return TabICLRegressor(**model_parameters())


def configure_runtime():
    active = {name: os.environ[name] for name in THREAD_VARS if name in os.environ}
    if active:
        raise RuntimeError(f"Unset thread overrides to retain the declared runtime: {active}")
    torch.set_num_threads(12)
    if not torch.backends.mps.is_available():
        raise RuntimeError("This preregistered comparison requires Apple MPS.")


def cleanup_memory():
    gc.collect()
    torch.mps.empty_cache()


def anchor_paths(fold_seed):
    directory = ROOT / "models/round5_ensembles/extension"
    if fold_seed != 42:
        raise ValueError("The original comparison uses fold seed 42.")
    oof = directory / "results" / f"{ANCHOR}_oof.csv"
    return oof, directory / "results" / f"{ANCHOR}_test_logs.csv"


def make_spec(fold_seed, input_path):
    oof_path, test_path = anchor_paths(fold_seed)
    params = model_parameters()
    params["model_path"] = local_path(CHECKPOINT)
    return {
        "id": MODEL_ID, "model": "TabICLv2", "target": "natural-log SalePrice",
        "prediction_summary": "mean", "feature_recipe": "79 original predictors, excluding Id",
        "model_parameters": params, "torch_num_threads": 12,
        "training_rows": 1460, "n_splits": 5, "fold_seed": fold_seed,
        "checkpoint": verify_checkpoint(), "versions": verify_dependencies(),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "source_sha256": sha256(__file__),
        "input_sha256": {"data/raw/train.csv": sha256(ROOT / "data/raw/train.csv"),
                          local_path(input_path): sha256(input_path)},
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARS},
        "external_labels_used": False, "preprocessing_fit_within_fold": True,
        "validation_rows_removed": False, "training_rows_removed": False,
        "fixed_blend": {"id": BLEND_ID, "space": "natural-log price",
                        "weights": {ANCHOR: 0.8, MODEL_ID: 0.2},
                        "anchor_sources": {local_path(p): sha256(p) for p in (oof_path, test_path)}},
        "selection_limitation": "Model selection uses the seed-42 development folds, not an untouched holdout evaluation.",
        "source_notebook": "https://www.kaggle.com/code/carlmcbrideellis/house-prices-using-tabicl-v2",
        "implementation_source": "https://github.com/soda-inria/tabicl",
    }


def read_anchor(fold_seed, train, test, expected_folds):
    oof_path, test_path = anchor_paths(fold_seed)
    frame, test_frame = pd.read_csv(oof_path), pd.read_csv(test_path)
    prediction_column = "log_prediction"
    if not np.array_equal(frame.Id, train.Id) or not np.array_equal(test_frame.Id, test.Id):
        raise ValueError("Anchor IDs differ from the original row order.")
    if not np.array_equal(frame.fold, expected_folds):
        raise ValueError("Anchor and TabICL fold assignments differ.")
    if not np.allclose(frame.log_SalePrice, np.log(train.SalePrice), rtol=0, atol=1e-12):
        raise ValueError("Anchor target values differ.")
    return frame[prediction_column].to_numpy(float), test_frame.log_prediction.to_numpy(float)


def score_predictions(predictions, y, fold_ids):
    if len(predictions) != 1460 or not np.isfinite(predictions).all():
        raise ValueError("Every labeled row must have a finite held-out prediction.")
    scores = [float(np.sqrt(np.mean((predictions[fold_ids == i] - y[fold_ids == i]) ** 2)))
              for i in range(1, 6)]
    return {"cv_mean": float(np.mean(scores)), "cv_std": float(np.std(scores, ddof=1)),
            "pooled_oof_log_rmse": float(np.sqrt(np.mean((predictions - y) ** 2))),
            "fold_scores": scores}


def save_test_outputs(out, name, test, logs):
    prices = np.exp(logs)
    if len(prices) != len(test) or not np.isfinite(prices).all() or not (prices > 0).all():
        raise ValueError("Submission prices must be finite, positive and aligned with test IDs.")
    log_path, submission = out / "results" / f"{name}_test_logs.csv", out / "submissions" / f"{name}.csv"
    pd.DataFrame({"Id": test.Id, "log_prediction": logs}).to_csv(log_path, index=False)
    pd.DataFrame({"Id": test.Id, "SalePrice": prices}).to_csv(submission, index=False)
    check = pd.read_csv(submission)
    if list(check.columns) != ["Id", "SalePrice"] or not np.array_equal(check.Id, test.Id):
        raise ValueError("The saved submission failed its schema/ID check.")
    return [log_path, submission]


def run_comparison(args):
    out = args.output_dir or BASE / f"seed{args.fold_seed}"
    for subdir in ("results", "submissions", "fold_cache"):
        (out / subdir).mkdir(parents=True, exist_ok=True)
    train, test, X, Xt, y = load_data(args.input)
    if Xt.shape != (1459, 79):
        raise ValueError("Cross-validation uses the official 1,459-row competition test set.")
    folds = list(KFold(5, shuffle=True, random_state=args.fold_seed).split(X))
    fold_ids = np.zeros(len(X), int)
    manifest = []
    for fold, (tr, va) in enumerate(folds, 1):
        fold_ids[va] = fold
        manifest.append({"fold": fold, "train_ids": train.Id.iloc[tr].tolist(),
                         "validation_ids": train.Id.iloc[va].tolist()})
    anchor_oof, anchor_test = read_anchor(args.fold_seed, train, test, fold_ids)
    spec = make_spec(args.fold_seed, args.input)
    plan_path = out / "preregistered_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != spec:
        raise ValueError("Existing plan differs. Preserve the old evidence and choose a new output directory.")
    write_json(plan_path, spec)
    write_json(out / "fold_manifest.json", manifest)
    pd.DataFrame({"Id": train.Id, "fold": fold_ids}).to_csv(out / "fold_assignments.csv", index=False)
    if args.plan_only:
        print(json.dumps({"status": "plan_only", "plan": str(plan_path), "mps_inference": False}), flush=True)
        return
    summary_path = out / "results" / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous["status"] == "complete" and previous["spec"] == spec:
            for name, digest in previous["artifact_sha256"].items():
                if sha256(out / name) != digest:
                    raise ValueError(f"Completed artifact was modified: {name}")
            print("SKIP hash-verified completed TabICL comparison", flush=True)
            return
    configure_runtime()
    start = time.perf_counter()
    oof, fold_rows = np.full(len(X), np.nan), []
    for fold, (tr, va) in enumerate(folds, 1):
        cache_path = out / "fold_cache" / f"fold{fold}.csv"
        cache_meta = cache_path.with_suffix(".json")
        if cache_path.exists() and cache_meta.exists():
            metadata = json.loads(cache_meta.read_text())
            if metadata["plan_sha256"] != sha256(plan_path) or metadata["sha256"] != sha256(cache_path):
                raise ValueError("Fold cache failed its source/hash check.")
            cached = pd.read_csv(cache_path)
            if not np.array_equal(cached.Id, train.Id.iloc[va]):
                raise ValueError("Fold cache IDs differ.")
            oof[va] = cached.log_prediction.to_numpy(float)
            fold_rows.append(metadata["fold_metrics"])
            print(f"Reused verified fold {fold}", flush=True)
            continue
        if set(tr) & set(va) or len(tr) != 1168 or len(va) != 292:
            raise ValueError("Invalid fold membership.")
        tick = time.perf_counter()
        model = make_model()
        model.fit(X.iloc[tr], y[tr])
        oof[va] = np.asarray(model.predict(X.iloc[va], output_type="mean"), float)
        if not np.isfinite(oof[va]).all():
            raise ValueError("Non-finite held-out prediction.")
        row = {"fold": fold, "n_train": len(tr), "n_valid": len(va),
               "train_validation_disjoint": True,
               "log_rmse": float(np.sqrt(np.mean((oof[va] - y[va]) ** 2))),
               "seconds": time.perf_counter() - tick}
        fold_rows.append(row)
        pd.DataFrame({"Id": train.Id.iloc[va], "log_prediction": oof[va]}).to_csv(cache_path, index=False)
        write_json(cache_meta, {"plan_sha256": sha256(plan_path), "sha256": sha256(cache_path),
                                "fold_metrics": row})
        pd.DataFrame(fold_rows).to_csv(out / "results/fold_execution.csv", index=False)
        print(f"{MODEL_ID} seed{args.fold_seed} fold{fold}: {row['log_rmse']:.9f}; {row['seconds']:.1f}s", flush=True)
        del model
        cleanup_memory()
    blend_oof = 0.8 * anchor_oof + 0.2 * oof
    results = {MODEL_ID: score_predictions(oof, y, fold_ids),
               BLEND_ID: score_predictions(blend_oof, y, fold_ids),
               ANCHOR: score_predictions(anchor_oof, y, fold_ids)}
    artifacts = [plan_path, out / "fold_manifest.json", out / "fold_assignments.csv"]
    for name, values in ((MODEL_ID, oof), (BLEND_ID, blend_oof)):
        path = out / "results" / f"{name}_oof.csv"
        pd.DataFrame({"Id": train.Id, "fold": fold_ids, "log_SalePrice": y,
                      "log_prediction": values}).to_csv(path, index=False)
        artifacts.append(path)
    tick = time.perf_counter()
    model = make_model()
    model.fit(X, y)
    test_logs = np.asarray(model.predict(Xt, output_type="mean"), float)
    final_seconds = time.perf_counter() - tick
    artifacts += save_test_outputs(out, MODEL_ID, test, test_logs)
    artifacts += save_test_outputs(out, BLEND_ID, test, 0.8 * anchor_test + 0.2 * test_logs)
    del model
    cleanup_memory()
    comparison = []
    for name, result in results.items():
        comparison.append({"id": name, "fold_seed": args.fold_seed,
                           **{key: value for key, value in result.items() if key != "fold_scores"},
                           "delta_pooled_vs_anchor": result["pooled_oof_log_rmse"] - results[ANCHOR]["pooled_oof_log_rmse"],
                           "folds_improved_vs_anchor": int(np.sum(np.array(result["fold_scores"]) < np.array(results[ANCHOR]["fold_scores"])))})
    pd.DataFrame(comparison).to_csv(out / "comparison.csv", index=False)
    artifacts += [out / "comparison.csv", out / "results/fold_execution.csv"]
    summary = {"status": "complete", "spec": spec,
               "results": results,
               "fold_execution": fold_rows, "final_fit_seconds": final_seconds,
               "runtime_seconds": time.perf_counter() - start,
               "residual_correlation_vs_anchor": float(np.corrcoef(oof-y, anchor_oof-y)[0, 1]),
               "artifact_sha256": {str(p.relative_to(out)): sha256(p) for p in artifacts},
               "public_score_observed": None}
    write_json(summary_path, summary)
    print(json.dumps({"status": summary["status"], "summary": str(summary_path), "results": results}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-seed", type=int, choices=[42], default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/test.csv")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--check-api", action="store_true")
    args = parser.parse_args()
    if args.check_api:
        versions = verify_dependencies()
        model = make_model()
        print(json.dumps({"versions": versions, "parameters": model.get_params(),
                          "checkpoint": verify_checkpoint(), "mps_inference": False}), flush=True)
    else:
        run_comparison(args)


if __name__ == "__main__":
    main()
