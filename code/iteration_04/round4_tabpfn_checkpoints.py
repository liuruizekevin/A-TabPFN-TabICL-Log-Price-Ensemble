"""Compare three public TabPFN-v2 checkpoints with fold-isolated prediction.

Use build/tabpfn-env/bin/python. Weights must match the official model file
manifest; no external targets, inference service, or test-label lookup is used.
"""
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion


ROOT = Path(__file__).resolve().parents[1]
FILES = ["tabpfn-v2-regressor.ckpt", "tabpfn-v2-regressor-09gpqh39.ckpt",
         "tabpfn-v2-regressor-2noar4o2.ckpt"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(estimators=16, output_dir=None):
    out = Path(output_dir) if output_dir is not None else ROOT / "models/round4_tabpfn_checkpoints"
    for d in ("results", "submissions"):
        (out / d).mkdir(parents=True, exist_ok=True)
    name = f"R4TC_v2_three_log_n{estimators}_s42"
    official = json.loads((ROOT / "references/tabpfn/remote_file_manifest.json").read_text())
    weights, manifest = [], []
    for filename in FILES:
        entry = next(r for r in official if r["path"] == filename)
        path = ROOT / "references/tabpfn" / filename
        digest = sha256(path)
        assert digest == entry["lfs"]["oid"] and path.stat().st_size == entry["size"]
        weights.append(path)
        manifest.append({"filename": filename, "bytes": path.stat().st_size,
                         "sha256": digest,
                         "url": "https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/" + filename})
    raw = pd.read_csv(ROOT / "data/raw/train.csv", keep_default_na=False, na_values=["NA", ""])
    test = pd.read_csv(ROOT / "data/raw/test.csv", keep_default_na=False, na_values=["NA", ""])
    X, Xt = raw.drop(columns=["Id", "SalePrice"]), test.drop(columns="Id")
    for col in X.select_dtypes(exclude=np.number).columns:
        X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
        Xt[col] = Xt[col].astype(object).where(Xt[col].notna(), np.nan)
    logs = np.log(raw.SalePrice.to_numpy(float))
    spec = {"id": name, "model": "TabPFN-v2 multi-checkpoint ensemble", "target": "log",
            "n_estimators": estimators, "random_state": 42, "device": "mps",
            "fold_seed": 42, "n_splits": 5, "rows": 1460, "features": 79,
            "feature_recipe": "raw", "filter_train": False,
            "checkpoint_manifest": manifest,
            "library_version": importlib.metadata.version("tabpfn"),
            "torch_version": torch.__version__, "source_sha256": sha256(__file__),
            "source": "https://huggingface.co/Prior-Labs/TabPFN-v2-reg",
            "selection_limitation": "Adaptive development; same frozen folds; no external test targets.",
            "preprocessing": "Fresh TabPFN instance fits internal preprocessing separately within each training fold."}
    (out / "results" / f"{name}_plan.json").write_text(json.dumps(spec, indent=2) + "\n")

    def make():
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, model_path=weights, n_estimators=estimators,
            device="mps", random_state=42, n_preprocessing_jobs=2)

    start = time.perf_counter()
    predictions = np.full(len(X), np.nan)
    foldids = np.zeros(len(X), dtype=int)
    foldrows = []
    for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(X), 1):
        t = time.perf_counter()
        assert not set(tr) & set(va)
        model = make()
        model.fit(X.iloc[tr], logs[tr])
        predictions[va] = np.asarray(model.predict(X.iloc[va]), float)
        foldids[va] = fold
        score = float(np.sqrt(np.mean((predictions[va] - logs[va]) ** 2)))
        foldrows.append({"fold": fold, "log_rmse": score, "n_train": len(tr),
                         "n_valid": len(va), "train_validation_disjoint": True,
                         "seconds": time.perf_counter()-t})
        print(f"{name} fold {fold}: {score:.6f}; {time.perf_counter()-t:.1f}s", flush=True)
        pd.DataFrame(foldrows).to_csv(out / "results" / f"{name}_fold_scores.csv", index=False)
        del model
        torch.mps.empty_cache()
    assert np.isfinite(predictions).all() and (foldids > 0).all()
    pd.DataFrame({"Id": raw.Id, "fold": foldids, "log_SalePrice": logs,
                  "log_prediction": predictions}).to_csv(out / "results" / f"{name}_oof.csv", index=False)
    cv_mean = float(np.mean([r["log_rmse"] for r in foldrows]))
    print(f"{name} complete CV mean={cv_mean:.8f}; fitting final model", flush=True)
    model = make()
    model.fit(X, logs)
    testlogs = np.asarray(model.predict(Xt), float)
    prices = np.exp(testlogs)
    assert np.isfinite(prices).all() and (prices > 0).all()
    pd.DataFrame({"Id": test.Id, "log_prediction": testlogs}).to_csv(
        out / "results" / f"{name}_test_logs.csv", index=False)
    submission = out / "submissions" / f"{name}.csv"
    pd.DataFrame({"Id": test.Id, "SalePrice": prices}).to_csv(submission, index=False)
    result = spec | {"status": "complete", "fold_scores": foldrows, "cv_mean": cv_mean,
                     "cv_std": float(np.std([r["log_rmse"] for r in foldrows], ddof=1)),
                     "pooled_oof_log_rmse": float(np.sqrt(np.mean((predictions-logs)**2))),
                     "runtime_seconds": time.perf_counter()-start,
                     "submission_sha256": sha256(submission)}
    (out / "results" / f"{name}_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimators", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional separate directory for reproducible CV and submission outputs.")
    args = parser.parse_args()
    run(args.estimators, args.output_dir)
