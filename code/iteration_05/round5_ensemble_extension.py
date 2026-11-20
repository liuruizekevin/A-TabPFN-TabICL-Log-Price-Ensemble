"""One explicitly adaptive fixed follow-up: 80% noY TabPFN + 20% domain TabPFN.

This preserves every original round5_ensembles result and exports the new result
plus combined integration manifests under its extension/ subdirectory.
Run with .venv/bin/python scripts/round5_ensemble_extension.py, then --verify.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from provenance import verify_source

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models/round5_ensembles"
OUT = BASE / "extension"
NAME = "R5E08_noY_domain20"
NOY = "R5TP_log_noYtransform_s42"
DOMAIN = "R4T_v2_log_n32_s42_domain"
ANCHOR = "R4TC_v2_three_log_n16_s42"
WEIGHTS = {NOY: 0.8, DOMAIN: 0.2}
LIMITATION = (
    "One explicitly adaptive follow-up chosen after reviewing noY and domain "
    "validation and public results. The established 80/20 weight is reused without "
    "a weight grid. These reused seed-42 folds are development evidence, not an "
    "independent holdout or a public-score prediction."
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(verify=False):
    source_sha256 = sha(__file__)
    if verify:
        # Replay recorded metadata only after checking current and historical code identity.
        source_sha256 = json.loads((OUT / "results/plan.json").read_text())["script_sha256"]
        verify_source(__file__, source_sha256, ROOT)
    train, test = pd.read_csv(ROOT / "data/raw/train.csv"), pd.read_csv(ROOT / "data/raw/test.csv")
    assert len(train) == 1460 and len(test) == 1459 and "SalePrice" not in test
    assert train.Id.is_unique and test.Id.is_unique
    y = np.log(train.SalePrice.to_numpy(float))
    folds = np.zeros(len(train), dtype=int)
    for f, (_, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(train), 1):
        folds[va] = f
    oofs, tests, sources = {}, {}, []
    directories = {NOY: "round5_tabpfn", DOMAIN: "round4_tabpfn", ANCHOR: "round4_tabpfn_checkpoints"}
    for name, directory in directories.items():
        p = ROOT / "models" / directory / "results" / f"{name}_oof.csv"
        d = pd.read_csv(p)
        assert np.array_equal(d.Id, train.Id) and np.array_equal(d.fold, folds)
        assert np.allclose(d.log_SalePrice, y, atol=1e-12, rtol=0)
        oofs[name] = d.log_prediction.to_numpy(float)
        sources.append({"source_id": name, "role": "OOF", "path": str(p.relative_to(ROOT)), "sha256": sha(p)})
        if name in WEIGHTS:
            p = p.with_name(f"{name}_test_logs.csv")
            d = pd.read_csv(p)
            assert np.array_equal(d.Id, test.Id)
            tests[name] = d.log_prediction.to_numpy(float)
            sources.append({"source_id": name, "role": "full-fit test logs", "path": str(p.relative_to(ROOT)), "sha256": sha(p)})
    assert all(np.isfinite(value).all() for value in [*oofs.values(), *tests.values()])
    pred = sum(weight * oofs[name] for name, weight in WEIGHTS.items())
    test_log = sum(weight * tests[name] for name, weight in WEIGHTS.items())
    prices = np.exp(test_log)
    assert np.isfinite(prices).all() and (prices > 0).all()
    per_fold = [float(np.sqrt(np.mean((pred[folds == f] - y[folds == f]) ** 2))) for f in range(1, 6)]
    base_folds = [float(np.sqrt(np.mean((oofs[ANCHOR][folds == f] - y[folds == f]) ** 2))) for f in range(1, 6)]
    metrics = {"cv_mean": float(np.mean(per_fold)), "cv_std": float(np.std(per_fold, ddof=1)),
               "pooled_oof_log_rmse": float(np.sqrt(np.mean((pred - y) ** 2)))}
    old_integration_path = BASE / "results/integration_candidates.json"
    old_candidates_path = BASE / "results/candidate_manifests.json"
    integration = json.loads(old_integration_path.read_text())
    candidates = json.loads(old_candidates_path.read_text())
    assert len(integration) == 7 and all(item["experiment_id"] != NAME for item in integration)
    payloads = {}

    def frame(path, value):
        payloads[path] = value.to_csv(index=False).encode()

    def obj(path, value):
        payloads[path] = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()

    params = {"combination_space": "natural logarithm of price", "weights": WEIGHTS,
              "selection_limitation": LIMITATION, "additional_weight_search": False,
              "submission_candidate": True}
    record = {"experiment_id": NAME, "model": "Weighted log-space blend", "stage": "round5_adaptive_followup",
              "target": "log", "runtime_seconds": 0.0, "params_json": json.dumps(params), **metrics}
    frame("results/experiment_results.csv", pd.DataFrame([record]))
    frame("results/oof_predictions.csv", pd.DataFrame({"Id": train.Id, "fold": folds, "log_SalePrice": y,
                                                       f"{NAME}__log_prediction": pred}))
    frame(f"results/{NAME}_oof.csv", pd.DataFrame({"Id": train.Id, "fold": folds, "log_SalePrice": y, "log_prediction": pred}))
    frame(f"results/{NAME}_test_logs.csv", pd.DataFrame({"Id": test.Id, "log_prediction": test_log}))
    frame("results/fold_scores.csv", pd.DataFrame([{"experiment_id": NAME, "fold": f,
                                                   "rows": int((folds == f).sum()), "log_rmse": score,
                                                   "delta_vs_original_winner": score - base_folds[f - 1]}
                                                  for f, score in enumerate(per_fold, 1)]))
    frame(f"submissions/{NAME}.csv", pd.DataFrame({"Id": test.Id, "SalePrice": prices}))
    submission_sha = hashlib.sha256(payloads[f"submissions/{NAME}.csv"]).hexdigest()
    candidate = {"experiment_id": NAME, "weights": WEIGHTS, "expanded_weights": WEIGHTS,
                 **metrics, "fold_scores": per_fold, "public_score": None,
                 "selection_limitation": LIMITATION,
                 "submission_path": f"models/round5_ensembles/extension/submissions/{NAME}.csv",
                 "submission_sha256": submission_sha}
    candidates.append(candidate)
    integration.append({"experiment_id": NAME, "model": record["model"], "target": "log",
                        "params": params, **metrics,
                        "oof_path": "models/round5_ensembles/extension/results/oof_predictions.csv",
                        "oof_column": f"{NAME}__log_prediction"})
    obj("results/candidate_manifests.json", candidates)
    obj("results/integration_candidates.json", integration)
    obj("results/source_manifest.json", sources)
    obj("results/plan.json", {"id": NAME, "weights": WEIGHTS, "new_recipes": 1,
                              "rationale": "Reuse the established 20 percent domain contribution with the independently evaluated configuration that omits the internal Y transform.",
                              "selection_limitation": LIMITATION,
                              "script_sha256": source_sha256,
                              "original_integration_manifest_sha256": sha(old_integration_path),
                              "original_candidate_manifest_sha256": sha(old_candidates_path),
                              "original_source_manifest_sha256": sha(BASE / "results/source_manifest.json"),
                              "reproduction_command": ".venv/bin/python scripts/round5_ensemble_extension.py --verify"})
    obj("results/summary.json", {"status": "complete", **candidate,
                                 "anchor_pooled_oof_log_rmse": float(np.sqrt(np.mean((oofs[ANCHOR] - y) ** 2))),
                                 "noY_pooled_oof_log_rmse": float(np.sqrt(np.mean((oofs[NOY] - y) ** 2))),
                                 "folds_improved_vs_original_winner": sum(a < b for a, b in zip(per_fold, base_folds)),
                                 "original_primary_recipes_preserved": 7, "combined_primary_recipe_count": len(integration),
                                 "training_rows_removed": 0})
    obj("results/validation.json", {"all1460_OOF_rows_retained": True, "all1459_test_rows_retained": True,
                                    "official_Id_order_and_seed42folds_verified": True,
                                    "source_script_sha256": source_sha256,
                                    "original_manifests_preserved": True,
                                    "generated_file_sha256": {path: hashlib.sha256(content).hexdigest() for path, content in payloads.items()}})
    for relative, content in payloads.items():
        path = OUT / relative
        if verify:
            assert path.read_bytes() == content, f"Extension reproduction mismatch: {path}"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    print(json.dumps({"verified" if verify else "generated": len(payloads), "candidate": candidate,
                      "combined_integration_manifest": str(OUT / "results/integration_candidates.json"),
                      "replay_manifest": str(OUT / "results/candidate_manifests.json")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    run(parser.parse_args().verify)
