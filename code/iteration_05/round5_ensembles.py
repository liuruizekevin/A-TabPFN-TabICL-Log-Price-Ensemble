"""Reproduce a bounded round-5 ensemble comparison from saved predictions.

Run with .venv/bin/python scripts/round5_ensembles.py --include-seeds, then add
--verify to check generated results against the saved component predictions.
All numbers use every official training row and natural-log prices.
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
OUT = ROOT / "models/round5_ensembles"
ANCHOR = "R4TC_v2_three_log_n16_s42"
LARGER = "R4TC_v2_three_log_n32_s42"
DOMAIN = "R4T_v2_log_n32_s42_domain"
RAW = "R4T_v2_raw_n32_s42"
CLASSICAL = "B14_cat_mae_d6_blend"
LIMITATION = (
    "Adaptive comparison on previously reused fixed folds; this is development "
    "evidence, not an unbiased holdout estimate or a public leaderboard score."
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(verify=False, include_seeds=False):
    source_sha256 = digest(__file__)
    if verify:
        # Replay recorded metadata only after checking current and historical code identity.
        source_sha256 = json.loads((OUT / "results/plan.json").read_text())["source_sha256"]
        verify_source(__file__, source_sha256, ROOT)
    train = pd.read_csv(ROOT / "data/raw/train.csv")
    test = pd.read_csv(ROOT / "data/raw/test.csv")
    y = np.log(train.SalePrice.to_numpy(float))
    fold = np.zeros(len(train), dtype=int)
    for number, (_, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(train), 1):
        fold[va] = number
    assert len(train) == 1460 and len(test) == 1459
    assert train.Id.is_unique and test.Id.is_unique and "SalePrice" not in test
    provenance = []
    oof, testing = {}, {}
    specs = {
        ANCHOR: ("round4_tabpfn_checkpoints", "log_prediction"),
        LARGER: ("round4_tabpfn_checkpoints", "log_prediction"),
        DOMAIN: ("round4_tabpfn", "log_prediction"),
        RAW: ("round4_tabpfn", "log_prediction"),
        CLASSICAL: ("round4_boosting", f"{CLASSICAL}__log_prediction"),
    }
    if include_seeds:
        specs.update({"R5TP_log_s17": ("round5_tabpfn", "log_prediction"),
                      "R5TP_log_s2026": ("round5_tabpfn", "log_prediction")})
    for name, (directory, column) in specs.items():
        prefix = ROOT / "models" / directory / "results"
        op = prefix / (f"{name}_oof.csv" if column == "log_prediction" else "oof_predictions.csv")
        tp = prefix / (f"{name}_test_logs.csv" if column == "log_prediction" else "test_log_predictions.csv")
        od, td = pd.read_csv(op), pd.read_csv(tp)
        assert np.array_equal(od.Id, train.Id) and np.array_equal(td.Id, test.Id)
        assert np.array_equal(od.fold, fold)
        assert np.allclose(od.log_SalePrice, y, atol=1e-12, rtol=0)
        oof[name], testing[name] = od[column].to_numpy(float), td[column].to_numpy(float)
        assert np.isfinite(oof[name]).all() and np.isfinite(testing[name]).all()
        provenance.append({
            "source_id": name, "oof_path": str(op.relative_to(ROOT)),
            "oof_sha256": digest(op), "test_path": str(tp.relative_to(ROOT)),
            "test_sha256": digest(tp), "column": column,
            "test_fit": "original full-training-data fit",
        })
    # Validate the saved classical subensemble's actual decomposition.
    classical_parts = {
        "B14_cat_mae_d6": ("round4_boosting", 0.4),
        "F03_robust_c20_g0003": ("followup", 0.4),
        "O22_xgboost_d3": ("optimization", 0.2),
    }
    actual = np.zeros(len(train))
    for name, (directory, weight) in classical_parts.items():
        path = ROOT / "models" / directory / "results/oof_predictions.csv"
        frame = pd.read_csv(path)
        assert np.array_equal(frame.Id, train.Id) and np.array_equal(frame.fold, fold)
        actual += weight * frame[f"{name}__log_prediction"].to_numpy(float)
    assert np.allclose(actual, oof[CLASSICAL], atol=1e-12, rtol=0)

    recipes = {
        "R5E01_n16_n32_equal": {ANCHOR: 0.5, LARGER: 0.5},
        "R5E02_anchor_raw20": {ANCHOR: 0.8, RAW: 0.2},
        "R5E03_anchor_classical15": {ANCHOR: 0.85, CLASSICAL: 0.15},
        "R5E04_anchor_domain20": {ANCHOR: 0.8, DOMAIN: 0.2},
        "R5E05_bagged_classical15": {ANCHOR: 0.425, LARGER: 0.425, CLASSICAL: 0.15},
    }
    proposed = ["R5E05_bagged_classical15", "R5E03_anchor_classical15", "R5E04_anchor_domain20"]
    if include_seeds:
        recipes["R5E06_seed42_17_2026_equal"] = {ANCHOR: 1 / 3, "R5TP_log_s17": 1 / 3, "R5TP_log_s2026": 1 / 3}
        recipes["R5E07_seedbag_domain20"] = {ANCHOR: 0.8 / 3, "R5TP_log_s17": 0.8 / 3, "R5TP_log_s2026": 0.8 / 3, DOMAIN: 0.2}
        proposed += ["R5E06_seed42_17_2026_equal", "R5E07_seedbag_domain20"]
    protocol = {
        "comparison_scope": f"{len(recipes)} simple convex recipes and six calibration diagnostics; no optimized-weight search.",
        "include_new_seed_ensembles": include_seeds,
        "seed_extension_chronology": "The pure equal-seed ensemble was requested before new public scores. The seedbag-domain20 combination was requested after observing the original anchor-domain20 submission improve the public score; its choice is leaderboard-adaptive." if include_seeds else None,
        "chronology": "Recipes were inspected interactively before this saved script; this is not a preregistered experiment.",
        "selection_limitation": LIMITATION,
        "training_rows": len(train), "test_rows": len(test),
        "fold_seed": 42, "n_splits": 5, "recipes": recipes,
        "calibration_diagnostics": {
            "input_ids": [ANCHOR, "R5E04_anchor_domain20", "R5E05_bagged_classical15"],
            "methods": ["half_bias", "affine_identity_ridge100"],
            "outer_operation": "Fit calibration on four saved OOF folds; apply to the fifth.",
            "limitation": "The calibrator uses cross-fitting over saved base OOF predictions. This evaluates the calibration stage rather than a fully nested pipeline.",
        },
        "external_test_labels_used": False, "training_rows_removed": 0,
        "reproduction_command": ".venv/bin/python scripts/round5_ensembles.py --verify" + (" --include-seeds" if include_seeds else ""),
        "source_sha256": source_sha256,
    }
    predictions = dict(oof)
    test_predictions = dict(testing)
    for name, weights in recipes.items():
        assert min(weights.values()) >= 0 and np.isclose(sum(weights.values()), 1)
        predictions[name] = sum(weight * oof[key] for key, weight in weights.items())
        test_predictions[name] = sum(weight * testing[key] for key, weight in weights.items())

    calibration_rows = []
    for source in protocol["calibration_diagnostics"]["input_ids"]:
        p = predictions[source]
        for method in protocol["calibration_diagnostics"]["methods"]:
            name = f"{source}__{method}"
            calibrated = np.full(len(y), np.nan)
            for number in range(1, 6):
                tr, va = fold != number, fold == number
                if method == "half_bias":
                    slope, intercept = 1.0, float(0.5 * np.mean(y[tr] - p[tr]))
                else:
                    z = p[tr] - 12.0
                    matrix = np.array([[z @ z + 100.0, z.sum()], [z.sum(), len(z)]])
                    rhs = np.array([z @ (y[tr] - 12.0) + 100.0, (y[tr] - 12.0).sum()])
                    slope, offset = np.linalg.solve(matrix, rhs)
                    intercept = 12.0 + offset - 12.0 * slope
                calibrated[va] = slope * p[va] + intercept
                calibration_rows.append({"experiment_id": name, "fold": number,
                                         "slope": slope, "intercept": intercept,
                                         "calibration_training_rows": int(tr.sum())})
            predictions[name] = calibrated

    def metrics(p):
        scores = [float(np.sqrt(np.mean((p[fold == f] - y[fold == f]) ** 2))) for f in range(1, 6)]
        return {"cv_mean": float(np.mean(scores)), "cv_std": float(np.std(scores, ddof=1)),
                "pooled_oof_log_rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                "fold_scores": scores}

    baseline = metrics(oof[ANCHOR])
    rows, fold_rows = [], []
    for name, p in predictions.items():
        scores = metrics(p)
        rows.append({"experiment_id": name, **{k: v for k, v in scores.items() if k != "fold_scores"},
                     "training_rows": len(y),
                     "delta_pooled_vs_anchor": scores["pooled_oof_log_rmse"] - baseline["pooled_oof_log_rmse"],
                     "folds_improved_vs_anchor": sum(a < b for a, b in zip(scores["fold_scores"], baseline["fold_scores"])),
                     "submission_candidate": name in proposed,
                     "kind": "calibration_diagnostic" if "__" in name else "fixed_blend" if name in recipes else "saved_reference"})
        for f, score in enumerate(scores["fold_scores"], 1):
            fold_rows.append({"experiment_id": name, "fold": f, "rows": int((fold == f).sum()),
                              "log_rmse": score, "delta_vs_anchor": score - baseline["fold_scores"][f - 1]})
    residuals = pd.DataFrame({name: value - y for name, value in oof.items()})
    details = []
    for name in proposed:
        weights = recipes[name]
        expanded = {}
        for key, weight in weights.items():
            parts = {key: 1.0} if key != CLASSICAL else {part: value[1] for part, value in classical_parts.items()}
            for part, pw in parts.items():
                expanded[part] = expanded.get(part, 0.0) + weight * pw
        gain = (oof[ANCHOR] - y) ** 2 - (predictions[name] - y) ** 2
        details.append({"experiment_id": name, "weights": weights, "expanded_weights": expanded,
                        **metrics(predictions[name]), "public_score": None,
                        "selection_limitation": LIMITATION,
                        "largest_squared_error_reductions": [
                            {"Id": int(train.Id.iloc[i]), "squared_error_reduction": float(gain[i])}
                            for i in np.argsort(gain)[-5:][::-1]],
                        "largest_squared_error_increases": [
                            {"Id": int(train.Id.iloc[i]), "squared_error_increase": float(-gain[i])}
                            for i in np.argsort(gain)[:5]],
                        "submission_path": f"models/round5_ensembles/submissions/{name}.csv"})

    payloads = {}

    def frame(name, value, index=False):
        payloads[name] = value.to_csv(index=index).encode()

    def obj(name, value):
        payloads[name] = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()

    obj("results/plan.json", protocol)
    obj("results/source_manifest.json", provenance)
    frame("results/experiment_results.csv", pd.DataFrame(rows))
    frame("results/fold_scores.csv", pd.DataFrame(fold_rows))
    frame("results/residual_correlations.csv", residuals.corr(), index=True)
    frame("results/calibration_parameters.csv", pd.DataFrame(calibration_rows))
    combined = pd.DataFrame({"Id": train.Id, "fold": fold, "log_SalePrice": y})
    for name, p in predictions.items():
        combined[f"{name}__log_prediction"] = p
    frame("results/oof_predictions.csv", combined)
    test_combined = pd.DataFrame({"Id": test.Id})
    for name in proposed:
        p = test_predictions[name]
        prices = np.exp(p)
        assert np.isfinite(prices).all() and (prices > 0).all()
        test_combined[f"{name}__log_prediction"] = p
        frame(f"submissions/{name}.csv", pd.DataFrame({"Id": test.Id, "SalePrice": prices}))
        frame(f"results/{name}_oof.csv", pd.DataFrame({"Id": train.Id, "fold": fold, "log_SalePrice": y, "log_prediction": predictions[name]}))
        frame(f"results/{name}_test_logs.csv", pd.DataFrame({"Id": test.Id, "log_prediction": p}))
        detail = next(d for d in details if d["experiment_id"] == name)
        detail["submission_sha256"] = hashlib.sha256(payloads[f"submissions/{name}.csv"]).hexdigest()
    frame("results/test_log_predictions.csv", test_combined)
    obj("results/candidate_manifests.json", details)
    integration = []
    for row in rows:
        name = row["experiment_id"]
        if name in specs or "__" in name:
            continue
        if name in recipes:
            params = {"combination_space": "natural logarithm of price", "weights": recipes[name],
                      "selection_limitation": LIMITATION, "submission_candidate": name in proposed}
            model = "Weighted log-space blend"
        else:
            source, method = name.split("__", 1)
            params = {"source_id": source, "method": method,
                      "calibration_only_cross_fitted": True, "fully_nested": False,
                      "selection_limitation": protocol["calibration_diagnostics"]["limitation"] + " " + LIMITATION,
                      "submission_candidate": False}
            model = "OOF calibration diagnostic (not fully nested)"
        integration.append({"experiment_id": name, "model": model, "params": params,
                            "oof_path": "models/round5_ensembles/results/oof_predictions.csv",
                            "oof_column": f"{name}__log_prediction", "target": "log",
                            "cv_mean": row["cv_mean"], "cv_std": row["cv_std"],
                            "pooled_oof_log_rmse": row["pooled_oof_log_rmse"]})
    obj("results/integration_candidates.json", integration)
    obj("results/summary.json", {
        "status": "complete", "selected_anchor": ANCHOR, "anchor_public_score_pre_round5": 0.11074,
        "new_fixed_recipes": len(recipes), "calibration_diagnostics": len(calibration_rows) // 5,
        "candidate_ids_in_local_priority_order": proposed,
        "local_best_candidate": details[0],
        "all_calibrations_failed_to_improve_their_uncalibrated_source": all(
            metrics(predictions[f"{source}__{method}"])["pooled_oof_log_rmse"] >= metrics(predictions[source])["pooled_oof_log_rmse"]
            for source in protocol["calibration_diagnostics"]["input_ids"]
            for method in protocol["calibration_diagnostics"]["methods"]),
        "selection_limitation": LIMITATION,
        "attribution": "Built with PriorLabs-TabPFN.",
    })
    obj("results/validation.json", {
        "all_1460_training_rows_retained": True, "all_1459_test_rows_retained": True,
        "fold_assignments_match_KFold5_shuffle_seed42": True,
        "source_ids_and_targets_match_official_train": True,
        "source_test_ids_match_official_test": True,
        "classical_blend_decomposition_verified": True,
        "weights_nonnegative_and_sum_one": True,
        "all_predictions_finite_and_prices_positive": True,
        "source_script_sha256": source_sha256,
        "generated_file_sha256": {path: hashlib.sha256(content).hexdigest() for path, content in payloads.items()},
    })
    for name, content in payloads.items():
        path = OUT / name
        if verify:
            assert path.read_bytes() == content, f"Reproduction mismatch: {path}"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    print(json.dumps({"verified" if verify else "generated": len(payloads), "output": str(OUT),
                      "candidates": [{"id": d["experiment_id"], "cv_mean": d["cv_mean"],
                                      "pooled_oof": d["pooled_oof_log_rmse"],
                                      "sha256": d["submission_sha256"]} for d in details]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--include-seeds", action="store_true", help="Include the two additional ensembles once both new seed runs are complete.")
    args = parser.parse_args()
    run(args.verify, args.include_seeds)
