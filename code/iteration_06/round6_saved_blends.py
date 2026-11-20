"""Recompute four bounded log-space blends, exporting two shortlisted submissions.

The configured weights combine saved predictions for the original test rows.
Run: .venv/bin/python scripts/round6_saved_blends.py [--verify]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models/round6_saved_blends"
BASE = "R5E08_noY_domain20"
NOY = "R5TP_log_noYtransform_s42"
DOMAIN = "R4T_v2_log_n32_s42_domain"
FP = "R5TP_log_nofingerprint_s42"
BOX = "R5TP_boxcox015_s42"
CLASSICAL = "B14_cat_mae_d6_blend"
RECIPES = {
    "R6E01_e08_classical15": {BASE: .85, CLASSICAL: .15},
    "R6E02_noY_domain50": {NOY: .5, DOMAIN: .5},
    "R6E03_e08_boxcox01520": {BASE: .8, BOX: .2},
    "R6E04_e08_nofingerprint20": {BASE: .8, FP: .2},
}
SHORTLIST = {"R6E01_e08_classical15", "R6E02_noY_domain50"}
LIMITATION = (
    "Four fixed recipes chosen adaptively after prior development CV and online feedback; "
    "no weight grid or optimized weights. The reused seed42 partition provides "
    "development evidence, not an independent holdout or prediction of Kaggle score. "
    "A majority of squared-error reduction is concentrated in two training rows, which remain included."
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(verify=False):
    train_path, test_path = ROOT / "data/raw/train.csv", ROOT / "data/raw/test.csv"
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    data_manifest_path = ROOT / "data/source_manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text())
    for label, path in [("train.csv", train_path), ("test.csv", test_path)]:
        assert sha(path) == data_manifest["files"][label]["sha256"]
    assert len(train) == 1460 and len(test) == 1459
    assert train.Id.is_unique and test.Id.is_unique and "SalePrice" not in test
    y = np.log(train.SalePrice.to_numpy(float))
    folds = np.zeros(len(y), int)
    for f, (_, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(train), 1):
        folds[va] = f
    source_manifest, oofs, tests = [], {}, {}

    def read_source(name, directory, combined=False):
        folder = ROOT / "models" / directory / "results"
        op = folder / ("oof_predictions.csv" if combined else f"{name}_oof.csv")
        tp = folder / ("test_log_predictions.csv" if combined else f"{name}_test_logs.csv")
        col = f"{name}__log_prediction" if combined else "log_prediction"
        a, b = pd.read_csv(op), pd.read_csv(tp)
        assert np.array_equal(a.Id, train.Id) and np.array_equal(b.Id, test.Id)
        assert np.array_equal(a.fold, folds)
        assert np.allclose(a.log_SalePrice, y, rtol=0, atol=1e-12)
        oofs[name], tests[name] = a[col].to_numpy(float), b[col].to_numpy(float)
        assert np.isfinite(oofs[name]).all() and np.isfinite(tests[name]).all()
        source_manifest.append({"id": name, "column": col,
            "oof_path": str(op.relative_to(ROOT)), "oof_sha256": sha(op),
            "test_path": str(tp.relative_to(ROOT)), "test_sha256": sha(tp),
            "test_fit": "original full-training-data fit"})

    read_source(BASE, "round5_ensembles/extension")
    read_source(DOMAIN, "round4_tabpfn")
    for name in [NOY, FP, BOX]:
        read_source(name, "round5_tabpfn")
    read_source(CLASSICAL, "round4_boosting", combined=True)
    assert np.allclose(oofs[BASE], .8*oofs[NOY]+.2*oofs[DOMAIN], rtol=0, atol=1e-12)
    assert np.allclose(tests[BASE], .8*tests[NOY]+.2*tests[DOMAIN], rtol=0, atol=1e-12)

    classical_parts = {"B14_cat_mae_d6": ("round4_boosting", .4),
                       "F03_robust_c20_g0003": ("followup", .4),
                       "O22_xgboost_d3": ("optimization", .2)}
    actual = np.zeros(len(y))
    part_sources = []
    for name, (directory, weight) in classical_parts.items():
        path = ROOT / "models" / directory / "results/oof_predictions.csv"
        frame = pd.read_csv(path)
        assert np.array_equal(frame.Id, train.Id) and np.array_equal(frame.fold, folds)
        assert np.allclose(frame.log_SalePrice, y, rtol=0, atol=1e-12)
        actual += weight * frame[f"{name}__log_prediction"].to_numpy(float)
        part_sources.append({"id":name, "weight":weight, "path":str(path.relative_to(ROOT)), "sha256":sha(path)})
    assert np.allclose(actual, oofs[CLASSICAL], rtol=0, atol=1e-12)

    def metric(p, f=folds):
        scores = [float(np.sqrt(np.mean((p[f==k]-y[f==k])**2))) for k in range(1,6)]
        return {"pooled_oof_log_rmse":float(np.sqrt(np.mean((p-y)**2))),
                "cv_mean":float(np.mean(scores)), "cv_std":float(np.std(scores,ddof=1)), "fold_scores":scores}

    payloads = {}

    def frame(path, value):
        payloads[path] = value.to_csv(index=False).encode()

    def obj(path, value):
        payloads[path] = (json.dumps(value,indent=2,allow_nan=False)+"\n").encode()

    base_metric = metric(oofs[BASE])
    metrics, candidates, diagnostics, all_folds = [], [], {}, []
    all_oof = {"Id":train.Id,"fold":folds,"log_SalePrice":y}
    all_test = {"Id":test.Id}
    for name, weights in RECIPES.items():
        assert min(weights.values()) >= 0 and np.isclose(sum(weights.values()), 1)
        p = sum(w*oofs[k] for k,w in weights.items())
        t = sum(w*tests[k] for k,w in weights.items())
        prices = np.exp(t)
        assert np.isfinite(p).all() and np.isfinite(prices).all() and (prices>0).all()
        scores = metric(p)
        gains = (oofs[BASE]-y)**2-(p-y)**2
        focus_gain = float(gains[np.isin(train.Id,[1299,524])].sum())
        total_gain = float(gains.sum())
        row = {"experiment_id":name, "kind":"fixed_log_space_blend", "weights":weights,
            **scores, "delta_pooled_vs_e08":scores["pooled_oof_log_rmse"]-base_metric["pooled_oof_log_rmse"],
            "folds_improved_vs_e08":sum(a<b for a,b in zip(scores["fold_scores"],base_metric["fold_scores"])),
            "submission_candidate":name in SHORTLIST,
            "recommendation":"shortlisted_for_online_test" if name in SHORTLIST else "rejected_worse_development_cv",
            "public_score":None, "training_rows_removed":0,
            "total_squared_error_reduction":total_gain,
            "id1299_and_524_squared_error_reduction":focus_gain,
            "two_row_share_of_total_gain":focus_gain/total_gain,
            "selection_limitation":LIMITATION}
        metrics.append({k:(json.dumps(v,sort_keys=True) if isinstance(v,dict) else v) for k,v in row.items() if k != "fold_scores"})
        for f, value in enumerate(scores["fold_scores"],1):
            all_folds.append({"experiment_id":name,"fold":f,"rows":int((folds==f).sum()),
                "log_rmse":value,"delta_vs_e08":value-base_metric["fold_scores"][f-1]})
        all_oof[f"{name}__log_prediction"] = p
        all_test[f"{name}__log_prediction"] = t
        diagnostics[name] = {"residual_correlation_vs_e08":float(np.corrcoef(p-y,oofs[BASE]-y)[0,1]),
            "top_squared_error_reductions":[{"Id":int(train.Id.iloc[i]),"reduction":float(gains[i])} for i in np.argsort(gains)[-8:][::-1]],
            "top_squared_error_increases":[{"Id":int(train.Id.iloc[i]),"increase":float(-gains[i])} for i in np.argsort(gains)[:8]],
            "test_log_rmse_difference_vs_e08":float(np.sqrt(np.mean((t-tests[BASE])**2))),
            "test_max_abs_log_difference_vs_e08":float(np.max(np.abs(t-tests[BASE])))}
        if name in SHORTLIST:
            path = f"submissions/{name}.csv"
            frame(path,pd.DataFrame({"Id":test.Id,"SalePrice":prices}))
            row.update({"submission_path":str((OUT/path).relative_to(ROOT)),
                        "submission_sha256":hashlib.sha256(payloads[path]).hexdigest()})
        candidates.append(row)

    frame("results/experiment_results.csv",pd.DataFrame(metrics))
    frame("results/fold_scores.csv",pd.DataFrame(all_folds))
    frame("results/oof_predictions.csv",pd.DataFrame(all_oof))
    frame("results/test_log_predictions.csv",pd.DataFrame(all_test))
    obj("results/candidate_manifests.json",candidates)
    obj("results/source_manifest.json",source_manifest)
    obj("results/residual_diagnostics.json",diagnostics)
    obj("results/plan.json",{"recipes":RECIPES,"shortlist":sorted(SHORTLIST),
        "chronology":"Four recipes were inspected during the round6 audit before saving this reproduction script; not preregistration.",
        "baseline":BASE,"baseline_recorded_public_score_before_round6":0.11064,
        "baseline_development_metrics":base_metric,"source_sha256":sha(__file__),
        "data_manifest_path":str(data_manifest_path.relative_to(ROOT)),"data_manifest_sha256":sha(data_manifest_path),
        "train_sha256":sha(train_path),"test_sha256":sha(test_path),"fold_seed":42,
        "classical_decomposition_oof_verified":part_sources,
        "weight_grid":False,"training_rows_removed":0,"external_test_labels_used":False,
        "selection_limitation":LIMITATION,
        "reproduction_command":".venv/bin/python scripts/round6_saved_blends.py --verify"})
    obj("results/validation.json",{"all1460_oof_rows_retained":True,"all1459_test_rows_retained":True,
        "input_data_hashes_match_preserved_source_manifest":True,"Id_order_targets_and_fold_assignments_verified":True,
        "finite_positive_submission_prices":True,"source_sha256":sha(__file__),
        "output_sha256":{path:hashlib.sha256(content).hexdigest() for path,content in payloads.items()}})
    for relative,content in payloads.items():
        path=OUT/relative
        if verify:
            assert path.read_bytes()==content,f"Reproduction mismatch: {path}"
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(content)
    print(json.dumps({"status":"verified" if verify else "generated","file_count":len(payloads),
        "shortlist":[row for row in candidates if row["submission_candidate"]]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true")
    run(parser.parse_args().verify)
