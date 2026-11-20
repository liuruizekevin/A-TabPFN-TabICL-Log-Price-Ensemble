"""Promote phase-two results without losing the first-stage evidence."""
import json
import shutil
from importlib.metadata import version
from pathlib import Path

import pandas as pd


def promote_optimization(phase1_dir, optimization_dir):
    root, optimization = Path(phase1_dir), Path(optimization_dir)
    original = json.loads((root/"results/summary.json").read_text())
    improved = json.loads((optimization/"results/summary.json").read_text())
    snapshot = root/"phase1_snapshot"
    # A fresh phase-one run must refresh the snapshot even if this destination
    # previously held a completed two-phase run. Re-promoting an already merged
    # run instead reads its preserved phase-one snapshot.
    if not snapshot.exists() or "optimization" not in original:
        snapshot.mkdir(parents=True, exist_ok=True)
        for directory in ["results","artifacts","submissions"]:
            shutil.copytree(root/directory,snapshot/directory,dirs_exist_ok=True)
        for filename in ["final_submission.csv","baseline_submission.csv","model_results.csv"]:
            shutil.copy2(root/filename,snapshot/filename)
    phase1 = json.loads((snapshot/"results/summary.json").read_text())
    if phase1["data"]["train_sha256"] != improved["train_sha256"] or phase1["data"]["test_sha256"] != improved["test_sha256"]:
        raise ValueError("Phase datasets differ; promotion refused.")
    first_scores = pd.read_csv(snapshot/"results/experiment_results.csv")
    second_scores = pd.read_csv(optimization/"results/experiment_results.csv")
    assert set(first_scores.experiment_id).isdisjoint(second_scores.experiment_id)
    first_scores["phase"] = 1
    second_scores["phase"] = 2
    merged = pd.concat([first_scores,second_scores],ignore_index=True)
    merged.to_csv(root/"results/experiment_results.csv",index=False)
    merged.to_csv(root/"model_results.csv",index=False)
    folds1 = pd.read_csv(snapshot/"results/fold_scores.csv")
    folds2 = pd.read_csv(optimization/"results/fold_scores.csv")
    pd.concat([folds1,folds2],ignore_index=True).to_csv(root/"results/fold_scores.csv",index=False)
    first_oof = pd.read_csv(snapshot/"results/oof_predictions.csv")
    second_oof = pd.read_csv(optimization/"results/oof_predictions.csv")
    assert first_oof.Id.equals(second_oof.Id)
    predictions2 = second_oof.filter(like="__log_prediction")
    pd.concat([first_oof,predictions2],axis=1).to_csv(root/"results/oof_predictions.csv",index=False)
    audit1 = json.loads((snapshot/"results/preprocessing_audit.json").read_text())
    audit2 = json.loads((optimization/"results/preprocessing_audit.json").read_text())
    (root/"results/preprocessing_audit.json").write_text(json.dumps(audit1+audit2,indent=2))
    plan1 = json.loads((snapshot/"results/experiment_plan.json").read_text())
    plan2 = json.loads((optimization/"results/optimization_plan.json").read_text())
    (root/"results/experiment_plan.json").write_text(json.dumps({"phase1":plan1,"phase2":plan2,"planning_scope":"Phase two was designed after reviewing phase-one results. Each phase has its recorded bounded plan; blend family components are selected using development CV and weights are fixed. This is adaptive development, not a single untouched confirmatory experiment."},indent=2))
    should_promote=improved["best_cv_mean"]<phase1["best_cv_mean"]
    summary=phase1.copy()
    summary["experiment_count"]=len(merged)
    summary["phase1_experiment_count"]=len(first_scores)
    summary["optimization_experiment_count"]=len(second_scores)
    summary["phase1_best_experiment"]=phase1["best_experiment"]
    summary["phase1_best_cv_mean"]=phase1["best_cv_mean"]
    summary["optimization"]=improved|{"promoted":should_promote}
    summary["runtime_seconds"]=phase1["runtime_seconds"]+improved["runtime_seconds"]
    summary["environment"]["packages"].update({p:version(p) for p in ["lightgbm","xgboost"]})
    summary["validation"]["selection_limitation"]="Two adaptive development phases reused the same five seed-42 folds for features, hyperparameters, family selection and fixed-weight blend selection. The selected estimate may be optimistic. No Kaggle score has been measured."
    if should_promote:
        for key in ["best_experiment","best_model","best_cv_mean","best_cv_std","best_pooled_oof_log_rmse","best_components","best_parameters"]:
            summary[key]=improved[key]
        shutil.copy2(optimization/"artifacts/best_model.joblib",root/"artifacts/best_model.joblib")
        shutil.copy2(optimization/"final_submission.csv",root/"final_submission.csv")
        shutil.copy2(optimization/"final_submission.csv",root/"submissions/final_submission.csv")
        submission=pd.read_csv(root/"final_submission.csv")
        summary["test_prediction_summary"]={"min":float(submission.SalePrice.min()),"median":float(submission.SalePrice.median()),"max":float(submission.SalePrice.max())}
    summary["absolute_improvement_over_baseline"]=summary["baseline"]["cv_mean"]-summary["best_cv_mean"]
    summary["relative_improvement_over_baseline_pct"]=100*summary["absolute_improvement_over_baseline"]/summary["baseline"]["cv_mean"]
    summary["absolute_improvement_over_phase1"]=phase1["best_cv_mean"]-summary["best_cv_mean"]
    (root/"results/summary.json").write_text(json.dumps(summary,indent=2))
    improved["promoted"]=should_promote
    (optimization/"results/summary.json").write_text(json.dumps(improved,indent=2))
    return summary
