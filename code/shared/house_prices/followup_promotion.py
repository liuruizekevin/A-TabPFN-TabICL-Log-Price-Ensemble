"""Merge adaptive follow-up evidence while retaining the complete two-phase state."""
import json
import shutil
from pathlib import Path

import pandas as pd


def promote_followup(main_dir,followup_dir,selected_experiment_id=None,selection_basis="Lowest primary development CV mean; leaderboard decision recorded separately"):
    main,followup=map(Path,[main_dir,followup_dir])
    current=json.loads((main/"results/summary.json").read_text())
    followup_summary=json.loads((followup/"results/summary.json").read_text())
    snapshot=main/"phase2_snapshot"
    if not snapshot.exists() or "followup" not in current:
        snapshot.mkdir(parents=True,exist_ok=True)
        for directory in ["results","artifacts","submissions"]:
            shutil.copytree(main/directory,snapshot/directory,dirs_exist_ok=True)
        for filename in ["final_submission.csv","baseline_submission.csv","model_results.csv"]:
            shutil.copy2(main/filename,snapshot/filename)
    phase2=json.loads((snapshot/"results/summary.json").read_text())
    old_scores=pd.read_csv(snapshot/"results/experiment_results.csv")
    new_scores=pd.read_csv(followup/"results/experiment_results.csv")
    assert set(old_scores.experiment_id).isdisjoint(new_scores.experiment_id)
    new_scores["phase"]=3
    combined=pd.concat([old_scores,new_scores],ignore_index=True)
    combined["baseline_absolute_improvement"]=phase2["baseline"]["cv_mean"]-combined.cv_mean
    combined["baseline_relative_improvement_pct"]=100*combined.baseline_absolute_improvement/phase2["baseline"]["cv_mean"]
    combined.to_csv(main/"results/experiment_results.csv",index=False)
    combined.to_csv(main/"model_results.csv",index=False)
    old_oof=pd.read_csv(snapshot/"results/oof_predictions.csv")
    new_oof=pd.read_csv(followup/"results/oof_predictions.csv")
    assert old_oof.Id.equals(new_oof.Id) and old_oof.fold.equals(new_oof.fold)
    pd.concat([old_oof,new_oof.filter(like="__log_prediction")],axis=1).to_csv(main/"results/oof_predictions.csv",index=False)
    pd.concat([pd.read_csv(snapshot/"results/fold_scores.csv"),pd.read_csv(followup/"results/fold_scores.csv")],ignore_index=True).to_csv(main/"results/fold_scores.csv",index=False)
    old_audit=json.loads((snapshot/"results/preprocessing_audit.json").read_text())
    new_audit=json.loads((followup/"results/preprocessing_audit.json").read_text())
    (main/"results/preprocessing_audit.json").write_text(json.dumps(old_audit+new_audit,indent=2))
    plan=json.loads((snapshot/"results/experiment_plan.json").read_text())
    plan["phase3"]=json.loads((followup/"results/followup_plan.json").read_text())
    (main/"results/experiment_plan.json").write_text(json.dumps(plan,indent=2))
    selected_id=selected_experiment_id or followup_summary["best_experiment"]
    if selected_id in new_scores.experiment_id.values:
        chosen=new_scores.set_index("experiment_id").loc[selected_id]
        should_promote=selected_experiment_id is not None or chosen.cv_mean<phase2["best_cv_mean"]
    elif selected_id==phase2["best_experiment"]:
        should_promote=False
    else:
        raise ValueError("Selected experiment must be a follow-up candidate or the previous winner.")
    summary=phase2.copy()
    summary["kaggle"]=current.get("kaggle",phase2.get("kaggle",{}))
    summary["experiment_count"]=len(combined)
    summary["followup_experiment_count"]=len(new_scores)
    summary["phase2_best_experiment"]=phase2["best_experiment"]
    summary["phase2_best_cv_mean"]=phase2["best_cv_mean"]
    summary["followup"]=followup_summary|{"promoted":should_promote,"selected_experiment":selected_id,"selection_basis":selection_basis}
    summary["runtime_seconds"]=phase2["runtime_seconds"]+followup_summary["runtime_seconds"]
    summary["best_cv_experiment"]=cv_best.experiment_id
    summary["best_cv_mean_across_experiments"]=float(cv_best.cv_mean)
    summary["best_cv_std_across_experiments"]=float(cv_best.cv_std)
    summary["best_cv_pooled_oof_log_rmse"]=float(cv_best.pooled_oof_log_rmse)
    summary["selected_experiment"]=summary["best_experiment"]
    summary["selected_cv_mean"]=summary["best_cv_mean"]
    summary["selection_criterion"]=selection_basis
    summary["best_experiment_field_meaning"]="Legacy field naming: best_experiment and best_cv_mean describe the selected delivered model. best_cv_experiment and best_cv_mean_across_experiments separately identify the lowest development-CV candidate."
    summary["validation"]["selection_limitation"]="Three adaptive development phases reused seed-42 folds. The small final follow-up was motivated by earlier CV and actual public leaderboard feedback. Its selected score is not an unbiased final test."
    (main/"results/summary.json").write_text(json.dumps(summary,indent=2))
    followup_summary["promoted"]=should_promote
    (followup/"results/summary.json").write_text(json.dumps(followup_summary,indent=2))
    return summary
