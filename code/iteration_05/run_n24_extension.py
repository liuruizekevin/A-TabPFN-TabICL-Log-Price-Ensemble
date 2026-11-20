"""Run one separately recorded follow-up after the eight fixed comparisons.

The 16-member, three-checkpoint ensemble allocates members 6:5:5. The library
uses model_index = i % num_models; 24 members allocate 8:8:8. This follow-up
changes only inference ensemble size relative to the original log/seed-42
baseline. It is adaptive, not part of the initial eight-configuration plan.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import round5_tabpfn as base


def main():
    base.configure_runtime()
    out = ROOT / "models/round5_tabpfn"
    input_path = ROOT / "data/raw/test.csv"
    completed = []
    for candidate in base.CANDIDATES:
        path = out / "results" / f"{candidate['id']}_summary.json"
        result = json.loads(path.read_text())
        assert result["status"] == "complete"
        completed.append({"id": result["id"], "cv_mean": result["cv_mean"],
                          "pooled_oof_log_rmse": result["pooled_oof_log_rmse"],
                          "summary_sha256": base.sha256(path)})
    candidate = {
        "id": "R5TP_log_n24_balanced_s42", "target": "log", "random_state": 42,
        "n_estimators": 24,
        "hypothesis": "Twenty-four total inference members balance the three checkpoints at 8:8:8 rather than the baseline 6:5:5 allocation, while retaining log targets and seed 42.",
        "extension_launcher_sha256": base.sha256(__file__),
        "planned_checkpoint_member_counts": [8, 8, 8],
        "follow_up_status": "One adaptive follow-up declared after the initial eight completed configurations; not part of their frozen plan.",
    }
    spec = base.make_spec(candidate, base.checkpoint_manifest(), input_path)
    plan = {"extension": "Exactly one balanced-checkpoint budget comparison",
            "initial_plan_sha256": base.sha256(out / "preregistered_plan.json"),
            "completed_initial_results": completed,
            "allocation_evidence": {
                "local_source": "build/tabpfn-env/lib/python3.12/site-packages/tabpfn/preprocessing/ensemble.py",
                "expression": "model_indices = [i % num_models for i in range(num_estimators)]",
                "source": "https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/preprocessing/ensemble.py"},
            "specification": spec}
    plan_path = out / "extension_n24_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("The recorded n24 extension plan differs from this invocation.")
    base.write_json(plan_path, plan)
    train, test, X, Xt = base.load_data(input_path)
    folds = list(KFold(5, shuffle=True, random_state=42).split(X))
    result = base.run_candidate(spec, out, train, test, X, Xt, folds)
    all_ids = [row["id"] for row in base.CANDIDATES] + [result["id"]]
    results = [json.loads((out / "results" / f"{name}_summary.json").read_text()) for name in all_ids]
    pd.DataFrame([{key: row[key] for key in ("id", "target", "random_state", "cv_mean", "cv_std",
                  "pooled_oof_log_rmse", "runtime_seconds")} for row in results]).to_csv(
        out / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
