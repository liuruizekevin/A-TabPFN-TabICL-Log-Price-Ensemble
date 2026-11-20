"""One explicitly adaptive fixed 80/20 subsample/domain mixture.

Run .venv/bin/python scripts/round6_subsample_blend.py, then --verify.
No additional model fit, weight search, row removal, or prediction edits.
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
OUT = ROOT / "models/round6_subsample_blend"
NAME = "R6E06_subsample_domain20"
SUBSAMPLE = "R6TP_noY_subsample090_s42"
DOMAIN = "R4T_v2_log_n32_s42_domain"
BASE = "R5E08_noY_domain20"
WEIGHTS = {SUBSAMPLE: .8, DOMAIN: .2}
LIMITATION = (
    "One explicitly adaptive follow-up chosen after R6TP_noY_subsample090_s42 scored "
    "0.11054 online, improving on E05 at 0.11060 and E08 at 0.11064. Reuse the established "
    "80/20 weights with the new 90-percent-context-subsampling component and original "
    "domain model; no weight search. Reused seed42 OOF is development evidence, not an "
    "independent holdout or a public-score prediction. All 1460 validation rows remain."
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(verify=False):
    source_sha256 = sha(__file__)
    if verify:
        # Replay recorded metadata only after checking current and historical code identity.
        source_sha256 = json.loads((OUT / "results/plan.json").read_text())["source_sha256"]
        verify_source(__file__, source_sha256, ROOT)
    train, test = pd.read_csv(ROOT/"data/raw/train.csv"),pd.read_csv(ROOT/"data/raw/test.csv")
    source_data = json.loads((ROOT/"data/source_manifest.json").read_text())
    for filename in ("train.csv","test.csv"):
        assert sha(ROOT/"data/raw"/filename)==source_data["files"][filename]["sha256"]
    assert len(train)==1460 and len(test)==1459 and train.Id.is_unique and test.Id.is_unique
    assert "SalePrice" not in test and not set(train.Id)&set(test.Id)
    y=np.log(train.SalePrice.to_numpy(float))
    fold=np.zeros(len(y),int)
    for f,(_,va) in enumerate(KFold(5,shuffle=True,random_state=42).split(train),1): fold[va]=f
    oof,testing,sources={},{},[]
    for name,directory in [(SUBSAMPLE,"round6_tabpfn"),(DOMAIN,"round4_tabpfn"),(BASE,"round5_ensembles/extension")]:
        op=ROOT/"models"/directory/"results"/f"{name}_oof.csv"
        tp=op.with_name(f"{name}_test_logs.csv")
        a,b=pd.read_csv(op),pd.read_csv(tp)
        assert np.array_equal(a.Id,train.Id) and np.array_equal(b.Id,test.Id)
        assert np.array_equal(a.fold,fold) and np.allclose(a.log_SalePrice,y,rtol=0,atol=1e-12)
        oof[name],testing[name]=a.log_prediction.to_numpy(float),b.log_prediction.to_numpy(float)
        assert np.isfinite(oof[name]).all() and np.isfinite(testing[name]).all()
        sources.append({"id":name,"oof_path":str(op.relative_to(ROOT)),"oof_sha256":sha(op),
                        "test_path":str(tp.relative_to(ROOT)),"test_sha256":sha(tp),
                        "test_fit":"original full-data fit"})
    summary_path=ROOT/"models/round6_tabpfn/results"/f"{SUBSAMPLE}_summary.json"
    subsample_summary=json.loads(summary_path.read_text())
    assert subsample_summary["status"]=="complete"
    assert subsample_summary["inference_config"]=={"REGRESSION_Y_PREPROCESS_TRANSFORMS":[None],"SUBSAMPLE_SAMPLES":0.9}
    for relative,digest in subsample_summary["artifact_sha256"].items():
        assert sha(ROOT/"models/round6_tabpfn"/relative)==digest
    p=sum(w*oof[name] for name,w in WEIGHTS.items())
    t=sum(w*testing[name] for name,w in WEIGHTS.items())
    prices=np.exp(t)
    assert np.isfinite(prices).all() and (prices>0).all()

    def metrics(values):
        fs=[float(np.sqrt(np.mean((values[fold==f]-y[fold==f])**2))) for f in range(1,6)]
        return {"pooled_oof_log_rmse":float(np.sqrt(np.mean((values-y)**2))),
                "cv_mean":float(np.mean(fs)),"cv_std":float(np.std(fs,ddof=1)),"fold_scores":fs}

    current,baseline=metrics(p),metrics(oof[BASE])
    subsample_metrics=metrics(oof[SUBSAMPLE])
    gain=(oof[BASE]-y)**2-(p-y)**2
    payloads={}
    def frame(path,value):payloads[path]=value.to_csv(index=False).encode()
    def obj(path,value):payloads[path]=(json.dumps(value,indent=2,allow_nan=False)+"\n").encode()
    frame(f"submissions/{NAME}.csv",pd.DataFrame({"Id":test.Id,"SalePrice":prices}))
    frame("results/oof_predictions.csv",pd.DataFrame({"Id":train.Id,"fold":fold,"log_SalePrice":y,
        "log_prediction":p,"baseline_log_prediction":oof[BASE],"squared_error_reduction":gain}))
    frame("results/test_log_predictions.csv",pd.DataFrame({"Id":test.Id,"log_prediction":t}))
    frame("results/fold_scores.csv",pd.DataFrame([{"fold":f,"rows":int((fold==f).sum()),"log_rmse":v,
        "delta_vs_e08":v-baseline["fold_scores"][f-1]} for f,v in enumerate(current["fold_scores"],1)]))
    summary={"status":"complete","experiment_id":NAME,"weights":WEIGHTS,**current,"baseline":baseline,
        "delta_pooled_vs_e08":current["pooled_oof_log_rmse"]-baseline["pooled_oof_log_rmse"],
        "subsample_only":subsample_metrics,
        "delta_pooled_vs_subsample":current["pooled_oof_log_rmse"]-subsample_metrics["pooled_oof_log_rmse"],
        "folds_improved_vs_subsample":sum(a<b for a,b in zip(current["fold_scores"],subsample_metrics["fold_scores"])),
        "folds_improved_vs_e08":sum(a<b for a,b in zip(current["fold_scores"],baseline["fold_scores"])),
        "training_rows":1460,"training_rows_removed":0,"test_rows":1459,"public_score":None,
        "selection_limitation":LIMITATION,
        "submission_path":str((OUT/f"submissions/{NAME}.csv").relative_to(ROOT)),
        "submission_sha256":hashlib.sha256(payloads[f"submissions/{NAME}.csv"]).hexdigest(),
        "total_squared_error_reduction":float(gain.sum()),
        "id1299_and_524_squared_error_reduction":float(gain[np.isin(train.Id,[1299,524])].sum()),
        "largest_squared_error_reductions":[{"Id":int(train.Id.iloc[i]),"reduction":float(gain[i])} for i in np.argsort(gain)[-5:][::-1]],
        "largest_squared_error_increases":[{"Id":int(train.Id.iloc[i]),"increase":float(-gain[i])} for i in np.argsort(gain)[:5]]}
    obj("results/summary.json",summary)
    obj("results/source_manifest.json",sources)
    obj("results/plan.json",{"experiment_id":NAME,"weights":WEIGHTS,"candidate_count":1,"weight_search":False,
        "selection_limitation":LIMITATION,"source_sha256":source_sha256,
        "subsample_model_summary_path":str(summary_path.relative_to(ROOT)),"subsample_model_summary_sha256":sha(summary_path),
        "train_sha256":sha(ROOT/"data/raw/train.csv"),"test_sha256":sha(ROOT/"data/raw/test.csv"),
        "fold_seed":42,"reproduction_command":".venv/bin/python scripts/round6_subsample_blend.py --verify"})
    obj("results/validation.json",{"all1460_oof_rows_retained":True,"all1459_test_rows_retained":True,
        "Id_targets_folds_and_data_hashes_checked":True,"subsample_source_artifact_hashes_checked":True,
        "original_subsample_test_logs_used":True,"finite_positive_prices":True,"source_sha256":source_sha256,
        "output_sha256":{path:hashlib.sha256(content).hexdigest() for path,content in payloads.items()}})
    for relative,content in payloads.items():
        path=OUT/relative
        if verify:assert path.read_bytes()==content,f"Reproduction mismatch: {path}"
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(content)
    print(json.dumps({"operation":"verified" if verify else "generated",**summary},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true")
    run(parser.parse_args().verify)
