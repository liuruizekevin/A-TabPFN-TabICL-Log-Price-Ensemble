"""Small, explicitly adaptive scaler/SVR follow-up after observed leaderboard scores."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits

from .models import PricePredictor
from .optimization import WeightedLogPredictor, make_optimization_pipeline
from .preprocessing import read_data


def followup_specs():
    return [
        dict(id="F01_robust_c10_g001",scaler="RobustScaler",C=10.,gamma=.001),
        dict(id="F02_robust_c10_scale",scaler="RobustScaler",C=10.,gamma="scale"),
        dict(id="F03_robust_c20_g0003",scaler="RobustScaler",C=20.,gamma=.0003),
        dict(id="F04_standard_c10_scale",scaler="StandardScaler",C=10.,gamma="scale"),
    ]


def followup_pipeline(spec):
    model_spec=dict(family="SVR",params=dict(C=spec["C"],gamma=spec["gamma"],epsilon=.008,kernel="rbf",cache_size=512),enhanced=True,skew=True,filter_train=False)
    pipeline=make_optimization_pipeline(model_spec)
    if spec["scaler"]=="RobustScaler":
        # ColumnTransformer is unfitted; replace its numerical pipeline scaler.
        numeric=pipeline.named_steps["preprocessor"].transformers[0][1]
        numeric.set_params(scale=RobustScaler())
    return pipeline


def run_followup(data_dir,output_dir,phase2_dir):
    start=time.perf_counter()
    data_dir,output_dir,phase2_dir=map(Path,[data_dir,output_dir,phase2_dir])
    for subdir in ["results","candidate_models","submissions"]:
        (output_dir/subdir).mkdir(parents=True,exist_ok=True)
    plan={"phase":3,"motivation":"Bounded adaptive follow-up requested after phase-two CV and actual Kaggle leaderboard feedback; not part of the original two-phase plan.","specifications":followup_specs(),"common_parameters":{"epsilon":.008,"kernel":"rbf","enhanced_features":True,"skew_transform":True},"blends":{"components":["O09_native_mae","one new SVR","O22_xgboost_d3"],"weights":[.4,.4,.2]},"validation":{"n_splits":5,"shuffle":True,"random_state":42,"all_1460_rows_retained":True},"scope":"Same-data adaptive development. Scores are not unbiased confirmation and do not replace leaderboard evidence."}
    (output_dir/"results/followup_plan.json").write_text(json.dumps(plan,indent=2))
    train,test=read_data(data_dir/"train.csv"),read_data(data_dir/"test.csv")
    X=train.drop(columns="SalePrice");logs=np.log(train.SalePrice.to_numpy(float))
    reference_oof=pd.read_csv(phase2_dir/"results/oof_predictions.csv")
    folds=pd.read_csv(phase2_dir/"results/fold_assignments.csv")
    assert reference_oof.Id.equals(train.Id)
    cat_id,xgb_id="O09_native_mae","O22_xgboost_d3"
    cat_model=joblib.load(phase2_dir/"candidate_models"/(cat_id+".joblib"))
    xgb_model=joblib.load(phase2_dir/"candidate_models"/(xgb_id+".joblib"))
    cat_oof=reference_oof[cat_id+"__log_prediction"].to_numpy()
    xgb_oof=reference_oof[xgb_id+"__log_prediction"].to_numpy()
    splits=list(KFold(5,shuffle=True,random_state=42).split(X))
    rows,fold_scores,audit=[],[],[]
    all_oof={};prediction_checks={}
    with threadpool_limits(limits=4):
        for i,spec in enumerate(followup_specs(),5):
            t0=time.perf_counter();oof=np.zeros(len(X));actual_gamma=[]
            for fold,(tr,va) in enumerate(splits,1):
                assert (folds.iloc[va].fold==fold).all()
                pipeline=followup_pipeline(spec).fit(X.iloc[tr],logs[tr])
                oof[va]=pipeline.predict(X.iloc[va])
                fitted_ids=pipeline.named_steps["features"].base_.fit_row_ids_
                assert set(fitted_ids).isdisjoint(X.iloc[va].Id)
                actual_gamma.append(float(pipeline.named_steps["model"]._gamma))
                audit.append(dict(experiment_id=spec["id"],fold=fold,training_rows=len(tr),validation_rows=len(va),train_validation_disjoint=True,scaler_class=type(pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["scale"]).__name__,effective_gamma=actual_gamma[-1]))
            fitted=followup_pipeline(spec).fit(X,logs)
            predictor=PricePredictor([(fitted,"log")],spec["id"])
            blend_id=f"F{i:02d}_blend_"+spec["id"].split("_",1)[1]
            blend=WeightedLogPredictor([cat_model,predictor,xgb_model],blend_id,[.4,.4,.2])
            blended=.4*cat_oof+.4*oof+.2*xgb_oof
            for id_,model,prediction,kind,params in [(spec["id"],predictor,oof,"SVR",{**spec,"epsilon":.008,"fold_effective_gamma":actual_gamma}),(blend_id,blend,blended,"Weighted log-space blend",{"components":[cat_id,spec["id"],xgb_id],"weights":[.4,.4,.2]})]:
                all_oof[id_]=prediction
                scores=[]
                for fold,(_,va) in enumerate(splits,1):
                    score=float(np.sqrt(np.mean((logs[va]-prediction[va])**2)));scores.append(score)
                    fold_scores.append(dict(experiment_id=id_,fold=fold,log_rmse=score,n_train=1168,n_valid=292))
                values=model.predict(test)
                assert np.isfinite(values).all() and (values>0).all()
                model_path=output_dir/"candidate_models"/(id_+".joblib")
                joblib.dump(model,model_path,compress=3)
                reloaded=joblib.load(model_path)
                difference=float(np.max(np.abs(reloaded.predict(test)-values)))
                assert difference<1e-6
                prediction_checks[id_]=difference
                pd.DataFrame({"Id":test.Id,"SalePrice":values}).to_csv(output_dir/"submissions"/(id_+".csv"),index=False)
                row=dict(experiment_id=id_,model=kind,stage="adaptive_followup" if kind=="SVR" else "adaptive_followup_blend",target="log",semantic_missing=True,feature_groups="area,age,baths,porch,quality,enhanced",skew_transform="component-specific" if kind!="SVR" else True,params_json=json.dumps(params,sort_keys=True),cv_mean=float(np.mean(scores)),cv_std=float(np.std(scores,ddof=1)),pooled_oof_log_rmse=float(np.sqrt(np.mean((logs-prediction)**2))),runtime_seconds=time.perf_counter()-t0 if kind=="SVR" else 0.,raw_predictions_clipped=0)
                rows.append(row)
                print(f"{id_}: {row['cv_mean']:.6f} +/- {row['cv_std']:.6f}",flush=True)
    results=pd.DataFrame(rows)
    results.to_csv(output_dir/"results/experiment_results.csv",index=False)
    pd.DataFrame(fold_scores).to_csv(output_dir/"results/fold_scores.csv",index=False)
    pd.DataFrame({"Id":train.Id,"fold":folds.fold,"SalePrice":train.SalePrice,"log_SalePrice":logs,**{k+"__log_prediction":v for k,v in all_oof.items()}}).to_csv(output_dir/"results/oof_predictions.csv",index=False)
    folds.to_csv(output_dir/"results/fold_assignments.csv",index=False)
    (output_dir/"results/preprocessing_audit.json").write_text(json.dumps(audit,indent=2))
    best=results.loc[results.cv_mean.idxmin()]
    summary={"phase":3,"experiment_count":len(results),"best_experiment":best.experiment_id,"best_cv_mean":float(best.cv_mean),"best_cv_std":float(best.cv_std),"best_pooled_oof_log_rmse":float(best.pooled_oof_log_rmse),"best_parameters":json.loads(best.params_json),"phase2_reference_cv_mean":float(np.mean([np.sqrt(np.mean((logs[va]-reference_oof.loc[va,"O26_cat_svr_xgb__log_prediction"])**2)) for _,va in splits])),"reload_prediction_checks":prediction_checks,"runtime_seconds":time.perf_counter()-start,"promoted":False,"limitation":plan["scope"]}
    (output_dir/"results/summary.json").write_text(json.dumps(summary,indent=2))
    return summary
