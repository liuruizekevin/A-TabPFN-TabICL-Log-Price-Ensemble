"""Isolated round-four boosting search with unchanged development folds.

Every configuration is declared before fitting. Test predictions are exported both
from a full-training fit and as the average of five fold fits. The latter is a
different deployment procedure; its shared OOF score is not a separate CV test.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .experiment import file_sha256
from .models import PricePredictor
from .optimization import EnhancedFeatures, make_optimization_pipeline, optimization_specs
from .followup import followup_pipeline, followup_specs
from .preprocessing import make_preprocessor, read_data


class CenteredRegressor(RegressorMixin, BaseEstimator):
    """Center the training target to initialize robust losses near its location."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.offset_ = float(np.mean(y))
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y) - self.offset_)
        return self

    def predict(self, X):
        return self.estimator_.predict(X) + self.offset_


class ArithmeticLogBlendPredictor(PricePredictor):
    """Retain the exact recorded arithmetic order and component dtypes."""
    def __init__(self, components, name, weights):
        self.components=components
        self.name=name
        self.weights=weights

    def predict_log(self, X):
        terms=[w*m.predict_log(X) for w,m in zip(self.weights,self.components)]
        result=terms[0]
        for term in terms[1:]:
            result=result+term
        return result


def specifications():
    result = []
    def add(id_, family, params, center=False):
        result.append(dict(id=id_, family=family, params=params, enhanced=True,
                           skew=False, filter_train=False, center_target=center))

    xgb = dict(n_estimators=2600, learning_rate=.018, max_depth=3,
               min_child_weight=2, subsample=.8, colsample_bytree=.7,
               reg_alpha=.01, reg_lambda=1., objective="reg:squarederror",
               random_state=42, n_jobs=4, tree_method="hist")
    for id_, changes in [
        ("B01_xgb_huber010", dict(objective="reg:pseudohubererror", huber_slope=.1)),
        ("B02_xgb_huber020", dict(objective="reg:pseudohubererror", huber_slope=.2)),
        ("B03_xgb_huber005", dict(objective="reg:pseudohubererror", huber_slope=.05)),
        ("B04_xgb_absolute", dict(objective="reg:absoluteerror")),
        ("B05_xgb_d2_long", dict(max_depth=2,n_estimators=4000,learning_rate=.02,min_child_weight=1)),
        ("B06_xgb_d3_regularized", dict(reg_lambda=8.,reg_alpha=.05,min_child_weight=3)),
        ("B07_xgb_d4_regularized", dict(max_depth=4,reg_lambda=5.,min_child_weight=3,colsample_bytree=.6)),
    ]:
        add(id_, "XGBoost", xgb | changes, center="huber" in id_)
    lgb = dict(n_estimators=3200, learning_rate=.012, num_leaves=8,
               max_depth=-1, min_child_samples=15, colsample_bytree=.8,
               subsample=.8, subsample_freq=1, reg_alpha=.05, reg_lambda=2.,
               random_state=42, n_jobs=4, verbosity=-1, deterministic=True,
               force_col_wise=True)
    for id_, changes in [
        ("B08_lgb_fair010", dict(objective="fair",fair_c=.1)),
        ("B09_lgb_fair020", dict(objective="fair",fair_c=.2)),
        ("B10_lgb_huber", dict(objective="huber",alpha=.85)),
        ("B11_lgb_absolute", dict(objective="regression_l1",num_leaves=12)),
    ]:
        add(id_, "LightGBM", lgb | changes)
    cat = dict(iterations=3000, depth=4, learning_rate=.025, l2_leaf_reg=8.,
               loss_function="MAE", random_seed=42, thread_count=4,
               verbose=False, allow_writing_files=False)
    for id_, changes, center, family in [
        ("B12_cat_mae_d3", dict(depth=3,iterations=4200), False, "CatBoostNative"),
        ("B13_cat_mae_d5", dict(depth=5), False, "CatBoostNative"),
        ("B14_cat_mae_d6", dict(depth=6,iterations=2400), False, "CatBoostNative"),
        ("B15_cat_mae_seed13", dict(random_seed=13), False, "CatBoostNative"),
        ("B16_cat_mae_seed2026", dict(random_seed=2026), False, "CatBoostNative"),
        ("B17_cat_mae_l2_3", dict(l2_leaf_reg=3.,iterations=4000), False, "CatBoostNative"),
        ("B18_cat_mae_l2_20", dict(l2_leaf_reg=20.,rsm=.8), False, "CatBoostNative"),
        ("B19_cat_huber010_center", dict(loss_function="Huber:delta=0.1",iterations=3600), True, "CatBoostNative"),
        ("B20_cat_huber020_center", dict(loss_function="Huber:delta=0.2",iterations=3600), True, "CatBoostNative"),
        ("B21_cat_rmse_d5_l2_3", dict(loss_function="RMSE",depth=5,l2_leaf_reg=3.), False, "CatBoostNative"),
        ("B22_cat_ohe_mae_d3", dict(depth=3,iterations=4200), False, "CatBoostOHE"),
        ("B23_cat_ohe_mae_d4", dict(depth=4,iterations=3500), False, "CatBoostOHE"),
    ]:
        add(id_, family, cat | changes, center)
    gbr=dict(n_estimators=3000,learning_rate=.05,max_depth=4,max_features="sqrt",
             min_samples_leaf=15,min_samples_split=10,loss="huber",random_state=42)
    add("B24_gbr_huber_classic", "GradientBoosting", gbr)
    add("B25_gbr_huber_shallow", "GradientBoosting", gbr|dict(max_depth=3,n_estimators=4000,learning_rate=.03,min_samples_leaf=10))
    # Adaptive extension, run separately after observing the depth3--6 trend.
    add("B28_cat_mae_d7", "CatBoostNative", cat|dict(depth=7,iterations=2200))
    add("B29_cat_mae_d6_long", "CatBoostNative", cat|dict(depth=6,iterations=3600,learning_rate=.02))
    return result


def pipeline(spec):
    if spec["family"] == "GradientBoosting":
        estimator=Pipeline([("features",EnhancedFeatures(True)),("preprocessor",make_preprocessor(False)),
                            ("model",GradientBoostingRegressor(**spec["params"]))])
    else:
        estimator = make_optimization_pipeline(spec)
    if spec.get("center_target"):
        estimator.set_params(model=CenteredRegressor(estimator.named_steps["model"]))
    return estimator


def run(data_dir, output_dir, phase2_dir, followup_dir, only=None):
    started = time.perf_counter()
    data_dir, output_dir, phase2_dir, followup_dir = map(Path, (data_dir,output_dir,phase2_dir,followup_dir))
    for folder in ("results","candidate_models","submissions","fold_predictions"):
        (output_dir/folder).mkdir(parents=True,exist_ok=True)
    specs=specifications()
    if only:
        requested=set(only.split(","));specs=[s for s in specs if s["id"] in requested]
    else:
        # The primary frozen batch remains B01--B23; extensions require --only.
        specs=[s for s in specs if int(s["id"][1:3])<=23]
    train,test=read_data(data_dir/"train.csv"),read_data(data_dir/"test.csv")
    X=train.drop(columns="SalePrice");y=np.log(train.SalePrice.to_numpy(float))
    folds=pd.read_csv(phase2_dir/"results/fold_assignments.csv")
    old_oof=pd.read_csv(phase2_dir/"results/oof_predictions.csv")
    svr_oof=pd.read_csv(followup_dir/"results/oof_predictions.csv")
    assert old_oof.Id.equals(train.Id) and svr_oof.Id.equals(train.Id)
    splits=list(KFold(5,shuffle=True,random_state=42).split(X))
    for fold,(_,va) in enumerate(splits,1):
        assert (folds.iloc[va].fold==fold).all()
    refs={}
    for id_,directory,source in [("O09_native_mae",phase2_dir,old_oof),
                                  ("O22_xgboost_d3",phase2_dir,old_oof),
                                  ("F03_robust_c20_g0003",followup_dir,svr_oof)]:
        model=joblib.load(directory/"candidate_models"/(id_+".joblib"))
        refs[id_]=(source[id_+"__log_prediction"].to_numpy(),model.predict_log(test))
    cat_oof,cat_test=refs["O09_native_mae"]
    xgb_oof,xgb_test=refs["O22_xgboost_d3"]
    sv_oof,sv_test=refs["F03_robust_c20_g0003"]
    baseline_oof=.4*cat_oof+.4*sv_oof+.2*xgb_oof
    plan={"phase":"round4_boosting","specifications":specs,
          "data_sha256":{p.name:file_sha256(p) for p in [data_dir/"train.csv",data_dir/"test.csv"]},
          "validation":{"folds":5,"shuffle":True,"seed":42,"all_rows":1460,"training_only_preprocessing":True},
          "deployment":"Full fit and mean of fold test log predictions, exported separately; fold-averaged deployment shares the base OOF validation score, not an independently evaluated variant.",
          "fixed_blends":"Native/OHE CatBoost replaces O09 at weight .4. XGB/LightGBM replaces O22 at weight .2. F03 SVR remains .4. Blend weights are fixed configuration values.",
          "sources":["https://catboost.ai/docs/en/concepts/loss-functions-regression","https://xgboost.readthedocs.io/en/stable/parameter.html","https://lightgbm.readthedocs.io/en/stable/Parameters.html"],
          "limitation":"Adaptive development reuses prior folds and leaderboard feedback; no unbiased final performance claim."}
    (output_dir/"results/plan.json").write_text(json.dumps(plan,indent=2))
    records=[];fold_records=[];audits=[];failures=[];oofs={};test_logs={};bagged_logs={}

    def write_checkpoint():
        pd.DataFrame(records).to_csv(output_dir/"results/experiment_results.csv",index=False)
        pd.DataFrame(fold_records).to_csv(output_dir/"results/fold_scores.csv",index=False)
        pd.DataFrame({"Id":train.Id,"fold":folds.fold,"SalePrice":train.SalePrice,"log_SalePrice":y,
                      **{k+"__log_prediction":v for k,v in oofs.items()}}).to_csv(output_dir/"results/oof_predictions.csv",index=False)
        pd.DataFrame({"Id":test.Id,**{k+"__log_prediction":v for k,v in test_logs.items()}}).to_csv(output_dir/"results/test_log_predictions.csv",index=False)
        pd.DataFrame({"Id":test.Id,**{k+"__log_prediction":v for k,v in bagged_logs.items()}}).to_csv(output_dir/"results/test_fold_average_log_predictions.csv",index=False)
        (output_dir/"results/preprocessing_audit.json").write_text(json.dumps(audits,indent=2))
        (output_dir/"results/failures.json").write_text(json.dumps(failures,indent=2))

    def record(id_,predictions,full_test,bagged_test,family,params,runtime):
        scores=[]
        for fold,(_,va) in enumerate(splits,1):
            score=float(np.sqrt(np.mean((y[va]-predictions[va])**2)));scores.append(score)
            fold_records.append(dict(experiment_id=id_,fold=fold,log_rmse=score,n_train=1168,n_valid=292))
        row=dict(experiment_id=id_,model=family,stage="round4_boosting",target="log",cv_mean=float(np.mean(scores)),
                 cv_std=float(np.std(scores,ddof=1)),pooled_oof_log_rmse=float(np.sqrt(np.mean((y-predictions)**2))),
                 runtime_seconds=runtime,params_json=json.dumps(params,sort_keys=True))
        records.append(row);oofs[id_]=predictions;test_logs[id_]=full_test;bagged_logs[id_]=bagged_test
        for suffix,values in [("",full_test),("_foldavg",bagged_test)]:
            values=np.exp(values)
            assert len(values)==1459 and np.isfinite(values).all() and (values>0).all()
            pd.DataFrame({"Id":test.Id,"SalePrice":values}).to_csv(output_dir/"submissions"/(id_+suffix+".csv"),index=False)
        print(f"{id_}: mean={row['cv_mean']:.8f} pooled={row['pooled_oof_log_rmse']:.8f} runtime={runtime:.1f}s",flush=True)
        return row

    with threadpool_limits(limits=4):
        for spec in specs:
            t0=time.perf_counter();id_=spec["id"]
            try:
                prediction=np.full(len(X),np.nan);fold_test=[]
                for fold,(tr,va) in enumerate(splits,1):
                    fitted=pipeline(spec).fit(X.iloc[tr],y[tr])
                    prediction[va]=fitted.predict(X.iloc[va]);fold_test.append(fitted.predict(test))
                    fit_ids=fitted.named_steps["features"].base_.fit_row_ids_
                    assert set(fit_ids)==set(X.iloc[tr].Id) and set(fit_ids).isdisjoint(X.iloc[va].Id)
                    audits.append(dict(experiment_id=id_,fold=fold,training_rows=len(tr),validation_rows=len(va),all_validation_rows_retained=True,train_validation_disjoint=True,target_centering_training_only=spec["center_target"]))
                assert np.isfinite(prediction).all()
                fitted=pipeline(spec).fit(X,y)
                predictor=PricePredictor([(fitted,"log")],id_)
                full_test=predictor.predict_log(test);average_test=np.mean(fold_test,axis=0)
                joblib.dump(predictor,output_dir/"candidate_models"/(id_+".joblib"),compress=3)
                pd.DataFrame({"Id":test.Id,**{f"fold_{j}":v for j,v in enumerate(fold_test,1)}}).to_csv(output_dir/"fold_predictions"/(id_+".csv"),index=False)
                record(id_,prediction,full_test,average_test,spec["family"],spec,time.perf_counter()-t0)
                if spec["family"].startswith("CatBoost"):
                    b_oof=.4*prediction+.4*sv_oof+.2*xgb_oof
                    b_test=.4*full_test+.4*sv_test+.2*xgb_test
                    b_bag=.4*average_test+.4*sv_test+.2*xgb_test
                    components=[id_,"F03_robust_c20_g0003","O22_xgboost_d3"]
                else:
                    b_oof=.4*cat_oof+.4*sv_oof+.2*prediction
                    b_test=.4*cat_test+.4*sv_test+.2*full_test
                    b_bag=.4*cat_test+.4*sv_test+.2*average_test
                    components=["O09_native_mae","F03_robust_c20_g0003",id_]
                record(id_+"_blend",b_oof,b_test,b_bag,"Fixed log blend",{"components":components,"weights":[.4,.4,.2]},0.)
            except Exception as exc:
                failures.append(dict(experiment_id=id_,error=repr(exc),runtime_seconds=time.perf_counter()-t0))
                print(f"FAILED {id_}: {exc!r}",flush=True)
            write_checkpoint()
    folds.to_csv(output_dir/"results/fold_assignments.csv",index=False)
    frame=pd.DataFrame(records)
    best=frame.loc[frame.cv_mean.idxmin()].to_dict() if len(frame) else None
    summary={"experiment_count":len(records),"base_specification_count":len(specs),"best":best,
             "reference_F07_cv_mean":float(np.mean([np.sqrt(np.mean((y[v]-baseline_oof[v])**2)) for _,v in splits])),
             "runtime_seconds":time.perf_counter()-started,"failures":failures,"promoted":False}
    (output_dir/"results/summary.json").write_text(json.dumps(summary,indent=2))
    return summary


def run_reference_bagging(data_dir, output_dir, phase2_dir, followup_dir):
    """Reproduce F07 OOF and average its five fold models for deployment."""
    started=time.perf_counter()
    data_dir,output_dir,phase2_dir,followup_dir=map(Path,(data_dir,output_dir,phase2_dir,followup_dir))
    output_dir=output_dir/"reference_bagging"
    for name in ["results","candidate_models","submissions"]:
        (output_dir/name).mkdir(parents=True,exist_ok=True)
    train,test=read_data(data_dir/"train.csv"),read_data(data_dir/"test.csv")
    X=train.drop(columns="SalePrice");y=np.log(train.SalePrice.to_numpy(float))
    splits=list(KFold(5,shuffle=True,random_state=42).split(X))
    phase_oof=pd.read_csv(phase2_dir/"results/oof_predictions.csv")
    followup_oof=pd.read_csv(followup_dir/"results/oof_predictions.csv")
    references=[("O09_native_mae",phase2_dir,phase_oof,.4),
                ("F03_robust_c20_g0003",followup_dir,followup_oof,.4),
                ("O22_xgboost_d3",phase2_dir,phase_oof,.2)]
    combined_oof=np.zeros(len(X));combined_test=np.zeros(len(test));combined_full=np.zeros(len(test))
    checks={};component_records={};all_models=[];all_weights=[];fold_ids=np.zeros(len(X),int)
    cache_path=output_dir/"candidate_models/B26_F07_fold_bag.joblib"
    cached=joblib.load(cache_path) if cache_path.exists() else None
    if cached is not None:
        assert cached.name=="B26_F07_fold_bag" and len(cached.components)==15
    with threadpool_limits(limits=4):
        for id_,directory,source,weight in references:
            assert source.Id.equals(train.Id)
            oof=np.zeros(len(X));test_logs=[]
            for fold,(tr,va) in enumerate(splits,1):
                if cached is not None:
                    fitted=cached.components[len(all_models)].components[0][0]
                elif id_.startswith("F"):
                    fitted=followup_pipeline(next(s for s in followup_specs() if s["id"]==id_))
                else:
                    fitted=make_optimization_pipeline(next(s for s in optimization_specs() if s["id"]==id_))
                if cached is None:
                    fitted.fit(X.iloc[tr],y[tr])
                oof[va]=fitted.predict(X.iloc[va]);test_logs.append(np.asarray(fitted.predict(test),dtype=np.float64));fold_ids[va]=fold
                fit_ids=fitted.named_steps["features"].base_.fit_row_ids_
                assert set(fit_ids)==set(X.iloc[tr].Id) and set(fit_ids).isdisjoint(X.iloc[va].Id)
                predictor=PricePredictor([(fitted,"log")],id_+f"_fold{fold}")
                all_models.append(predictor);all_weights.append(weight/5)
            difference=float(np.max(np.abs(oof-source[id_+"__log_prediction"].to_numpy())))
            assert difference<1e-8, (id_,difference)
            checks[id_]=difference
            full=joblib.load(directory/"candidate_models"/(id_+".joblib"))
            average=np.mean(test_logs,axis=0)
            full_log=full.predict_log(test)
            combined_oof+=weight*oof;combined_test+=weight*average;combined_full+=weight*full_log
            component_records[id_]=dict(weight=weight,fold_train_rows=1168,validation_rows=292,oof_max_abs_difference=difference)
            pd.DataFrame({"Id":test.Id,"fullfit":full_log,"fold_average":average,
                          **{f"fold{fold}":values for fold,values in enumerate(test_logs,1)}}).to_csv(output_dir/"results"/(id_+"_test_logs.csv"),index=False)
            print(f"Reference bagging {id_}: OOF maximum difference {difference:.3g}",flush=True)
    from .optimization import WeightedLogPredictor
    model=WeightedLogPredictor(all_models,"B26_F07_fold_bag",all_weights)
    joblib.dump(model,output_dir/"candidate_models/B26_F07_fold_bag.joblib",compress=3)
    assert np.max(np.abs(model.predict_log(test)-combined_test))<1e-10
    for id_,logs in [("B26_F07_fold_bag",combined_test),("B27_F07_half_full_half_bag",.5*combined_test+.5*combined_full)]:
        pd.DataFrame({"Id":test.Id,"SalePrice":np.exp(logs)}).to_csv(output_dir/"submissions"/(id_+".csv"),index=False)
    full_models=[joblib.load(directory/"candidate_models"/(id_+".joblib")) for id_,directory,_,_ in references]
    half=WeightedLogPredictor([model]+full_models,"B27_F07_half_full_half_bag",[.5,.2,.2,.1])
    joblib.dump(half,output_dir/"candidate_models/B27_F07_half_full_half_bag.joblib",compress=3)
    pd.DataFrame({"Id":train.Id,"fold":fold_ids,"SalePrice":train.SalePrice,"log_SalePrice":y,
                  "F07_blend_robust_c20_g0003__log_prediction":combined_oof}).to_csv(output_dir/"results/oof_predictions.csv",index=False)
    cv_scores=[float(np.sqrt(np.mean((y[va]-combined_oof[va])**2))) for _,va in splits]
    summary={"components":component_records,"oof_checks":checks,"cv_mean":float(np.mean(cv_scores)),
             "cv_std":float(np.std(cv_scores,ddof=1)),"pooled_oof_log_rmse":float(np.sqrt(np.mean((y-combined_oof)**2))),
             "deployment_variants":{"B26_F07_fold_bag":"Mean log predictions of each component's five fold models", "B27_F07_half_full_half_bag":"Equal-weight average of F07 full fit and B26 log predictions"},
             "limitation":"OOF predictions reproduce the original F07 score. They do not independently estimate the performance of either deployment averaging variant; Kaggle submission is required.",
             "reused_fold_model_cache":cached is not None,
             "runtime_seconds":time.perf_counter()-started}
    (output_dir/"results/summary.json").write_text(json.dumps(summary,indent=2))
    return summary


def run_pfn_blends(output_dir, pfn_dir):
    """Evaluate fixed convex combinations with available TabPFN predictions."""
    output_dir,pfn_dir=(Path(p).resolve() for p in (output_dir,pfn_dir))
    destination=output_dir/"pfn_blends"
    for folder in ["results","submissions"]:
        (destination/folder).mkdir(parents=True,exist_ok=True)
    records=[];oof_export=None;test_export=None;manifests={};hash_cache={}
    project=Path(__file__).resolve().parents[2]
    def artifact(path):
        path=Path(path)
        if str(path) not in hash_cache:
            hash_cache[str(path)]=file_sha256(path)
        return {"path":str(path.resolve().relative_to(project)),"sha256":hash_cache[str(path)]}
    base_roots=[(output_dir,["B14_cat_mae_d6","B14_cat_mae_d6_blend"]),
                (output_dir/"deep_extension",["B28_cat_mae_d7","B28_cat_mae_d7_blend","B29_cat_mae_d6_long","B29_cat_mae_d6_long_blend"])]
    for pfn_file in sorted((pfn_dir/"results").glob("R4T_v2_log_n*_s42*_oof.csv")):
        pfn_id=pfn_file.name.removesuffix("_oof.csv")
        test_file=pfn_dir/"results"/(pfn_id+"_test_logs.csv")
        if not test_file.exists():
            continue
        pfn_spec=json.loads((pfn_dir/"results"/(pfn_id+"_summary.json")).read_text())
        pfn=pd.read_csv(pfn_file);pfn_test=pd.read_csv(test_file)
        y=pfn.log_SalePrice.to_numpy();folds=pfn.fold.to_numpy()
        if oof_export is None:
            oof_export=pfn[["Id","fold","log_SalePrice"]].copy()
            test_export=pfn_test[["Id"]].copy()
        for root,ids in base_roots:
            if not (root/"results/oof_predictions.csv").exists():
                continue
            base=pd.read_csv(root/"results/oof_predictions.csv")
            tests=pd.read_csv(root/"results/test_log_predictions.csv")
            assert base.Id.equals(pfn.Id) and tests.Id.equals(pfn_test.Id) and np.array_equal(base.fold.to_numpy(),folds)
            for base_id in ids:
                key=base_id+"__log_prediction"
                if key not in base:
                    continue
                for weight in [.10,.25,.5]:
                    id_=f"BP_{pfn_id}_{base_id}_w{int(weight*100):02}"
                    prediction=(1-weight)*pfn.log_prediction.to_numpy()+weight*base[key].to_numpy()
                    test_prediction=(1-weight)*pfn_test.log_prediction.to_numpy()+weight*tests[key].to_numpy()
                    scores=[float(np.sqrt(np.mean((y[folds==fold]-prediction[folds==fold])**2))) for fold in sorted(set(folds))]
                    if base_id.endswith("_blend"):
                        boost_ids=[base_id.removesuffix("_blend"),"F03_robust_c20_g0003","O22_xgboost_d3"]
                        inner_weights=[.4,.4,.2]
                        model_paths=[root/"candidate_models"/(boost_ids[0]+".joblib"),
                                     project/"models/followup/candidate_models/F03_robust_c20_g0003.joblib",
                                     project/"models/optimization/candidate_models/O22_xgboost_d3.joblib"]
                    else:
                        boost_ids=[base_id];inner_weights=[1.];model_paths=[root/"candidate_models"/(base_id+".joblib")]
                    params={"components":[pfn_id]+boost_ids,"weights":[1-weight]+[weight*w for w in inner_weights],"combination_space":"natural logarithm of price"}
                    records.append(dict(experiment_id=id_,model="Weighted log-space blend",stage="round4_fixed_PFN_boost_blend",target="log",
                                        cv_mean=float(np.mean(scores)),cv_std=float(np.std(scores,ddof=1)),params_json=json.dumps(params,sort_keys=True),runtime_seconds=0.,
                                        pooled_oof_log_rmse=float(np.sqrt(np.mean((y-prediction)**2))),pfn_id=pfn_id,boost_id=base_id,boost_weight=weight))
                    oof_export[id_+"__log_prediction"]=prediction
                    test_export[id_+"__log_prediction"]=test_prediction
                    submission=destination/"submissions"/(id_+".csv")
                    pd.DataFrame({"Id":pfn_test.Id,"SalePrice":np.exp(test_prediction)}).to_csv(submission,index=False)
                    manifests[id_]={"experiment_id":id_,"parameters":params,"submission":artifact(submission),
                                    "boosting_models":[artifact(p) for p in model_paths],
                                    "boost_ensemble_model":artifact(root/"candidate_models"/(base_id+".joblib")) if (root/"candidate_models"/(base_id+".joblib")).exists() else None,
                                    "pfn_predictions":artifact(test_file),"pfn_oof":artifact(pfn_file),
                                    "pfn_specification":pfn_spec,"pfn_checkpoint_manifest":artifact(pfn_dir/"results/checkpoint_manifest.json"),
                                    "boost_oof":artifact(root/"results/oof_predictions.csv"),
                                    "reproduction":"Combine the saved TabPFN and boosting predictions with the listed weights in log space, then exponentiate. The saved prediction combination is implemented by scripts/round4_boosting.py --pfn-blends.",
                                    "verification_scope":"OOF metrics recomputed for all1460 rows; candidate is a fixed convex combination. No independent full rerun of all underlying components claimed."}
    frame=pd.DataFrame(records)
    if len(frame):
        frame.to_csv(destination/"results/experiment_results.csv",index=False)
        oof_export.to_csv(destination/"results/oof_predictions.csv",index=False)
        test_export.to_csv(destination/"results/test_log_predictions.csv",index=False)
    summary={"experiment_count":len(frame),"weights":[.1,.25,.5],"selection":"Adaptive fixed-weight convex combinations after standalone base-model development; all1460OOF rows retained; not a new holdout.",
             "best":frame.loc[frame.cv_mean.idxmin()].to_dict() if len(frame) else None}
    (destination/"results/summary.json").write_text(json.dumps(summary,indent=2))
    (destination/"results/candidate_manifests.json").write_text(json.dumps(manifests,indent=2))
    return summary


def materialize_blend(output_dir, id_):
    """Save a completed fixed boosting blend and verify its submitted CSV."""
    output_dir=Path(output_dir).resolve()
    project=Path(__file__).resolve().parents[2]
    results=pd.read_csv(output_dir/"results/experiment_results.csv")
    row=results.set_index("experiment_id").loc[id_]
    params=json.loads(row.params_json)
    models=[];paths=[]
    for component in params["components"]:
        if component.startswith("F"):
            path=project/"models/followup/candidate_models"/(component+".joblib")
        elif component.startswith("O"):
            path=project/"models/optimization/candidate_models"/(component+".joblib")
        else:
            path=output_dir/"candidate_models"/(component+".joblib")
        models.append(joblib.load(path));paths.append(str(path.relative_to(project)))
    model=ArithmeticLogBlendPredictor(models,id_,params["weights"])
    path=output_dir/"candidate_models"/(id_+".joblib")
    joblib.dump(model,path,compress=3)
    loaded=joblib.load(path)
    test=read_data(project/"data/raw/test.csv")
    submitted=pd.read_csv(output_dir/"submissions"/(id_+".csv"))
    difference=float(np.max(np.abs(loaded.predict(test)-submitted.SalePrice.to_numpy())))
    assert difference<1e-6,difference
    evidence={"experiment_id":id_,"model_path":str(path.relative_to(project)),"model_sha256":file_sha256(path),
              "components":paths,"weights":params["weights"],"arithmetic":"Termwise scalar multiplication, then left-to-right addition, preserving the exported arithmetic dtypes.",
              "reload_max_absolute_price_difference":difference}
    (output_dir/"results"/(id_+"_model_validation.json")).write_text(json.dumps(evidence,indent=2))
    return evidence


def audit_outputs(output_dir):
    """Recompute saved metrics and verify all prediction schemas and fit audits."""
    output_dir=Path(output_dir).resolve();project=Path(__file__).resolve().parents[2]
    train=read_data(project/"data/raw/train.csv");test=read_data(project/"data/raw/test.csv")
    folds=pd.read_csv(project/"models/optimization/results/fold_assignments.csv")
    y=np.log(train.SalePrice.to_numpy(float));reports=[]
    for root in [output_dir,output_dir/"gbr_recipe",output_dir/"deep_extension",output_dir/"pfn_blends"]:
        rows=pd.read_csv(root/"results/experiment_results.csv")
        oof=pd.read_csv(root/"results/oof_predictions.csv")
        assert oof.Id.equals(train.Id) and oof.fold.equals(folds.fold)
        differences=[]
        for row in rows.to_dict("records"):
            predictions=oof[row["experiment_id"]+"__log_prediction"].to_numpy(float)
            assert len(predictions)==1460 and np.isfinite(predictions).all()
            scores=[float(np.sqrt(np.mean((y[folds.fold.to_numpy()==f]-predictions[folds.fold.to_numpy()==f])**2))) for f in range(1,6)]
            metric=dict(cv_mean=float(np.mean(scores)),cv_std=float(np.std(scores,ddof=1)),pooled_oof_log_rmse=float(np.sqrt(np.mean((y-predictions)**2))))
            differences.extend(abs(metric[k]-row[k]) for k in metric)
        assert max(differences)<1e-10
        files=list((root/"submissions").glob("*.csv"))
        for path in files:
            values=pd.read_csv(path)
            assert values.columns.tolist()==["Id","SalePrice"] and values.Id.equals(test.Id)
            assert np.isfinite(values.SalePrice).all() and values.SalePrice.gt(0).all()
        audit=root/"results/preprocessing_audit.json"
        if audit.exists():
            entries=json.loads(audit.read_text())
            assert all(x["all_validation_rows_retained"] and x["train_validation_disjoint"] and x["training_rows"]==1168 and x["validation_rows"]==292 for x in entries)
        else:
            entries=[]
        reports.append(dict(directory=str(root.relative_to(project)),experiments=len(rows),submission_files=len(files),fold_fit_audits=len(entries),
                            maximum_metric_absolute_difference=max(differences),results_sha256=file_sha256(root/"results/experiment_results.csv")))
    summary=json.loads((output_dir/"results/summary.json").read_text())
    assert summary["base_specification_count"]==23 and summary["experiment_count"]==46 and not summary["failures"]
    result={"status":"passed","runs":reports,"all_1460_rows_retained":True,"folds_match_seed42":True,
            "validation_scope":"Saved OOF metrics, schema checks and fold fit audits.",
            "selected_boosting_model_reload":json.loads((output_dir/"results/B14_cat_mae_d6_blend_model_validation.json").read_text()),
            "reference_bagging":json.loads((output_dir/"reference_bagging/results/summary.json").read_text()),
            "interrupted_extension":json.loads((output_dir/"deep_extension/results/summary.json").read_text())}
    (output_dir/"results/round4_boosting_validation.json").write_text(json.dumps(result,indent=2))
    return result
