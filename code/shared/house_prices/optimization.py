"""Bounded second-phase search; retains every validation row and failed trial."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from .experiment import file_sha256
from .models import PricePredictor, experiment_specs, make_pipeline
from .preprocessing import ALL_GROUPS, FeatureBuilder, make_preprocessor, read_data


QUALITY_COLUMNS = ("ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC")


class EnhancedFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, enhanced=True):
        self.enhanced = enhanced

    def fit(self, X, y=None):
        self.base_ = FeatureBuilder(semantic=True, groups=ALL_GROUPS).fit(X)
        return self

    def transform(self, X):
        out = self.base_.transform(X)
        if not self.enhanced:
            return out
        # Consolidate blocks before adding another batch of row-local columns.
        out = out.copy()
        def num(c):
            return pd.to_numeric(out[c], errors="coerce").fillna(0)
        for c in QUALITY_COLUMNS:
            out[c+"Ordinal"] = out[c].map({"None":0,"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5}).astype(float)
        for c, values in {
            "BsmtExposure": {"None":0,"No":1,"Mn":2,"Av":3,"Gd":4},
            "BsmtFinType1": {"None":0,"Unf":1,"LwQ":2,"Rec":3,"BLQ":4,"ALQ":5,"GLQ":6},
            "BsmtFinType2": {"None":0,"Unf":1,"LwQ":2,"Rec":3,"BLQ":4,"ALQ":5,"GLQ":6},
            "GarageFinish": {"None":0,"Unf":1,"RFn":2,"Fin":3},
            "Functional": {"Sal":0,"Sev":1,"Maj2":2,"Maj1":3,"Mod":4,"Min2":5,"Min1":6,"Typ":7},
        }.items():
            out[c+"Ordinal"] = out[c].map(values).astype(float)
        for c in ["TotalBsmtSF", "GarageArea", "PoolArea", "Fireplaces", "2ndFlrSF", "MasVnrArea"]:
            out["Has"+c] = (num(c)>0).astype(int)
        out["OverallQualCond"] = num("OverallQual") * num("OverallCond")
        out["LivingAreaPerRoom"] = num("GrLivArea") / num("TotRmsAbvGrd").clip(lower=1)
        out["LotAreaPerLivingArea"] = num("LotArea") / num("GrLivArea").clip(lower=1)
        out["FinishedBasementShare"] = (num("BsmtFinSF1") + num("BsmtFinSF2")) / num("TotalBsmtSF").clip(lower=1)
        out["GarageAreaPerCar"] = num("GarageArea") / num("GarageCars").clip(lower=1)
        out["QualityAge"] = num("OverallQual") / (1 + num("HouseAge"))
        out["NeighborhoodClass"] = out["Neighborhood"].astype(str) + "_" + out["MSSubClass"].astype(str)
        out["NeighborhoodQual"] = out["Neighborhood"].astype(str) + "_" + out["OverallQual"].astype(str)
        return out


class NativeCatRegressor(RegressorMixin, BaseEstimator):
    """CatBoost categories are built within each training fold, with no OHE."""
    def __init__(self, params=None):
        self.params = params

    @staticmethod
    def clean(X):
        out = X.copy()
        for c in out.select_dtypes(exclude=np.number).columns:
            out[c] = out[c].fillna("__MISSING__").astype(str)
        return out

    def fit(self, X, y):
        X = self.clean(X)
        self.feature_names_in_ = np.asarray(X.columns)
        self.cat_columns_ = X.select_dtypes(exclude=np.number).columns.tolist()
        self.model_ = CatBoostRegressor(**self.params)
        self.model_.fit(X, y, cat_features=self.cat_columns_)
        return self

    def predict(self, X):
        return self.model_.predict(self.clean(X))


class WeightedLogPredictor(PricePredictor):
    def __init__(self, components, name, weights):
        self.components = components
        self.name = name
        self.weights = weights

    def predict_log(self, X):
        logs = [m.predict_log(X) for m in self.components]
        return np.average(logs, axis=0, weights=self.weights)


def optimization_specs():
    specs = []
    base = dict(iterations=2400, depth=4, learning_rate=.025, l2_leaf_reg=8., loss_function="RMSE", random_seed=42, thread_count=4, verbose=False, allow_writing_files=False)
    for id_, changes, enhanced, family in [
        ("O01_native_base", {}, False, "CatBoostNative"),
        ("O02_native_d4", {}, True, "CatBoostNative"),
        ("O03_native_d5", {"depth":5}, True, "CatBoostNative"),
        ("O04_native_d3", {"depth":3,"iterations":3000}, True, "CatBoostNative"),
        ("O05_native_d6", {"depth":6,"iterations":2200}, True, "CatBoostNative"),
        ("O06_native_d4_l2", {"l2_leaf_reg":20.,"rsm":.8}, True, "CatBoostNative"),
        ("O07_native_d5_l2", {"depth":5,"l2_leaf_reg":20.,"rsm":.8}, True, "CatBoostNative"),
        ("O08_native_huber", {"loss_function":"Huber:delta=0.15"}, True, "CatBoostNative"),
        ("O09_native_mae", {"loss_function":"MAE","iterations":3000}, True, "CatBoostNative"),
        ("O10_onehot_d4", {}, True, "CatBoostOHE"),
        ("O11_onehot_d5", {"depth":5}, True, "CatBoostOHE"),
    ]:
        specs.append(dict(id=id_, family=family, params=base|changes, enhanced=enhanced, skew=False, filter_train=False))
    for id_, C, gamma in [("O12_svr_c10",10.,.001),("O13_svr_c20_g5",20.,.0005),("O14_svr_c20_g10",20.,.001),("O15_svr_c30_g3",30.,.0003),("O16_svr_c50_g5",50.,.0005)]:
        specs.append(dict(id=id_,family="SVR",params=dict(C=C,gamma=gamma,epsilon=.008,kernel="rbf",cache_size=512),enhanced=True,skew=True,filter_train=False))
    specs.append(dict(id="O17_kernel_ridge",family="KernelRidge",params=dict(alpha=.6,kernel="polynomial",degree=2,gamma=.001,coef0=2.5),enhanced=True,skew=True,filter_train=False))
    lgb = dict(n_estimators=2600,learning_rate=.015,num_leaves=8,max_depth=-1,min_child_samples=15,colsample_bytree=.8,subsample=.8,subsample_freq=1,reg_alpha=.05,reg_lambda=2.,random_state=42,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
    for id_,changes in [("O18_lightgbm_leaf8",{}),("O19_lightgbm_leaf15",{"num_leaves":15,"min_child_samples":20}),("O20_lightgbm_huber",{"objective":"huber","alpha":.9,"num_leaves":15})]:
        specs.append(dict(id=id_,family="LightGBM",params=lgb|changes,enhanced=True,skew=False,filter_train=False))
    xgb = dict(n_estimators=2200,learning_rate=.02,max_depth=2,min_child_weight=1,subsample=.8,colsample_bytree=.7,reg_alpha=.01,reg_lambda=1.,objective="reg:squarederror",random_state=42,n_jobs=4,tree_method="hist")
    for id_,changes in [("O21_xgboost_d2",{}),("O22_xgboost_d3",{"max_depth":3,"min_child_weight":2})]:
        specs.append(dict(id=id_,family="XGBoost",params=xgb|changes,enhanced=True,skew=False,filter_train=False))
    specs.append(dict(id="O23_native_train_filter",family="CatBoostNative",params=base.copy(),enhanced=True,skew=False,filter_train=True))
    specs.append(dict(id="O24_svr_train_filter",family="SVR",params=dict(C=20.,gamma=.0005,epsilon=.008,kernel="rbf",cache_size=512),enhanced=True,skew=True,filter_train=True))
    return specs


def make_optimization_pipeline(spec):
    family, params = spec["family"], spec["params"]
    steps = [("features", EnhancedFeatures(spec["enhanced"]))]
    if family == "CatBoostNative":
        steps.append(("model", NativeCatRegressor(params)))
    else:
        steps.append(("preprocessor",make_preprocessor(spec["skew"])))
        if family == "CatBoostOHE":
            estimator = CatBoostRegressor(**params)
        elif family == "SVR":
            estimator = SVR(**params)
        elif family == "KernelRidge":
            estimator = KernelRidge(**params)
        elif family == "LightGBM":
            from lightgbm import LGBMRegressor
            estimator = LGBMRegressor(**params)
        elif family == "XGBoost":
            from xgboost import XGBRegressor
            estimator = XGBRegressor(**params)
        steps.append(("model",estimator))
    return Pipeline(steps)


def training_indices(spec, X, y, indices):
    if spec.get("filter_train",False):
        excluded = (X.iloc[indices].GrLivArea.to_numpy()>4000) & (y[indices]<300000)
        return indices[~excluded]
    return indices


def run_optimization(data_dir, output_dir, phase1_dir):
    start = time.perf_counter()
    data_dir, output_dir, phase1_dir = map(Path, [data_dir, output_dir, phase1_dir])
    for d in ["results","artifacts","candidate_models","submissions"]:
        (output_dir/d).mkdir(parents=True,exist_ok=True)
    raw, test = read_data(data_dir/"train.csv"), read_data(data_dir/"test.csv")
    X, y = raw.drop(columns="SalePrice"), raw.SalePrice.to_numpy(dtype=float)
    logs = np.log(y)
    phase_summary = json.loads((phase1_dir/"results/summary.json").read_text())
    phase_oof = pd.read_csv(phase1_dir/"results/oof_predictions.csv")
    phase_scores = pd.read_csv(phase1_dir/"results/experiment_results.csv")
    fold_ids = pd.read_csv(phase1_dir/"results/fold_assignments.csv")
    splits = list(KFold(5,shuffle=True,random_state=42).split(X))
    for f,(_,v) in enumerate(splits,1):
        assert (fold_ids.iloc[v].fold==f).all()
    specs = optimization_specs()
    plan = {"phase":2,"specifications":specs,"all_1460_validation_rows_retained":True,"filter_rule":"Only candidates O23/O24 remove training-fold rows with GrLivArea>4000 and training SalePrice<300000. Validation rows never filtered.","blend_rule":"Select lowest mean-fold score within eligible native-CatBoost, SVR, XGBoost and LightGBM families, excluding filtered variants. Use predeclared equal or 40/40/20 or 60/40 weights, using fixed configuration weights.","blend_weights":{"O25_cat_svr_equal":[.5,.5],"O26_cat_svr_xgb":[.4,.4,.2],"O27_conservative_phase1_svr":[.6,.4],"O28_cat_svr_lgb":[.4,.4,.2],"O29_cat_svr_elastic":[1/3,1/3,1/3]},"selection_limitation":"All search decisions reuse development folds; no unbiased final generalization estimate."}
    (output_dir/"results/optimization_plan.json").write_text(json.dumps(plan,indent=2))
    records, fold_records, failures, fit_audit = [],[],[],[]
    oof_logs, fitted_models, test_predictions = {},{},{}
    with threadpool_limits(limits=4):
        for spec in specs:
            t0=time.perf_counter()
            try:
                predictions=np.zeros(len(X)); scores=[]; current_folds=[]; current_audit=[]
                for fold,(tr,va) in enumerate(splits,1):
                    actual_tr=training_indices(spec,X,y,tr)
                    estimator=make_optimization_pipeline(spec)
                    estimator.fit(X.iloc[actual_tr],logs[actual_tr])
                    predictions[va]=estimator.predict(X.iloc[va])
                    score=float(np.sqrt(np.mean((logs[va]-predictions[va])**2)))
                    scores.append(score)
                    current_folds.append(dict(experiment_id=spec["id"],fold=fold,log_rmse=score,n_train=len(actual_tr),n_valid=len(va)))
                    ids=estimator.named_steps["features"].base_.fit_row_ids_
                    assert set(ids).isdisjoint(X.iloc[va].Id)
                    current_audit.append(dict(experiment_id=spec["id"],fold=fold,training_rows=len(ids),excluded_training_ids=X.iloc[np.setdiff1d(tr,actual_tr)].Id.tolist(),validation_rows=len(va),all_validation_rows_retained=True,train_validation_disjoint=True))
                full_tr=training_indices(spec,X,y,np.arange(len(X)))
                final_model=make_optimization_pipeline(spec).fit(X.iloc[full_tr],logs[full_tr])
                predictor=PricePredictor([(final_model,"log")],spec["id"])
                fitted_models[spec["id"]]=predictor
                oof_logs[spec["id"]]=predictions
                test_predictions[spec["id"]]=predictor.predict(test)
                joblib.dump(predictor,output_dir/"candidate_models"/(spec["id"]+".joblib"),compress=3)
                pd.DataFrame({"Id":test.Id,"SalePrice":test_predictions[spec["id"]]}).to_csv(output_dir/"submissions"/(spec["id"]+".csv"),index=False)
                record=dict(experiment_id=spec["id"],model=spec["family"],stage="optimization",target="log",semantic_missing=True,feature_groups="area,age,baths,porch,quality"+(",enhanced" if spec["enhanced"] else ""),skew_transform=spec["skew"],params_json=json.dumps(spec["params"]|{"enhanced_features":spec["enhanced"],"training_outlier_filter":spec["filter_train"]},sort_keys=True),cv_mean=float(np.mean(scores)),cv_std=float(np.std(scores,ddof=1)),pooled_oof_log_rmse=float(np.sqrt(np.mean((logs-predictions)**2))),runtime_seconds=time.perf_counter()-t0,raw_predictions_clipped=0)
                records.append(record);fold_records.extend(current_folds);fit_audit.extend(current_audit)
                pd.DataFrame(records).to_csv(output_dir/"results/experiment_results.csv",index=False)
                pd.DataFrame(fold_records).to_csv(output_dir/"results/fold_scores.csv",index=False)
                print(f"{spec['id']}: {record['cv_mean']:.6f} +/- {record['cv_std']:.6f} ({record['runtime_seconds']:.1f}s)",flush=True)
            except Exception as error:
                failures.append(dict(experiment_id=spec["id"],error=repr(error),runtime_seconds=time.perf_counter()-t0))
                (output_dir/"results/failed_experiments.json").write_text(json.dumps(failures,indent=2))
                print(f"FAILED {spec['id']}: {error}",flush=True)
        results=pd.DataFrame(records)
        def winner(family):
            eligible=results[(results.model==family)&~results.experiment_id.str.contains("train_filter")]
            return None if eligible.empty else eligible.loc[eligible.cv_mean.idxmin(),"experiment_id"]
        cat,svr,xgb,lgb=map(winner,["CatBoostNative","SVR","XGBoost","LightGBM"])
        phase_best=phase_summary["best_experiment"]
        fitted_models[phase_best]=joblib.load(phase1_dir/"artifacts/best_model.joblib")
        oof_logs[phase_best]=phase_oof[phase_best+"__log_prediction"].to_numpy()
        elastic="E07_elastic_0.0005"
        elastic_spec=next(s for s in experiment_specs() if s["id"]==elastic)
        elastic_pipeline=make_pipeline(elastic_spec).fit(X,logs)
        fitted_models[elastic]=PricePredictor([(elastic_pipeline,"log")],elastic)
        oof_logs[elastic]=phase_oof[elastic+"__log_prediction"].to_numpy()
        blend_definitions=[("O25_cat_svr_equal",[cat,svr],[.5,.5]),("O26_cat_svr_xgb",[cat,svr,xgb],[.4,.4,.2]),("O27_conservative_phase1_svr",[phase_best,svr],[.6,.4]),("O28_cat_svr_lgb",[cat,svr,lgb],[.4,.4,.2]),("O29_cat_svr_elastic",[cat,svr,elastic],[1/3]*3)]
        resolved_blends=[]
        for id_,components,weights in blend_definitions:
            if any(c is None for c in components):
                continue
            prediction=np.average([oof_logs[c] for c in components],axis=0,weights=weights)
            scores=[float(np.sqrt(np.mean((logs[v]-prediction[v])**2))) for _,v in splits]
            predictor=WeightedLogPredictor([fitted_models[c] for c in components],id_,weights)
            fitted_models[id_]=predictor;oof_logs[id_]=prediction
            test_predictions[id_]=predictor.predict(test)
            joblib.dump(predictor,output_dir/"candidate_models"/(id_+".joblib"),compress=3)
            pd.DataFrame({"Id":test.Id,"SalePrice":test_predictions[id_]}).to_csv(output_dir/"submissions"/(id_+".csv"),index=False)
            params={"components":components,"weights":weights}
            resolved_blends.append({"id":id_,**params})
            records.append(dict(experiment_id=id_,model="Weighted log-space blend",stage="optimization_blend",target="log",semantic_missing=True,feature_groups="component-specific",skew_transform="component-specific",params_json=json.dumps(params),cv_mean=float(np.mean(scores)),cv_std=float(np.std(scores,ddof=1)),pooled_oof_log_rmse=float(np.sqrt(np.mean((logs-prediction)**2))),runtime_seconds=0.,raw_predictions_clipped=0))
            for fold,score in enumerate(scores,1):
                fold_records.append(dict(experiment_id=id_,fold=fold,log_rmse=score,n_train=1168,n_valid=292))
            print(f"{id_}: {np.mean(scores):.6f} +/- {np.std(scores,ddof=1):.6f}; {components}",flush=True)
    results=pd.DataFrame(records)
    results["baseline_absolute_improvement"]=phase_summary["baseline"]["cv_mean"]-results.cv_mean
    results["baseline_relative_improvement_pct"]=100*results.baseline_absolute_improvement/phase_summary["baseline"]["cv_mean"]
    results["phase1_best_absolute_improvement"]=phase_summary["best_cv_mean"]-results.cv_mean
    results.to_csv(output_dir/"results/experiment_results.csv",index=False)
    pd.DataFrame(fold_records).to_csv(output_dir/"results/fold_scores.csv",index=False)
    pd.DataFrame({"Id":raw.Id,"fold":fold_ids.fold,"SalePrice":y,"log_SalePrice":logs,**{r.experiment_id+"__log_prediction":oof_logs[r.experiment_id] for r in results.itertuples()}}).to_csv(output_dir/"results/oof_predictions.csv",index=False)
    fold_ids.to_csv(output_dir/"results/fold_assignments.csv",index=False)
    (output_dir/"results/preprocessing_audit.json").write_text(json.dumps(fit_audit,indent=2))
    (output_dir/"results/failed_experiments.json").write_text(json.dumps(failures,indent=2))
    (output_dir/"results/resolved_blends.json").write_text(json.dumps(resolved_blends,indent=2))
    best=results.loc[results.cv_mean.idxmin()]
    best_id=best.experiment_id
    joblib.dump(fitted_models[best_id],output_dir/"artifacts/best_model.joblib",compress=3)
    pd.DataFrame({"Id":test.Id,"SalePrice":test_predictions[best_id]}).to_csv(output_dir/"final_submission.csv",index=False)
    pd.DataFrame({"Id":test.Id,"SalePrice":test_predictions[best_id]}).to_csv(output_dir/"submissions/final_submission.csv",index=False)
    conservative_id="O27_conservative_phase1_svr"
    if conservative_id in test_predictions:
        pd.DataFrame({"Id":test.Id,"SalePrice":test_predictions[conservative_id]}).to_csv(output_dir/"conservative_blend_submission.csv",index=False)
    filtered=results[results.experiment_id.str.contains("train_filter")]
    if not filtered.empty:
        filtered_best=filtered.loc[filtered.cv_mean.idxmin(),"experiment_id"]
        pd.DataFrame({"Id":test.Id,"SalePrice":test_predictions[filtered_best]}).to_csv(output_dir/"train_filter_sensitivity_submission.csv",index=False)
    summary={"phase":2,"best_experiment":best_id,"best_model":best.model,"best_cv_mean":float(best.cv_mean),"best_cv_std":float(best.cv_std),"best_pooled_oof_log_rmse":float(best.pooled_oof_log_rmse),"best_parameters":json.loads(best.params_json),"best_components":json.loads(best.params_json).get("components",[best_id]),"experiment_count":len(results),"failed_experiment_count":len(failures),"phase1_best_experiment":phase_best,"phase1_best_cv_mean":phase_summary["best_cv_mean"],"absolute_improvement_over_phase1":phase_summary["best_cv_mean"]-float(best.cv_mean),"relative_improvement_over_phase1_pct":100*(phase_summary["best_cv_mean"]-float(best.cv_mean))/phase_summary["best_cv_mean"],"all_1460_validation_rows_retained":True,"selection_limitation":plan["selection_limitation"],"runtime_seconds":time.perf_counter()-start,"train_sha256":file_sha256(data_dir/"train.csv"),"test_sha256":file_sha256(data_dir/"test.csv"),"promoted":False,"conservative_submission_experiment":conservative_id,"sensitivity_submission_experiment":filtered_best if not filtered.empty else None}
    (output_dir/"results/summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary
