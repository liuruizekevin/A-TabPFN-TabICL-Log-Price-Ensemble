"""Local TabPFN-v2 comparison, public weights, all-row fold-isolated validation.

Run with build/tabpfn-env/bin/python; no cloud inference or external labels.
Inspired by Carl McBride Ellis's raw-target TabPFN notebook, not copied outputs.
"""
import argparse, hashlib, importlib.metadata, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

ROOT=Path(__file__).resolve().parents[1]

def run(target, estimators=8, seed=42, features='raw', filter_train=False, output_dir=None):
    out=Path(output_dir) if output_dir else ROOT/'models/round4_tabpfn';out.mkdir(parents=True,exist_ok=True)
    for d in ['results','submissions']: (out/d).mkdir(exist_ok=True)
    name=f'R4T_v2_{target}_n{estimators}_s{seed}'
    if features!='raw': name+='_'+features
    if filter_train: name+='_trainfilter'
    raw=pd.read_csv(ROOT/'data/raw/train.csv',keep_default_na=False,na_values=['NA',''])
    test=pd.read_csv(ROOT/'data/raw/test.csv',keep_default_na=False,na_values=['NA',''])
    X=raw.drop(columns=['Id','SalePrice']);Xt=test.drop(columns='Id')
    if features=='domain':
        sys.path.insert(0,str(ROOT/'src'))
        from house_prices.preprocessing import deterministic_features, ALL_GROUPS
        X=deterministic_features(X,semantic=True,groups=ALL_GROUPS)
        Xt=deterministic_features(Xt,semantic=True,groups=ALL_GROUPS)
    # Make native string categories explicit; TabPFN learns all encodings per fit.
    for col in X.select_dtypes(exclude=np.number).columns:
        X[col]=X[col].astype(object).where(X[col].notna(),np.nan)
        Xt[col]=Xt[col].astype(object).where(Xt[col].notna(),np.nan)
    y=raw.SalePrice.to_numpy(float); logs=np.log(y)
    labels=y if target=='raw' else logs
    spec=dict(id=name,model='TabPFN-v2',target=target,n_estimators=estimators,random_state=seed,device='mps',fold_seed=42,n_splits=5,rows=1460,features=X.shape[1],feature_recipe=features,filter_train=filter_train,filter_rule='GrLivArea>4000 and SalePrice<300000, training partitions only' if filter_train else None,source='https://www.kaggle.com/code/carlmcbrideellis/house-prices-using-tabpfn',weight_license='Prior Labs License (Apache 2.0 with additional attribution)',library_version=importlib.metadata.version('tabpfn'),torch_version=torch.__version__,selection_limitation='Adaptive development CV; no external test targets used.')
    (out/'results'/f'{name}_plan.json').write_text(json.dumps(spec,indent=2))
    def make(): return TabPFNRegressor.create_default_for_version(ModelVersion.V2,n_estimators=estimators,device='mps',random_state=seed,n_preprocessing_jobs=2)
    def to_log(pred):
        pred=np.asarray(pred,float)
        if target=='raw':
            assert (pred>0).all()
            return np.log(pred)
        return pred
    preds=np.zeros(len(X));foldids=np.zeros(len(X),int);foldrows=[];start=time.perf_counter()
    for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=42).split(X),1):
        if filter_train: tr=tr[~((raw.GrLivArea.to_numpy()[tr]>4000)&(y[tr]<300000))]
        t=time.perf_counter();model=make();model.fit(X.iloc[tr],labels[tr]);preds[va]=to_log(model.predict(X.iloc[va]));foldids[va]=fold
        score=float(np.sqrt(np.mean((preds[va]-logs[va])**2)))
        foldrows.append(dict(fold=fold,log_rmse=score,n_train=len(tr),n_valid=len(va),train_validation_disjoint=not bool(set(tr)&set(va)),seconds=time.perf_counter()-t))
        print(name,fold,score,'sec',time.perf_counter()-t,flush=True)
        del model;torch.mps.empty_cache()
    final_indices=np.arange(len(X))
    if filter_train: final_indices=final_indices[~((raw.GrLivArea.to_numpy()>4000)&(y<300000))]
    model=make();model.fit(X.iloc[final_indices],labels[final_indices]);testlogs=to_log(model.predict(Xt))
    pd.DataFrame({'Id':raw.Id,'fold':foldids,'log_SalePrice':logs,'log_prediction':preds}).to_csv(out/'results'/f'{name}_oof.csv',index=False)
    pd.DataFrame({'Id':test.Id,'log_prediction':testlogs}).to_csv(out/'results'/f'{name}_test_logs.csv',index=False)
    pd.DataFrame({'Id':test.Id,'SalePrice':np.exp(testlogs)}).to_csv(out/'submissions'/f'{name}.csv',index=False)
    result=spec|dict(fold_scores=foldrows,cv_mean=float(np.mean([r['log_rmse'] for r in foldrows])),cv_std=float(np.std([r['log_rmse'] for r in foldrows],ddof=1)),pooled_oof_log_rmse=float(np.sqrt(np.mean((preds-logs)**2))),runtime_seconds=time.perf_counter()-start)
    (out/'results'/f'{name}_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',choices=['raw','log'],required=True);p.add_argument('--estimators',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--features',choices=['raw','domain'],default='raw');p.add_argument('--filter-train',action='store_true');p.add_argument('--output-dir',type=Path);a=p.parse_args();run(a.target,a.estimators,a.seed,a.features,a.filter_train,a.output_dir)
