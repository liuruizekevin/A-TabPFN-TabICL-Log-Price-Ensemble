"""Fold-isolated round-four kernel/linear research. No external target data."""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, PowerTransformer
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits
from .optimization import EnhancedFeatures, WeightedLogPredictor, QUALITY_COLUMNS
from .models import PricePredictor
from .preprocessing import read_data

class ResearchFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, ordinal=False, enhanced=True, extra=False):
        self.ordinal=ordinal; self.enhanced=enhanced; self.extra=extra
    def fit(self,X,y=None):
        self.base_=EnhancedFeatures(self.enhanced).fit(X)
        return self
    def transform(self,X):
        out=self.base_.transform(X)
        if self.ordinal and self.enhanced:
            for c in list(QUALITY_COLUMNS)+['BsmtExposure','BsmtFinType1','BsmtFinType2','GarageFinish','Functional']:
                out=out.drop(columns=c)
        if self.extra:
            def num(c): return pd.to_numeric(out[c],errors='coerce').fillna(0.)
            out['TotalAreaIncludingGarage']=num('TotalSF')+num('GarageArea')
            out['TotalPlace']=num('TotalSF')+num('GarageArea')+num('TotalPorchSF')
            out['BasementQualityArea']=num('TotalBsmtSF')*num('BsmtQualOrdinal')
            out['GarageQualityArea']=num('GarageArea')*num('GarageQualOrdinal')
            out['RoomsWithBath']=num('TotRmsAbvGrd')+num('FullBath')
        return out

class PowerSkew(TransformerMixin, BaseEstimator):
    def __init__(self,power=0.,threshold=.75,all_skew=False):
        self.power=power;self.threshold=threshold;self.all_skew=all_skew
    def fit(self,X,y=None):
        frame=pd.DataFrame(X)
        self.n_features_in_=frame.shape[1]
        self.indices_=[i for i,c in enumerate(frame) if frame[c].min()>=0 and frame[c].nunique()>=(3 if self.all_skew else 10) and frame[c].skew()>self.threshold]
        self.learned_=None
        if self.power=='yeo-johnson' and self.indices_:
            self.learned_=PowerTransformer(method='yeo-johnson',standardize=False).fit(frame.iloc[:,self.indices_])
        return self
    def transform(self,X):
        out=np.asarray(X,float).copy()
        if self.indices_:
            a=np.maximum(out[:,self.indices_],0)
            out[:,self.indices_]=self.learned_.transform(a) if self.learned_ is not None else boxcox1p(a,self.power)
        return out

def make_pipeline(spec):
    num=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('power',PowerSkew(spec.get('power',0.),all_skew=spec.get('all_skew',False))),('scale',RobustScaler(quantile_range=tuple(spec.get('quantiles',[25.,75.]))) if spec.get('scaler','robust')=='robust' else StandardScaler())])
    cat=Pipeline([('impute',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    prep=ColumnTransformer([('num',num,make_column_selector(dtype_include=np.number)),('cat',cat,make_column_selector(dtype_exclude=np.number))],transformer_weights={'num':1.,'cat':spec.get('cat_weight',1.)})
    model={'SVR':SVR,'Ridge':Ridge,'Lasso':Lasso,'ElasticNet':ElasticNet,'KernelRidge':KernelRidge,'BayesianRidge':BayesianRidge}[spec['family']](**spec['params'])
    steps=[('features',ResearchFeatures(spec.get('ordinal',False),spec.get('enhanced',True),spec.get('extra',False))),('preprocessor',prep)]
    if spec.get('pca'): steps.append(('pca',PCA(n_components=spec['pca'],svd_solver='full')))
    return Pipeline(steps+[('model',model)])

def specifications():
    specs=[]
    def add(label,family='SVR',params=None,**kw):
        specs.append(dict(id=f'R4L{len(specs)+1:02d}_{label}',family=family,params=params or dict(C=20.,gamma=.0003,epsilon=.008,kernel='rbf',cache_size=512),**kw))
    for C,g,eps in [(5,.001,.008),(10,.0003,.008),(30,.0003,.008),(50,.0003,.008),(100,.0001,.008),(20,.0005,.008),(10,.001,.02),(20,.0003,.02),(20,.0003,.04),(10,.0005,.008),(13,.0004,.009),(5,'scale',.008),(20,'scale',.008),(30,.0001,.008)]:
        add(f'svr_C{C}_g{g}_e{eps}'.replace('.','p'),params=dict(C=C,gamma=g,epsilon=eps,kernel='rbf',cache_size=512))
    for p in [.15,.3,'yeo-johnson']:
        for g in [.0003,'scale']:
            add(f'power{p}_g{g}'.replace('.','p'),power=p,params=dict(C=20.,gamma=g,epsilon=.008,kernel='rbf',cache_size=512))
    for weight in [.5,1.5,2.]: add(f'catweight{weight}'.replace('.','p'),cat_weight=weight)
    for q in [[10.,90.],[20.,80.],[35.,65.]]: add(f'quantile{int(q[0])}',quantiles=q)
    for p in [0.,.15]:
        add(f'ordinal_p{p}'.replace('.','p'),ordinal=True,power=p)
        add(f'extra_p{p}'.replace('.','p'),extra=True,power=p)
    add('simple_base',enhanced=False)
    add('ordinal_pca99',ordinal=True,pca=.99,power=.15)
    add('allskew_p15',power=.15,all_skew=True)
    for a in [.0003,.0005,.001]:
        add(f'lasso{a}'.replace('.','p'),family='Lasso',params=dict(alpha=a,max_iter=30000,tol=.0001),power=.15,ordinal=True)
    for a in [5.,20.,60.]:
        add(f'ridge{a}'.replace('.','p'),family='Ridge',params=dict(alpha=a),power=.15,ordinal=True)
    for a,l1 in [(.0005,.9),(.005,.08),(.001,.5)]:
        add(f'elastic{a}l1{l1}'.replace('.','p'),family='ElasticNet',params=dict(alpha=a,l1_ratio=l1,max_iter=30000,tol=.0001),power=.15,ordinal=True)
    for degree,a,coef in [(2,.6,2.5),(3,.2,.8),(2,.2,2.5)]:
        add(f'krr{degree}_a{a}'.replace('.','p'),family='KernelRidge',params=dict(alpha=a,kernel='polynomial',degree=degree,coef0=coef,gamma=.001),power=.15,ordinal=True)
    add('svr_trainfilter',filter_train=True)
    add('boxcox_trainfilter',power=.15,filter_train=True)
    add('pca995',pca=.995,power=.15)
    add('bayesian',family='BayesianRidge',params=dict(max_iter=500),power=.15,ordinal=True)
    return specs

def run(data_dir,output_dir,reference_dir,specs=None):
    start=time.perf_counter(); output_dir=Path(output_dir);reference_dir=Path(reference_dir)
    for sub in ['results','submissions','candidate_models']: (output_dir/sub).mkdir(parents=True,exist_ok=True)
    specs=specifications() if specs is None else specs
    plan={'specifications':specs,'validation':'KFold5 shuffle True seed42; all1460 validation rows retained; preprocessing fit within each training fold; exact natural log RMSE','filter_rule':'Training-only GrLivArea >4000 and SalePrice <300000 when requested; no validation deletion','blends':'Fixed0.4O09Cat+0.4newcandidate+0.2O22XGB; alltested candidates retained','selection_limitation':'Adaptive development comparisons, not unbiased generalization estimates','sources':[{'url':'https://github.com/massquantity/Kaggle-HousePrices','inspiration':'Ordinal qualities, extra sums, PCA, SVR/KRR/linear diversity; original full-data preprocessing replaced with fold-only fitting'},{'url':'https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard','inspiration':'BoxCox0.15 and robust linear models; official notebook page text unavailable via web, matching details seen in attributed public transcription; independently implemented'}]}
    (output_dir/'results/plan.json').write_text(json.dumps(plan,indent=2))
    train=read_data(Path(data_dir)/'train.csv');test=read_data(Path(data_dir)/'test.csv');X=train.drop(columns='SalePrice');y=np.log(train.SalePrice.to_numpy(float))
    folds=list(KFold(5,shuffle=True,random_state=42).split(X));fold_ids=np.zeros(len(X),int)
    for f,(_,v) in enumerate(folds,1):fold_ids[v]=f
    ref=pd.read_csv(reference_dir/'results/oof_predictions.csv');assert ref.Id.equals(train.Id)
    catid='O09_native_mae';xgbid='O22_xgboost_d3';coof=ref[catid+'__log_prediction'].to_numpy();xoof=ref[xgbid+'__log_prediction'].to_numpy()
    cat=joblib.load(reference_dir/'candidate_models'/f'{catid}.joblib');xgb=joblib.load(reference_dir/'candidate_models'/f'{xgbid}.joblib')
    ctest=cat.predict_log(test);xtest=xgb.predict_log(test)
    results=[];scores=[];audits=[];fails=[];oof={};tp={}
    with threadpool_limits(limits=2):
        for spec in specs:
            t=time.perf_counter()
            try:
                pred=np.zeros(len(X))
                for fold,(tr,va) in enumerate(folds,1):
                    if spec.get('filter_train',False):tr=tr[~((X.iloc[tr].GrLivArea.to_numpy()>4000)&(train.iloc[tr].SalePrice.to_numpy()<300000))]
                    pipe=make_pipeline(spec).fit(X.iloc[tr],y[tr]);pred[va]=pipe.predict(X.iloc[va])
                    ids=pipe.named_steps['features'].base_.base_.fit_row_ids_;assert set(ids).isdisjoint(X.iloc[va].Id)
                    audits.append(dict(experiment_id=spec['id'],fold=fold,n_train=len(tr),n_valid=len(va),train_validation_disjoint=True,all_validation_rows_retained=True))
                ix=np.arange(len(X))
                if spec.get('filter_train',False):ix=ix[~((X.GrLivArea.to_numpy()>4000)&(train.SalePrice.to_numpy()<300000))]
                pipe=make_pipeline(spec).fit(X.iloc[ix],y[ix]);model=PricePredictor([(pipe,'log')],spec['id']);testlog=model.predict_log(test)
                joblib.dump(model,output_dir/'candidate_models'/f'{spec["id"]}.joblib',compress=3)
                for name,prediction,tlog,kind in [(spec['id'],pred,testlog,spec['family']),(spec['id']+'_blend',.4*coof+.4*pred+.2*xoof,.4*ctest+.4*testlog+.2*xtest,'Fixed40/40/20blend')]:
                    foldscore=[float(np.sqrt(np.mean((y[v]-prediction[v])**2))) for _,v in folds]
                    row=dict(experiment_id=name,model=kind,cv_mean=float(np.mean(foldscore)),cv_std=float(np.std(foldscore,ddof=1)),pooled_oof_log_rmse=float(np.sqrt(np.mean((y-prediction)**2))),runtime_seconds=time.perf_counter()-t,params_json=json.dumps(spec,sort_keys=True))
                    results.append(row);oof[name+'__log_prediction']=prediction;tp[name+'__log_prediction']=tlog
                    scores.extend([dict(experiment_id=name,fold=f,log_rmse=s) for f,s in enumerate(foldscore,1)])
                    values=np.exp(tlog);assert np.isfinite(values).all() and (values>0).all()
                    if kind=='Fixed40/40/20blend' and spec['id'] in ['R4L12_svr_C5_gscale_e0p008','R4L20_poweryeo-johnson_gscale','R4L23_catweight2p0','R4L24_quantile10','R4L46_svr_trainfilter']:
                        joblib.dump(ResearchFixedBlend(cat,model,xgb,name),output_dir/'candidate_models'/f'{name}.joblib',compress=3)
                    pd.DataFrame({'Id':test.Id,'SalePrice':values}).to_csv(output_dir/'submissions'/f'{name}.csv',index=False)
                pd.DataFrame(results).to_csv(output_dir/'results/experiment_results.csv',index=False)
                pd.DataFrame({'Id':train.Id,'fold':fold_ids,'log_SalePrice':y,**oof}).to_csv(output_dir/'results/oof_predictions.csv',index=False)
                pd.DataFrame({'Id':test.Id,**tp}).to_csv(output_dir/'results/test_log_predictions.csv',index=False)
                print(f'{spec["id"]}: base {results[-2]["cv_mean"]:.6f} blend {results[-1]["cv_mean"]:.6f} time{time.perf_counter()-t:.1f}s',flush=True)
            except Exception as exc:
                fails.append({'experiment_id':spec['id'],'error':repr(exc)});print(f'FAILED{spec["id"]}: {exc}',flush=True)
    pd.DataFrame(scores).to_csv(output_dir/'results/fold_scores.csv',index=False)
    pd.DataFrame({'Id':train.Id,'fold':fold_ids}).to_csv(output_dir/'results/fold_assignments.csv',index=False)
    (output_dir/'results/preprocessing_audit.json').write_text(json.dumps(audits,indent=2));(output_dir/'results/failed_experiments.json').write_text(json.dumps(fails,indent=2))
    result=pd.DataFrame(results);best=result.loc[result.cv_mean.idxmin()].to_dict()
    summary={'best':best,'count':len(results),'runtime_seconds':time.perf_counter()-start,'failures':fails,'candidates':result.sort_values('cv_mean').head(15).to_dict('records'),'limitation':plan['selection_limitation']}
    (output_dir/'results/summary.json').write_text(json.dumps(summary,indent=2));return summary


def extension_specs():
    specs=[]
    def add(label,**kw):
        family=kw.pop('family','SVR');params=kw.pop('params',dict(C=10.,gamma='scale',epsilon=.008,kernel='rbf',cache_size=512))
        specs.append(dict(id=f'R4LE{len(specs)+1:02d}_{label}',family=family,params=params,**kw))
    for C in [2.,5.,10.,30.,50.]:
        add(f'yj_C{int(C)}',power='yeo-johnson',params=dict(C=C,gamma='scale',epsilon=.008,kernel='rbf',cache_size=512))
    for C,g in [(5,.001),(10,.001),(20,.001),(10,.0005),(20,.0005)]:
        add(f'yj_C{C}_g{g}'.replace('.','p'),power='yeo-johnson',params=dict(C=C,gamma=g,epsilon=.008,kernel='rbf',cache_size=512))
    for w in [1.5,2.,3.]:add(f'yj_w{w}'.replace('.','p'),power='yeo-johnson',cat_weight=w)
    for C in [1.,2.,3.]:add(f'log_C{int(C)}',params=dict(C=C,gamma='scale',epsilon=.008,kernel='rbf',cache_size=512))
    add('yj_e02',power='yeo-johnson',params=dict(C=10.,gamma='scale',epsilon=.02,kernel='rbf',cache_size=512))
    add('yj_extra',power='yeo-johnson',extra=True)
    add('krr_defaultgamma',family='KernelRidge',params=dict(alpha=.6,kernel='polynomial',degree=2,coef0=2.5),power=.15,ordinal=True)
    add('krr_d3_defaultgamma',family='KernelRidge',params=dict(alpha=.2,kernel='polynomial',degree=3,coef0=.8),power=.15,ordinal=True)
    return specs



class ResearchFixedBlend(PricePredictor):
    """Use the same component arithmetic as the CSV-generating research runner.

    XGBoost returns float32; retaining the runner's multiply-then-add order also
    reproduces its harmless float32 rounding exactly after serialization.
    """
    def __init__(self, cat, candidate, xgb, name):
        self.cat=cat;self.candidate=candidate;self.xgb=xgb;self.name=name
    def predict_log(self,X):
        return .4*self.cat.predict_log(X)+.4*self.candidate.predict_log(X)+.2*self.xgb.predict_log(X)
