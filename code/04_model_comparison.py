"""Supervised model comparison: {LR, RF, XGB} x {original, class_weight, undersample, smote} x {chronological, random}.
Usage: 04_model_comparison.py CSV RES SEEDS(comma) [MODELS(comma)]
Protocol (pre-specified): features=6 base (no step); StandardScaler+OneHot(type) fit on train only; samplers fit on train only;
undersample -> fraud:legit 1:1; smote -> fraud:legit 0.1 (k=5); threshold = max validation F1; one test evaluation;
class-conditional bootstrap 95% CI (1000 resamples chrono test, 200 for the 954k random test)."""
import sys, os, time, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
CSV,RES=sys.argv[1],sys.argv[2]; SEEDS=[int(x) for x in sys.argv[3].split(',')]
MODELS=sys.argv[4].split(',') if len(sys.argv)>4 else ['LR','XGB','RF']
CONDS=['original','class_weight','undersample','smote']
NUM=['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']; CAT=['type']
OUTCSV=f'{RES}/model_comparison_metrics.csv'; os.makedirs(f'{RES}/scores',exist_ok=True)
df=pd.read_csv(CSV,usecols=NUM+CAT+['isFraud'],dtype={'type':'category'}); ix=np.load(f'{RES}/split_indices.npz')
def raw(k): d=df.iloc[ix[k]]; return d[NUM+CAT],d.isFraud.values
done=set()
if os.path.exists(OUTCSV):
    o=pd.read_csv(OUTCSV); done=set(zip(o.model,o.condition,o.design,o.seed))
def prep(tr,others):
    pre=ColumnTransformer([('n',StandardScaler(),NUM),('c',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)])
    return pre.fit_transform(tr).astype(np.float32),[pre.transform(o).astype(np.float32) for o in others]
D={}
Xc,yc=raw('chrono_train'); Xcv,ycv=raw('chrono_val'); Xct,yct=raw('chrono_test')
Xr,yr=raw('rand_train'); Xrv,yrv=raw('rand_val'); Xrt,yrt=raw('rand_test')
a,(b,c)=prep(Xc,[Xcv,Xct]); D['chronological']=dict(Xtr=a,ytr=yc.copy(),Xva=b,yva=ycv,tests={'chronological':(c,yct)})
a,(b,c,d)=prep(Xr,[Xrv,Xrt,Xct]); D['random']=dict(Xtr=a,ytr=yr.copy(),Xva=b,yva=yrv,tests={'random':(c,yrt),'chronological':(d,yct)})
def build(model,cond,seed,ytr):
    neg,pos=int((ytr==0).sum()),int(ytr.sum())
    if model=='LR': return LogisticRegression(max_iter=1000,class_weight='balanced' if cond=='class_weight' else None,random_state=seed)
    if model=='RF': return RandomForestClassifier(n_estimators=100,min_samples_leaf=3,class_weight='balanced' if cond=='class_weight' else None,n_jobs=2,random_state=seed)
    if model=='XGB': return XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,tree_method='hist',n_jobs=2,random_state=seed,
                                          scale_pos_weight=(neg/pos if cond=='class_weight' else 1.0),eval_metric='aucpr',verbosity=0)
def resample(cond,X,y,seed):
    if cond=='undersample': return RandomUnderSampler(sampling_strategy=1.0,random_state=seed).fit_resample(X,y)
    if cond=='smote': return SMOTE(sampling_strategy=0.1,k_neighbors=5,random_state=seed).fit_resample(X,y)
    return X,y
t0=time.time()
for seed in SEEDS:
  for model in MODELS:
    for cond in CONDS:
      if model in('LR',) and cond in('original','class_weight') and seed!=SEEDS[0]: continue   # deterministic
      for design,d in D.items():
        if (model,cond,design,seed) in done: continue
        cast=(lambda a:a.astype(np.float64)) if model=='LR' else (lambda a:a)   # LR in float64: float32 changes the optimizer path (see 07_lr_discrepancy.py)
        Xtr,ytr=resample(cond,cast(d['Xtr']),d['ytr'],seed); n_train=len(ytr); n_train_fraud=int(ytr.sum())
        m=build(model,cond,seed,ytr)
        if model=='RF' and len(ytr)>600000: m.set_params(max_samples=600000/len(ytr))   # computational cap on per-tree sample
        m.fit(Xtr,ytr); sv=m.predict_proba(cast(d['Xva']))[:,1]; thr,fval=best_f1_threshold(d['yva'],sv)
        rows=[]
        for pop,(Xt,yt) in d['tests'].items():
            st=m.predict_proba(cast(Xt))[:,1]; np.savez_compressed(f'{RES}/scores/{model}_{cond}_{design}_{pop}_seed{seed}.npz',test=st.astype(np.float32),val=sv.astype(np.float32))
            tp,fp,tn,fn=counts(yt,st,thr); p,r,f,fpr=prf(tp,fp,tn,fn)
            row=dict(model=model,condition=cond,design=design,test_population=pop,seed=seed,train_rows_after_resampling=n_train,train_fraud_after_resampling=n_train_fraud,
                     test_rows=len(yt),test_fraud=int(yt.sum()),test_prevalence_pct=round(100*yt.mean(),4),threshold=thr,val_f1=fval,
                     pr_auc=average_precision_score(yt,st),roc_auc=roc_auc_score(yt,st),precision=p,recall=r,f1=f,fpr=fpr,TP=tp,FP=fp,TN=tn,FN=fn,alerts=tp+fp)
            b=Boot(yt,st,thr).run(1000 if len(yt)<500000 else 200,seed)
            for c_ in b.columns: row[f'{c_}_ci_low']=float(b[c_].quantile(.025)); row[f'{c_}_ci_high']=float(b[c_].quantile(.975))
            rows.append(row)
        pd.DataFrame(rows).to_csv(OUTCSV,mode='a',header=not os.path.exists(OUTCSV),index=False)
        print(f'{model} {cond} {design} seed{seed} thr={thr:.4g} '+' | '.join(f"{r['test_population']}: PR-AUC {r['pr_auc']:.3f} P {r['precision']:.3f} R {r['recall']:.3f}" for r in rows)+f' [{time.time()-t0:.0f}s]',flush=True)
