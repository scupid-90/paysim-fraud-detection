"""SHAP stability for XGBoost (chronological design, seeds given).
Usage: 06_shap_stability.py CSV RES SEEDS(comma)
Part A (imbalance): full 6 features; conditions original/class_weight/undersample/smote.
Part B (ablation): original data; configs full/no_origin/no_dest/no_balances.
SHAP = TreeSHAP in log-odds space via xgboost pred_contribs (exact TreeSHAP); one-hot 'type' columns are summed into one 'type' contribution.
Same fixed chronological-test samples for every model: 'mixed_population' (uniform random 20,000 rows, seed 0) and 'fraud_only' (all test fraud rows).
Stability = Spearman of mean|SHAP| rankings, top-5 overlap (actual common features reported), directional agreement.
Direction of feature f = sign of Spearman(feature value, SHAP_f) on the fraud_only sample ('type': coded TRANSFER=1 vs CASH_OUT=0 within sample;
other types ordinal-coded by category order). Raw SHAP magnitudes are never compared across models."""
import sys, os, itertools, time, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, DMatrix
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from common import *
CSV,RES=sys.argv[1],sys.argv[2]; SEEDS=[int(x) for x in sys.argv[3].split(',')]
BAL_O=['oldbalanceOrg','newbalanceOrig']; BAL_D=['oldbalanceDest','newbalanceDest']
CFG={'full':['amount']+BAL_O+BAL_D,'no_origin':['amount']+BAL_D,'no_dest':['amount']+BAL_O,'no_balances':['amount']}
CONDS=['original','class_weight','undersample','smote']
df=pd.read_csv(CSV,usecols=['type','amount']+BAL_O+BAL_D+['isFraud'],dtype={'type':'category'}); ix=np.load(f'{RES}/split_indices.npz')
tr=df.iloc[ix['chrono_train']]; te=df.iloc[ix['chrono_test']]
rng0=np.random.default_rng(0)
pop_idx=np.sort(rng0.choice(len(te),20000,replace=False)); fr_idx=np.where(te.isFraud.values==1)[0]
SAMPLES={'mixed_population':te.iloc[pop_idx],'fraud_only':te.iloc[fr_idx]}
pd.DataFrame({'sample':['mixed_population','fraud_only'],'rows':[len(pop_idx),len(fr_idx)],'fraud_rows':[int(SAMPLES['mixed_population'].isFraud.sum()),len(fr_idx)],'legit_rows':[len(pop_idx)-int(SAMPLES['mixed_population'].isFraud.sum()),0],'sampling':['simple random without replacement, seed 0, not stratified','all fraud rows in chronological test (steps 611-718)'],'same_rows_all_models_and_seeds':[True,True]}).to_csv(f'{RES}/shap_samples.csv',index=False)
def run(seed,cond,cfg):
    num=CFG[cfg]; cols=num+['type']
    pre=ColumnTransformer([('n',StandardScaler(),num),('c',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['type'])])
    X=pre.fit_transform(tr[cols]).astype(np.float32); y=tr.isFraud.values
    if cond=='undersample': X,y=RandomUnderSampler(sampling_strategy=1.0,random_state=seed).fit_resample(X,y)
    if cond=='smote': X,y=SMOTE(sampling_strategy=0.1,k_neighbors=5,random_state=seed).fit_resample(X,y)
    neg,pos=int((y==0).sum()),int(y.sum())
    m=XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,tree_method='hist',n_jobs=2,random_state=seed,
                    scale_pos_weight=(neg/pos if cond=='class_weight' else 1.0),eval_metric='aucpr',verbosity=0).fit(X,y)
    names=num+[f'type_{c}' for c in pre.named_transformers_['c'].categories_[0]]; out={}
    for sname,d in SAMPLES.items():
        Xs=pre.transform(d[cols]).astype(np.float32); c=m.get_booster().predict(DMatrix(Xs),pred_contribs=True)[:,:-1]
        sh=pd.DataFrame(c,columns=names); sh['type']=sh[[n for n in names if n.startswith('type_')]].sum(axis=1)
        sh=sh[num+['type']]; out[sname]=(sh,d)
    return out
def direction(sh,d):
    r={}
    for f in sh.columns:
        v=d['type'].cat.codes.values if f=='type' else d[f].values
        rho=spearmanr(v,sh[f].values)[0] if np.std(sh[f].values)>0 and np.std(v)>0 else 0.0
        r[f]=0 if abs(rho)<0.05 else int(np.sign(rho))
        if f=='type': r[f]=np.nan   # grouped one-hot: no simple direction; report rank/magnitude only
    return r
rows=[];dirs=[];t0=time.time(); store={}
jobs=[('imbalance',c,'full') for c in CONDS]+[('ablation','original',g) for g in ['no_origin','no_dest','no_balances']]
for seed in SEEDS:
    for part,cond,cfg in jobs:
        out=run(seed,cond,cfg); label=cond if part=='imbalance' else cfg
        for sname,(sh,d) in out.items():
            ma=sh.abs().mean(); rk=ma.rank(ascending=False,method='min'); dr=direction(sh,d)
            for f in sh.columns:
                rows.append(dict(seed=seed,part=part,model=label,sample=sname,feature=f,mean_abs_shap=ma[f],rank=int(rk[f]),mean_signed_shap=sh[f].mean(),direction=dr[f]))
        print(seed,part,label,'done',f'[{time.time()-t0:.0f}s]',flush=True)
        pd.DataFrame(rows).to_csv(f'{RES}/shap_mean_abs.csv',index=False)
M=pd.DataFrame(rows)
def compare(a,b,sub):
    s=sub[sub.model.isin([a,b])]; fa=s[s.model==a].set_index('feature'); fb=s[s.model==b].set_index('feature')
    shared=[f for f in fa.index if f in fb.index]
    ra=fa.loc[shared,'mean_abs_shap'].rank(ascending=False); rb=fb.loc[shared,'mean_abs_shap'].rank(ascending=False)
    rho=spearmanr(ra,rb)[0] if len(shared)>2 else np.nan
    k=min(5,len(shared)); ta=set(fa.loc[shared,'mean_abs_shap'].nlargest(k).index); tb=set(fb.loc[shared,'mean_abs_shap'].nlargest(k).index)
    ag=np.mean([fa.loc[f,'direction']==fb.loc[f,'direction'] for f in shared if f!='type']) if any(f!='type' for f in shared) else np.nan
    return dict(model_a=a,model_b=b,n_shared_features=len(shared),shared_features=','.join(shared),removed_in_b=','.join(f for f in fa.index if f not in fb.index),
                spearman=rho,top_k=k,top_k_overlap=len(ta&tb)/k,common_topk=','.join(sorted(ta&tb)),directional_agreement=ag,
                top3_a=','.join(fa.loc[shared,'mean_abs_shap'].nlargest(3).index),top3_b=','.join(fb.loc[shared,'mean_abs_shap'].nlargest(3).index))
P=[]
for (seed,part,sname),sub in M.groupby(['seed','part','sample']):
    models=list(dict.fromkeys(sub.model))
    pairs=list(itertools.combinations(models,2)) if part=='imbalance' else [('full',m) for m in models]
    if part=='ablation':   # full model comes from the imbalance/original run
        full=M[(M.seed==seed)&(M.part=='imbalance')&(M.model=='original')&(M['sample']==sname)].assign(model='full')
        sub=pd.concat([sub,full]); pairs=[('full',m) for m in models]
    for a,b in pairs: P.append(dict(seed=seed,part=part,sample=sname,**compare(a,b,sub)))
pd.DataFrame(P).to_csv(f'{RES}/shap_pairwise_stability.csv',index=False); print('DONE')
