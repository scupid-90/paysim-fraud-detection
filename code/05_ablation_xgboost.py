"""Pre-specified feature ablation on XGBoost (original-data condition), seed 42.
Configs: full | no_origin | no_dest | no_balances | engineered (full + 2 balance-error features).
Evaluations: chronological->chronological, random->random, random->chronological; subgroups ALL / TRANSFER / CASH_OUT
(retrained within subgroup: train on subgroup train rows, threshold on subgroup validation rows, evaluate on subgroup test rows).
Also grouped permutation importance (PR-AUC drop) for the full model."""
import sys, os, time, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from common import *
CSV,RES=sys.argv[1],sys.argv[2]; SEED=int(sys.argv[3]) if len(sys.argv)>3 else 42
SUF='' if SEED==42 else f'_seed{SEED}'
ONLY=sys.argv[4].split(',') if len(sys.argv)>4 else None   # subset of configs (key ablations for extra seeds)
BAL_O=['oldbalanceOrg','newbalanceOrig']; BAL_D=['oldbalanceDest','newbalanceDest']
CONFIGS={'full':['amount']+BAL_O+BAL_D,'no_origin':['amount']+BAL_D,'no_dest':['amount']+BAL_O,'no_balances':['amount'],
         'engineered':['amount']+BAL_O+BAL_D+['origin_balance_error','destination_balance_error']}
df=pd.read_csv(CSV,usecols=['type','amount']+BAL_O+BAL_D+['isFraud'],dtype={'type':'category'})
df['origin_balance_error']=df.newbalanceOrig+df.amount-df.oldbalanceOrg; df['destination_balance_error']=df.oldbalanceDest+df.amount-df.newbalanceDest
ix=np.load(f'{RES}/split_indices.npz')
pd.DataFrame([{'configuration':k,'numeric_features':','.join(v),'includes_type':True} for k,v in CONFIGS.items()]).to_csv(f'{RES}/ablation_feature_groups{SUF}.csv',index=False)
def part(k): d=df.iloc[ix[k]]; return d
P={k:part(k) for k in ['chrono_train','chrono_val','chrono_test','rand_train','rand_val','rand_test']}
DESIGNS={'chronological':('chrono_train','chrono_val',{'chronological':'chrono_test'}),
         'random':('rand_train','rand_val',{'random':'rand_test','chronological':'chrono_test'})}
XGB=lambda: XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,tree_method='hist',n_jobs=2,random_state=SEED,eval_metric='aucpr',verbosity=0)
def make_pre(num,use_type):
    tr=[('n',StandardScaler(),num)]+([('c',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['type'])] if use_type else [])
    return ColumnTransformer(tr)
def evaluate(model,pre,dtr_,dva,dte,mask_tr=None):
    pass
rows=[];cms=[];perm=[];t0=time.time()
def metrics_row(meta,y,s,thr,fval,nb):
    tp,fp,tn,fn=counts(y,s,thr); p,r,f,fpr=prf(tp,fp,tn,fn)
    row=dict(meta,test_rows=len(y),test_fraud=int(y.sum()),test_prevalence_pct=round(100*y.mean(),4),threshold=thr,val_f1=fval,
             pr_auc=average_precision_score(y,s),roc_auc=roc_auc_score(y,s),precision=p,recall=r,f1=f,fpr=fpr,TP=tp,FP=fp,TN=tn,FN=fn,alerts=tp+fp)
    b=Boot(y,s,thr).run(nb,SEED)
    for c in b.columns: row[f'{c}_ci_low']=float(b[c].quantile(.025)); row[f'{c}_ci_high']=float(b[c].quantile(.975))
    return row
SUBS=[('ALL',None),('TRANSFER','TRANSFER'),('CASH_OUT','CASH_OUT')]
for cfg,num in CONFIGS.items():
  if ONLY and cfg not in ONLY: continue
  for design,(trk,vak,tests) in DESIGNS.items():
    for sub,typ in SUBS:
        sel=lambda d:(d if typ is None else d[d.type==typ])
        dtr,dva=sel(P[trk]),sel(P[vak]); use_type=(typ is None)
        pre=make_pre(num,use_type); cols=num+(['type'] if use_type else [])
        Xtr=pre.fit_transform(dtr[cols]).astype(np.float32); ytr=dtr.isFraud.values
        m=XGB().fit(Xtr,ytr); sv=m.predict_proba(pre.transform(dva[cols]).astype(np.float32))[:,1]; thr,fval=best_f1_threshold(dva.isFraud.values,sv)
        for pop,tk in tests.items():
            dte=sel(P[tk]); y=dte.isFraud.values
            if y.sum()<5: continue
            s=m.predict_proba(pre.transform(dte[cols]).astype(np.float32))[:,1]
            r=metrics_row(dict(configuration=cfg,design=design,test_population=pop,subgroup=sub,evaluation_mode='retrained_within_subgroup' if typ else 'primary',seed=SEED,
                               train_rows=len(ytr),train_fraud=int(ytr.sum())),y,s,thr,fval,1000 if len(y)<500000 else 200)
            rows.append(r)
            if cfg=='full' and sub=='ALL':   # grouped permutation importance on full model
                groups={'origin_balances':BAL_O,'destination_balances':BAL_D,'amount':['amount'],'transaction_type':['type']}
                rng=np.random.default_rng(SEED); base=average_precision_score(y,s)
                for g,cs in groups.items():
                    drops=[]
                    for rep in range(5):
                        dp=dte[cols].copy(); idx=rng.permutation(len(dp))
                        for c in cs: dp[c]=dp[c].values[idx] if c!='type' else dp[c].cat.set_categories(dp[c].cat.categories).values[idx]
                        sp=m.predict_proba(pre.transform(dp).astype(np.float32))[:,1]; drops.append(base-average_precision_score(y,sp))
                    perm.append(dict(design=design,test_population=pop,group=g,baseline_pr_auc=base,mean_pr_auc_drop=float(np.mean(drops)),sd_drop=float(np.std(drops)),repeats=5))
        print(cfg,design,sub,f'thr={thr:.4g}',' | '.join(f"{r['test_population']}: PR-AUC {r['pr_auc']:.3f} R {r['recall']:.3f}" for r in rows[-len(tests):]),f'[{time.time()-t0:.0f}s]',flush=True)
        pd.DataFrame(rows).to_csv(f'{RES}/ablation_xgboost_seed{SEED}.csv',index=False); pd.DataFrame(perm).to_csv(f'{RES}/grouped_permutation_importance{SUF}.csv',index=False)
R=pd.DataFrame(rows)
key=['design','test_population','subgroup']
full=R[R.configuration=='full'].set_index(key)['pr_auc']
R['pr_auc_full']=R.set_index(key).index.map(full); R['delta_pr_auc_vs_full']=R.pr_auc-R.pr_auc_full; R['relative_decline']=(R.pr_auc_full-R.pr_auc)/R.pr_auc_full
R.to_csv(f'{RES}/ablation_xgboost_seed{SEED}.csv',index=False)
R[['configuration','design','test_population','subgroup','threshold','TP','FP','TN','FN','alerts']].to_csv(f'{RES}/ablation_confusion_matrices{SUF}.csv',index=False)
R[R.subgroup!='ALL'].to_csv(f'{RES}/ablation_subgroups{SUF}.csv',index=False)
print('DONE')
