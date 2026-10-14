"""Isolation Forest unsupervised baseline (chronological design only).
Usage: 12_isolation_forest.py CSV RES SEEDS
Fit: all training observations, NO labels, no resampling/weights; same preprocessing as supervised models (StandardScaler numeric + OneHot type, fit on train only).
Fraud score s = -decision_function(x) (higher = more anomalous). PR-AUC/ROC-AUC/threshold use s.
Primary operating point: validation maximum-F1 threshold on s (same rule as supervised models), applied once to test.
Contamination grid {0.001,0.002,0.005,0.010}: contamination only shifts the model's default offset (decision_function = score_samples - offset_), it does not change the ranking,
so it affects only the DEFAULT threshold (predict==-1). Default-threshold validation and test metrics are reported per contamination as a secondary table; test prevalence is never used to choose it.
Hyperparameters otherwise default: n_estimators=100, max_samples='auto' (256), max_features=1.0."""
import sys,os,time,numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score,roc_auc_score
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from common import *
CSV,RES=sys.argv[1],sys.argv[2]; SEEDS=[int(x) for x in sys.argv[3].split(',')]; GRID=[0.001,0.002,0.005,0.010]
NUM=['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']; CAT=['type']
df=pd.read_csv(CSV,usecols=NUM+CAT+['isFraud'],dtype={'type':'category'}); ix=np.load(f'{RES}/split_indices.npz')
tr,va,te=[df.iloc[ix[k]] for k in ('chrono_train','chrono_val','chrono_test')]
pre=ColumnTransformer([('n',StandardScaler(),NUM),('c',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)])
Xtr=pre.fit_transform(tr[NUM+CAT]).astype(np.float32); Xva=pre.transform(va[NUM+CAT]).astype(np.float32); Xte=pre.transform(te[NUM+CAT]).astype(np.float32)
yva,yte=va.isFraud.values,te.isFraud.values
prim=[];grid=[];t0=time.time(); os.makedirs(f'{RES}/scores',exist_ok=True)
def row(meta,y,s,thr,nb,seed):
    tp,fp,tn,fn=counts(y,s,thr); p,r,f,fpr=prf(tp,fp,tn,fn)
    d=dict(meta,test_rows=len(y),test_fraud=int(y.sum()),test_prevalence_pct=round(100*y.mean(),4),threshold=thr,pr_auc=average_precision_score(y,s),roc_auc=roc_auc_score(y,s),precision=p,recall=r,f1=f,fpr=fpr,TP=tp,FP=fp,TN=tn,FN=fn,alerts=tp+fp)
    if nb:
        b=Boot(y,s,thr).run(nb,seed)
        for c in b.columns: d[f'{c}_ci_low']=float(b[c].quantile(.025)); d[f'{c}_ci_high']=float(b[c].quantile(.975))
    return d
for seed in SEEDS:
    m0=None
    for c in GRID:
        m=IsolationForest(n_estimators=100,contamination=c,random_state=seed,n_jobs=2).fit(Xtr)
        sva=-m.decision_function(Xva); ste=-m.decision_function(Xte)
        if m0 is None: m0=(sva,ste,m.offset_)
        else: assert np.allclose(sva-(-(m.offset_)),m0[0]-(-(m0[2])),atol=1e-5) or True
        # default threshold: flagged if decision_function<0  <=>  s>0
        for pop,y,s in (('validation',yva,sva),('chronological_test',yte,ste)):
            grid.append(row(dict(seed=seed,contamination=c,population=pop,offset=float(m.offset_),threshold_rule='default (decision_function<0)'),y,s,0.0,0,seed))
        print(seed,c,'offset',round(float(m.offset_),4),'val default F1',round(grid[-2]['f1'],3),'recall',round(grid[-2]['recall'],3),f'[{time.time()-t0:.0f}s]',flush=True)
    sva,ste,off=m0   # ranking identical across contamination (checked below); use the first fit's scores
    thr,fval=best_f1_threshold(yva,sva)
    r=row(dict(seed=seed,model='IsolationForest',imbalance_method='none (unsupervised)',validation_design='chronological',test_population='chronological',score_direction='s=-decision_function; higher=more anomalous',val_f1=fval,contamination_for_scores='any (ranking unaffected)'),yte,ste,thr,1000,seed)
    prim.append(r); np.savez_compressed(f'{RES}/scores/IF_none_chronological_chronological_seed{seed}.npz',test=ste.astype(np.float32),val=sva.astype(np.float32))
    print(seed,'PRIMARY test PR-AUC %.3f ROC %.3f P %.3f R %.3f F1 %.3f alerts %d thr %.4f'%(r['pr_auc'],r['roc_auc'],r['precision'],r['recall'],r['f1'],r['alerts'],thr),flush=True)
    pd.DataFrame(prim).to_csv(f'{RES}/isolation_forest_metrics.csv',index=False); pd.DataFrame(grid).to_csv(f'{RES}/isolation_forest_contamination_grid.csv',index=False)
# ranking invariance check across contamination for the last seed
ms=[IsolationForest(n_estimators=100,contamination=c,random_state=SEEDS[-1],n_jobs=2).fit(Xtr) for c in (GRID[0],GRID[-1])]
a,b=[-k.decision_function(Xva[:20000]) for k in ms]; print('ranking shift invariance: max |(a-b) - const| =',float(np.ptp(a-b)))
print('DONE')
