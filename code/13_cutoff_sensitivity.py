"""Cutoff sensitivity (seed 42): chronological 70/15/15 by step on steps 1..CUTOFF for CUTOFF in 718 (reproduction check), 700 (nearby cutoff), 743 (full file incl. all-fraud tail).
Models: XGBoost original with full 6 features, and XGBoost original without balance fields (amount + type). Same hyperparameters, preprocessing, threshold rule (validation max-F1), bootstrap (1000).
Partition rule: n=CUTOFF steps; train = steps 1..(7n)//10; validation next round-half-up(.15n)=(3n+10)//20 steps; test = the rest (718 reproduces 502/108/108).
Usage: 13_cutoff_sensitivity.py CSV RES"""
import sys,os,time,numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import average_precision_score,roc_auc_score
from xgboost import XGBClassifier
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from common import *
CSV,RES=sys.argv[1],sys.argv[2]; SEED=42
BAL=['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']
df=pd.read_csv(CSV,usecols=['step','type','amount']+BAL+['isFraud'],dtype={'type':'category'})
CFG={'full':['amount']+BAL,'no_balances':['amount']}
rows=[];t0=time.time()
for cut in (718,700,743):
    # integer arithmetic (floating-point .70*700 gives 489.99..)
    n=cut; a=(7*n)//10; b=a+(3*n+10)//20; d=df[df.step<=cut]
    parts={'train':d[d.step<=a],'validation':d[(d.step>a)&(d.step<=b)],'test':d[d.step>b]}
    info={k:(int(v.step.min()),int(v.step.max()),len(v),int(v.isFraud.sum())) for k,v in parts.items()}
    print(cut,info,flush=True)
    for cfg,num in CFG.items():
        pre=ColumnTransformer([('n',StandardScaler(),num),('c',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['type'])]); cols=num+['type']
        Xtr=pre.fit_transform(parts['train'][cols]).astype(np.float32); ytr=parts['train'].isFraud.values
        m=XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,tree_method='hist',n_jobs=2,random_state=SEED,eval_metric='aucpr',verbosity=0).fit(Xtr,ytr)
        sv=m.predict_proba(pre.transform(parts['validation'][cols]).astype(np.float32))[:,1]; yv=parts['validation'].isFraud.values; thr,fv=best_f1_threshold(yv,sv)
        for pop,dd in (('test',parts['test']),) + ((('test_excluding_tail_steps_719_743',parts['test'][parts['test'].step<=718]),) if cut==743 else ()):
            y=dd.isFraud.values; s=m.predict_proba(pre.transform(dd[cols]).astype(np.float32))[:,1]
            tp,fp,tn,fn=counts(y,s,thr); p,r,f,fpr=prf(tp,fp,tn,fn)
            row=dict(cutoff=cut,configuration=cfg,test_population=pop,train_steps=f'{info["train"][0]}-{info["train"][1]}',val_steps=f'{info["validation"][0]}-{info["validation"][1]}',test_steps=f'{int(dd.step.min())}-{int(dd.step.max())}',
                     train_rows=info['train'][2],train_fraud=info['train'][3],val_rows=info['validation'][2],val_fraud=info['validation'][3],test_rows=len(y),test_fraud=int(y.sum()),test_prevalence_pct=round(100*y.mean(),4),
                     seed=SEED,threshold=thr,val_f1=fv,pr_auc=average_precision_score(y,s),roc_auc=roc_auc_score(y,s) if (y==0).any() else np.nan,precision=p,recall=r,f1=f,fpr=fpr,TP=tp,FP=fp,TN=tn,FN=fn,alerts=tp+fp)
            if (y==0).any():
                bb=Boot(y,s,thr).run(1000,SEED)
                for c in ('pr_auc','precision','recall'): row[f'{c}_ci_low']=float(bb[c].quantile(.025)); row[f'{c}_ci_high']=float(bb[c].quantile(.975))
            rows.append(row); print(cut,cfg,pop,'PR-AUC %.3f P %.3f R %.3f FPR %.5f prev %.3f%% [%ds]'%(row['pr_auc'],p,r,fpr,row['test_prevalence_pct'],time.time()-t0),flush=True)
        pd.DataFrame(rows).to_csv(f'{RES}/cutoff_sensitivity_seed42.csv',index=False)
print('DONE')
