"""Baseline: Logistic Regression (class_weight='balanced'), validation-F1 threshold, class-conditional bootstrap CIs.
Groups: chrono->chrono, random->random, random->chrono (transferability)."""
import sys, time, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
CSV, RES = sys.argv[1], sys.argv[2]
SEED=42; NB=1000
NUM=['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']; CAT=['type']
df=pd.read_csv(CSV,usecols=NUM+CAT+['isFraud'],dtype={'type':'category'})
ix=np.load(f'{RES}/split_indices.npz')
def part(k): d=df.iloc[ix[k]]; return d[NUM+CAT],d.isFraud.values
def make():
    pre=ColumnTransformer([('n',StandardScaler(),NUM),('c',OneHotEncoder(handle_unknown='ignore'),CAT)])
    return Pipeline([('pre',pre),('lr',LogisticRegression(class_weight='balanced',max_iter=1000,random_state=SEED))])
def best_f1_threshold(y,s):
    p,r,t=precision_recall_curve(y,s); f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
    i=int(np.argmax(f)); return float(t[i]),float(f[i])
def counts(y,s,thr):
    pred=s>=thr; tp=int((pred&(y==1)).sum()); fp=int((pred&(y==0)).sum()); fn=int((~pred&(y==1)).sum()); tn=int((~pred&(y==0)).sum())
    return tp,fp,tn,fn
def prf(tp,fp,tn,fn):
    p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0; f=2*p*r/(p+r) if p+r else 0.0; fpr=fp/(fp+tn) if fp+tn else 0.0
    return p,r,f,fpr
# weighted (bootstrap) metrics on pre-sorted scores
class Boot:
    def __init__(s,y,sc,thr,thr2):
        o=np.argsort(-sc,kind='stable'); s.y=y[o]; s.sc=sc[o]; s.pos=np.where(s.y==1)[0]; s.neg=np.where(s.y==0)[0]
        s.pred=[(s.sc>=t) for t in (thr,thr2)]
    def wmetrics(s,w):
        y=s.y; tpc=np.cumsum(w*y); fpc=np.cumsum(w*(1-y)); P=tpc[-1]; N=fpc[-1]
        ap=float(np.sum(w*y*tpc/np.maximum(tpc+fpc,1e-12))/P)
        tpr=np.concatenate([[0],tpc/P]); fpr=np.concatenate([[0],fpc/N]); auc=float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2))
        out={'pr_auc':ap,'roc_auc':auc}
        for nm,pr in zip(('val_f1_thr','fixed_0.5'),s.pred):
            tp=float(np.sum(w*y*pr)); fp=float(np.sum(w*(1-y)*pr)); fn=P-tp; tn=N-fp
            p,r,f,fr=prf(tp,fp,tn,fn); out.update({f'{nm}_precision':p,f'{nm}_recall':r,f'{nm}_f1':f,f'{nm}_fpr':fr})
        return out
    def run(s,nb,seed):
        rng=np.random.default_rng(seed); res=[]
        for _ in range(nb):
            w=np.zeros(len(s.y)); w[s.pos]=rng.multinomial(len(s.pos),np.full(len(s.pos),1/len(s.pos))); w[s.neg]=rng.multinomial(len(s.neg),np.full(len(s.neg),1/len(s.neg)))
            res.append(s.wmetrics(w))
        return pd.DataFrame(res)
# sanity: weighted metrics with w=1 must equal sklearn
def sanity(y,sc):
    b=Boot(y,sc,0.5,0.5); m=b.wmetrics(np.ones(len(y)))
    assert abs(m['pr_auc']-average_precision_score(y,sc))<2e-3,(m['pr_auc'],average_precision_score(y,sc))
    assert abs(m['roc_auc']-roc_auc_score(y,sc))<1e-6
t0=time.time(); X={}
Xtr,ytr=part('chrono_train'); Xva,yva=part('chrono_val'); Xte,yte=part('chrono_test')
mc=make().fit(Xtr,ytr); print('chrono fit',time.time()-t0,flush=True)
Rtr,rtr=part('rand_train'); Rva,rva=part('rand_val'); Rte,rte=part('rand_test')
mr=make().fit(Rtr,rtr); print('rand fit',time.time()-t0,flush=True)
jobs=[('chronological_model_on_chronological_test',mc,Xva,yva,Xte,yte),
      ('random_model_on_random_test',mr,Rva,rva,Rte,rte),
      ('random_model_on_chronological_test',mr,Rva,rva,Xte,yte)]
M=[];C=[];T=[]
for name,model,Va,yv,Te,yt in jobs:
    sv=model.predict_proba(Va)[:,1]; st=model.predict_proba(Te)[:,1]
    thr,fval=best_f1_threshold(yv,sv); sanity(yt,st)
    row={'group':name,'test_rows':len(yt),'test_fraud':int(yt.sum()),'test_fraud_prevalence_pct':round(100*yt.mean(),4),
         'threshold_val_f1':thr,'val_f1':fval,'pr_auc':average_precision_score(yt,st),'roc_auc':roc_auc_score(yt,st)}
    for nm,th in (('val_f1_thr',thr),('fixed_0.5',0.5)):
        tp,fp,tn,fn=counts(yt,st,th); p,r,f,fpr=prf(tp,fp,tn,fn)
        row.update({f'{nm}_precision':p,f'{nm}_recall':r,f'{nm}_f1':f,f'{nm}_fpr':fpr,f'{nm}_alerts':tp+fp})
        C.append({'group':name,'threshold_rule':nm,'threshold':th,'TP':tp,'FP':fp,'TN':tn,'FN':fn,'alerts':tp+fp})
    b=Boot(yt,st,thr,0.5).run(NB,SEED)
    for c in b.columns: row[f'{c}_ci_low']=float(b[c].quantile(.025)); row[f'{c}_ci_high']=float(b[c].quantile(.975))
    M.append(row); T.append({'group':name,'threshold_val_f1':thr,'val_f1':fval,'validation_rows':len(yv),'validation_fraud':int(yv.sum())})
    print(name,'done',time.time()-t0,flush=True)
pd.DataFrame(M).to_csv(f'{RES}/baseline_metrics.csv',index=False); pd.DataFrame(C).to_csv(f'{RES}/baseline_confusion_matrices.csv',index=False); pd.DataFrame(T).to_csv(f'{RES}/baseline_thresholds.csv',index=False)
print(pd.DataFrame(M).T.to_string())
