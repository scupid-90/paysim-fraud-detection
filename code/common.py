import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_curve
def best_f1_threshold(y,s):
    p,r,t=precision_recall_curve(y,s); f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
    i=int(np.argmax(f)); return float(t[i]),float(f[i])
def counts(y,s,thr):
    pred=s>=thr; tp=int((pred&(y==1)).sum()); fp=int((pred&(y==0)).sum()); fn=int((~pred&(y==1)).sum()); tn=int((~pred&(y==0)).sum())
    return tp,fp,tn,fn
def prf(tp,fp,tn,fn):
    p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0; f=2*p*r/(p+r) if p+r else 0.0; fpr=fp/(fp+tn) if fp+tn else 0.0
    return p,r,f,fpr
class Boot:
    """Class-conditional bootstrap of test predictions via multinomial weights on pre-sorted scores."""
    def __init__(s,y,sc,thr):
        o=np.argsort(-sc,kind='stable'); s.y=y[o].astype(np.float64); s.pos=np.where(s.y==1)[0]; s.neg=np.where(s.y==0)[0]; s.pred=(sc[o]>=thr)
    def wmetrics(s,w):
        y=s.y; tpc=np.cumsum(w*y); fpc=np.cumsum(w*(1-y)); P=tpc[-1]; N=fpc[-1]
        ap=float(np.sum(w*y*tpc/np.maximum(tpc+fpc,1e-12))/P)
        tpr=np.concatenate([[0],tpc/P]); fpr=np.concatenate([[0],fpc/N]); auc=float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2))
        tp=float(np.sum(w*y*s.pred)); fp=float(np.sum(w*(1-y)*s.pred)); fn=P-tp; tn=N-fp
        p,r,f,fr=prf(tp,fp,tn,fn)
        return {'pr_auc':ap,'roc_auc':auc,'precision':p,'recall':r,'f1':f,'fpr':fr}
    def run(s,nb,seed=0):
        rng=np.random.default_rng(seed); res=[]
        for _ in range(nb):
            w=np.zeros(len(s.y)); w[s.pos]=rng.multinomial(len(s.pos),np.full(len(s.pos),1/len(s.pos))); w[s.neg]=rng.multinomial(len(s.neg),np.full(len(s.neg),1/len(s.neg)))
            res.append(s.wmetrics(w))
        return pd.DataFrame(res)
