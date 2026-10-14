"""Diagnose LR random-split PR-AUC discrepancy (baseline 0.542 vs comparison 0.529): float64 sparse (baseline) vs float32 dense (comparison)."""
import sys,numpy as np,pandas as pd,warnings
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
CSV,RES=sys.argv[1],sys.argv[2]
NUM=['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest'];CAT=['type']
df=pd.read_csv(CSV,usecols=NUM+CAT+['isFraud'],dtype={'type':'category'});ix=np.load(f'{RES}/split_indices.npz')
g=lambda k:(df.iloc[ix[k]][NUM+CAT],df.iloc[ix[k]].isFraud.values)
Xtr,ytr=g('rand_train');Xte,yte=g('rand_test')
rows=[]
for name,sparse,dt in [('sparse_float64 (baseline)',True,np.float64),('dense_float32 (comparison)',False,np.float32),('dense_float64',False,np.float64)]:
    pre=ColumnTransformer([('n',StandardScaler(),NUM),('c',OneHotEncoder(handle_unknown='ignore',sparse_output=sparse),CAT)])
    A=pre.fit_transform(Xtr); B=pre.transform(Xte)
    if not sparse: A=A.astype(dt);B=B.astype(dt)
    elif dt==np.float32: pass
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always'); m=LogisticRegression(class_weight='balanced',max_iter=1000,random_state=42).fit(A,ytr)
    s=m.predict_proba(B)[:,1]
    rows.append(dict(variant=name,n_iter=int(m.n_iter_[0]),converged=int(m.n_iter_[0])<1000 and not any('converge' in str(x.message).lower() for x in w),
                     pr_auc=average_precision_score(yte,s),roc_auc=roc_auc_score(yte,s),coef_norm=float(np.linalg.norm(m.coef_)),intercept=float(m.intercept_[0])))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(f'{RES}/lr_discrepancy_diagnosis.csv',index=False)
