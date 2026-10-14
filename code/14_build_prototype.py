"""Train and persist the prototype models (seed 42): XGBoost on ORIGINAL unresampled chronological training data (steps 1-502).
 full: type + amount + four balance fields;  comparator: type + amount (no balances). Threshold = validation (503-610) max-F1.
Writes results/prototype/ (JSON models, joblib preprocessors, manifest.json with hashes, params, versions, thresholds, reproduced test metrics).
Usage: 14_build_prototype.py CSV RES"""
import sys,os,json,platform,hashlib,numpy as np,pandas as pd,joblib,sklearn,xgboost
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from common import *
CSV,RES=sys.argv[1],sys.argv[2]; OUT=f'{RES}/prototype'; os.makedirs(OUT,exist_ok=True); SEED=42
BAL=['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']; CFG={'full':['amount']+BAL,'no_balances':['amount']}
ALLOWED=['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER']
df=pd.read_csv(CSV,usecols=['type','amount']+BAL+['isFraud']); df['type']=pd.Categorical(df['type'],categories=ALLOWED); ix=np.load(f'{RES}/split_indices.npz')
tr,va,te=[df.iloc[ix[k]] for k in ('chrono_train','chrono_val','chrono_test')]
man={'purpose':'research prototype; PaySim synthetic data; not for production use','training_data':'chronological training partition, steps 1-502, original (unresampled)','validation_data':'steps 503-610 (threshold selection)','seed':SEED,
     'threshold_rule':'validation maximum F1','xgb_params':dict(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,tree_method='hist',eval_metric='aucpr'),
     'software':{'python':platform.python_version(),'scikit_learn':sklearn.__version__,'xgboost':xgboost.__version__,'numpy':np.__version__,'pandas':pd.__version__,'joblib':joblib.__version__},'artifacts':{}}
for key,num in CFG.items():
    cols=num+['type']
    pre=ColumnTransformer([('n',StandardScaler(),num),('c',OneHotEncoder(categories=[ALLOWED],handle_unknown='error',sparse_output=False),['type'])])
    X=pre.fit_transform(tr[cols]).astype(np.float32); y=tr.isFraud.values
    m=XGBClassifier(**{k:v for k,v in man['xgb_params'].items()},n_jobs=2,random_state=SEED,verbosity=0).fit(X,y)
    sv=m.predict_proba(pre.transform(va[cols]).astype(np.float32))[:,1]; thr,fv=best_f1_threshold(va.isFraud.values,sv)
    st=m.predict_proba(pre.transform(te[cols]).astype(np.float32))[:,1]; yt=te.isFraud.values; tp,fp,tn,fn=counts(yt,st,thr); p,r,f,fpr=prf(tp,fp,tn,fn)
    name=f"xgboost_original_{'full_features' if key=='full' else 'no_balances'}"; mf=f'xgb_{key}_original.json'; pf=f'preprocessor_{key}.joblib'
    m.save_model(f'{OUT}/{mf}'); joblib.dump(pre,f'{OUT}/{pf}')
    for fn_ in (mf,pf): man['artifacts'][fn_]={'sha256':hashlib.sha256(open(f'{OUT}/{fn_}','rb').read()).hexdigest()}
    man[key]={'name':name,'columns':cols,'feature_names':list(num)+[f'type_{c}' for c in ALLOWED],'threshold':thr,'val_f1':fv,
              'reproduced_chronological_test':{'pr_auc':float(average_precision_score(yt,st)),'precision':p,'recall':r,'f1':f,'fpr':fpr,'TP':tp,'FP':fp,'TN':tn,'FN':fn}}
    print(key,'thr %.5f val F1 %.4f | test PR-AUC %.4f P %.3f R %.3f'%(thr,fv,average_precision_score(yt,st),p,r),flush=True)

json.dump(man,open(f'{OUT}/manifest.json','w'),indent=2); print('saved',OUT)
