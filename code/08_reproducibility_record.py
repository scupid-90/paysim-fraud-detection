"""Write results/reproducibility_record.json + .csv: seeds, dtype, hyperparameters, thresholds, bootstrap, software versions, hardware."""
import sys,json,platform,os,sklearn,xgboost,imblearn,shap,numpy,pandas,scipy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
RES=sys.argv[1]; lr=LogisticRegression(); 
rec=dict(
 random_seeds=[42,123,2024,31415,8675309], split_seed=42, shap_sample_seed=0,
 features=['type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest'], feature_order_after_encoding='[amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest, type one-hot in category order]',
 scaling='StandardScaler fit on training partition only', encoding='OneHotEncoder(handle_unknown=ignore) fit on training partition only',
 dtype=dict(LR='float64 (float32 version superseded)',RF='float32',XGB='float32'),
 LR=dict(solver=lr.solver,C=lr.C,tol=lr.tol,max_iter=1000,penalty=lr.penalty,class_weight='balanced only in class_weight condition'),
 RF=dict(n_estimators=100,min_samples_leaf=3,max_samples_cap_rows=600000,n_jobs=2,class_weight='balanced only in class_weight condition'),
 XGB=dict(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,tree_method='hist',n_jobs=2,scale_pos_weight='neg/pos from training partition only in class_weight condition'),
 resampling=dict(undersample='RandomUnderSampler fraud:legit 1:1',smote='SMOTE sampling_strategy=0.1,k_neighbors=5, applied to encoded features, training only'),
 threshold_rule='validation maximum F1, applied once to test', bootstrap_resamples=dict(chronological_test=1000,random_test=200), bootstrap='class-conditional multinomial weights',
 chronological_split=dict(train='steps 1-502',validation='steps 503-610',test='steps 611-718'), random_split='stratified 70/15/15 on steps 1-718, seed 42',
 software=dict(python=platform.python_version(),scikit_learn=sklearn.__version__,xgboost=xgboost.__version__,imbalanced_learn=imblearn.__version__,shap=shap.__version__,numpy=numpy.__version__,pandas=pandas.__version__,scipy=scipy.__version__),
 hardware=dict(cpu_threads_used=2,cpu_count=os.cpu_count(),platform=platform.platform()))
json.dump(rec,open(f'{RES}/reproducibility_record.json','w'),indent=2)
def flat(d,p=''):
    for k,v in d.items():
        if isinstance(v,dict): yield from flat(v,p+k+'.')
        else: yield p+k,str(v)
import pandas as pd; pd.DataFrame(list(flat(rec)),columns=['item','value']).to_csv(f'{RES}/reproducibility_record.csv',index=False); print(open(f'{RES}/reproducibility_record.csv').read()[:600])
