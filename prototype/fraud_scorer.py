"""Research prototype: transaction-scoring workflow (PaySim). NOT a production system.
Pipeline: validate -> encode type + scale (training-fitted) -> XGBoost score -> validation-selected threshold -> review/approve -> warnings -> optional SHAP explanation -> no-balance comparator.
Artifacts: XGBoost models are stored as native JSON; the sklearn preprocessor is stored with joblib (pickle-based): load ONLY trusted artifacts in the recorded environment.
The artifact hashes in manifest.json are verified before loading."""
import json,hashlib,math,os,numpy as np,pandas as pd
ALLOWED_TYPES=['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER']
REQUIRED=['type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']
NUMERIC=REQUIRED[1:]
DATA_WARNING='PaySim is synthetic; performance may not transfer to real transactions. The score is not a calibrated probability.'
BALANCE_WARNING='The full model relies heavily on balance-related variables.'
class ValidationError(ValueError): pass
def _sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for ch in iter(lambda:f.read(1<<20),b''): h.update(ch)
    return h.hexdigest()
def validate(tx):
    if not isinstance(tx,dict): raise ValidationError('transaction must be a mapping of field names to values')
    missing=[k for k in REQUIRED if k not in tx or tx[k] is None]
    if missing: raise ValidationError(f'missing required fields: {missing}')
    unknown=[k for k in tx if k not in REQUIRED]
    if unknown: raise ValidationError(f'unexpected fields (identifiers and labels must not be supplied): {unknown}')
    if tx['type'] not in ALLOWED_TYPES: raise ValidationError(f"unknown transaction type {tx['type']!r}; allowed: {ALLOWED_TYPES}")
    out={'type':tx['type']}
    for k in NUMERIC:
        v=tx[k]
        if isinstance(v,bool) or not isinstance(v,(int,float,np.integer,np.floating)): raise ValidationError(f'{k} must be numeric, got {type(v).__name__}')
        v=float(v)
        if not math.isfinite(v): raise ValidationError(f'{k} must be finite')
        if v<0: raise ValidationError(f'{k} must be non-negative')
        out[k]=v
    return out
class FraudScorer:
    def __init__(self,d,man,pre,full,comp): self.dir,self.manifest,self.pre,self.full,self.comp=d,man,pre,full,comp
    @classmethod
    def load(cls,d):
        import joblib; from xgboost import XGBClassifier
        man=json.load(open(f'{d}/manifest.json'))
        for name,meta in man['artifacts'].items():
            if _sha(f'{d}/{name}')!=meta['sha256']: raise RuntimeError(f'artifact {name} failed its checksum; refusing to load')
        pre=joblib.load(f'{d}/preprocessor_full.joblib'); prec=joblib.load(f'{d}/preprocessor_no_balances.joblib')
        full=XGBClassifier(); full.load_model(f'{d}/xgb_full_original.json'); comp=XGBClassifier(); comp.load_model(f'{d}/xgb_no_balances_original.json')
        return cls(d,man,{'full':pre,'comp':prec},full,comp)
    def _score(self,key,model,cols,df):
        X=self.pre[key].transform(df[cols]).astype(np.float32); s=float(model.predict_proba(X)[:,1][0]); return X,s
    def explain(self,model,X,names,cat_prefix='type_'):
        from xgboost import DMatrix
        c=model.get_booster().predict(DMatrix(X),pred_contribs=True)[0][:-1]
        d={}
        for n,v in zip(names,c): k='type' if n.startswith(cat_prefix) else n; d[k]=d.get(k,0.0)+float(v)
        return sorted(({'feature':k,'shap_log_odds':round(v,4)} for k,v in d.items()),key=lambda r:-abs(r['shap_log_odds']))
    def score(self,tx,explain=False):
        t=validate(tx); df=pd.DataFrame([t]); m=self.manifest
        Xf,sf=self._score('full',self.full,m['full']['columns'],df); Xc,sc=self._score('comp',self.comp,m['no_balances']['columns'],df)
        tf,tc=m['full']['threshold'],m['no_balances']['threshold']
        r={'model':m['full']['name'],'fraud_score':round(sf,6),'threshold':tf,'decision':'review' if sf>=tf else 'approve',
           'data_warning':DATA_WARNING,'balance_feature_warning':BALANCE_WARNING,
           'comparator':{'model':m['no_balances']['name'],'fraud_score':round(sc,6),'threshold':tc,'decision':'review' if sc>=tc else 'approve'}}
        if explain: r['explanation']=self.explain(self.full,Xf,m['full']['feature_names'])
        return r
