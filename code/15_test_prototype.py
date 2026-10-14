"""Prototype tests: valid hand-built transactions, invalid inputs, unseen types, tampered artifact. Usage: 15_test_prototype.py PROTO_DIR"""
import sys,os,json,shutil,tempfile,math
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','prototype'))
from fraud_scorer import FraudScorer,ValidationError
D=sys.argv[1]; sc=FraudScorer.load(D); res=[]
def ok(name,cond,detail=''): res.append((name,bool(cond),detail)); print(('PASS ' if cond else 'FAIL ')+name,detail)
# valid transactions
fraud_like=dict(type='TRANSFER',amount=181.0,oldbalanceOrg=181.0,newbalanceOrig=0.0,oldbalanceDest=0.0,newbalanceDest=0.0)      # origin drained, destination unchanged (PaySim fraud pattern)
legit_pay=dict(type='PAYMENT',amount=9839.64,oldbalanceOrg=170136.0,newbalanceOrig=160296.36,oldbalanceDest=0.0,newbalanceDest=0.0)
legit_cashin=dict(type='CASH_IN',amount=5000.0,oldbalanceOrg=20000.0,newbalanceOrig=25000.0,oldbalanceDest=100000.0,newbalanceDest=95000.0)
r1=sc.score(fraud_like,explain=True); r2=sc.score(legit_pay); r3=sc.score(legit_cashin)
print(json.dumps(r1,indent=1)[:900])
ok('fraud-like TRANSFER is flagged for review (full model)',r1['decision']=='review',f"score={r1['fraud_score']}")
ok('legitimate PAYMENT approved',r2['decision']=='approve',f"score={r2['fraud_score']}")
ok('legitimate CASH_IN approved',r3['decision']=='approve',f"score={r3['fraud_score']}")
ok('response has warnings and comparator',all(k in r1 for k in('data_warning','balance_feature_warning','comparator')) and r1['comparator']['model']=='xgboost_original_no_balances')
ok('score is in [0,1] and not labelled probability',0<=r1['fraud_score']<=1 and 'probability' not in json.dumps(r1).replace('not a calibrated probability',''))
ok('explanation lists six features',len(r1['explanation'])==6,str([e['feature'] for e in r1['explanation']][:3]))
# comparator ignores balances
a=dict(fraud_like); b=dict(fraud_like,oldbalanceOrg=999999.0,newbalanceOrig=500.0)
ok('no-balance comparator score independent of balance fields',sc.score(a)['comparator']['fraud_score']==sc.score(b)['comparator']['fraud_score'])
# invalid inputs
bad=[('missing field',{k:v for k,v in fraud_like.items() if k!='amount'}),('unseen type',dict(fraud_like,type='WITHDRAWAL')),('negative amount',dict(fraud_like,amount=-5.0)),
     ('string amount',dict(fraud_like,amount='181')),('NaN balance',dict(fraud_like,oldbalanceOrg=float('nan'))),('infinite amount',dict(fraud_like,amount=float('inf'))),
     ('boolean amount',dict(fraud_like,amount=True)),('identifier supplied',dict(fraud_like,nameOrig='C123')),('label supplied',dict(fraud_like,isFraud=1)),('None value',dict(fraud_like,newbalanceDest=None)),('not a mapping',[1,2,3])]
for n,t in bad:
    try: sc.score(t); ok(f'reject: {n}',False,'accepted')
    except ValidationError as e: ok(f'reject: {n}',True,str(e)[:70])
# determinism & tamper check
ok('deterministic scoring',sc.score(fraud_like)['fraud_score']==sc.score(fraud_like)['fraud_score'])
tmp=tempfile.mkdtemp(); shutil.copytree(D,tmp+'/p'); open(tmp+'/p/xgb_full_original.json','a').write(' ')
try: FraudScorer.load(tmp+'/p'); ok('tampered artifact refused',False)
except RuntimeError as e: ok('tampered artifact refused',True,str(e)[:60])
# reproduction of recorded test metrics is in manifest
m=json.load(open(f'{D}/manifest.json')); ok('manifest records reproduced PR-AUC',abs(m['full']['reproduced_chronological_test']['pr_auc']-0.9888)<5e-4)
n=sum(1 for _,c,_ in res if c); print(f'{n}/{len(res)} checks passed'); json.dump([dict(test=a,passed=b,detail=c) for a,b,c in res],open(f'{D}/test_results.json','w'),indent=1)
sys.exit(0 if n==len(res) else 1)
