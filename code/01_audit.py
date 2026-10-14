"""PaySim data-quality audit. Writes reproducible CSV outputs to results/audit/."""
import sys, pandas as pd, numpy as np
CSV = sys.argv[1]; OUT = sys.argv[2]
dt = {'step':'int16','type':'category','amount':'float64','nameOrig':'object','oldbalanceOrg':'float64',
      'newbalanceOrig':'float64','nameDest':'object','oldbalanceDest':'float64','newbalanceDest':'float64',
      'isFraud':'int8','isFlaggedFraud':'int8'}
df = pd.read_csv(CSV, dtype=dt); df['isFraud']=df['isFraud'].astype('int64')
n = len(df); fr = int(df.isFraud.sum())
S = {}
S['total_rows']=n; S['fraud_rows']=fr; S['fraud_pct']=round(100*fr/n,4)
S['min_step']=int(df.step.min()); S['max_step']=int(df.step.max()); S['unique_steps']=int(df.step.nunique())
allsteps=set(range(S['min_step'],S['max_step']+1)); S['missing_steps']=str(sorted(allsteps-set(df.step.unique())))
S['isFlaggedFraud_positives']=int(df.isFlaggedFraud.sum())
S['fraud_caught_by_isFlaggedFraud']=int(((df.isFlaggedFraud==1)&(df.isFraud==1)).sum())
S['duplicate_rows']=int(df.drop(columns=[]).duplicated().sum())
S['missing_values_total']=int(df.isna().sum().sum())
S['negative_amounts']=int((df.amount<0).sum()); S['zero_amounts']=int((df.amount==0).sum())
S['zero_amount_fraud']=int(((df.amount==0)&(df.isFraud==1)).sum())
for c in ['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']:
    S[f'negative_{c}']=int((df[c]<0).sum())
S['unique_nameOrig']=int(df.nameOrig.nunique()); S['unique_nameDest']=int(df.nameDest.nunique())
S['rows_where_nameOrig_seen_before']=int(n-df.nameOrig.nunique())
# by type
bt=df.groupby('type',observed=True).agg(rows=('isFraud','size'),fraud=('isFraud','sum')); bt['fraud_pct']=(100*bt.fraud/bt.rows).round(4)
for c,nm in [('oldbalanceDest','oldDest'),('newbalanceDest','newDest'),('oldbalanceOrg','oldOrg'),('newbalanceOrig','newOrig')]:
    z=df[df[c]==0].groupby('type',observed=True).size()
    zf=df[(df[c]==0)&(df.isFraud==1)].groupby('type',observed=True).size()
    bt[f'{nm}_zero_all_pct']=(100*z.reindex(bt.index).fillna(0)/bt.rows).round(2)
    bt[f'{nm}_zero_fraud_pct']=(100*zf.reindex(bt.index).fillna(0)/bt.fraud.replace(0,np.nan)).round(2)
bt.to_csv(f'{OUT}/fraud_by_type.csv')
# by step
bs=df.groupby('step').agg(rows=('isFraud','size'),fraud=('isFraud','sum')).astype('int64'); bs['fraud_pct']=(100*bs.fraud/bs.rows).round(4)
bs['legit']=bs.rows-bs.fraud
bs.to_csv(f'{OUT}/fraud_by_step.csv'); bs[['rows']].to_csv(f'{OUT}/volume_by_step.csv')
# tail rule (pre-specified, data-based): the tail is the maximal terminal run of steps containing ZERO legitimate transactions
zl=(bs.legit==0); tail_start=None
for s_ in sorted(bs.index):
    if zl.loc[s_:].all(): tail_start=int(s_); break
med=float(bs.loc[1:700,'rows'].median()); thr=float('nan'); col=zl
S['median_rows_per_step_1_700']=med
S['steps_with_zero_legit_total']=int(zl.sum())
S['tail_start_step']=tail_start
if tail_start is not None:
    t=df[df.step>=tail_start]; S['tail_rows']=len(t); S['tail_fraud_rows']=int(t.isFraud.sum())
    S['tail_fraud_pct']=round(100*t.isFraud.mean(),2)
    S['tail_types']=str(t.type.value_counts().to_dict())
# first collapsed step regardless of persistence, and rows around the boundary
# tail comparison: full vs truncated at 718 and neighbours
rows=[]
for cut in [None,700,718,719,720]:
    d=df if cut is None else df[df.step<=cut]
    rows.append({'dataset':'full' if cut is None else f'steps<= {cut}','rows':len(d),'fraud':int(d.isFraud.sum()),'fraud_pct':round(100*d.isFraud.mean(),4),'max_step':int(d.step.max())})
pd.DataFrame(rows).to_csv(f'{OUT}/tail_comparison.csv',index=False)
# feature summary / missingness
df.isna().sum().rename('missing').to_csv(f'{OUT}/feature_missingness.csv')
df[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']].describe().T.to_csv(f'{OUT}/feature_summary.csv')
# balance identity checks (artifact evidence)
oe=(df.newbalanceOrig+df.amount-df.oldbalanceOrg).abs()>0.01
de=(df.oldbalanceDest+df.amount-df.newbalanceDest).abs()>0.01
chk=df[['type','isFraud']].copy(); chk['orig_mismatch']=oe; chk['dest_mismatch']=de
cm=chk.groupby(['type','isFraud'],observed=True)[['orig_mismatch','dest_mismatch']].mean().mul(100).round(2)
cm.to_csv(f'{OUT}/balance_identity_mismatch_pct.csv')
pd.Series(S).to_csv(f'{OUT}/audit_summary.csv',header=['value'],index_label='measure')
print(pd.Series(S).to_string())
print(bt.to_string()); print(bs.tail(30).to_string()); print(cm.to_string()); print(pd.read_csv(f'{OUT}/tail_comparison.csv').to_string())
print(bs.head(30).to_string()); print(df[df.amount==0][['step','type','isFraud','isFlaggedFraud']].describe(include='all').to_string()); print(bs.loc[1:718,'fraud_pct'].describe().to_string())

bs['day']=(bs.index-1)//24+1
dd=bs.groupby('day').agg(rows=('rows','sum'),fraud=('fraud','sum'),zero_legit_steps=('legit',lambda x:int((x==0).sum()))); dd['fraud_pct']=(100*dd.fraud/dd.rows).round(3)
dd.to_csv(f'{OUT}/fraud_by_day.csv')
