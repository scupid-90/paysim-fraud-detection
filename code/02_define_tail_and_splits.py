"""Define primary data (tail excluded), chronological and random splits. Writes split tables and saves split indices."""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
CSV, OUT = sys.argv[1], sys.argv[2]
SEED = 42
dt={'step':'int16','type':'category','amount':'float64','nameOrig':'object','oldbalanceOrg':'float64','newbalanceOrig':'float64',
    'nameDest':'object','oldbalanceDest':'float64','newbalanceDest':'float64','isFraud':'int64','isFlaggedFraud':'int8'}
df=pd.read_csv(CSV,dtype=dt)
# tail rule: maximal terminal run of steps with zero legitimate rows
leg=df[df.isFraud==0].groupby('step').size().reindex(range(df.step.min(),df.step.max()+1),fill_value=0)
tail_start=None
for s in leg.index:
    if (leg.loc[s:]==0).all(): tail_start=int(s); break
cut=tail_start-1
prim=df[df.step<=cut]
def row(design,part,d):
    return dict(design=design,partition=part,step_min=int(d.step.min()),step_max=int(d.step.max()),unique_steps=int(d.step.nunique()),
                rows=len(d),fraud=int(d.isFraud.sum()),legitimate=int((d.isFraud==0).sum()),fraud_rate_pct=round(100*d.isFraud.mean(),4))
def chrono(b1,b2,label):
    return [row(label,'train',prim[prim.step<=b1]),row(label,'validation',prim[(prim.step>b1)&(prim.step<=b2)]),row(label,'test',prim[prim.step>b2])]
rows=[row('primary data (steps 1-%d)'%cut,'all',prim),row('tail excluded','all',df[df.step>cut])]
rows+=chrono(502,610,'chronological (proposed 70/15/15 by step)')
# comparison designs, reported for transparency only (not selected on model results)
rows+=chrono(408,563,'alt A: train = high-volume regime (steps<=408)')
# alt B: boundaries giving equal-fraud 70/15/15 (cumulative fraud)
cf=prim[prim.isFraud==1].groupby('step').size().reindex(range(1,cut+1),fill_value=0).cumsum()/prim.isFraud.sum()
b1=int(cf[cf>=0.70].index[0]); b2=int(cf[cf>=0.85].index[0])
rows+=chrono(b1,b2,f'alt B: equal-fraud 70/15/15 (cuts {b1},{b2})')
# random stratified 70/15/15 on primary data
idx=np.arange(len(prim)); y=prim.isFraud.values
tr,tmp=train_test_split(idx,test_size=0.30,stratify=y,random_state=SEED)
va,te=train_test_split(tmp,test_size=0.50,stratify=y[tmp],random_state=SEED)
for nm,i in [('train',tr),('validation',va),('test',te)]: rows.append(row('random stratified 70/15/15',nm,prim.iloc[i]))
st=pd.DataFrame(rows); st.to_csv(f'{OUT}/split_table.csv',index=False)
st[['design','partition','fraud_rate_pct']].to_csv(f'{OUT}/prevalence_by_partition.csv',index=False)
# save indices of the primary-data rows (row labels in the original file) for later scripts
np.savez(f'{OUT}/split_indices.npz',tail_start=tail_start,cutoff=cut,
         chrono_train=prim.index[prim.step<=502].values,chrono_val=prim.index[(prim.step>502)&(prim.step<=610)].values,chrono_test=prim.index[prim.step>610].values,
         rand_train=prim.index[tr].values,rand_val=prim.index[va].values,rand_test=prim.index[te].values,seed=SEED)
pd.set_option('display.width',250); print('tail_start',tail_start,'cutoff',cut); print(st.to_string())
