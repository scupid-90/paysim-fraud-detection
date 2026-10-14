"""Figure templates. Run AFTER aggregation. Usage: 11_make_figures.py RES OUTDIR [CSV] [--seed 42]
Fig1 prevalence & volume by step (+tail); Fig2 PR curves (chrono test, XGB/RF/LR x original) for chosen seed; Fig3 ablation PR-AUC (mean+-SD across seeds);
Fig4 SHAP rank heatmap (mixed_population and fraud_only, seed-level mean rank); Fig5 seed variability (PR-AUC by model/condition); Fig6 confusion matrices (selected)."""
import sys,os,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
RES,OUT=sys.argv[1],sys.argv[2]; os.makedirs(OUT,exist_ok=True); SEED=int(sys.argv[sys.argv.index('--seed')+1]) if '--seed' in sys.argv else 42
CSV=sys.argv[3] if len(sys.argv)>3 and not sys.argv[3].startswith('--') else None
plt.rcParams.update({'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False,'font.size':9})
def save(f,n): f.tight_layout(); f.savefig(f'{OUT}/{n}.png'); plt.close(f); print('wrote',n)
# Fig1
fs=pd.read_csv(f'{RES}/audit/fraud_by_step.csv'); tail=719
f,ax=plt.subplots(2,1,figsize=(7,5),sharex=True)
ax[0].semilogy(fs.step,fs.rows.clip(lower=1),lw=.8); ax[0].set_ylabel('transactions per step (log)')
ax[1].plot(fs.step,fs.fraud_pct,lw=.8); ax[1].set_ylabel('fraud rate per step (%)'); ax[1].set_xlabel('step (hour)')
for a in ax: a.axvline(tail,ls='--',c='k',lw=.8); 
ax[0].text(tail-5,ax[0].get_ylim()[1],'tail (719-743)',ha='right',va='top',fontsize=8); save(f,'fig1_prevalence_volume_by_step')
# Fig2
if CSV and os.path.exists(f'{RES}/scores'):
    y=pd.read_csv(CSV,usecols=['isFraud']).isFraud.values[np.load(f'{RES}/split_indices.npz')['chrono_test']]
    f,a=plt.subplots(figsize=(5,4))
    for m in ['LR','RF','XGB']:
        p=f'{RES}/scores/{m}_original_chronological_chronological_seed{SEED}.npz'
        if os.path.exists(p): s=np.load(p)['test']; pr,rc,_=precision_recall_curve(y,s); a.plot(rc,pr,label=m)
    a.axhline(y.mean(),ls=':',c='gray',label=f'prevalence {y.mean():.4f}'); a.set_xlabel('recall'); a.set_ylabel('precision'); a.legend(); a.set_title(f'Chronological test, original data, seed {SEED}'); save(f,'fig2_pr_curves')
# Fig3
ab=f'{RES}/final_ablation_summary.csv'
if os.path.exists(ab):
    A=pd.read_csv(ab); A=A[(A.validation_design=='chronological')&(A.subgroup=='ALL')]
    f,a=plt.subplots(figsize=(5,3.5)); o=['full','no_dest','no_origin','no_balances','engineered']; A=A.set_index('feature_configuration').reindex([x for x in o if x in set(A.feature_configuration)])
    bars=a.bar(A.index,A.pr_auc_mean,yerr=A.pr_auc_sd.fillna(0))
    for b_,n_ in zip(bars,A.n_seeds):
        if n_==1: b_.set_hatch('//'); b_.set_alpha(.6)
    a.text(0.98,0.97,'hatched = seed 42 only',transform=a.transAxes,ha='right',va='top',fontsize=7); a.set_ylabel('PR-AUC (mean ± SD over seeds)'); a.set_ylim(0,1.15); plt.xticks(rotation=20); save(f,'fig3_ablation_pr_auc')
# Fig4
sh=f'{RES}/shap_mean_abs.csv'
if os.path.exists(sh):
    M=pd.read_csv(sh); M=M[M.part=='imbalance']
    f,ax=plt.subplots(1,2,figsize=(9,3.2))
    for a,sname in zip(ax,sorted(M['sample'].unique())):
        T=M[M['sample']==sname].groupby(['feature','model'])['rank'].mean().unstack().reindex(columns=['original','class_weight','undersample','smote'])
        im=a.imshow(T.values,cmap='viridis_r',vmin=1,vmax=6); a.set_xticks(range(T.shape[1])); a.set_xticklabels(T.columns,rotation=30,ha='right'); a.set_yticks(range(len(T))); a.set_yticklabels(T.index); a.set_title(f'{sname} (mean rank over seeds)')
        for i in range(T.shape[0]):
            for j in range(T.shape[1]): a.text(j,i,f'{T.values[i,j]:.1f}',ha='center',va='center',color='w',fontsize=8)
    save(f,'fig4_shap_rank_heatmap')
# Fig5
fm=f'{RES}/final_results_long.csv'
if os.path.exists(fm):
    F=pd.read_csv(fm); F=F[(F.feature_configuration=='full_6_features')&(F.validation_design=='chronological')&(F.test_population=='chronological')]
    f,a=plt.subplots(figsize=(7,3.5)); k=0; tick=[]
    for m in ['LR','RF','XGB']:
        for c in ['original','class_weight','undersample','smote']:
            v=F[(F.model==m)&(F.imbalance_method==c)].pr_auc.values; a.scatter(np.full(len(v),k),v,s=14); a.hlines(v.mean(),k-.3,k+.3,color='k',lw=1); tick.append(f'{m}\n{c[:5]}'); k+=1
    a.set_xticks(range(k)); a.set_xticklabels(tick,fontsize=6); a.set_ylabel('PR-AUC per seed'); save(f,'fig5_seed_variability')
# Fig6
cm=f'{RES}/final_results_long.csv'
if os.path.exists(cm):
    F=pd.read_csv(cm); G=F[(F.feature_configuration=='full_6_features')&(F.validation_design=='chronological')&(F.test_population=='chronological')&(F.seed==SEED)&(F.model=='XGB')]
    f,ax=plt.subplots(1,4,figsize=(10,2.6))
    for a,(c,g) in zip(ax,G.groupby('imbalance_method')):
        r=g.iloc[0]; a.imshow([[r.true_negatives,r.false_positives],[r.false_negatives,r.true_positives]],cmap='Blues',norm=matplotlib.colors.LogNorm(vmin=1))
        for i,j,v in [(0,0,r.true_negatives),(0,1,r.false_positives),(1,0,r.false_negatives),(1,1,r.true_positives)]: a.text(j,i,int(v),ha='center',va='center',fontsize=8)
        a.set_title(f'XGB {c}',fontsize=8); a.set_xticks([0,1]); a.set_xticklabels(['pred 0','pred 1']); a.set_yticks([0,1]); a.set_yticklabels(['true 0','true 1'])
    save(f,'fig6_confusion_matrices')
