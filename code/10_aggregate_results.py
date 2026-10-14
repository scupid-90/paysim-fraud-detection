"""Final aggregation into the single results schema. STRICT by default: errors if any expected seed/model/condition/design is missing.
Usage: 10_aggregate_results.py RES [--allow-partial]
Inputs : results/model_comparison_metrics.csv (float64 LR only), results/ablation_xgboost_seed*.csv
Outputs: results/final_results_long.csv (schema), results/final_multiseed_summary.csv, results/final_main_table.csv, results/final_ablation_summary.csv"""
import sys,glob,numpy as np,pandas as pd
RES=sys.argv[1]; PARTIAL='--allow-partial' in sys.argv
SEEDS=[42,123,2024,31415,8675309]; CONDS=['original','class_weight','undersample','smote']; DESIGNS=['chronological','random']; MODELS=['LR','RF','XGB']
PER={'chronological':('steps 1-502','steps 503-610'),'random':('stratified 70% of steps 1-718','stratified 15% of steps 1-718')}
TEST={'chronological':'steps 611-718','random':'stratified 15% of steps 1-718'}
def schema(d,model,cond,design,pop,cfg,sub='ALL'):
    return pd.DataFrame(dict(model=model,imbalance_method=cond,seed=d.seed,validation_design=design,train_period=PER[design][0],validation_period=PER[design][1],
        test_period=TEST[pop],test_population=pop,feature_configuration=cfg,subgroup=sub,test_prevalence=d.test_prevalence_pct/100,threshold=d.threshold,pr_auc=d.pr_auc,
        pr_auc_ci_low=d.pr_auc_ci_low,pr_auc_ci_high=d.pr_auc_ci_high,roc_auc=d.roc_auc,precision=d.precision,recall=d.recall,f1=d.f1,false_positive_rate=d.fpr,
        true_positives=d.TP,false_positives=d.FP,true_negatives=d.TN,false_negatives=d.FN,alerts=d.alerts))
C=pd.read_csv(f'{RES}/model_comparison_metrics.csv')
# ---- completeness checks
exp=set()
for s in SEEDS:
    for m in MODELS:
        for c in CONDS:
            if m=='LR' and c in('original','class_weight') and s!=SEEDS[0]: continue   # deterministic: seed 42 only
            for d in DESIGNS:
                for p in (['chronological'] if d=='chronological' else ['random','chronological']): exp.add((m,c,d,p,s))
got=set(zip(C.model,C.condition,C.design,C.test_population,C.seed))
missing=sorted(exp-got); extra=sorted(got-exp); dup=int(C.duplicated(['model','condition','design','test_population','seed']).sum())
print(f'expected rows {len(exp)}, found {len(got)}, missing {len(missing)}, unexpected {len(extra)}, duplicates {dup}')
if (missing or extra or dup) and not PARTIAL:
    for x in missing[:20]: print('MISSING',x)
    raise SystemExit('INCOMPLETE/INCONSISTENT RESULTS - refusing to aggregate (use --allow-partial only for inspection)')
assert C.model.isin(MODELS).all() and C.condition.isin(CONDS).all()
L=[schema(g,k[0],k[1],k[2],k[3],'full_6_features') for k,g in C.groupby(['model','condition','design','test_population'])]
for f in sorted(glob.glob(f'{RES}/ablation_xgboost_seed*.csv')):
    A=pd.read_csv(f)
    for k,g in A.groupby(['design','test_population','configuration','subgroup']):
        L.append(schema(g,'XGB','original',k[0],k[1],k[2],k[3]))
F=pd.concat(L,ignore_index=True); F.to_csv(f'{RES}/final_results_long.csv',index=False)
# ---- multi-seed summary (primary comparison rows only)
P=F[F.feature_configuration=='full_6_features']; keys=['model','imbalance_method','validation_design','test_population']
S=P.groupby(keys).agg(n_seeds=('seed','nunique'),test_prevalence=('test_prevalence','first'),pr_auc_mean=('pr_auc','mean'),pr_auc_sd=('pr_auc','std'),roc_auc_mean=('roc_auc','mean'),
   precision_mean=('precision','mean'),precision_sd=('precision','std'),recall_mean=('recall','mean'),recall_sd=('recall','std'),f1_mean=('f1','mean'),f1_sd=('f1','std'),
   fpr_mean=('false_positive_rate','mean'),threshold_mean=('threshold','mean'),alerts_mean=('alerts','mean')).reset_index()
S.loc[(S.model=='LR')&S.imbalance_method.isin(['original','class_weight']),['pr_auc_sd','precision_sd','recall_sd','f1_sd']]=np.nan   # deterministic: one fit
S.to_csv(f'{RES}/final_multiseed_summary.csv',index=False)
S[(S.validation_design==S.test_population)].to_csv(f'{RES}/final_main_table.csv',index=False)   # chrono->chrono and random->random; random->chrono stress test stays in the summary file
# ---- ablation summary
Ab=F[F.feature_configuration.isin(['full','no_origin','no_dest','no_balances','engineered'])]
if len(Ab):
    k=['feature_configuration','validation_design','test_population','subgroup']
    AS=Ab.groupby(k).agg(n_seeds=('seed','nunique'),pr_auc_mean=('pr_auc','mean'),pr_auc_sd=('pr_auc','std'),recall_mean=('recall','mean'),fpr_mean=('false_positive_rate','mean')).reset_index()
    full=AS[AS.feature_configuration=='full'].set_index(k[1:])['pr_auc_mean']
    AS['change_from_full']=AS.pr_auc_mean-AS.set_index(k[1:]).index.map(full); AS.to_csv(f'{RES}/final_ablation_summary.csv',index=False)
print(S[S.validation_design=='chronological'].round(3).to_string())
