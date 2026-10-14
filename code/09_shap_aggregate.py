"""Aggregate per-seed SHAP stability: score computed per seed first, then mean/SD/min/max across seeds. Reads results/shap_pairwise_stability.csv."""
import sys,pandas as pd
RES=sys.argv[1]; P=pd.read_csv(f'{RES}/shap_pairwise_stability.csv')
g=P.groupby(['part','sample','model_a','model_b'],sort=False)
A=g.agg(n_seeds=('seed','nunique'),n_shared_features=('n_shared_features','first'),shared_features=('shared_features','first'),removed_in_b=('removed_in_b','first'),
        spearman_mean=('spearman','mean'),spearman_sd=('spearman','std'),spearman_min=('spearman','min'),spearman_max=('spearman','max'),
        top5_overlap_mean=('top_k_overlap','mean'),top5_overlap_sd=('top_k_overlap','std'),
        dir_agree_mean=('directional_agreement','mean'),dir_agree_sd=('directional_agreement','std')).reset_index()
A.to_csv(f'{RES}/shap_stability_aggregate.csv',index=False)
# reference-vs-others summary for imbalance part (original vs resampled; resampled among themselves)
I=P[P.part=='imbalance'].copy(); I['pair_type']=I.apply(lambda r:'original_vs_resampled' if 'original' in (r.model_a,r.model_b) else 'resampled_vs_resampled',axis=1)
S=I.groupby(['sample','pair_type']).agg(n_scores=('spearman','size'),spearman_mean=('spearman','mean'),spearman_sd=('spearman','std'),spearman_min=('spearman','min'),spearman_max=('spearman','max'),top5_mean=('top_k_overlap','mean'),top5_sd=('top_k_overlap','std')).reset_index()
S.to_csv(f'{RES}/shap_stability_pairtype_summary.csv',index=False); print(S.to_string())
