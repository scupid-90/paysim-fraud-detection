# Experiment Protocol — FROZEN 2026-10-07
No change to anything below after this date. Any deviation must be logged as a dated amendment at the bottom, with the reason.

**Data.** PaySim, 6,362,620 rows, steps 1–743. Primary data: steps 1–718 (tail rule: maximal terminal run of steps with zero legitimate rows starts at step 719). Sensitivity: full steps 1–743, plus one nearby alternative cutoff.
**Splits.** Chronological: train 1–502 / validation 503–610 / test 611–718. Random: stratified 70/15/15 on steps 1–718, seed 42. Split indices fixed in `split_indices.npz`.
**Features.** `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`. Excluded: `step`, `isFlaggedFraud`, `nameOrig`, `nameDest`, `isFraud`. Balance-error features: separate sensitivity condition only.
**Preprocessing.** StandardScaler + OneHot(type), fit on the training partition only.
**Imbalance conditions (supervised models).** original; class weighting (balanced / scale_pos_weight = neg/pos); random undersampling 1:1; SMOTE to fraud:legit 0.1 (k=5). Resampling on training data only.
**Models.** LR (lbfgs, C=1, tol=1e-4, max_iter=1000, **float64**); RF (100 trees, min_samples_leaf 3, 600k-row per-tree cap); XGBoost (200 trees, depth 6, lr 0.1, subsample/colsample 0.8, hist). Isolation Forest: separate unsupervised design (below).
**Threshold.** Maximum F1 on validation, applied once to test. Fixed 0.5 is a diagnostic only.
**Metrics.** PR-AUC (average precision), ROC-AUC, precision, recall, F1, FPR, confusion matrix (TN, FP, FN, TP), alerts; fraud class is positive; zero-division → 0. Always report test prevalence beside PR-AUC and precision.
**Uncertainty.** Class-conditional bootstrap 95% CI: 1000 resamples (chronological test), 200 (random test). Seeds: 42, 123, 2024, 31415, 8675309. LR original/class_weight are deterministic (one fit; no SD reported).
**Interpretation.** Random and chronological results are different test populations and are not ranked against each other. Random model → chronological test is a contaminated cross-period stress test. No significance tests applied mechanically; report mean ± SD across seeds and bootstrap intervals; large consistent effects only.
**Ablation.** XGBoost, original data: full / no_origin / no_dest / no_balances (five seeds) and engineered balance-error (seed 42, sensitivity); aggregate and TRANSFER / CASH_OUT subgroups. Shows model dependence, not proof of leakage.
**SHAP.** XGBoost TreeSHAP, log-odds space. Samples fixed: `mixed_population` (20,000 rows, simple random, seed 0) and `fraud_only` (all 1,192 chronological-test fraud rows), reused for all models and seeds. Per-seed Spearman, top-5 overlap, directional agreement (numeric features only); aggregate mean/SD/min/max across seeds; no pooling of SHAP values across seeds.
**Isolation Forest (design only, not yet run).** Fit without labels on training features; no SMOTE / undersampling / class weights; contamination grid {0.001, 0.005, 0.01} and threshold chosen on validation only; evaluate once on chronological test; anomaly-score direction reported.
**Prototype (not yet built).** Uses the final selected model and threshold fixed before test inspection.
**Archive rules.** Superseded float32 LR results kept in `model_comparison_LR_float32_superseded.csv`, never aggregated. Aggregation (`10_aggregate_results.py`) refuses to run if any expected seed/model/condition/design is missing.

## Amendments
(none)
