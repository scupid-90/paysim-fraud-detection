#!/usr/bin/env bash
# Reproduce every result in results/ from the raw PaySim CSV.
# Usage: ./run_all.sh [path/to/PaySim.csv]
# Runtime: about 2 hours on 2 CPU cores for the five-seed model comparison,
# plus 1-2 hours for ablation and SHAP. Peak memory about 7 GB.
set -euo pipefail

CSV="${1:-data/PS_20174392719_1491204439457_log.csv}"
RES=results
SEEDS=42,123,2024,31415,8675309

if [ ! -f "$CSV" ]; then
  echo "PaySim CSV not found at $CSV. See data/README.md." >&2
  exit 1
fi
mkdir -p "$RES/audit" figures

python code/01_audit.py "$CSV" "$RES/audit"                      # data audit, tail detection
python code/02_define_tail_and_splits.py "$CSV" "$RES"           # chronological + random splits -> split_indices.npz
python code/03_baseline_logistic_regression.py "$CSV" "$RES"     # LR baseline
python code/04_model_comparison.py "$CSV" "$RES" "$SEEDS"        # LR / RF / XGBoost x 4 imbalance conditions x 2 designs
python code/05_ablation_xgboost.py "$CSV" "$RES" 42              # all ablation configs, seed 42
for s in 123 2024 31415 8675309; do                              # key ablation configs, remaining seeds
  python code/05_ablation_xgboost.py "$CSV" "$RES" "$s" full,no_dest,no_balances
done
python code/06_shap_stability.py "$CSV" "$RES" "$SEEDS"          # SHAP rank stability
python code/07_lr_discrepancy.py "$CSV" "$RES"                   # float32 vs float64 LR diagnosis
python code/08_reproducibility_record.py "$RES"                  # versions, params, seeds
python code/09_shap_aggregate.py "$RES"
python code/10_aggregate_results.py "$RES"                       # fails if any seed/model/condition is missing
python code/12_isolation_forest.py "$CSV" "$RES" "$SEEDS"        # unsupervised baseline
python code/13_cutoff_sensitivity.py "$CSV" "$RES"               # alternative tail cutoffs
python code/11_make_figures.py "$RES" figures "$CSV" --seed 42
python code/14_build_prototype.py "$CSV" "$RES"                  # train + persist prototype models
python code/15_test_prototype.py "$RES/prototype"                # 21 prototype checks
