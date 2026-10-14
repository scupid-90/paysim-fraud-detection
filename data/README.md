# Data

The PaySim CSV (~470 MB) is not included in this repository.

1. Download it from Kaggle: https://www.kaggle.com/datasets/ealaxi/paysim1
2. Place it here:

```
data/PS_20174392719_1491204439457_log.csv
```

Or with the Kaggle CLI:

```bash
kaggle datasets download -d ealaxi/paysim1 -p data --unzip
```

The file used for the reported results has 6,362,620 rows over steps 1 to 743.

## Source

PaySim was created by Dr. Edgar Lopez-Rojas. Simulator code: https://github.com/EdgarLopezPhD/PaySim (GPL-3.0). Please cite:

Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). PaySim: A financial mobile money simulator for fraud detection. *28th European Modeling and Simulation Symposium (EMSS)*, Larnaca, Cyprus.
