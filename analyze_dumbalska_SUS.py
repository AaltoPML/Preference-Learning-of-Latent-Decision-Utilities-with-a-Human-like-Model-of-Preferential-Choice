import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import json

# DUMBALSKA

N_EXPERIMENTS = [100, 100, 100, 100, 100]
EXP_PATH = 'dumbalska'
EXP_FOLDER = 'experiment_data'
EXP_NAMES = ["BT", "CRCS", "LCL", "BBBT", "LC-CRCS"]
VARS = ["posterior_entropy", "utility_marginal_posterior_entropy", "hyperparameter_marginal_posterior_entropy", "n_unique_particles", "mean_ll_val", "mean_l_val", "mle_mean_ll_val"]

EXP_DATA = {}

for (EXP_NAME, N) in zip(EXP_NAMES, N_EXPERIMENTS):
    EXP_DATA[EXP_NAME] = {}
    for i in range(N):
        if i in [9, 14, 15, 20, 29, 32, 40, 46, 49, 53, 54, 57, 60, 61, 66, 75, 79, 82, 88, 94, 99, 104, 108, 111, 116, 121, 123, 127, 129, 130, 146, 147, 150, 159, 160, 170, 183, 184, 195, 197, 198, 200, 203, 218]:
            # participants excluded from original study
            continue
        with open(f"{EXP_PATH}/{EXP_FOLDER}/SUS_{EXP_NAME}_{i}.json", "r") as f:
            DATA = json.load(f)
        for VAR in VARS:
            if VAR not in DATA.keys():
                continue
            if VAR not in EXP_DATA[EXP_NAME].keys():
                EXP_DATA[EXP_NAME][VAR] = []
            EXP_DATA[EXP_NAME][VAR].append(DATA[VAR])
    for VAR in VARS:
        if VAR not in DATA.keys():
            continue
        var_raw = np.array(EXP_DATA[EXP_NAME][VAR])
        var_mean = var_raw.mean(axis=0)
        var_stderr = var_raw.std(axis=0) / np.sqrt(N)
        EXP_DATA[EXP_NAME][VAR] = {"raw": var_raw, "mean": var_mean, "stderr": var_stderr}


plt.figure(figsize=(3.2*1.333,3.2))
x = [i for i in range(len(EXP_DATA["BT"]["mean_l_val"]["mean"]))]
plt.errorbar(x, y = EXP_DATA["CRCS"]["mean_l_val"]["mean"], yerr=EXP_DATA["CRCS"]["mean_l_val"]["stderr"] * 2, label = "CRCS")
plt.errorbar(x, y = EXP_DATA["LC-CRCS"]["mean_l_val"]["mean"], yerr=EXP_DATA["LC-CRCS"]["mean_l_val"]["stderr"] * 2, label = "LC-CRCS")
plt.errorbar(x, y = EXP_DATA["BT"]["mean_l_val"]["mean"], yerr=EXP_DATA["BT"]["mean_l_val"]["stderr"] * 2, label = "Bradley-Terry")
plt.errorbar(x, y = EXP_DATA["LCL"]["mean_l_val"]["mean"], yerr=EXP_DATA["LCL"]["mean_l_val"]["stderr"] * 2, label = "LCL")
plt.errorbar(x, y = EXP_DATA["BBBT"]["mean_l_val"]["mean"], yerr=EXP_DATA["BBBT"]["mean_l_val"]["stderr"] * 2, label = "Bower & Balzano")
plt.legend()
plt.ylabel("expected likelihood")
plt.xlabel("Number of preference queries")
plt.grid()
plt.ylim(top=0.8)
plt.tight_layout(pad=0)
plt.show()