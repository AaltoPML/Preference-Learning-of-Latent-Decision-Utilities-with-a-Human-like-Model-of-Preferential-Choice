import numpy as np
import matplotlib.pyplot as plt
import json

N_EXPERIMENTS = [100]
EXP_PATH = 'water_management'
EXP_FOLDER = 'experiment_data'
EXP_NAMES = ["CRCS"]
VARS = ["posterior_entropy", "utility_marginal_posterior_entropy", "hyperparameter_marginal_posterior_entropy", "n_unique_particles", "utility_inference_error", "hyperparameter_inference_error", "recommendation_regret", "context_effect_min_wasserstein", "context_effect_min_prob_ratio", "context_effect_mean_wasserstein", "context_effect_mean_prob_ratio"]

EXP_DATA = {}

for (EXP_NAME, N) in zip(EXP_NAMES, N_EXPERIMENTS):
    EXP_DATA[EXP_NAME] = {}
    for i in range(N):
        with open(f"{EXP_PATH}/{EXP_FOLDER}/{EXP_NAME}_{i}.json", "r") as f:
            DATA = json.load(f)
        for VAR in VARS:
            if VAR not in EXP_DATA[EXP_NAME].keys():
                EXP_DATA[EXP_NAME][VAR] = []
            EXP_DATA[EXP_NAME][VAR].append(DATA[VAR][:51])
    for VAR in VARS:
        var_raw = np.array(EXP_DATA[EXP_NAME][VAR])
        var_mean = var_raw.mean(axis=0)
        var_stderr = var_raw.std(axis=0) / np.sqrt(N)
        EXP_DATA[EXP_NAME][VAR] = {"raw": var_raw, "mean": var_mean, "stderr": var_stderr}

x = [i for i in range(len(EXP_DATA["CRCS"]["posterior_entropy"]["mean"]))]
plt.errorbar(x, y = EXP_DATA["CRCS"]["posterior_entropy"]["mean"], yerr=EXP_DATA["CRCS"]["posterior_entropy"]["stderr"] * 2)
plt.title("Entropy reduction over time (mean +- 2 std.err.)")
plt.ylabel("Posterior Entropy")
plt.grid()
plt.show()

x = [i for i in range(len(EXP_DATA["CRCS"]["utility_inference_error"]["mean"]))]
plt.errorbar(x, y = EXP_DATA["CRCS"]["utility_inference_error"]["mean"], yerr=EXP_DATA["CRCS"]["utility_inference_error"]["stderr"] * 2)
plt.ylabel("Mean utility inference error")
plt.xlabel("Number of preference queries presented to user.")
plt.grid()
plt.show()

x = [i for i in range(len(EXP_DATA["CRCS"]["hyperparameter_inference_error"]["mean"]))]
plt.errorbar(x, y = EXP_DATA["CRCS"]["hyperparameter_inference_error"]["mean"], yerr=EXP_DATA["CRCS"]["hyperparameter_inference_error"]["stderr"] * 2)
plt.ylabel("Mean model parameter inference error")
plt.xlabel("Number of preference queries.")
plt.grid()
plt.show()

x = [i for i in range(len(EXP_DATA["CRCS"]["recommendation_regret"]["mean"]))]
plt.errorbar(x, y = EXP_DATA["CRCS"]["recommendation_regret"]["mean"], yerr=EXP_DATA["CRCS"]["recommendation_regret"]["stderr"] * 2)
plt.ylabel("Recommendation regret")
plt.xlabel("Number of preference queries.")
plt.grid()
plt.show()