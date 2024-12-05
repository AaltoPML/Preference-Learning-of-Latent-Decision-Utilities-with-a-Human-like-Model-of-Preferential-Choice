import numpy as np
from scipy.stats import wilcoxon
import json

N_EXPERIMENTS = 233
EXP_PATH = 'dumbalska'
EXP_FOLDER = 'experiment_data'

EXP_DATA = {"BT": {"ll": [], "expected_l": []}, 
            "BBBT": {"ll": [], "expected_l": []}, 
            "LCLBT": {"ll": [], "expected_l": []}, 
            "CR": {"ll": [], "expected_l": []}, 
            "LCLCR": {"ll": [], "expected_l": []}}

for i in range(N_EXPERIMENTS):
    if i in [9, 14, 15, 20, 29, 32, 40, 46, 49, 53, 54, 57, 60, 61, 66, 75, 79, 82, 88, 94, 99, 104, 108, 111, 116, 121, 123, 127, 129, 130, 146, 147, 150, 159, 160, 170, 183, 184, 195, 197, 198, 200, 203, 218]:
        continue
    with open(f"{EXP_PATH}/{EXP_FOLDER}/CV_{i}.json", "r") as f:
        DATA = json.load(f)
    for EXP_NAME in EXP_DATA.keys():
        for VAR in EXP_DATA[EXP_NAME].keys():
            EXP_DATA[EXP_NAME][VAR].append(DATA[EXP_NAME][VAR])

print(f"""BT: {np.sum(EXP_DATA["BT"]["ll"])} \t BBBT: {np.sum(EXP_DATA["BBBT"]["ll"])} \t LCL: {np.sum(EXP_DATA["LCLBT"]["ll"])} \t CR {np.sum(EXP_DATA["CR"]["ll"])} \t LCLCR {np.sum(EXP_DATA["LCLCR"]["ll"])}""")
print("BT vs BBBT", wilcoxon(EXP_DATA["BT"]["ll"], EXP_DATA["BBBT"]["ll"]).pvalue)
print("LCL vs BT", wilcoxon(EXP_DATA["LCLBT"]["ll"], EXP_DATA["BT"]["ll"]).pvalue)
print("LCL vs BBBT", wilcoxon(EXP_DATA["LCLBT"]["ll"], EXP_DATA["BBBT"]["ll"]).pvalue)
print("CRCS vs BT", wilcoxon(EXP_DATA["CR"]["ll"], EXP_DATA["BT"]["ll"]).pvalue)
print("CRCS vs BBBT", wilcoxon(EXP_DATA["CR"]["ll"], EXP_DATA["BBBT"]["ll"]).pvalue)
print("CRCS vs LCL", wilcoxon(EXP_DATA["CR"]["ll"], EXP_DATA["LCLBT"]["ll"]).pvalue)
print("LC-CRCS vs BT", wilcoxon(EXP_DATA["LC-CRCS"]["ll"], EXP_DATA["BT"]["ll"]).pvalue)
print("LC-CRCS vs BBBT", wilcoxon(EXP_DATA["LC-CRCS"]["ll"], EXP_DATA["BBBT"]["ll"]).pvalue)
print("LC-CRCS vs LCL", wilcoxon(EXP_DATA["LC-CRCS"]["ll"], EXP_DATA["LCLBT"]["ll"]).pvalue)
print("LC-CRCS vs CRCS", wilcoxon(EXP_DATA["LC-CRCS"]["ll"], EXP_DATA["CR"]["ll"]).pvalue)