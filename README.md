# Preference Learning of Latent Decision Utilities with a Human-like Model of Preferential Choice

This repository contains the experiments for the paper
> Sebastiaan De Peuter, Shibei Zhu, Yujia Guo, Andrew Howes & Samuel Kaski (2024) **Preference Learning of Latent Decision Utilities with a Human-like Model of Preferential Choice** In The Thirty-eighth Annual Conference on Neural Information Processing Systems.  
[[NeurIPS]](https://neurips.cc/virtual/2024/poster/93675) | [[openreview]](https://openreview.net/pdf?id=nfq3GKfb4h)

Please refer to the appendices for data sources and references.

## Requirements

The experiments are implemented in python. The implementations uses the following packages:
- Scikit-learn
- numpy
- scipy
- tqdm
- PyTorch
- Pandas
- wandb

## Choice model training

The paper proposes two surrogates, an expected utility surrogate and a computationally rational choice surrogate (CRCS). Both are trained using the supplied main.py file. Training works the same for all choice tasks and use cases in the paper. To train these models run

```bash
python main.py --use_case $USECASE --train
```

Where `$USECASE` is one of:
- `RiskyChoice`: The risky choice task as defined in [Howes et al. 2016].
- `WaterManagement`: The water drainage network design use case presented in [Tanabe & Ishibuchi 2020].
- `CarCrash`: The car crash structure design use case presented in [Liao et al. 2008].
- `RonayneBrownHotels`: The "Hotels" choice task introduced in [Ronayne & Brown, 2017].
- `Car-Alt`: The "Car-Alt" choice task introduced in [Brownstone et al. 1996].
- `District-Smart`: The "District-Smart" choice task introduced in [Kaufman et al. 2021].
- `Dumbalska`: The "Dumbalska" choice task introduced in [Dumbalska et al. 2020].

The above command will train the expected utility surrogate and CRCS in sequence. When CRCS training starts, the following message will be printed to the terminal:

> Starting new training run with PC model at `$CR_MODEL_DIR`

where `CR_MODEL_DIR` will be some path within the current folder, for example `"car-alt/checkpoints/1hd6aowg_3C/j1r8v5yl/"` for the Car-Alt task. This is the location where the trained CRCS model will be saved, and is specific for each use case or choice task and each training run. Please make a note of this as you will need it later when running the experiments.

## Running Experiments

### Validation on risky choice

To run the validation experiments on the risky choice task, train a choice model on `RiskyChoice` and then run the following command with the resulting `CR_MODEL_DIR`:

```bash
python main.py --use_case RiskyChoice --CR_model_dir $CR_MODEL_DIR --validate
```

This command will print the reversal rate minus the inverse reversal rate for a range of values for \sigma^2_{calc} (the utilty calculation noise). These can be compared directly to the results obtained in [Howes et al. 2016].

### Evaluations on static choice data

To run the evaluations on static choice data sets, train CRCS as instructed above and run

```bash
python main.py --use_case $CHOICE_TASK --CR_model_dir $CR_MODEL_DIR --validate
```

For the Dumbalska choice task, where inference and prediction was run per individual study participant, you need to specify an additional `$ID` as follows:
```bash
python main.py --use_case Dumbalska --CR_model_dir $CR_MODEL_DIR --validate --ID $ID
```
where `$ID` is an integer between 0 and 232 (inclusive). This ID identifies the participant in the original study. Please see [Dumbalska et al. 2020] for details. Unlike for the other choice tasks, the results will not be printed to the terminal, but will instead be written to the `dumbalska/experiment_data` folder under the name `CV_$ID.json`.

Finally, to validate how consistent the rankings implied by the inferred utilty function for each choice model are with separately collected rankings on the District-Smart task, run:
```bash
python main.py --use_case District-Smart --CR_model_dir $CR_MODEL_DIR --validate_rankings
```

### Active elicitation experiments

To evaluate the choice models in an active learning setting with Dumbalska use the following command:
```bash
python main.py --use_case Dumbalska --CR_model_dir $CR_MODEL_DIR --ID $ID --elicit_with $CHOICE_MODEL
```
Here, `$ID` specifies a participant ID as before, and `$CHOICE_MODEL` specifies the choice model you want to evaluate. Results will be written to `dumbalska/experiment_data`. The following values can be used for `$CHOICE_MODEL`:
- `CRCS`: use our CRCS model. The results will be written to a file named `SUS_CRCS_$ID.json`.
- `LC-CRCS`: use our LC-CRCS model. The results will be written to a file named `SUS_LC-CRCS_$ID.json`.
- `BT`: use a standard Bradley-Terry model. The results will be written to a file named `SUS_BT_$ID.json`.
- `BBBT`: use the model introduced in [Bower & Balzano 2020]. The results will be written to a file named `SUS_BBBT_$ID.json`.
- `LCLBT`use the LCL model introduced in [Tomlinson & Benson 2021]. The results will be written to a file named `SUS_LCL_$ID.json`.

Note that the `--CR_model_dir` option can be omitted when not using use `CRCS` or `LC-CRCS`.

### Simulated use cases
To run CRCS-based preference learning on the `WaterManagement` and `CarCrash` design use cases, first start by seting up the use case with the following command:

```bash
python main.py --use_case $USE_CASE --setup
python main.py --use_case $USE_CASE --find_pareto_front
```

where `$USE_CASE` is the chosen use case. This will precompute the pareto fron for the use case, and create 500 random experiment specifications which vary in terms of the simulated user's utility parameters and choice model paraemters. Next, run the following command to run the experiment with a chosen `$ID` between 0 and 499 (inclusive):

```bash
python main.py --use_case $USE_CASE --CR_model_dir $CR_MODEL_DIR --ID $ID --elicit_with CRCS
```

The results can be found in a file named `CRCS_$ID.json` in one the following folders, depending on the selected use case:
- `car_crash/experiment_data`
- `water_management/experiment_data`

The implementation of the retrosynthesis planning experiment is included, but depends on non-public data. Please contact us directly for details regarding this experiment.

# Analysis
For some experiments some post-processing of the results is required to obtain the graphs and tables presented in the paper. Example code to process the results of various experiments is included in this repository and described below:
* `analyze_dumbalska_CV.py` contains code to collect and process the results from the Dumbalska task when treating it as a static data set.
* `analyze_dumbalska_CV.py` contains code to collect and process the results from running actice preference learning on the Dumbalska task.
* `analyze_use_case.py` contains code to analyze and process results for the use cases.
