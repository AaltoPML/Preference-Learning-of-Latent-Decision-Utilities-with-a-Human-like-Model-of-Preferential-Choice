import sys
import os
import argparse
import numpy as np
import torch
import json
import tqdm
from functools import partial

from scipy.stats import wilcoxon, ttest_ind
from sklearn.model_selection import StratifiedKFold

from models.models import AmortizedConditionalChoicePredictionModel
from user_models.bradley_terry import *
from user_models.LCLCR_model import *

from training.train_choice_model import train, train_EU_model, train_PC_model, load_PC_model

from tasks.ronayne_brown_hotels import RonayneBrownHotelsTask
from tasks.risky_choice_task import StandardizedRiskyChoiceTask, Howes2016RiskyChoiceTask
from tasks.car_crash_MOO import ParameterizedCarCrashMOOTask
from tasks.water_management import WaterManagementMOOTask
from tasks.retrosynthesis_planning import RetrosynthesisPlanningTask
from tasks.car_alt import CarAltTask
from tasks.district_smart import DistrictSmartLargeTask
from tasks.dumbalska_property_task import DumbalskaPropertyTask, DumbalskaUserStudySimulator

from elicitation.BOED import create_DATA, create_SUS_DATA, create_joint_partcile_sets, run_BOED, run_simulated_elicitation, calculate_regret_random_recommendation

from utils.TruncatedNormal import PositiveNormal
from utils.eval_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps')

N_CHOICE = 3

################################################################################
#                                 MAIN LOOPS                                   #
################################################################################

#
# Dumbalska property choice
#
DP_PATH = 'dumbalska'
DP_CHECKPOINT_FOLDER = 'checkpoints'
DP_EXP_FOLDER = 'experiment_data'
DP_N_PARTICLES_PER_UPARAM_DIM = 13
DP_N_PARTICLES_PER_HPARAM_DIM = 13
DP_N_ELICITATION_STEPS = 25

def get_task_dumbalska(device = device):
    sigma_e_prior = torch.distributions.half_normal.HalfNormal(1.0)
    tau_pv_prior = PositiveNormal(torch.tensor([200.0, 180.0]), torch.tensor([200.0, 180.0]))
    p_error_prior = torch.distributions.Beta(1.0, 3.0)
    return DumbalskaPropertyTask(sigma_e_prior, tau_pv_prior, p_error_prior, device = device, normalize_weights=True, data_location=DP_PATH)

def get_model_kwargs_dumbalska():
    EU_kwargs = {"n_epochs": 40_000,
                 "batch_size": 4096,
                 "lr_start": 1e-3,
                 "lr_end": 1e-4,
                 "model_kwargs": {"main_sizes": [512, 512, 256], 
                                  "control_latent_dim": 128, 
                                  "controller_sizes": [256, 128],
                                  "dropout_rate": 0.01
                                  }}
    PC_kwargs = {"n_epochs": 25_000, 
                 "batch_size": 8192,
                 "lr_start": 1e-3,
                 "lr_end": 1e-3,
                 "model_kwargs": {"main_sizes": [512,256,128],
                                  "control_latent_dim": 128,
                                  "controller_sizes": [128, 128],
                                  "dropout_rate": 0.01,
                                  "u_dropout_rate": 0.01
                                  }}
    return EU_kwargs, PC_kwargs

def train_dumbalska():
    ptask = get_task_dumbalska()
    EU_kwargs, PC_kwargs = get_model_kwargs_dumbalska()
    train(ptask, f'{DP_PATH}/{DP_CHECKPOINT_FOLDER}/', device = device, EU_kwargs=EU_kwargs, PC_kwargs=PC_kwargs)

def train_EU_dumbalska():
    ptask = get_task_dumbalska()
    kwargs, _ = get_model_kwargs_dumbalska()
    train_EU_model(ptask, f'{DP_PATH}/{DP_CHECKPOINT_FOLDER}/', device=device, EU_kwargs=kwargs)

def train_PC_dumbalska(EUID):
    ptask = get_task_dumbalska()
    _, kwargs = get_model_kwargs_dumbalska()
    train_PC_model(ptask, f'{DP_PATH}/{DP_CHECKPOINT_FOLDER}/', EUID, device=device, PC_kwargs=kwargs)

def SUS_CR_dumbalska(exp_ID, CR_model_dir):
    ptask = get_task_dumbalska()
    sim = DumbalskaUserStudySimulator(ptask, exp_ID)
    CR_model = load_PC_model(ptask, CR_model_dir, device=device)

    original_uparam_particle_set = ptask.generate_parameter_batch(int(DP_N_PARTICLES_PER_UPARAM_DIM**(ptask.parameter_dim-0.4)))
    original_hparam_particle_set = ptask.generate_hyperparameter_batch(int(DP_N_PARTICLES_PER_HPARAM_DIM**ptask.hyperparameter_dim))
    uparam_particle_set, hparam_particle_set = create_joint_partcile_sets(original_uparam_particle_set, original_hparam_particle_set)

    DATA = create_SUS_DATA(original_uparam_particle_set.shape[0], original_hparam_particle_set.shape[0])
    DATA = run_simulated_elicitation(ptask, sim, CR_model, uparam_particle_set, hparam_particle_set, DP_N_ELICITATION_STEPS, DATA, particle_batch_size = 4096)

    f_out = f"{DP_PATH}/{DP_EXP_FOLDER}/SUS_CRCS_{str(exp_ID)}.json"
    with open(f_out, "w+") as f:
        json.dump(DATA, f)

def SUS_LCLCR_dumbalska(exp_ID, CR_model_dir):
    ptask = get_task_dumbalska()
    sim = DumbalskaUserStudySimulator(ptask, exp_ID)
    CR_model = load_LCLCR_model(ptask, CR_model_dir, device=device)

    original_uparam_particle_set = ptask.generate_parameter_batch(25)
    n_hyperparam = int(4.6**8)
    original_hparam_particle_set = torch.concat([ptask.generate_hyperparameter_batch(n_hyperparam), torch.randn((n_hyperparam, ptask.n_attributes**2), device=device)], dim=1)
    uparam_particle_set, hparam_particle_set = create_joint_partcile_sets(original_uparam_particle_set, original_hparam_particle_set)

    DATA = create_SUS_DATA(original_uparam_particle_set.shape[0], original_hparam_particle_set.shape[0])
    DATA = run_simulated_elicitation(ptask, sim, CR_model, uparam_particle_set, hparam_particle_set, DP_N_ELICITATION_STEPS, DATA, particle_batch_size = 4096)

    f_out = f"{DP_PATH}/{DP_EXP_FOLDER}/SUS_LC-CRCS_{str(exp_ID)}.json"
    with open(f_out, "w+") as f:
        json.dump(DATA, f)

def SUS_CR_no_BOED_dumbalska(exp_ID, CR_model_dir):
    ptask = get_task_dumbalska()
    sim = DumbalskaUserStudySimulator(ptask, exp_ID)
    CR_model = load_PC_model(ptask, CR_model_dir, device=device)

    original_uparam_particle_set = ptask.generate_parameter_batch(int(DP_N_PARTICLES_PER_UPARAM_DIM**(ptask.parameter_dim-0.4)))
    original_hparam_particle_set = ptask.generate_hyperparameter_batch(int(DP_N_PARTICLES_PER_HPARAM_DIM**ptask.hyperparameter_dim))
    uparam_particle_set, hparam_particle_set = create_joint_partcile_sets(original_uparam_particle_set, original_hparam_particle_set)

    DATA = create_SUS_DATA(original_uparam_particle_set.shape[0], original_hparam_particle_set.shape[0])
    DATA = run_simulated_elicitation(ptask, sim, CR_model, uparam_particle_set, hparam_particle_set, DP_N_ELICITATION_STEPS, DATA, particle_batch_size = 4096, ED_strategy="random")

    f_out = f"{DP_PATH}/{DP_EXP_FOLDER}/SUS_CRCS_no_BOED_{str(exp_ID)}.json"
    with open(f_out, "w+") as f:
        json.dump(DATA, f)

def SUS_LCLBT_dumbalska(exp_ID):
    ptask = get_task_dumbalska()
    sim = DumbalskaUserStudySimulator(ptask, exp_ID)
    CR_model = LCLBradleyTerryModel(ptask)

    original_uparam_particle_set, original_hparam_particle_set = torch.randn((int(DP_N_PARTICLES_PER_UPARAM_DIM**ptask.parameter_dim),ptask.parameter_dim)) * 0.5, torch.randn((int(DP_N_PARTICLES_PER_HPARAM_DIM**ptask.hyperparameter_dim),ptask.parameter_dim**2)) * 0.1
    uparam_particle_set, hparam_particle_set = create_joint_partcile_sets(original_uparam_particle_set, original_hparam_particle_set)

    DATA = create_SUS_DATA(original_uparam_particle_set.shape[0], original_hparam_particle_set.shape[0])
    DATA = run_simulated_elicitation(ptask, sim, CR_model, uparam_particle_set, hparam_particle_set, DP_N_ELICITATION_STEPS, DATA, particle_batch_size = 4096)

    f_out = f"{DP_PATH}/{DP_EXP_FOLDER}/SUS_LCL_{str(exp_ID)}.json"
    with open(f_out, "w+") as f:
        json.dump(DATA, f)

def SUS_BT_dumbalska(exp_ID):
    ptask = get_task_dumbalska()
    sim = DumbalskaUserStudySimulator(ptask, exp_ID)
    CR_model = CanonicalBradleyTerryModel(ptask)

    original_uparam_particle_set, original_hparam_particle_set = torch.randn((int(DP_N_PARTICLES_PER_UPARAM_DIM**ptask.parameter_dim), ptask.parameter_dim)) * 0.5, torch.zeros((1,0))
    uparam_particle_set, hparam_particle_set = create_joint_partcile_sets(original_uparam_particle_set, original_hparam_particle_set)

    DATA = create_SUS_DATA(original_uparam_particle_set.shape[0], original_hparam_particle_set.shape[0])
    DATA = run_simulated_elicitation(ptask, sim, CR_model, uparam_particle_set, hparam_particle_set, DP_N_ELICITATION_STEPS, DATA, particle_batch_size = 4096)

    f_out = f"{DP_PATH}/{DP_EXP_FOLDER}/SUS_BT_{str(exp_ID)}.json"
    with open(f_out, "w+") as f:
        json.dump(DATA, f)

def SUS_BBBT_dumbalska(exp_ID):
    ptask = get_task_dumbalska()
    sim = DumbalskaUserStudySimulator(ptask, exp_ID)
    CR_model = CanonicalBradleyTerryModel(ptask)

    original_uparam_particle_set, original_hparam_particle_set = torch.randn((int(DP_N_PARTICLES_PER_UPARAM_DIM**ptask.parameter_dim), ptask.parameter_dim)) * 0.5, torch.arange(0, ptask.n_attributes+1, 1, dtype=torch.int32)[:,None]
    uparam_particle_set, hparam_particle_set = create_joint_partcile_sets(original_uparam_particle_set, original_hparam_particle_set)

    DATA = create_SUS_DATA(original_uparam_particle_set.shape[0], original_hparam_particle_set.shape[0])
    DATA = run_simulated_elicitation(ptask, sim, CR_model, uparam_particle_set, hparam_particle_set, DP_N_ELICITATION_STEPS, DATA, particle_batch_size = 4096)

    f_out = f"{DP_PATH}/{DP_EXP_FOLDER}/SUS_BBBT_{str(exp_ID)}.json"
    with open(f_out, "w+") as f:
        json.dump(DATA, f)

def validate_dumbalska(exp_ID, CR_model_dir):
    ptask = get_task_dumbalska()
    sim = DumbalskaUserStudySimulator(ptask, exp_ID)
    rate_bin_IDs, cost_bin_IDs = sim.rate_bin_IDs, sim.cost_bin_IDs
    x, y = sim.recorded_queries, sim.recorded_choices

    BT_model = CanonicalBradleyTerryModel(ptask)
    BBBT_model = BowerBalzanoBradleyTerryModel(ptask)
    LCLBT_model = LCLBradleyTerryModel(ptask)
    CR_model = load_PC_model(ptask, CR_model_dir, device=device)
    CR_model_kwargs = {"transform_params": transform_params_nd, "transform_hparams": transform_hparams, "n_iter": 2000, "lr": 0.1, "aggregation": "mean"}
    LCLCR_model = load_LCLCR_model(ptask, CR_model_dir, device=device)
    transform_hparams_LCLCR = partial(transform_hparams_aux, n_aux = LCLCR_model.weight_scaler_dim_flat)
    inv_transform_hparams_LCLCR = partial(inv_transform_hparams_aux, n_aux = LCLCR_model.weight_scaler_dim_flat)
    LCLCR_model_kwargs = {"transform_params": transform_params_nd, "transform_hparams": transform_hparams_LCLCR, "n_iter": 2000, "lr": 0.1, "aggregation": "mean"}

    BT_fold_lls, BBBT_fold_lls, LCLBT_fold_lls, CR_fold_lls, LCLCR_fold_lls = [], [], [], [], []
    BT_acc, BBBT_acc, LCLBT_acc, CR_acc, LCLCR_acc = 0.0, 0.0, 0.0, 0.0, 0.0

    n_folds = 10
    idxs = torch.cartesian_prod(torch.arange(0, 10), torch.arange(0,10))
    perm = torch.randperm(100)
    in_train_set = torch.zeros((n_folds,10,10), dtype=torch.bool)
    for i in range(n_folds):
        for (j,k) in idxs[torch.cat([perm[:(100//n_folds)*i], perm[(100//n_folds)*(i+1):]])]:
            in_train_set[i,j,k] = True

    for fold_i in tqdm.tqdm(range(n_folds)):
        x_train = x[[in_train_set[fold_i,cost_bin_IDs[i],rate_bin_IDs[i]] for i in range(x.shape[0])]]
        y_train = y[[in_train_set[fold_i,cost_bin_IDs[i],rate_bin_IDs[i]] for i in range(x.shape[0])]]
        x_test = x[[torch.logical_not(in_train_set[fold_i,cost_bin_IDs[i],rate_bin_IDs[i]]) for i in range(x.shape[0])]]
        y_test = y[[torch.logical_not(in_train_set[fold_i,cost_bin_IDs[i],rate_bin_IDs[i]]) for i in range(x.shape[0])]]

        w_init_BT, h_init_BT = torch.randn(ptask.parameter_dim) * 1.0, None
        BT_fold_ll, BT_fold_acc, _, _ = infer_and_validate_SGD(BT_model, x_train, y_train, x_test, y_test, w_init_BT, h_init_BT, model_kwargs = {"lr": 0.05, "aggregation": "mean"})
        BT_fold_lls.append(BT_fold_ll.item())
        BT_acc += BT_fold_acc.item() * y_test.shape[0]

        w_init_BBBT, h_init_BBBT = torch.randn(ptask.parameter_dim) * 1.0, torch.tensor([2], dtype=torch.int32)
        BBBT_fold_ll, BBBT_fold_acc, _, _ = infer_and_validate_SGD(BBBT_model, x_train, y_train, x_test, y_test, w_init_BBBT, h_init_BBBT, model_kwargs = {"lr": 0.5, "aggregation": "mean"})
        BBBT_fold_lls.append(BBBT_fold_ll.item())
        BBBT_acc += BBBT_fold_acc.item() * y_test.shape[0]

        w_init_LCLBT, h_init_LCLBT = torch.randn(ptask.parameter_dim) * 0.5, torch.randn(ptask.parameter_dim**2) * 0.1
        LCLBT_fold_ll, LCLBT_fold_acc, _, _ = infer_and_validate_SGD(LCLBT_model, x_train, y_train, x_test, y_test, w_init_LCLBT, h_init_LCLBT, model_kwargs = {"lr": 0.05, "aggregation": "mean"})
        LCLBT_fold_lls.append(LCLBT_fold_ll.item())
        LCLBT_acc += LCLBT_fold_acc.item() * y_test.shape[0]

        init_generator_CR = lambda : CR_find_starting_point(ptask, CR_model, x_train, y_train, inv_transform_params_nd, inv_transform_hparams, N_test=10)
        CR_fold_ll, CR_fold_acc, _, _ = infer_and_validate_SGD_multistart(CR_model, x_train, y_train, x_test, y_test, init_generator_CR, 20, test_portion=0.2, model_kwargs = CR_model_kwargs)
        CR_fold_lls.append(CR_fold_ll.item())
        CR_acc += CR_fold_acc.item() * y_test.shape[0]

        init_generator_LCLCR = lambda : (inv_transform_params_nd(ptask.generate_parameter_batch(1).flatten()), inv_transform_hparams_LCLCR(torch.concat([ptask.generate_hyperparameter_batch(1), torch.randn(1, ptask.n_attributes**2)], dim=1).flatten()))
        LCLCR_fold_ll, LCLCR_fold_acc, _, _ = infer_and_validate_SGD_multistart(LCLCR_model, x_train, y_train, x_test, y_test, init_generator_LCLCR, 50, test_portion=0.2, model_kwargs = LCLCR_model_kwargs)
        LCLCR_fold_lls.append(LCLCR_fold_ll.item())
        LCLCR_acc += LCLCR_fold_acc.item() * y_test.shape[0]
    
    DATA = {"BT": {"ll": np.sum(BT_fold_lls), "expected_l": BT_acc / y.shape[0]}, 
            "BBBT": {"ll": np.sum(BBBT_fold_lls), "expected_l": BBBT_acc / y.shape[0]}, 
            "LCLBT": {"ll": np.sum(LCLBT_fold_lls), "expected_l": LCLBT_acc / y.shape[0]}, 
            "CRCS": {"ll": np.sum(CR_fold_lls), "expected_l": CR_acc / y.shape[0]}, 
            "LC-CRCS": {"ll": np.sum(LCLCR_fold_lls), "expected_l": LCLCR_acc / y.shape[0]}}

    f_out = f"{DP_PATH}/{DP_EXP_FOLDER}/CV_{str(exp_ID)}.json"
    with open(f_out, "w+") as f:
        json.dump(DATA, f)
#
# District-Smart
#

DS_N_ELICITATION_STEPS = 30
DS_EVAL_N_FOLDS = 15
DS_EVAL_RANK_N_FOLDS = 25
DS_PATH = 'district-smart'
DS_CHECKPOINT_FOLDER = "checkpoints"
DS_EXP_FOLDER = "experiment_data"

def get_task_district_smart():
    sigma_e_prior = torch.distributions.half_normal.HalfNormal(2.0)
    tau_pv_prior = PositiveNormal(torch.tensor([0.06, 0.08, 0.05, 0.08, 0.14, 0.1]), torch.tensor([0.12, 0.16, 0.10, 0.16, 0.28, 0.2]))
    p_error_prior = torch.distributions.Beta(1.0, 3.0)
    return DistrictSmartLargeTask(sigma_e_prior, tau_pv_prior, p_error_prior, device = device, normalize_weights=True, data_location=DS_PATH)

def get_model_kwargs_district_smart():
    EU_kwargs = {"n_epochs": 40_000,
                 "batch_size": 8192,
                 "lr_start": 1e-2,
                 "lr_end": 1e-4, 
                 "model_kwargs": {"main_sizes": [1024, 512, 128], 
                                  "control_latent_dim": 128, 
                                  "controller_sizes": [256, 256],
                                  "dropout_rate": 0.01
                                  }}
    PC_kwargs = {"n_epochs": 15_000, 
                 "batch_size": 4096,
                 "lr_start": 1e-3,
                 "lr_end": 1e-4,
                 "model_kwargs": {"main_sizes": [1024, 1024, 256],
                                  "control_latent_dim": 256,
                                  "controller_sizes": [512, 256],
                                  "dropout_rate": 0.0,
                                  "u_dropout_rate": 0.0
                                  }}
    return EU_kwargs, PC_kwargs

def train_district_smart():
    ptask = get_task_district_smart()
    EU_kwargs, PC_kwargs = get_model_kwargs_district_smart()
    train(ptask, f'{DS_PATH}/{DS_CHECKPOINT_FOLDER}/', device = device, EU_kwargs=EU_kwargs, PC_kwargs=PC_kwargs)

def train_EU_district_smart():
    ptask = get_task_district_smart()
    kwargs, _ = get_model_kwargs_district_smart()
    train_EU_model(ptask, f'{DS_PATH}/{DS_CHECKPOINT_FOLDER}/', device=device, EU_kwargs=kwargs)

def train_PC_district_smart(EUID):
    ptask = get_task_district_smart()
    _, kwargs = get_model_kwargs_district_smart()
    train_PC_model(ptask, f'{DS_PATH}/{DS_CHECKPOINT_FOLDER}/', EUID, device=device, PC_kwargs=kwargs)

def validate_district_smart(CR_model_dir):
    ptask = get_task_district_smart()

    BT_model = CanonicalBradleyTerryModel(ptask)
    BBBT_model = BowerBalzanoBradleyTerryModel(ptask)
    LCLBT_model = LCLBradleyTerryModel(ptask)
    CR_model = load_PC_model(ptask, CR_model_dir, device=device)
    CR_model_kwargs = {"transform_params": transform_params_nd, "transform_hparams": transform_hparams, "n_iter": 1000, "lr": 0.1, "aggregation": "mean"}
    LCLCR_model = load_LCLCR_model(ptask, CR_model_dir, device=device)
    transform_hparams_LCLCR = partial(transform_hparams_aux, n_aux = LCLCR_model.weight_scaler_dim_flat)
    inv_transform_hparams_LCLCR = partial(inv_transform_hparams_aux, n_aux = LCLCR_model.weight_scaler_dim_flat)
    LCLCR_model_kwargs = {"transform_params": transform_params_nd, "transform_hparams": transform_hparams_LCLCR, "n_iter": 1000, "lr": 0.1, "aggregation": "mean"}

    BT_fold_lls, BBBT_fold_lls, LCLBT_fold_lls, CR_fold_lls, LCLCR_fold_lls = [], [], [], [], []
    BT_acc, BBBT_acc, LCLBT_acc, CR_acc, LCLCR_acc = 0.0, 0.0, 0.0, 0.0, 0.0
    BT_fold_accs, BBBT_fold_accs, LCLBT_fold_accs, CR_fold_accs, LCLCR_fold_accs = [], [], [], [], []

    cv = StratifiedKFold(n_splits=DS_EVAL_N_FOLDS, shuffle=True)
    for (train_index, test_index) in tqdm.tqdm(cv.split(ptask.x.numpy(), ptask.y.numpy())):
        x_train, y_train = ptask.x[train_index,:,:], ptask.y[train_index]
        x_test, y_test = ptask.x[test_index,:,:], ptask.y[test_index]

        w_init_BT, h_init_BT = torch.randn(ptask.parameter_dim) * 1.0, None
        BT_fold_ll, BT_fold_acc, _, _ = infer_and_validate_SGD(BT_model, x_train, y_train, x_test, y_test, w_init_BT, h_init_BT, model_kwargs = {"lr": 0.005, "aggregation": "sum"})
        BT_fold_lls.append(BT_fold_ll)
        BT_acc += BT_fold_acc * y_test.shape[0]
        BT_fold_accs.append(BT_fold_acc)

        w_init_BBBT, h_init_BBBT = torch.randn(ptask.parameter_dim) * 1.0, torch.tensor([2], dtype=torch.int32)
        BBBT_fold_ll, BBBT_fold_acc, _, _ = infer_and_validate_SGD(BBBT_model, x_train, y_train, x_test, y_test, w_init_BBBT, h_init_BBBT, model_kwargs = {"lr": 0.01, "aggregation": "sum"})
        BBBT_fold_lls.append(BBBT_fold_ll)
        BBBT_acc += BBBT_fold_acc * y_test.shape[0]
        BBBT_fold_accs.append(BBBT_fold_acc)

        w_init_LCLBT, h_init_LCLBT = torch.randn(ptask.parameter_dim) * 1.0, torch.randn(ptask.parameter_dim**2) * 0.1
        LCLBT_fold_ll, LCLBT_fold_acc, _, _ = infer_and_validate_SGD(LCLBT_model, x_train, y_train, x_test, y_test, w_init_LCLBT, h_init_LCLBT, model_kwargs = {"lr": 0.005, "aggregation": "sum"})
        LCLBT_fold_lls.append(LCLBT_fold_ll)
        LCLBT_acc += LCLBT_fold_acc * y_test.shape[0]
        LCLBT_fold_accs.append(LCLBT_fold_acc)

        init_generator_CR = lambda : CR_find_starting_point(ptask, CR_model, x_train, y_train, inv_transform_params_nd, inv_transform_hparams, N_test=10)
        CR_fold_ll, CR_fold_acc, _, _ = infer_and_validate_SGD_multistart(CR_model, x_train, y_train, x_test, y_test, init_generator_CR, 15, model_kwargs = CR_model_kwargs)
        CR_fold_lls.append(CR_fold_ll)
        CR_acc += CR_fold_acc * y_test.shape[0]
        CR_fold_accs.append(CR_fold_acc)

        init_generator_LCLCR = lambda : (inv_transform_params_nd(ptask.generate_parameter_batch(1).flatten()), inv_transform_hparams_LCLCR(torch.concat([ptask.generate_hyperparameter_batch(1), torch.randn(1, ptask.n_attributes**2)], dim=1).flatten()))
        LCLCR_fold_ll, LCLCR_fold_acc, _, _ = infer_and_validate_SGD_multistart(LCLCR_model, x_train, y_train, x_test, y_test, init_generator_LCLCR, 15, model_kwargs = LCLCR_model_kwargs)
        LCLCR_fold_lls.append(LCLCR_fold_ll)
        LCLCR_acc += LCLCR_fold_acc * y_test.shape[0]
        LCLCR_fold_accs.append(LCLCR_fold_acc)

        print(y_test.shape, BT_fold_ll, BBBT_fold_ll, LCLBT_fold_ll, CR_fold_ll, LCLCR_fold_ll)
        print(y_test.shape, BT_fold_acc, BBBT_fold_acc, LCLBT_fold_acc, CR_fold_acc, LCLCR_fold_acc)

    print("Test-set CVd likelihoods:")
    print(f"BT: {np.sum(BT_fold_lls)} \t BBBT: {np.sum(BBBT_fold_lls)} \t LCL: {np.sum(LCLBT_fold_lls)} \t CRCS {np.sum(CR_fold_lls)} \t LC-CRCS {np.sum(LCLCR_fold_lls)}")
    print("Paired t-test p-values:")
    print("BT vs BBBT", wilcoxon(BT_fold_lls, BBBT_fold_lls).pvalue)
    print("LCL vs BT", wilcoxon(LCLBT_fold_lls, BT_fold_lls).pvalue)
    print("LCL vs BBBT", wilcoxon(LCLBT_fold_lls, BBBT_fold_lls).pvalue)
    print("CRCS vs BT", wilcoxon(CR_fold_lls, BT_fold_lls).pvalue)
    print("CRCS vs BBBT", wilcoxon(CR_fold_lls, BBBT_fold_lls).pvalue)
    print("CRCS vs LCL", wilcoxon(CR_fold_lls, LCLBT_fold_lls).pvalue)
    print("LC-CRCS vs BT", wilcoxon(LCLCR_fold_lls, BT_fold_lls).pvalue)
    print("LC-CRCS vs BBBT", wilcoxon(LCLCR_fold_lls, BBBT_fold_lls).pvalue)
    print("LC-CRCS vs LCL", wilcoxon(LCLCR_fold_lls, LCLBT_fold_lls).pvalue)
    print("LC-CRCS vs CRCS", wilcoxon(LCLCR_fold_lls, CR_fold_lls).pvalue)

    print("Test-set CVd accuracies:")
    print(f"BT: {BT_acc / ptask.y.shape[0]} \t BBBT: {BBBT_acc / ptask.y.shape[0]} \t LCL: {LCLBT_acc / ptask.y.shape[0]} \t CRCS {CR_acc / ptask.y.shape[0]} \t LC-CRCS {LCLCR_acc / ptask.y.shape[0]}")
    print("Paired t-test p-values:")
    print("BT vs BBBT", wilcoxon(BT_fold_accs, BBBT_fold_accs).pvalue)
    print("LCL vs BT", wilcoxon(LCLBT_fold_accs, BT_fold_accs).pvalue)
    print("LCL vs BBBT", wilcoxon(LCLBT_fold_accs, BBBT_fold_accs).pvalue)
    print("CRCS vs BT", wilcoxon(CR_fold_accs, BT_fold_accs).pvalue)
    print("CRCS vs BBBT", wilcoxon(CR_fold_accs, BBBT_fold_accs).pvalue)
    print("CRCS vs LCL", wilcoxon(CR_fold_accs, LCLBT_fold_accs).pvalue)
    print("LC-CRCS vs BT", wilcoxon(LCLCR_fold_accs, BT_fold_accs).pvalue)
    print("LC-CRCS vs BBBT", wilcoxon(LCLCR_fold_accs, BBBT_fold_accs).pvalue)
    print("LC-CRCS vs LCL", wilcoxon(LCLCR_fold_accs, LCLBT_fold_accs).pvalue)
    print("LC-CRCS vs CRCS", wilcoxon(LCLCR_fold_accs, CR_fold_accs).pvalue)

def validate_rankings_district_smart(CR_model_dir):
    ptask = get_task_district_smart()

    BT_model = CanonicalBradleyTerryModel(ptask)
    BBBT_model = BowerBalzanoBradleyTerryModel(ptask)
    LCLBT_model = LCLBradleyTerryModel(ptask)

    sigma_e_prior = torch.distributions.Normal(25.0, 0.1)
    p_error_prior = torch.distributions.Beta(1.0, 1000.0)
    stdn = torch.distributions.Normal(0.0, 1.0)
    tau_pv_means = torch.tensor([0.06, 0.08, 0.05, 0.08, 0.14, 0.1])
    tau_pv_stds = torch.tensor([0.12, 0.16, 0.10, 0.16, 0.28, 0.2])
    tau_pv_normalizers_log = - tau_pv_stds.log() - (1 - stdn.cdf(-tau_pv_means / tau_pv_stds)).log()
    CR_logprior = lambda h: - sigma_e_prior.log_prob(h[0]) - p_error_prior.log_prob(h[1]) - (stdn.log_prob((h[2:] - tau_pv_means)/tau_pv_stds).sum() - tau_pv_normalizers_log.sum())
    CR_model = load_PC_model(ptask, CR_model_dir, device=device)
    LCLCR_model = load_LCLCR_model(ptask, CR_model_dir, device=device)

    transform_hparams_LCLCR = partial(transform_hparams_aux, n_aux = LCLCR_model.weight_scaler_dim_flat)
    inv_transform_hparams_LCLCR = partial(inv_transform_hparams_aux, n_aux = LCLCR_model.weight_scaler_dim_flat)
    LCLCR_logprior = lambda h: - sigma_e_prior.log_prob(h[0]) - p_error_prior.log_prob(h[1])  - (stdn.log_prob((h[2:-LCLCR_model.weight_scaler_dim_flat] - tau_pv_means)/tau_pv_stds).sum() - tau_pv_normalizers_log.sum())

    BT_rank_consistencies = []
    for _ in range(DS_EVAL_RANK_N_FOLDS):
        w_init, h_init = torch.randn(ptask.parameter_dim) * 1.0, None
        w_inferred, _ = BT_model.infer(ptask.x, ptask.y, w_init, h_init, lr = 0.005, aggregation="sum")
        BT_rank_consistencies.append(ptask.calculate_ground_truth_rank_consistency(w_inferred))

    BBBT_rank_consistencies = []
    for _ in range(DS_EVAL_RANK_N_FOLDS):
        w_init, h_init = torch.randn(ptask.parameter_dim) * 1.0, torch.tensor([2], dtype=torch.int32)
        w_inferred, _ = BBBT_model.infer(ptask.x, ptask.y, w_init, h_init, lr = 0.01, aggregation="sum")
        BBBT_rank_consistencies.append(ptask.calculate_ground_truth_rank_consistency(w_inferred))

    LCLBT_rank_consistencies = []
    for _ in range(DS_EVAL_RANK_N_FOLDS):
        w_init, h_init = torch.randn(ptask.parameter_dim) * 1.0, torch.randn(ptask.parameter_dim**2) * 0.1
        w_inferred, _ = LCLBT_model.infer(ptask.x, ptask.y, w_init, h_init, lr = 0.005, aggregation="sum", early_stopping=False, n_iter=2500, alpha = 75.0)
        LCLBT_rank_consistencies.append(ptask.calculate_ground_truth_rank_consistency(w_inferred))

    CR_rank_consistencies = []
    for _ in range(DS_EVAL_RANK_N_FOLDS):
        w_init, h_init = inv_transform_params_nd(ptask.generate_parameter_batch(1).flatten()), inv_transform_hparams(ptask.generate_hyperparameter_batch(1).flatten())
        w_inferred, _ = CR_model.infer(ptask.x, ptask.y, w_init, h_init, lr = 0.1, n_iter=1000, transform_params=transform_params_nd, transform_hparams=transform_hparams, hparam_regularizer = CR_logprior)
        CR_rank_consistencies.append(ptask.calculate_ground_truth_rank_consistency(w_inferred))

    LCLCR_rank_consistencies = []
    for _ in range(DS_EVAL_RANK_N_FOLDS):
        w_init, h_init = inv_transform_params_nd(ptask.generate_parameter_batch(1).flatten()), inv_transform_hparams_LCLCR(torch.concat([ptask.generate_hyperparameter_batch(1), torch.randn(1, ptask.n_attributes**2)], dim=1).flatten())
        w_inferred, _ = LCLCR_model.infer(ptask.x, ptask.y, w_init, h_init, lr = 0.1, n_iter=1000, transform_params=transform_params_nd, transform_hparams=transform_hparams_LCLCR, verbose = True, hparam_regularizer=LCLCR_logprior)
        LCLCR_rank_consistencies.append(ptask.calculate_ground_truth_rank_consistency(w_inferred))

    BT_mean, BT_stderr = np.mean(BT_rank_consistencies), np.std(BT_rank_consistencies) / np.sqrt(len(BT_rank_consistencies))
    BBBT_mean, BBBT_stderr = np.mean(BBBT_rank_consistencies), np.std(BBBT_rank_consistencies) / np.sqrt(len(BBBT_rank_consistencies))
    LCLBT_mean, LCLBT_stderr = np.mean(LCLBT_rank_consistencies), np.std(LCLBT_rank_consistencies) / np.sqrt(len(LCLBT_rank_consistencies))
    CR_mean, CR_stderr = np.mean(CR_rank_consistencies), np.std(CR_rank_consistencies) / np.sqrt(len(CR_rank_consistencies))
    LCLCR_mean, LCLCR_stderr = np.mean(LCLCR_rank_consistencies), np.std(LCLCR_rank_consistencies) / np.sqrt(len(LCLCR_rank_consistencies))
    print(f"Mean Kendall's Taus (average of 6 folds) over {DS_EVAL_RANK_N_FOLDS} inference runs:")
    print(f"BT: {BT_mean} +-  {BT_stderr} \t BBBT: {BBBT_mean} +-  {BBBT_stderr} \t LCL: {LCLBT_mean} +-  {LCLBT_stderr} \t CRCS {CR_mean} +-  {CR_stderr} \t LC-CRCS {LCLCR_mean} +-  {LCLCR_stderr}")
    print("Paired t-test p-values:")
    print("BT vs BBBT", wilcoxon(BT_rank_consistencies, BBBT_rank_consistencies).pvalue)
    print("LCL vs BT", wilcoxon(LCLBT_rank_consistencies, BT_rank_consistencies).pvalue)
    print("LCL vs BBBT", wilcoxon(LCLBT_rank_consistencies, BBBT_rank_consistencies).pvalue)
    print("CRCS vs BT", wilcoxon(CR_rank_consistencies, BT_rank_consistencies).pvalue)
    print("CRCS vs BBBT", wilcoxon(CR_rank_consistencies, BBBT_rank_consistencies).pvalue)
    print("CRCS vs LCL", wilcoxon(CR_rank_consistencies, LCLBT_rank_consistencies).pvalue)
    print("LC-CRCS vs BT", wilcoxon(LCLCR_rank_consistencies, BT_rank_consistencies).pvalue)
    print("LC-CRCS vs BBBT", wilcoxon(LCLCR_rank_consistencies, BBBT_rank_consistencies).pvalue)
    print("LC-CRCS vs LCL", wilcoxon(LCLCR_rank_consistencies, LCLBT_rank_consistencies).pvalue)
    print("LC-CRCS vs CRCS", wilcoxon(LCLCR_rank_consistencies, CR_rank_consistencies).pvalue)


#
# Car-Alt
#

CA_EVAL_N_FOLDS = 20
CA_PATH = 'car-alt'
CA_CHECKPOINT_FOLDER = "checkpoints"

def get_task_car_alt():
    sigma_e_prior = torch.distributions.half_normal.HalfNormal(2.0)
    tau_pv_prior = PositiveNormal(torch.tensor([0.7, 45, 1.0, 8.5, .15, 0.07, 1.8, 0.2]), torch.tensor([1.4, 90, 2.0, 17.0, 0.3, 0.15, 3.0, 0.4]))
    p_error_prior = torch.distributions.Beta(1.0, 3.0)
    return CarAltTask(sigma_e_prior, tau_pv_prior, p_error_prior, device = device, normalize_weights=True, data_location=CA_PATH)

def get_model_kwargs_car_alt():
    EU_kwargs= {"n_epochs": 30_000,
                "batch_size": 8192,
                "lr_start": 1e-3,
                "lr_end": 1e-3,
                "model_kwargs": {"main_sizes": [512, 256, 128],
                                 "control_latent_dim": 256,
                                 "controller_sizes": [256, 256],
                                 "dropout_rate": 0.01
                                 }}
    PC_kwargs={"n_epochs": 100_000,
               "batch_size": 4096,
               "lr_start": 1e-3,
               "lr_end": 1e-3,
               "model_kwargs": {"main_sizes": [1024, 1024, 256],
                                "control_latent_dim": 256,
                                "controller_sizes": [256, 256],
                                "dropout_rate": 0.0,
                                "u_dropout_rate": 0.0
                                }}
    return EU_kwargs, PC_kwargs

def train_car_alt():
    ptask = get_task_car_alt()
    EU_kwargs, PC_kwargs = get_model_kwargs_car_alt()
    train(ptask, f'{CA_PATH}/{CA_CHECKPOINT_FOLDER}/', device = device, EU_kwargs=EU_kwargs, PC_kwargs=PC_kwargs)

def train_EU_car_alt():
    ptask = get_task_car_alt()
    kwargs, _ = get_model_kwargs_car_alt()
    train_EU_model(ptask, f'{CA_PATH}/{CA_CHECKPOINT_FOLDER}/', device=device, EU_kwargs=kwargs)

def train_PC_car_alt(EUID):
    ptask = get_task_car_alt()
    _, kwargs = get_model_kwargs_car_alt()
    train_PC_model(ptask, f'{CA_PATH}/{CA_CHECKPOINT_FOLDER}/', EUID, device=device, PC_kwargs=kwargs)

def validate_car_alt(CR_model_dir):
    ptask = get_task_car_alt()

    BT_model = CanonicalBradleyTerryModel(ptask)
    BBBT_model = BowerBalzanoBradleyTerryModel(ptask)
    LCLBT_model = LCLBradleyTerryModel(ptask)
    CR_model = load_PC_model(ptask, CR_model_dir, device=device)
    CR_model_kwargs = {"transform_params": transform_params_nd, "transform_hparams": transform_hparams, "n_iter": 2000, "lr": 0.1, "aggregation": "mean"}
    LCLCR_model = load_LCLCR_model(ptask, CR_model_dir, device=device)
    transform_hparams_LCLCR = partial(transform_hparams_aux, n_aux = LCLCR_model.weight_scaler_dim_flat)
    inv_transform_hparams_LCLCR = partial(inv_transform_hparams_aux, n_aux = LCLCR_model.weight_scaler_dim_flat)
    LCLCR_model_kwargs = {"transform_params": transform_params_nd, "transform_hparams": transform_hparams_LCLCR, "n_iter": 2000, "lr": 0.1, "aggregation": "mean"}

    BT_fold_lls, BBBT_fold_lls, LCLBT_fold_lls, CR_fold_lls, LCLCR_fold_lls = [], [], [], [], []
    BT_acc, BBBT_acc, LCLBT_acc, CR_acc, LCLCR_acc = 0.0, 0.0, 0.0, 0.0, 0.0
    BT_fold_accs, BBBT_fold_accs, LCLBT_fold_accs, CR_fold_accs, LCLCR_fold_accs = [], [], [], [], []

    cv = StratifiedKFold(n_splits=CA_EVAL_N_FOLDS, shuffle=True)
    for (train_index, test_index) in tqdm.tqdm(cv.split(ptask.x.numpy(), ptask.y.numpy())):
        x_train, y_train = ptask.x[train_index,:,:], ptask.y[train_index]
        x_test, y_test = ptask.x[test_index,:,:], ptask.y[test_index]

        w_init_BT, h_init_BT = torch.randn(ptask.parameter_dim) * 1.0, None
        BT_fold_ll, BT_fold_acc, _, _ = infer_and_validate_SGD(BT_model, x_train, y_train, x_test, y_test, w_init_BT, h_init_BT, model_kwargs = {"lr": 0.001, "aggregation": "sum"})
        BT_fold_lls.append(BT_fold_ll)
        BT_fold_accs.append(BT_fold_acc)
        BT_acc += BT_fold_acc * y_test.shape[0]

        w_init_BBBT, h_init_BBBT = torch.randn(ptask.parameter_dim) * 1.0, torch.tensor([2], dtype=torch.int32)
        BBBT_fold_ll, BBBT_fold_acc, _, _ = infer_and_validate_SGD(BBBT_model, x_train, y_train, x_test, y_test, w_init_BBBT, h_init_BBBT, model_kwargs = {"lr": 0.001, "aggregation": "sum"})
        BBBT_fold_lls.append(BBBT_fold_ll)
        BBBT_fold_accs.append(BBBT_fold_acc)
        BBBT_acc += BBBT_fold_acc * y_test.shape[0]

        w_init_LCLBT, h_init_LCLBT = torch.randn(ptask.parameter_dim) * 1.0, torch.randn(ptask.parameter_dim**2) * 0.1
        LCLBT_fold_ll, LCLBT_fold_acc, _, _ = infer_and_validate_SGD(LCLBT_model, x_train, y_train, x_test, y_test, w_init_LCLBT, h_init_LCLBT, model_kwargs = {"lr": 0.01, "aggregation": "sum"})
        LCLBT_fold_lls.append(LCLBT_fold_ll)
        LCLBT_fold_accs.append(LCLBT_fold_acc)
        LCLBT_acc += LCLBT_fold_acc * y_test.shape[0]

        init_generator_CR = lambda : CR_find_starting_point(ptask, CR_model, x_train, y_train, inv_transform_params_nd, inv_transform_hparams, N_test=10)
        CR_fold_ll, CR_fold_acc, _, _ = infer_and_validate_SGD_multistart(CR_model, x_train, y_train, x_test, y_test, init_generator_CR, 30, test_portion=0.2, model_kwargs = CR_model_kwargs)
        CR_fold_lls.append(CR_fold_ll)
        CR_fold_accs.append(CR_fold_acc)
        CR_acc += CR_fold_acc * y_test.shape[0]

        init_generator_LCLCR = lambda : (inv_transform_params_nd(ptask.generate_parameter_batch(1).flatten()), inv_transform_hparams_LCLCR(torch.concat([ptask.generate_hyperparameter_batch(1), torch.randn(1, ptask.n_attributes**2)], dim=1).flatten()))
        LCLCR_fold_ll, LCLCR_fold_acc, _, _ = infer_and_validate_SGD_multistart(LCLCR_model, x_train, y_train, x_test, y_test, init_generator_LCLCR, 30, test_portion=0.2, model_kwargs = LCLCR_model_kwargs)
        LCLCR_fold_lls.append(LCLCR_fold_ll)
        LCLCR_fold_accs.append(LCLCR_fold_acc)
        LCLCR_acc += LCLCR_fold_acc * y_test.shape[0]

        print(y_test.shape, BT_fold_ll, BBBT_fold_ll, LCLBT_fold_ll, CR_fold_ll, LCLCR_fold_ll)
        print(y_test.shape, BT_fold_acc, BBBT_fold_acc, LCLBT_fold_acc, CR_fold_acc, LCLCR_fold_acc)
        
    print("Test-set CVd likelihoods:")
    print(f"BT: {np.sum(BT_fold_lls)} \t BBBT: {np.sum(BBBT_fold_lls)} \t LCL: {np.sum(LCLBT_fold_lls)} \t CRCS {np.sum(CR_fold_lls)} \t LC-CRCS {np.sum(LCLCR_fold_lls)}")
    print("Paired t-test p-values:")
    print("BT vs BBBT", wilcoxon(BT_fold_lls, BBBT_fold_lls).pvalue)
    print("LCL vs BT", wilcoxon(LCLBT_fold_lls, BT_fold_lls).pvalue)
    print("LCL vs BBBT", wilcoxon(LCLBT_fold_lls, BBBT_fold_lls).pvalue)
    print("CR vs BT", wilcoxon(CR_fold_lls, BT_fold_lls).pvalue)
    print("CR vs BBBT", wilcoxon(CR_fold_lls, BBBT_fold_lls).pvalue)
    print("CR vs LCL", wilcoxon(CR_fold_lls, LCLBT_fold_lls).pvalue)
    print("LC-CRCS vs BT", wilcoxon(LCLCR_fold_lls, BT_fold_lls).pvalue)
    print("LC-CRCS vs BBBT", wilcoxon(LCLCR_fold_lls, BBBT_fold_lls).pvalue)
    print("LC-CRCS vs LCL", wilcoxon(LCLCR_fold_lls, LCLBT_fold_lls).pvalue)
    print("LC-CRCS vs CR", wilcoxon(LCLCR_fold_lls, CR_fold_lls).pvalue)

    print("Test-set CVd accuracies:")
    print(f"BT: {BT_acc / ptask.y.shape[0]} \t BBBT: {BBBT_acc / ptask.y.shape[0]} \t LCL: {LCLBT_acc / ptask.y.shape[0]} \t CRCS {CR_acc / ptask.y.shape[0]} \t LC-CRCS {LCLCR_acc / ptask.y.shape[0]}")
    print("Paired t-test p-values:")
    print("BT vs BBBT", wilcoxon(BT_fold_accs, BBBT_fold_accs).pvalue)
    print("LCL vs BT", wilcoxon(LCLBT_fold_accs, BT_fold_accs).pvalue)
    print("LCL vs BBBT", wilcoxon(LCLBT_fold_accs, BBBT_fold_accs).pvalue)
    print("CRCS vs BT", wilcoxon(CR_fold_accs, BT_fold_accs).pvalue)
    print("CRCS vs BBBT", wilcoxon(CR_fold_accs, BBBT_fold_accs).pvalue)
    print("CRCS vs LCL", wilcoxon(CR_fold_accs, LCLBT_fold_accs).pvalue)
    print("LC-CRCS vs BT", wilcoxon(LCLCR_fold_accs, BT_fold_accs).pvalue)
    print("LC-CRCS vs BBBT", wilcoxon(LCLCR_fold_accs, BBBT_fold_accs).pvalue)
    print("LC-CRCS vs LCL", wilcoxon(LCLCR_fold_accs, LCLBT_fold_accs).pvalue)
    print("LC-CRCS vs CRCS", wilcoxon(LCLCR_fold_accs, CR_fold_accs).pvalue)

#
# Hotels (Ronayne & Brown)
#

RBH_EVAL_N_FOLDS = 50
RBH_PATH = 'ronayne_brown_hotels'
RBH_CHECKPOINT_FOLDER = "checkpoints2"

def get_task_ronayne_brown_hotels():
    sigma_e_prior = torch.distributions.uniform.Uniform(0.0, 5.0)
    tau_pv_prior = torch.distributions.uniform.Uniform(torch.tensor([0.0, 0.0]), torch.tensor([100.0, 1.0]))
    p_error_prior = torch.distributions.Beta(1.0, 3.0)
    return RonayneBrownHotelsTask(sigma_e_prior, tau_pv_prior, p_error_prior, device = device, normalize_weights=True, data_location=RBH_PATH)

def get_model_kwargs_ronayne_brown_hotels():
    EU_kwargs= {"n_epochs": 10_000,
                "batch_size": 8192,
                "lr_start": 1e-3,
                "lr_end": 1e-3,
                "model_kwargs": {"main_sizes": [512, 512, 256],
                                 "control_latent_dim": 128,
                                 "controller_sizes": [256, 256],
                                 "dropout_rate": 0.01
                                 }}
    PC_kwargs={"n_epochs": 50_000,
               "batch_size": 2048,
               "lr_start": 1e-3,
               "lr_end": 1e-4,
               "model_kwargs": {"main_sizes": [512, 512, 256],
                                "control_latent_dim": 128,
                                "controller_sizes": [256, 256],
                                "dropout_rate": 0.0,
                                "u_dropout_rate": 0.0
                                }}
    return EU_kwargs, PC_kwargs

def train_ronayne_brown_hotels():
    ptask = get_task_ronayne_brown_hotels()
    EU_kwargs, PC_kwargs = get_model_kwargs_ronayne_brown_hotels()
    train(ptask, f'{RBH_PATH}/{RBH_CHECKPOINT_FOLDER}/', device = device, EU_kwargs=EU_kwargs, PC_kwargs=PC_kwargs)

def train_EU_ronayne_brown_hotels():
    ptask = get_task_ronayne_brown_hotels()
    kwargs, _ = get_model_kwargs_ronayne_brown_hotels()
    train_EU_model(ptask, f'{RBH_PATH}/{RBH_CHECKPOINT_FOLDER}/', device=device, EU_kwargs=kwargs)

def train_PC_ronayne_brown_hotels(EUID):
    ptask = get_task_ronayne_brown_hotels()
    _, kwargs = get_model_kwargs_ronayne_brown_hotels()
    train_PC_model(ptask, f'{RBH_PATH}/{RBH_CHECKPOINT_FOLDER}/', EUID, device=device, PC_kwargs=kwargs)

def validate_ronayne_brown_hotels(CR_model_dir):
    ptask = get_task_ronayne_brown_hotels()

    BT_model = CanonicalBradleyTerryModel(ptask)
    BBBT_model = BowerBalzanoBradleyTerryModel(ptask)
    LCLBT_model = LCLBradleyTerryModel(ptask)
    CR_model = load_PC_model(ptask, CR_model_dir, device=device)
    CR_model_kwargs = {"transform_params": transform_params_nd, "transform_hparams": transform_hparams, "n_iter": 1000, "lr": 0.05, "aggregation": "mean"}
    LCLCR_model = load_LCLCR_model(ptask, CR_model_dir, device=device)
    transform_hparams_LCLCR = partial(transform_hparams_aux, n_aux = LCLCR_model.weight_scaler_dim_flat)
    inv_transform_hparams_LCLCR = partial(inv_transform_hparams_aux, n_aux = LCLCR_model.weight_scaler_dim_flat)
    LCLCR_model_kwargs = {"transform_params": transform_params_nd, "transform_hparams": transform_hparams_LCLCR, "n_iter": 5000, "lr": 0.05, "aggregation": "mean"}

    BT_fold_lls, BBBT_fold_lls, LCLBT_fold_lls, CR_fold_lls, LCLCR_fold_lls = [], [], [], [], []
    BT_acc, BBBT_acc, LCLBT_acc, CR_acc, LCLCR_acc = 0.0, 0.0, 0.0, 0.0, 0.0
    BT_fold_accs, BBBT_fold_accs, LCLBT_fold_accs, CR_fold_accs, LCLCR_fold_accs = [], [], [], [], []

    cv = StratifiedKFold(n_splits=RBH_EVAL_N_FOLDS, shuffle=True)
    for (train_index, test_index) in tqdm.tqdm(cv.split(ptask.x.numpy(), ptask.y.numpy())):
        x_train, y_train = ptask.x[train_index,:,:], ptask.y[train_index]
        x_test, y_test = ptask.x[test_index,:,:], ptask.y[test_index]

        w_init_BT, h_init_BT = torch.randn(ptask.parameter_dim), None
        BT_fold_ll, BT_fold_acc, _, _ = infer_and_validate_SGD(BT_model, x_train, y_train, x_test, y_test, w_init_BT, h_init_BT, model_kwargs = {"lr": 0.001, "aggregation": "sum"})
        BT_fold_lls.append(BT_fold_ll)
        BT_fold_accs.append(BT_fold_acc)
        BT_acc += BT_fold_acc * y_test.shape[0]

        w_init_BBBT, h_init_BBBT = torch.randn(ptask.parameter_dim), torch.tensor([2], dtype=torch.int32)
        BBBT_fold_ll, BBBT_fold_acc, _, _ = infer_and_validate_SGD(BBBT_model, x_train, y_train, x_test, y_test, w_init_BBBT, h_init_BBBT, model_kwargs = {"lr": 0.001, "aggregation": "sum"})
        BBBT_fold_lls.append(BBBT_fold_ll)
        BBBT_fold_accs.append(BBBT_fold_acc)
        BBBT_acc += BBBT_fold_acc * y_test.shape[0]

        w_init_LCLBT, h_init_LCLBT = torch.randn(ptask.parameter_dim), torch.randn(ptask.parameter_dim**2)
        LCLBT_fold_ll, LCLBT_fold_acc, _, _ = infer_and_validate_SGD(LCLBT_model, x_train, y_train, x_test, y_test, w_init_LCLBT, h_init_LCLBT, model_kwargs = {"lr": 0.001, "aggregation": "sum"})
        LCLBT_fold_lls.append(LCLBT_fold_ll)
        LCLBT_fold_accs.append(LCLBT_fold_acc)
        LCLBT_acc += LCLBT_fold_acc * y_test.shape[0]

        init_generator_CR = lambda : CR_find_starting_point(ptask, CR_model, x_train, y_train, inv_transform_params_nd, inv_transform_hparams, N_test=2000)
        CR_fold_ll, CR_fold_acc, _, _ = infer_and_validate_SGD_multistart(CR_model, x_train, y_train, x_test, y_test, init_generator_CR, 10, model_kwargs = CR_model_kwargs)
        CR_fold_lls.append(CR_fold_ll)
        CR_fold_accs.append(CR_fold_acc)
        CR_acc += CR_fold_acc * y_test.shape[0]

        init_generator_LCLCR = lambda : (inv_transform_params_nd(ptask.generate_parameter_batch(1).flatten()), inv_transform_hparams_LCLCR(torch.concat([ptask.generate_hyperparameter_batch(1), torch.randn(1, ptask.n_attributes**2)], dim=1).flatten()))
        LCLCR_fold_ll, LCLCR_fold_acc, _, _ = infer_and_validate_SGD_multistart(LCLCR_model, x_train, y_train, x_test, y_test, init_generator_LCLCR, 10, model_kwargs = LCLCR_model_kwargs)
        LCLCR_fold_lls.append(LCLCR_fold_ll)
        LCLCR_fold_accs.append(LCLCR_fold_acc)
        LCLCR_acc += LCLCR_fold_acc * y_test.shape[0]

        print(y_test.shape, BT_fold_ll, BBBT_fold_ll, LCLBT_fold_ll, CR_fold_ll, LCLCR_fold_ll)
        print(y_test.shape, BT_fold_acc, BBBT_fold_acc, LCLBT_fold_acc, CR_fold_acc, LCLCR_fold_acc)
            
    print("Test-set CVd likelihoods:")
    print(f"BT: {np.sum(BT_fold_lls)} \t BBBT: {np.sum(BBBT_fold_lls)} \t LCL: {np.sum(LCLBT_fold_lls)} \t CRCS {np.sum(CR_fold_lls)} \t LC-CRCS {np.sum(LCLCR_fold_lls)}")
    print("Paired t-test p-values:")
    print("BT vs BBBT", wilcoxon(BT_fold_lls, BBBT_fold_lls).pvalue)
    print("LCL vs BT", wilcoxon(LCLBT_fold_lls, BT_fold_lls).pvalue)
    print("LCL vs BBBT", wilcoxon(LCLBT_fold_lls, BBBT_fold_lls).pvalue)
    print("CR vs BT", wilcoxon(CR_fold_lls, BT_fold_lls).pvalue)
    print("CR vs BBBT", wilcoxon(CR_fold_lls, BBBT_fold_lls).pvalue)
    print("CR vs LCL", wilcoxon(CR_fold_lls, LCLBT_fold_lls).pvalue)
    print("LC-CRCS vs BT", wilcoxon(LCLCR_fold_lls, BT_fold_lls).pvalue)
    print("LC-CRCS vs BBBT", wilcoxon(LCLCR_fold_lls, BBBT_fold_lls).pvalue)
    print("LC-CRCS vs LCL", wilcoxon(LCLCR_fold_lls, LCLBT_fold_lls).pvalue)
    print("LC-CRCS vs CR", wilcoxon(LCLCR_fold_lls, CR_fold_lls).pvalue)

    print("Test-set CVd accuracies:")
    print(f"BT: {BT_acc / ptask.y.shape[0]} \t BBBT: {BBBT_acc / ptask.y.shape[0]} \t LCL: {LCLBT_acc / ptask.y.shape[0]} \t CRCS {CR_acc / ptask.y.shape[0]} \t LC-CRCS {LCLCR_acc / ptask.y.shape[0]}")
    print("Paired t-test p-values:")
    print("BT vs BBBT", wilcoxon(BT_fold_accs, BBBT_fold_accs).pvalue)
    print("LCL vs BT", wilcoxon(LCLBT_fold_accs, BT_fold_accs).pvalue)
    print("LCL vs BBBT", wilcoxon(LCLBT_fold_accs, BBBT_fold_accs).pvalue)
    print("CRCS vs BT", wilcoxon(CR_fold_accs, BT_fold_accs).pvalue)
    print("CRCS vs BBBT", wilcoxon(CR_fold_accs, BBBT_fold_accs).pvalue)
    print("CRCS vs LCL", wilcoxon(CR_fold_accs, LCLBT_fold_accs).pvalue)
    print("LC-CRCS vs BT", wilcoxon(LCLCR_fold_accs, BT_fold_accs).pvalue)
    print("LC-CRCS vs BBBT", wilcoxon(LCLCR_fold_accs, BBBT_fold_accs).pvalue)
    print("LC-CRCS vs LCL", wilcoxon(LCLCR_fold_accs, LCLBT_fold_accs).pvalue)
    print("LC-CRCS vs CRCS", wilcoxon(LCLCR_fold_accs, CR_fold_accs).pvalue)

#
# Risky choice (normalized and original of howes et al. 2016)
#

RC_N_PARTICLES_PER_DIM = 10
RC_N_QUERY_CANDIDATES = 1000
RC_N_ELICITATION_STEPS = 100
RC_PATH = 'risky_choice'
RC_EXP_FOLDER = 'experiment_data'
RC_CHECKPOINT_FOLDER = "checkpoints"

def get_task_risky_choice():
    sigma_e_prior = torch.distributions.half_normal.HalfNormal(1.0)
    return Howes2016RiskyChoiceTask(N_CHOICE, sigma_e_prior, [0.011, 1.1], 0.0, device)

def get_model_kwargs_risky_choice():
    EU_kwargs= {"n_epochs": 35_000,
                "batch_size": 1024,
                "lr_start": 1e-3,
                "lr_end": 1e-5,
                "model_kwargs": {"main_sizes": [512, 256, 128],
                                 "control_latent_dim": 64,
                                 "controller_sizes": [128, 64],
                                 "dropout_rate": 0.01
                                 }}
    PC_kwargs={"n_epochs": 50_000,
               "batch_size": 1024,
               "lr_start": 1e-3,
               "lr_end": 1e-6,
               "model_kwargs": {"main_sizes": [1024, 256, 128],
                                "control_latent_dim": 128,
                                "controller_sizes": [128, 64],
                                "dropout_rate": 0.0,
                                "u_dropout_rate": 0.0
                                }}
    return EU_kwargs, PC_kwargs

def train_risky_choice():
    ptask = get_task_risky_choice()
    EU_kwargs, PC_kwargs = get_model_kwargs_risky_choice()
    train(ptask, f'{RC_PATH}/{RC_CHECKPOINT_FOLDER}/', device = device, EU_kwargs=EU_kwargs, PC_kwargs=PC_kwargs)

def train_EU_risky_choice():
    ptask = get_task_risky_choice()
    kwargs, _ = get_model_kwargs_risky_choice()
    train_EU_model(ptask, f'{RC_PATH}/{RC_CHECKPOINT_FOLDER}/', device=device, EU_kwargs=kwargs)

def train_PC_risky_choice(EUID):
    ptask = get_task_risky_choice()
    _, kwargs = get_model_kwargs_risky_choice()
    train_PC_model(ptask, f'{RC_PATH}/{RC_CHECKPOINT_FOLDER}/', EUID, device=device, PC_kwargs=kwargs)

def validate_risky_choice(CR_model_dir):
    from tasks.risky_choice_task import generate_decoy_choice_batch

    ptask = get_task_risky_choice()
    CR_model = load_PC_model(ptask, CR_model_dir, device=device)

    N_BATCHES = 500
    batch_size = 512

    for sigma_e in torch.arange(0.0, 5.1, 0.25):
        hparam = sigma_e[None,None].repeat((batch_size,1))
        w = ptask.generate_parameter_batch(batch_size)
        reversal_rate = []
        for _ in range(N_BATCHES):
            choice_sets = generate_decoy_choice_batch(ptask, w, alpha = 0.2)
            reversal_rate1, inverse_reversal_rate1, _ = CR_model(choice_sets[:,[0,1,2],:], w, hparam).exp().sum(dim=0)
            inverse_reversal_rate2, reversal_rate2, _ = CR_model(choice_sets[:,[0,1,3],:], w, hparam).exp().sum(dim=0)
            reversal_rate.append((reversal_rate1 + reversal_rate2 - inverse_reversal_rate1 - inverse_reversal_rate2).item() / 2)

        print(f"reversal rate - inverse reversal rate @ SD={sigma_e.item()}: {np.sum(reversal_rate) / (N_BATCHES * batch_size)}")

#
# Car crash structure design
#

CC_N_PARTICLES_PER_UPARAM_DIM = 10
CC_N_PARTICLES_PER_HPARAM_DIM = 6
CC_N_QUERY_CANDIDATES = 1000
CC_N_ELICITATION_STEPS = 50
CC_N_PRIOR_ELICITATION_STEPS = 3
CC_SCALARIZATION = "Chebyshev"
CC_PATH = 'car_crash'
CC_EXP_FOLDER = 'experiment_data'
CC_CHECKPOINT_FOLDER = "checkpoints"

def get_task_car_crash():
    sigma_e_prior = torch.distributions.Beta(1.0, 4.0)
    tau_pv_prior = torch.distributions.Beta(1.0, 9.0)
    p_error_prior = torch.distributions.Beta(1.0, 19.0)
    return ParameterizedCarCrashMOOTask(N_CHOICE, sigma_e_prior, tau_pv_prior, p_error_prior, device = device, scalarization=CC_SCALARIZATION, data_location=CC_PATH)

def find_pareto_front_car_crash():
    task = get_task_car_crash()
    task.find_pareto_front()

def generate_exp_file_car_crash(N_EXP = 500):
    ptask = get_task_car_crash()

    exps = []
    for _ in range(N_EXP):
        original_uparam_particle_set = ptask.generate_parameter_batch(int(CC_N_PARTICLES_PER_UPARAM_DIM**ptask.parameter_dim))
        original_hparam_particle_set = original_hparam_particle_set = ptask.generate_hyperparameter_batch(int(CC_N_PARTICLES_PER_HPARAM_DIM**ptask.hyperparameter_dim))
        true_w = ptask.generate_parameter_batch(1)
        true_h = ptask.generate_hyperparameter_batch(1)
        exps.append((original_uparam_particle_set,original_hparam_particle_set,true_w,true_h))
    torch.save(exps, f"{CC_PATH}/{CC_EXP_FOLDER}/experiment_definitions.pth")

def get_model_kwargs_car_crash():
    EU_kwargs= {"n_epochs": 35_000,
                "batch_size": 1024,
                "lr_start": 1e-3,
                "lr_end": 1e-5,
                "model_kwargs": {"main_sizes": [512, 256, 128],
                                 "control_latent_dim": 64,
                                 "controller_sizes": [128, 64],
                                 "dropout_rate": 0.01
                                 }}
    PC_kwargs={"n_epochs": 50_000,
               "batch_size": 1024,
               "lr_start": 1e-3,
               "lr_end": 1e-6,
               "model_kwargs": {"main_sizes": [1024, 256, 128],
                                "control_latent_dim": 128,
                                "controller_sizes": [128, 64],
                                "dropout_rate": 0.0,
                                "u_dropout_rate": 0.0
                                }}
    return EU_kwargs, PC_kwargs

def train_car_crash():
    ptask = get_task_car_crash()
    EU_kwargs, PC_kwargs = get_model_kwargs_car_crash()
    train(ptask, f'{CC_PATH}/{CC_CHECKPOINT_FOLDER}/', device = device, EU_kwargs=EU_kwargs, PC_kwargs=PC_kwargs)

def train_EU_car_crash():
    ptask = get_task_car_crash()
    kwargs, _ = get_model_kwargs_car_crash()
    train_EU_model(ptask, f'{CC_PATH}/{CC_CHECKPOINT_FOLDER}/', device=device, EU_kwargs=kwargs)

def train_PC_car_crash(EUID):
    ptask = get_task_car_crash()
    _, kwargs = get_model_kwargs_car_crash()
    train_PC_model(ptask, f'{CC_PATH}/{CC_CHECKPOINT_FOLDER}/', EUID, device=device, PC_kwargs=kwargs)

def CRCS_car_crash(exp_ID, CR_model_dir):
    ptask = get_task_car_crash()
    CR_model = load_PC_model(ptask, CR_model_dir, device=device)

    original_uparam_particle_set,original_hparam_particle_set,true_w,true_h = torch.load(f"{CC_PATH}/{CC_EXP_FOLDER}/experiment_definitions.pth")[exp_ID]
    uparam_particle_set, hparam_particle_set = create_joint_partcile_sets(original_uparam_particle_set, original_hparam_particle_set)

    DATA = create_DATA(CC_N_QUERY_CANDIDATES, original_uparam_particle_set.shape[0], original_hparam_particle_set.shape[0])
    DATA, _ = run_BOED(ptask, CR_model, CR_model, uparam_particle_set, hparam_particle_set, true_w, true_h, CC_N_ELICITATION_STEPS, DATA)

    f_out = f"{CC_PATH}/{CC_EXP_FOLDER}/CRCS_{str(exp_ID)}.json"
    with open(f_out, "w+") as f:
        json.dump(DATA, f)

def random_car_crash(exp_ID):
    ptask = get_task_car_crash()

    original_uparam_particle_set,original_hparam_particle_set,true_w,true_h = torch.load(f"{CC_PATH}/{CC_EXP_FOLDER}/experiment_definitions.pth")[exp_ID]

    DATA = create_DATA(CC_N_QUERY_CANDIDATES, original_uparam_particle_set.shape[0], original_hparam_particle_set.shape[0])
    DATA["recommendation_regret"] = [calculate_regret_random_recommendation(ptask, true_w)] * (CC_N_ELICITATION_STEPS + 1)

    f_out = f"{CC_PATH}/{CC_EXP_FOLDER}/random_{str(exp_ID)}.json"
    with open(f_out, "w+") as f:
        json.dump(DATA, f)

def bradley_terry_car_crash(exp_ID, CR_model_dir):
    ptask = get_task_car_crash()

    beta_prior = torch.distributions.Uniform(5.0, 45.0)
    BT_model = ParameterizedBradleyTerryModel(ptask, beta_max=80.0)
    original_hparam_particle_set = beta_prior.sample((20,1))
    CR_model = load_PC_model(ptask, CR_model_dir, device=device)

    original_uparam_particle_set,_,true_w,true_h = torch.load(f"{CC_PATH}/{CC_EXP_FOLDER}/experiment_definitions.pth")[exp_ID]
    uparam_particle_set, hparam_particle_set = create_joint_partcile_sets(original_uparam_particle_set, original_hparam_particle_set)

    DATA = create_DATA(CC_N_QUERY_CANDIDATES, original_uparam_particle_set.shape[0], original_hparam_particle_set.shape[0])
    DATA, _ = run_BOED(ptask, BT_model, CR_model, uparam_particle_set, hparam_particle_set, true_w, true_h, CC_N_ELICITATION_STEPS, DATA, calculate_hparam_error = False)

    f_out = f"{CC_PATH}/{CC_EXP_FOLDER}/bradley_terry_{str(exp_ID)}.json"
    with open(f_out, "w+") as f:
        json.dump(DATA, f)

#
# Water network structure design design
#

WM_N_PARTICLES_PER_DIM = 4.3
WM_N_QUERY_CANDIDATES = 1000
WM_N_ELICITATION_STEPS = 100
WM_N_PRIOR_ELICITATION_STEPS = 3
WM_SCALARIZATION = "WeightedSum"
WM_PATH = 'water_management'
WM_EXP_FOLDER = 'experiment_data'
WM_CHECKPOINT_FOLDER = "checkpoints"

def get_task_water_management():
    sigma_e_prior = torch.distributions.Beta(1.0, 4.0)
    tau_pv_prior = torch.distributions.Beta(1.0, 9.0)
    p_error_prior = torch.distributions.Beta(1.0, 19.0)
    return WaterManagementMOOTask(N_CHOICE, sigma_e_prior, tau_pv_prior, p_error_prior, device = device, scalarization=WM_SCALARIZATION, data_location=WM_PATH)

def find_pareto_front_water_management():
    task = get_task_water_management()
    task.find_pareto_front()

def generate_exp_file_water_management(N_EXP = 500):
    ptask = get_task_water_management()

    exps = []
    for _ in range(N_EXP):
        original_uparam_particle_set = ptask.generate_parameter_batch(int(WM_N_PARTICLES_PER_DIM**ptask.parameter_dim))
        original_hparam_particle_set = original_hparam_particle_set = ptask.generate_hyperparameter_batch(int(WM_N_PARTICLES_PER_DIM**ptask.hyperparameter_dim))
        true_w = ptask.generate_parameter_batch(1)
        true_h = ptask.generate_hyperparameter_batch(1)
        exps.append((original_uparam_particle_set,original_hparam_particle_set,true_w,true_h))
    torch.save(exps, f"{WM_PATH}/{WM_EXP_FOLDER}/experiment_definitions.pth")

def get_model_kwargs_water_management():
    EU_kwargs= {"n_epochs": 35_000,
                "batch_size": 1024,
                "lr_start": 1e-3,
                "lr_end": 1e-5,
                "model_kwargs": {"main_sizes": [512, 256, 128],
                                 "control_latent_dim": 64,
                                 "controller_sizes": [128, 64],
                                 "dropout_rate": 0.01
                                 }}
    PC_kwargs={"n_epochs": 50_000,
               "batch_size": 1024,
               "lr_start": 1e-3,
               "lr_end": 1e-6,
               "model_kwargs": {"main_sizes": [1024, 256, 128],
                                "control_latent_dim": 128,
                                "controller_sizes": [128, 64],
                                "dropout_rate": 0.0,
                                "u_dropout_rate": 0.0
                                }}
    return EU_kwargs, PC_kwargs

def train_water_management():
    ptask = get_task_water_management()
    EU_kwargs, PC_kwargs = get_model_kwargs_water_management()
    train(ptask, f'{WM_PATH}/{WM_CHECKPOINT_FOLDER}/', device = device, EU_kwargs=EU_kwargs, PC_kwargs=PC_kwargs)

def train_EU_water_management():
    ptask = get_task_water_management()
    kwargs, _ = get_model_kwargs_water_management()
    train_EU_model(ptask, f'{WM_PATH}/{WM_CHECKPOINT_FOLDER}/', device=device, EU_kwargs=kwargs)

def train_PC_water_management(EUID):
    ptask = get_task_water_management()
    _, kwargs = get_model_kwargs_water_management()
    train_PC_model(ptask, f'{WM_PATH}/{WM_CHECKPOINT_FOLDER}/', EUID, device=device, PC_kwargs=kwargs)

def CRCS_water_management(exp_ID, CR_model_dir):
    ptask = get_task_water_management()
    CR_model = load_PC_model(ptask, CR_model_dir, device=device)

    original_uparam_particle_set,original_hparam_particle_set,true_w,true_h = torch.load(f"{WM_PATH}/{WM_EXP_FOLDER}/experiment_definitions.pth")[exp_ID]
    uparam_particle_set, hparam_particle_set = create_joint_partcile_sets(original_uparam_particle_set, original_hparam_particle_set)

    DATA = create_DATA(WM_N_QUERY_CANDIDATES, original_uparam_particle_set.shape[0], original_hparam_particle_set.shape[0])
    DATA, _ = run_BOED(ptask, CR_model, CR_model, uparam_particle_set, hparam_particle_set, true_w, true_h, WM_N_ELICITATION_STEPS, DATA)

    f_out = f"{WM_PATH}/{WM_EXP_FOLDER}/CRCS_{str(exp_ID)}.json"
    with open(f_out, "w+") as f:
        json.dump(DATA, f)

def random_water_management(exp_ID):
    ptask = get_task_water_management()

    original_uparam_particle_set,original_hparam_particle_set,true_w,true_h = torch.load(f"{WM_PATH}/{WM_EXP_FOLDER}/experiment_definitions.pth")[exp_ID]

    DATA = create_DATA(WM_N_QUERY_CANDIDATES, original_uparam_particle_set.shape[0], original_hparam_particle_set.shape[0])
    DATA["recommendation_regret"] = [calculate_regret_random_recommendation(ptask, true_w)] * (WM_N_ELICITATION_STEPS + 1)

    f_out = f"{WM_PATH}/{WM_EXP_FOLDER}/random_{str(exp_ID)}.json"
    with open(f_out, "w+") as f:
        json.dump(DATA, f)

def bradley_terry_water_management(exp_ID, CR_model_dir):
    ptask = get_task_water_management()

    beta_prior = torch.distributions.Uniform(0.0, 60.0)
    BT_model = ParameterizedBradleyTerryModel(ptask, beta_max=80.0)
    original_hparam_particle_set = beta_prior.sample((20,1))
    CR_model = load_PC_model(ptask, CR_model_dir, device=device)

    original_uparam_particle_set,_,true_w,true_h = torch.load(f"{WM_PATH}/{WM_EXP_FOLDER}/experiment_definitions.pth")[exp_ID]
    uparam_particle_set, hparam_particle_set = create_joint_partcile_sets(original_uparam_particle_set, original_hparam_particle_set)

    DATA = create_DATA(WM_N_QUERY_CANDIDATES, original_uparam_particle_set.shape[0], original_hparam_particle_set.shape[0])
    DATA, _ = run_BOED(ptask, BT_model, CR_model, uparam_particle_set, hparam_particle_set, true_w, true_h, WM_N_ELICITATION_STEPS, DATA, calculate_hparam_error = False)

    f_out = f"{WM_PATH}/{WM_EXP_FOLDER}/bradley_terry_{str(exp_ID)}.json"
    with open(f_out, "w+") as f:
        json.dump(DATA, f)

#
# Retrosynthesis planning
#

RP_N_PARTICLES_PER_DIM = 4.3
RP_N_QUERY_CANDIDATES = 1000
RP_N_ELICITATION_STEPS = 50
RP_SCALARIZATION = "WeightedSum"
RP_PATH = 'retrosynthesis_planning'
RP_EXP_FOLDER = 'experiment_data'
RP_CHECKPOINT_FOLDER = "checkpoints"

def get_task_retrosynthesis_planning():
    sigma_e_prior = torch.distributions.Beta(1.0, 4.0)
    tau_pv_prior = torch.distributions.Beta(1.0, 9.0)
    p_error_prior = torch.distributions.Beta(1.0, 19.0)
    return RetrosynthesisPlanningTask(N_CHOICE, sigma_e_prior, tau_pv_prior, p_error_prior, device = device, scalarization=RP_SCALARIZATION, data_location=RP_PATH)

def generate_exp_file_retrosynthesis_planning(N_EXP = 500):
    ptask = get_task_retrosynthesis_planning()

    exps = []
    for _ in range(N_EXP):
        original_uparam_particle_set = ptask.generate_parameter_batch(int(RP_N_PARTICLES_PER_DIM**ptask.parameter_dim))
        original_hparam_particle_set = original_hparam_particle_set = ptask.generate_hyperparameter_batch(int(RP_N_PARTICLES_PER_DIM**ptask.hyperparameter_dim))
        true_w = ptask.generate_parameter_batch(1)
        true_h = ptask.generate_hyperparameter_batch(1)
        exps.append((original_uparam_particle_set,original_hparam_particle_set,true_w,true_h))
    torch.save(exps, f"{RP_PATH}/{RP_EXP_FOLDER}/experiment_definitions.pth")

def get_model_kwargs_retrosynthesis_planning():
    EU_kwargs= {"n_epochs": 35_000,
                "batch_size": 1024,
                "lr_start": 1e-3,
                "lr_end": 1e-5,
                "model_kwargs": {"main_sizes": [512, 256, 128],
                                 "control_latent_dim": 64,
                                 "controller_sizes": [128, 64],
                                 "dropout_rate": 0.01
                                 }}
    PC_kwargs={"n_epochs": 50_000,
               "batch_size": 1024,
               "lr_start": 1e-3,
               "lr_end": 1e-6,
               "model_kwargs": {"main_sizes": [1024, 256, 128],
                                "control_latent_dim": 128,
                                "controller_sizes": [128, 64],
                                "dropout_rate": 0.0,
                                "u_dropout_rate": 0.0
                                }}
    return EU_kwargs, PC_kwargs

def train_retrosynthesis_planning():
    ptask = get_task_retrosynthesis_planning()
    EU_kwargs, PC_kwargs = get_model_kwargs_retrosynthesis_planning()
    train(ptask, f'{RP_PATH}/{RP_CHECKPOINT_FOLDER}/', device = device, EU_kwargs=EU_kwargs, PC_kwargs=PC_kwargs)

def train_EU_retrosynthesis_planning():
    ptask = get_task_retrosynthesis_planning()
    kwargs, _ = get_model_kwargs_retrosynthesis_planning()
    train_EU_model(ptask, f'{RP_PATH}/{RP_CHECKPOINT_FOLDER}/', device=device, EU_kwargs=kwargs)

def train_PC_retrosynthesis_planning(EUID):
    ptask = get_task_retrosynthesis_planning()
    _, kwargs = get_model_kwargs_retrosynthesis_planning()
    train_PC_model(ptask, f'{RP_PATH}/{RP_CHECKPOINT_FOLDER}/', EUID, device=device, PC_kwargs=kwargs)

def CRCS_retrosynthesis_planning(exp_ID, CR_model_dir):
    ptask = get_task_retrosynthesis_planning()
    CR_model = load_PC_model(ptask, CR_model_dir, device=device)

    original_uparam_particle_set,original_hparam_particle_set,true_w,true_h = torch.load(f"{RP_PATH}/{RP_EXP_FOLDER}/experiment_definitions.pth")[exp_ID]
    uparam_particle_set, hparam_particle_set = create_joint_partcile_sets(original_uparam_particle_set, original_hparam_particle_set)

    DATA = create_DATA(RP_N_QUERY_CANDIDATES, original_uparam_particle_set.shape[0], original_hparam_particle_set.shape[0])
    DATA, _ = run_BOED(ptask, CR_model, CR_model, uparam_particle_set, hparam_particle_set, true_w, true_h, RP_N_ELICITATION_STEPS, DATA, calculate_regret=False, save_posterior=True, calculate_context_effect=False)

    f_out = f"{RP_PATH}/{RP_EXP_FOLDER}/CRCS_{str(exp_ID)}.json"
    with open(f_out, "w+") as f:
        json.dump(DATA, f)

################################################################################
#                             ARGUMENT PROCESSING                              #
################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_case", type=str, choices=["RiskyChoice", "WaterManagement", "CarCrash", "RetrosynthesisPlanning", "RonayneBrownHotels", "Car-Alt", "District-Smart", "Dumbalska"], required = True)
    parser.add_argument("--elicit_with", type=str, nargs="*", choices=["CRCS", "BT", "random", "LCLBT", "BBBT", "LC-CRCS", "CRCS_no_BOED"], default=[None])
    parser.add_argument("--ID", type=int, default=None)
    parser.add_argument("--n_options", type=int, default=3)
    parser.add_argument("--setup", action='store_true')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--train_EU", action='store_true')
    parser.add_argument("--train_PC", action='store_true')
    parser.add_argument("--EUID", type=str, default=None)
    parser.add_argument("--CR_model_dir", type=str)
    parser.add_argument("--validate", action='store_true')
    parser.add_argument("--validate_rankings", action='store_true')
    parser.add_argument("--find_pareto_front", action="store_true")
    args = parser.parse_args()

    global N_CHOICE
    N_CHOICE = args.n_options
    
    if args.use_case == "RiskyChoice":
        if args.train:
            os.makedirs(f'{RC_PATH}/{RC_CHECKPOINT_FOLDER}/', exist_ok=True)
            train_risky_choice()
        if args.train_EU:
            os.makedirs(f'{RC_PATH}/{RC_CHECKPOINT_FOLDER}/', exist_ok=True)
            train_EU_risky_choice()
        if args.train_PC:
            assert args.EUID is not None, "must provide EU model ID to train PC model"
            train_PC_risky_choice(args.EUID)
        if args.validate:
            validate_risky_choice(args.CR_model_dir)
    elif args.use_case == "WaterManagement":
        os.makedirs(f"{WM_PATH}/{WM_EXP_FOLDER}", exist_ok=True)
        if args.find_pareto_front:
            find_pareto_front_water_management()
        if args.setup:
            generate_exp_file_water_management()
        if args.train:
            os.makedirs(f'{WM_PATH}/{WM_CHECKPOINT_FOLDER}/', exist_ok=True)
            train_water_management()
        if args.train_EU:
            os.makedirs(f'{WM_PATH}/{WM_CHECKPOINT_FOLDER}/', exist_ok=True)
            train_EU_water_management()
        if args.train_PC:
            assert args.EUID is not None, "must provide EU model ID to train PC model"
            train_PC_water_management(args.EUID)
        if "CRCS" in args.elicit_with:
            CRCS_water_management(args.ID, args.CR_model_dir)
        if "BT" in args.elicit_with:
            bradley_terry_water_management(args.ID, args.CR_model_dir)
        if "random" in args.elicit_with:
            random_water_management(args.ID)
    elif args.use_case == "CarCrash":
        os.makedirs(f"{CC_PATH}/{CC_EXP_FOLDER}", exist_ok=True)
        if args.find_pareto_front:
            find_pareto_front_car_crash()
        if args.setup:
            generate_exp_file_car_crash()
        if args.train:
            os.makedirs(f'{CC_PATH}/{CC_CHECKPOINT_FOLDER}/', exist_ok=True)
            train_car_crash()
        if args.train_EU:
            os.makedirs(f'{CC_PATH}/{CC_CHECKPOINT_FOLDER}/', exist_ok=True)
            train_EU_car_crash()
        if args.train_PC:
            assert args.EUID is not None, "must provide EU model ID to train PC model"
            train_PC_car_crash(args.EUID)
        if "CRCS" in args.elicit_with:
            CRCS_car_crash(args.ID, args.CR_model_dir)
        if "BT" in args.elicit_with:
            bradley_terry_car_crash(args.ID, args.CR_model_dir)
        if "random" in args.elicit_with:
            random_car_crash(args.ID)
    elif args.use_case == "RetrosynthesisPlanning":
        os.makedirs(f"{RP_PATH}/{RP_EXP_FOLDER}", exist_ok=True)
        if args.setup:
            generate_exp_file_retrosynthesis_planning()
        if args.train:
            os.makedirs(f'{RP_PATH}/{RP_CHECKPOINT_FOLDER}/', exist_ok=True)
            train_retrosynthesis_planning()
        if args.train_EU:
            os.makedirs(f'{RP_PATH}/{RP_CHECKPOINT_FOLDER}/', exist_ok=True)
            train_EU_retrosynthesis_planning()
        if args.train_PC:
            assert args.EUID is not None, "must provide EU model ID to train PC model"
            train_PC_retrosynthesis_planning(args.EUID)
        if "CRCS" in args.elicit_with:
            CRCS_retrosynthesis_planning(args.ID, args.CR_model_dir)
    elif args.use_case == "RonayneBrownHotels":
        if args.train:
            os.makedirs(f'{RBH_PATH}/{RBH_CHECKPOINT_FOLDER}/', exist_ok=True)
            train_ronayne_brown_hotels()
        if args.train_EU:
            os.makedirs(f'{RBH_PATH}/{RBH_CHECKPOINT_FOLDER}/', exist_ok=True)
            train_EU_ronayne_brown_hotels()
        if args.train_PC:
            assert args.EUID is not None, "must provide EU model ID to train PC model"
            train_PC_ronayne_brown_hotels(args.EUID)
        if args.validate:
            validate_ronayne_brown_hotels(args.CR_model_dir)
    elif args.use_case == "Car-Alt":
        if args.train:
            os.makedirs(f'{CA_PATH}/{CA_CHECKPOINT_FOLDER}/', exist_ok=True)
            train_car_alt()
        if args.train_EU:
            os.makedirs(f'{CA_PATH}/{CA_CHECKPOINT_FOLDER}/', exist_ok=True)
            train_EU_car_alt()
        if args.train_PC:
            assert args.EUID is not None, "must provide EU model ID to train PC model"
            train_PC_car_alt(args.EUID)
        if args.validate:
            validate_car_alt(args.CR_model_dir)
    elif args.use_case == "District-Smart":
        if args.train:
            os.makedirs(f'{DS_PATH}/{DS_CHECKPOINT_FOLDER}/', exist_ok=True)
            train_district_smart()
        if args.train_EU:
            os.makedirs(f'{DS_PATH}/{DS_CHECKPOINT_FOLDER}/', exist_ok=True)
            train_EU_district_smart()
        if args.train_PC:
            assert args.EUID is not None, "must provide EU model ID to train PC model"
            train_PC_district_smart(args.EUID)
        if args.validate:
            validate_district_smart(args.CR_model_dir)
        if args.validate_rankings:
            validate_rankings_district_smart(args.CR_model_dir)
    elif args.use_case == "Dumbalska":
        if args.train:
            os.makedirs(f'{DP_PATH}/{DP_CHECKPOINT_FOLDER}/', exist_ok=True)
            train_dumbalska()
        if args.train_EU:
            os.makedirs(f'{DP_PATH}/{DP_CHECKPOINT_FOLDER}/', exist_ok=True)
            train_EU_dumbalska()
        if args.train_PC:
            assert args.EUID is not None, "must provide EU model ID to train PC model"
            train_PC_dumbalska(args.EUID)
        if args.validate:
            os.makedirs(f'{DP_PATH}/{DP_EXP_FOLDER}/', exist_ok=True)
            validate_dumbalska(args.ID, args.CR_model_dir)
        if "CRCS" in args.elicit_with:
            os.makedirs(f'{DP_PATH}/{DP_EXP_FOLDER}/', exist_ok=True)
            SUS_CR_dumbalska(args.ID, args.CR_model_dir)
        if "LC-CRCS" in args.elicit_with:
            os.makedirs(f'{DP_PATH}/{DP_EXP_FOLDER}/', exist_ok=True)
            SUS_LCLCR_dumbalska(args.ID, args.CR_model_dir)
        if "CRCS_no_BOED" in args.elicit_with:
            os.makedirs(f'{DP_PATH}/{DP_EXP_FOLDER}/', exist_ok=True)
            SUS_CR_no_BOED_dumbalska(args.ID, args.CR_model_dir)
        if "LCLBT" in args.elicit_with:
            os.makedirs(f'{DP_PATH}/{DP_EXP_FOLDER}/', exist_ok=True)
            SUS_LCLBT_dumbalska(args.ID)
        if "BT" in args.elicit_with:
            os.makedirs(f'{DP_PATH}/{DP_EXP_FOLDER}/', exist_ok=True)
            SUS_BT_dumbalska(args.ID)
        if "BBBT" in args.elicit_with:
            os.makedirs(f'{DP_PATH}/{DP_EXP_FOLDER}/', exist_ok=True)
            SUS_BBBT_dumbalska(args.ID)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()