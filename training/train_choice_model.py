import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
import tqdm
from os.path import join
from os import makedirs

from models.models import *

import wandb

"""
parameters:
    task: task to train on.
    CHECKPOINT_DIR [str]: the directory where the checkpoint to load is saved (you should find "EU_model.pth" there).
"""
def load_EU_model(task, CHECKPOINT_DIR, device=torch.device("cpu")):
    model_kwargs = torch.load(join(CHECKPOINT_DIR, "model_kwargs.pth"))

    EU_model = AmortizedConditionedExpectedUtilityModel(task.observation_dim, task.n_choices, task.parameter_dim + task.hyperparameter_dim, **model_kwargs).to(device=device)
    EU_model.load_state_dict(torch.load(join(CHECKPOINT_DIR, "EU_model.pth"), map_location=device))
    EU_model.eval()
    return EU_model

"""
parameters:
    task: task to train on.
    CHECKPOINT_DIR [str]: the directory where the checkpoint to load is saved  (you should find "PC_model.pth" there).
"""
def load_PC_model(task, CHECKPOINT_DIR, device=torch.device("cpu")):
    model_kwargs = torch.load(join(CHECKPOINT_DIR, "model_kwargs.pth"))

    CR_model = AmortizedConditionalChoicePredictionModel(task.n_choices, task.n_attributes, task.parameter_dim + task.hyperparameter_dim, **model_kwargs).to(device=device)
    CR_model.load_state_dict(torch.load(join(CHECKPOINT_DIR, "PC_model.pth"), map_location=device))
    CR_model.eval()
    return CR_model

"""
parameters:
    task: task to train on.
    CHECKPOINT_DIR [str]: the base directory for where all checkpoints for this task are stored.
"""
def train_EU_model(task, CHECKPOINT_DIR, device = torch.device("cpu"), EU_kwargs = {}):
    EUID = wandb.util.generate_id() + f"_{task.n_choices}C"
    if "model_kwargs" not in EU_kwargs.keys():
        EU_kwargs["model_kwargs"] = {}

    save_dir = join(CHECKPOINT_DIR,EUID)
    print(save_dir)
    makedirs(save_dir, exist_ok=True)
    
    print(f"Starting new training run with EU model ID {EUID}")
    wandb.init(project="CR multi-attribute choice", group=EUID, tags = [task.__class__.__qualname__, "EU"])
    _train_EU_model(task, save_dir, device = device, **EU_kwargs)
    wandb.finish()
    return EUID

"""
parameters:
    task: task to train on.
    CHECKPOINT_DIR [str]: the base directory for where all checkpoints for this task are stored.
    EUID [str]: the EU model ID on which to train the PC model. A corresponding directory must exist within CHECKPOINT_DIR!
"""
def train_PC_model(task, CHECKPOINT_DIR, EUID, device = torch.device("cpu"), PC_kwargs = {}, reinit=False):
    if "model_kwargs" not in PC_kwargs.keys():
        PC_kwargs["model_kwargs"] = {}

    PCID = wandb.util.generate_id()
    save_dir = join(CHECKPOINT_DIR,EUID,PCID)
    makedirs(save_dir, exist_ok=True)
    print(f"Starting new training run with PC model at {save_dir}")

    wandb.init(project="CR multi-attribute choice", group=EUID, tags = [task.__class__.__qualname__, "PC"], reinit=reinit)
    EU_model = load_EU_model(task, join(CHECKPOINT_DIR,EUID))
    _train_PC_model(task, EU_model, save_dir, device = device, **PC_kwargs)
    wandb.finish()

def train(task, CHECKPOINT_DIR, device = torch.device("cpu"), EU_kwargs = {}, PC_kwargs = {}):
    EUID = train_EU_model(task, CHECKPOINT_DIR, device=device, EU_kwargs=EU_kwargs)
    train_PC_model(task, CHECKPOINT_DIR, EUID, device=device, PC_kwargs=PC_kwargs, reinit=True)

#
# HYPERPARAMETERS
#

EU_N_EPOCHS = 35000
EU_BATCH_SIZE = 1024
EU_N_EVAL_BATCHES = 100
EU_LR_START = 1e-3
EU_LR_END = 1e-5

def _eval_EU_model(task, EU_model, batch_size):
    EU_model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for _ in range(EU_N_EVAL_BATCHES):
            w_batch = task.generate_parameter_batch(batch_size)
            hyperparameter_batch = task.generate_hyperparameter_batch(batch_size)
            batch = task.generate_choice_batch(batch_size)
            x_f = task.utility(batch, w_batch)
            batch_obs = task.observe_batch(batch, x_f, hyperparameter_batch)
            
            # forward pass
            pred_util = EU_model(batch_obs, w_batch, hyperparameter_batch)
            eval_loss += torch.pow(pred_util - x_f, 2.0).sum().item()
    EU_model.train()
    return eval_loss / (batch_size * task.n_choices * EU_N_EVAL_BATCHES)

def _train_EU_model(task, CHECKPOINT_SAVE_DIR, device = torch.device("cpu"), lr_start = EU_LR_START, lr_end = EU_LR_END, n_epochs = EU_N_EPOCHS, batch_size = EU_BATCH_SIZE, model_kwargs = {}):
    EU_model = AmortizedConditionedExpectedUtilityModel(task.observation_dim, task.n_choices, task.parameter_dim + task.hyperparameter_dim, **model_kwargs)
    EU_model.to(device=device)
    EU_model.train()

    wandb.config.task_name = task.__class__.__qualname__
    wandb.config.surrogate_type = "EU"
    wandb.config.checkpoint_dir = CHECKPOINT_SAVE_DIR
    wandb.config.control_latent_dim = EU_model.control_latent_dim
    wandb.config.main_sizes = EU_model.main_sizes
    wandb.config.controller_sizes = EU_model.controller_sizes
    wandb.config.dropout = EU_model.dropout_rate
    wandb.config.lr_start = lr_start
    wandb.config.lr_end = lr_end
    wandb.config.batch_size = batch_size

    optimizer = AdamW(EU_model.parameters(), lr=lr_start)
    scheduler = ExponentialLR(optimizer, np.exp(np.log(lr_end / lr_start) / n_epochs))
    accumulated_losses = []
    for epoch in (tqdm_epoch := tqdm.tqdm(range(n_epochs+1))):
        w_batch = task.generate_parameter_batch(batch_size)
        hyperparameter_batch = task.generate_hyperparameter_batch(batch_size)
        batch = task.generate_choice_batch(batch_size)
        x_f = task.utility(batch, w_batch)
        batch_obs = task.observe_batch(batch, x_f, hyperparameter_batch)
        
        # forward pass
        pred_util = EU_model(batch_obs, w_batch, hyperparameter_batch)
        loss = torch.pow(pred_util - x_f, 2.0).mean()
        accumulated_losses.append(loss.item())
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # weight update
        optimizer.step()
        scheduler.step()

        if (epoch % 500 == 0) and (epoch != 0):
            wandb.log({"EUloss": loss.item(), "EUloss_val": _eval_EU_model(task, EU_model, batch_size)})
            torch.save(EU_model.state_dict(), join(CHECKPOINT_SAVE_DIR, "EU_model.pth"))
            torch.save(model_kwargs, join(CHECKPOINT_SAVE_DIR, "model_kwargs.pth"))
        else:
            wandb.log({"EUloss": loss.item()})

        if epoch % 25 == 1:
            tqdm_epoch.set_description('Average Loss: {:5f}'.format(np.mean(accumulated_losses)))
            accumulated_losses = []
    torch.save(EU_model.state_dict(), join(CHECKPOINT_SAVE_DIR, "EU_model.pth"))
    torch.save(model_kwargs, join(CHECKPOINT_SAVE_DIR, "model_kwargs.pth"))
    return EU_model


PC_N_EPOCHS = 50_000
PC_BATCH_SIZE = 1024
PC_N_EVAL_BATCHES = 100
PC_LR_START = 1e-3
PC_LR_END = 1e-6

def _eval_PC_model(task, PC_model, EU_model, batch_size):
    PC_model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for _ in range(PC_N_EVAL_BATCHES):
            w_batch = task.generate_parameter_batch(batch_size)
            hyperparameter_batch = task.generate_hyperparameter_batch(batch_size)
            batch = task.generate_choice_batch(batch_size)
            x_f = task.utility(batch, w_batch)
            batch_obs = task.observe_batch(batch, x_f, hyperparameter_batch)
            utils = EU_model(batch_obs, w_batch, hyperparameter_batch)

            max_utils = torch.argmax(utils, dim=1)
            weights = torch.zeros((batch_size, task.n_choices), device=utils.device)
            for (i,c_i) in enumerate(max_utils.cpu().numpy()):
                weights[i,c_i] = 1.0
            eval_loss -= (PC_model(batch, w_batch, hyperparameter_batch) * weights).sum().item()

    PC_model.train()
    return eval_loss / (batch_size * PC_N_EVAL_BATCHES)

def _train_PC_model(task, EU_model, CHECKPOINT_SAVE_DIR, device = torch.device("cpu"), lr_start = EU_LR_START, lr_end = EU_LR_END, n_epochs = PC_N_EPOCHS, batch_size = PC_BATCH_SIZE, model_kwargs = {}):
    PC_model = AmortizedConditionalChoicePredictionModel(task.n_choices, task.n_attributes, task.parameter_dim + task.hyperparameter_dim, **model_kwargs).to(device=device)
    PC_model.train()

    wandb.config.task_name = task.__class__.__qualname__
    wandb.config.surrogate_type = "PC"
    wandb.config.checkpoint_dir = CHECKPOINT_SAVE_DIR
    wandb.config.control_latent_dim = PC_model.control_latent_dim
    wandb.config.main_sizes = PC_model.main_sizes
    wandb.config.controller_sizes = PC_model.controller_sizes
    wandb.config.dropout = PC_model.dropout_rate
    wandb.config.enc_dropout = PC_model.u_dropout_rate
    wandb.config.lr_start = lr_start
    wandb.config.lr_end = lr_end
    wandb.config.batch_size = batch_size

    optimizer = AdamW(PC_model.parameters(), lr=lr_start)
    scheduler = ExponentialLR(optimizer, np.exp(np.log(lr_end / lr_start) / n_epochs))
    accumulated_losses = []
    for epoch in (tqdm_epoch := tqdm.tqdm(range(n_epochs+1))):
        w_batch = task.generate_parameter_batch(batch_size)
        hyperparameter_batch = task.generate_hyperparameter_batch(batch_size)
        batch = task.generate_choice_batch(batch_size)
        x_f = task.utility(batch, w_batch)
        batch_obs = task.observe_batch(batch, x_f, hyperparameter_batch)
        utils = EU_model(batch_obs, w_batch, hyperparameter_batch)

        max_utils = torch.argmax(utils, dim=1)
        weights = torch.zeros((batch_size, task.n_choices), device=device)
        for (i,c_i) in enumerate(max_utils.cpu().numpy()):
            weights[i,c_i] = 1.0
        loss = -(PC_model(batch, w_batch, hyperparameter_batch) * weights).sum(dim=1).mean()
        accumulated_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch % 500 == 0) and (epoch != 0):
            wandb.log({"CPloss": loss.item(), "CPloss_val": _eval_PC_model(task, PC_model, EU_model, batch_size)})
            torch.save(PC_model.state_dict(), join(CHECKPOINT_SAVE_DIR, "PC_model.pth"))
            torch.save(model_kwargs, join(CHECKPOINT_SAVE_DIR, "model_kwargs.pth"))
        else:
            wandb.log({"CPloss": loss.item()})

        if epoch % 25 == 1:
            tqdm_epoch.set_description('Average Loss: {:5f}'.format(np.mean(accumulated_losses)))
            accumulated_losses = []

    torch.save(PC_model.state_dict(), join(CHECKPOINT_SAVE_DIR, "PC_model.pth"))
    torch.save(model_kwargs, join(CHECKPOINT_SAVE_DIR, "model_kwargs.pth"))
    return PC_model