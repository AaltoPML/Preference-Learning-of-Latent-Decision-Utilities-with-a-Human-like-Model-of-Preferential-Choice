import torch
from .task import AbstractTask
from math import factorial
import pickle
import pandas as pd

"""
Implements the district-smart choice task.

Features: ['hull', 'bbox', 'reock', 'polsby', 'sym_x', 'sym_y']

The prepared user experiment data for this task was reused from the implementation of:
    Kiran Tomlinson and Austin R. Benson. Learning Interpretable Feature Context Effects in Discrete Choice. KDD 2021. https://arxiv.org/abs/2009.03417
"""
class DistrictSmartTask(AbstractTask):
    def __init__(self, sigma_e_prior, tau_pv_prior, p_error_prior, device = torch.device("cpu"), normalize_weights = False, data_location = "district-smart"):
        self.sigma_e_prior = sigma_e_prior  # samples expected to be size (N)
        self.tau_pv_prior = tau_pv_prior    # samples expected to be size (N,6)
        self.p_error_prior = p_error_prior  # samples expected to be size (N)
        self.hyperparameter_dim = 8

        file = open(f"{data_location}/district-smart.pickle", "rb")
        data = pickle.load(file)
        all_choice_idxs = torch.vstack([data[1][2], data[2][2], data[3][2]])
        all_choice_sets = torch.vstack([data[1][3], data[2][3], data[3][3]])
        all_choices = torch.cat([data[1][5], data[2][5], data[3][5]])
        options = []
        option_idxs = []
        for i in range(all_choice_idxs.shape[0]):
            for j in range(2):
                if all_choice_idxs[i,j].item() not in option_idxs:
                    options.append(all_choice_sets[i,j,:])
                    option_idxs.append(all_choice_idxs[i,j].item())
        self.options = torch.stack(options)
        self.y = all_choices
        self.x = all_choice_sets

        self.ranking_folds = [] # contains 6 arrays (one per fold) of features, each sorted according to prefrences given by one group (fold) of participants in the original user study.
        ranking_data = pd.read_csv(f"{data_location}/ranking_data.csv")[['fold_id', 'district', 'pca']].set_index('fold_id')
        feature_data = pd.read_csv(f"{data_location}/feature_data.csv")[['district', 'hull', 'bbox', 'reock', 'polsby', 'sym_x', 'sym_y']].set_index('district')
        ranking_data = ranking_data.join(feature_data, on='district')
        for fold_str in ["set1.", "set2.", "set3.", "set4.", "set5.", "set6."]:
            fold = ranking_data[[(fold_str in idx) for idx in ranking_data.index.array]]
            self.ranking_folds.append(torch.tensor(fold.sort_values("pca")[['hull', 'bbox', 'reock', 'polsby', 'sym_x', 'sym_y']].to_numpy(), dtype=torch.float32, device=device))

        self.normalize_weights = normalize_weights
        self.n_choices = self.x.shape[1]
        self.n_rankings = factorial(self.n_choices)
        self.n_attributes = self.x.shape[2]
        self.device = device
        self.parameter_dim = self.x.shape[2]

        self.x_normalization_means = self.x.reshape(-1, self.n_attributes).mean(dim=0)
        self.x_normalization_stds = self.x.reshape(-1, self.n_attributes).std(dim=0)

        self.parameter_prior = torch.distributions.Normal(torch.zeros(self.n_attributes), torch.ones(self.n_attributes) * 1.0)

        # there are (self.n_choices ^ 2 + self.n_choices) / 2 comparisons for each attribute, of which there are n_attributes, and n_choices observations of utility
        self.observation_dim = (self.n_choices * (self.n_choices - 1) // 2) * self.n_attributes + self.n_choices
    
    def _normalize_data(self, x):
        return (x - self.x_normalization_means) / self.x_normalization_stds
    
    def normalize_choice_batch(self, x):
        return self._normalize_data(x.reshape((-1, self.n_attributes))).reshape(x.shape)

    def generate_choice_batch(self, batch_size, device = None):
        return self.options[torch.multinomial(torch.ones((batch_size, self.options.shape[0])), self.n_choices)].to(device=self.get_device(device))
        
    def generate_parameter_batch(self, batch_size, device = None):
        if self.normalize_weights:
            return torch.nn.functional.normalize(self.parameter_prior.sample((batch_size,)), p=2.0,dim=1).to(device=self.get_device(device))
        else:
            return self.parameter_prior.sample((batch_size,)).to(device=self.get_device(device))
    
    def generate_hyperparameter_batch(self, batch_size, device = None):
        sigma_e_params = self.sigma_e_prior.sample((batch_size,1))
        p_error_params = self.p_error_prior.sample((batch_size,1))
        tau_pv_params = self.tau_pv_prior.sample((batch_size,))

        return torch.cat([sigma_e_params, p_error_params, tau_pv_params], dim=1).to(self.get_device(device))
    
    def utility(self, x, w):
        """
        Calculates the utilities of a number of choices
        input:
            x: (batch_size, n_choices, n_attributes)
            w: (batch_size, parameter_dim)
        returns: (batch_size, n_choices)
        """
        NC = x.shape[1]
        x_flat = self._normalize_data(x.reshape((-1, self.n_attributes)))
        return torch.multiply(x_flat, w.repeat_interleave(NC, 0)).sum(dim=1).reshape(-1, NC)
        
    """
    input:
        hyperparameter_batch: (batch_size, hyperparameter_dim)
        expand_tau_pv: bool. Determines whether to expand tau_pv with 0s where there are binary attributes, or whether to return only the parameters for the continuous attributes contained in hyperparameter_batch.
    """
    def _parse_hyperparams(self, hyperparameter_batch):
        sigma_e_prior_batch = hyperparameter_batch[:,0]
        p_error_batch = hyperparameter_batch[:,1]
        tau_pv_batch = hyperparameter_batch[:,2:]
        return sigma_e_prior_batch, p_error_batch, tau_pv_batch

    def observe_batch(self, x, x_f, hyperparameter_batch, device = None):
        sigma_e_prior_batch, p_error_batch, tau_pv_batch = self._parse_hyperparams(hyperparameter_batch)

        batch_size = x.shape[0]
        e_obs = x_f + torch.multiply(torch.randn(x.shape[:2], device=self.get_device(device)), sigma_e_prior_batch[:,None].repeat([1,self.n_choices]))
        obs = [e_obs]
        for i in range(self.n_choices):
            for j in range(i+1,self.n_choices):
                for k in range(0, x.shape[2]):
                    a_obs = torch.zeros((batch_size))
                    a_obs[x[:,i,k] < x[:,j,k] - tau_pv_batch[:,k]] = -1.0
                    a_obs[x[:,i,k] > x[:,j,k] + tau_pv_batch[:,k]] = 1.0
                    random_mask = torch.bernoulli(p_error_batch)
                    random_obs = torch.multinomial(torch.tensor([1/3, 1/3, 1/3]), batch_size, replacement=True) - 1.0
                    a_obs = a_obs * (1.0 - random_mask) + random_mask * random_obs
                    obs.append(a_obs[...,None].to(self.get_device(device)))
        return torch.cat(obs, dim=1)
    
    """
    Calculate's Kendall's Tau between rankings collected in original user experiment, and rankings implied by weights w
    input:
        w [batch_size, n_parameters] OR [n_parameters]
    Returns:
        Either vector of kendall's taus of size [batch_size] or single number, depending on shape of w.
    
    REFERENCE: M. Kendall. A new measure of rank correlation. Biometrica, 30(1–2):81–89, 1938.
    """
    def calculate_ground_truth_rank_consistency(self, w):
        flatten_result = False
        if len(w.shape) == 1:
            w = w[None,:]
            avg_kendall_tau_batch = torch.zeros(1, device=w.device)
            flatten_result = True
        else:
            avg_kendall_tau_batch = torch.zeros(w.shape[0], device=w.device)
        for w_idx in range(w.shape[0]):
            for fold in self.ranking_folds:
                utils = self.utility(fold[:,None,:], w[[w_idx],:].repeat(fold.shape[0],1))
                n_correct_rels = 0
                for i in range(fold.shape[0]-1):
                    n_correct_rels += (utils[i+1:] < utils[i]).sum()
                avg_kendall_tau_batch[w_idx] += (4*n_correct_rels) / (utils.shape[0] * (utils.shape[0]-1)) - 1
        avg_kendall_tau_batch /= len(self.ranking_folds)
        if flatten_result:
            return avg_kendall_tau_batch[0]
        return avg_kendall_tau_batch


class DistrictSmartLargeTask(DistrictSmartTask):
    def __init__(self, sigma_e_prior, tau_pv_prior, p_error_prior, device = torch.device("cpu"), normalize_weights = False, data_location = "district-smart"):
        super().__init__(sigma_e_prior, tau_pv_prior, p_error_prior, device = device, normalize_weights = normalize_weights, data_location = data_location)

        self.options = torch.tensor(pd.read_csv(f"{data_location}/feature_data.csv")[['hull', 'bbox', 'reock', 'polsby', 'sym_x', 'sym_y']].to_numpy(), dtype=torch.float32)

        self.x_normalization_means = self.options.mean(dim=0)
        self.x_normalization_stds = self.options.std(dim=0)