import torch
from .task import AbstractTask
from .user_study_simulator import AbstractElicitationStudySimulator
from math import factorial
import numpy as np
from scipy.io import loadmat

"""
Implements the property choice task from Dumbalska et al:
    Dumbalska, Tsvetomira, et al. "A map of decoy influence in human multialternative choice." Proceedings of the National Academy of Sciences 117.40 (2020): 25169-25178.

The attributes of the choices presented here are [cost, rate] where cost is the asking rent price presented to the participant in phase 2, and rate is the average rental rate estimate that the participant assigned the property in phase 1.
"""
class DumbalskaPropertyTask(AbstractTask):
    def __init__(self, sigma_e_prior, tau_pv_prior, p_error_prior, device = torch.device("cpu"), normalize_weights = False, data_location = "dumbalska"):
        self.sigma_e_prior = sigma_e_prior  # samples expected to be size (N)
        self.tau_pv_prior = tau_pv_prior    # samples expected to be size (N,2)
        self.p_error_prior = p_error_prior  # samples expected to be size (N)
        self.hyperparameter_dim = 4

        self.data_location = data_location
        data = loadmat(f"{self.data_location}/decoy_233_participants.mat", squeeze_me=True)['data']

        Acost = data["Acost"].item().flatten()
        Arate = data["Arate"].item().flatten()
        Bcost = data["Bcost"].item().flatten()
        Brate = data["Brate"].item().flatten()
        Dcost = data["Dcost"].item().flatten()
        Drate = data["Drate"].item().flatten()
        not_nan_locs = np.logical_not(np.isnan(Acost))
        A = np.concatenate([Acost[not_nan_locs,None,None], Arate[not_nan_locs,None,None]],2)
        B = np.concatenate([Bcost[not_nan_locs,None,None], Brate[not_nan_locs,None,None]],2)
        D = np.concatenate([Dcost[not_nan_locs,None,None], Drate[not_nan_locs,None,None]],2)
        x = np.concatenate([A,B,D], 1)
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(np.argmin(data["choice"].item(), 1).flatten()[not_nan_locs], dtype=torch.int32)

        self.normalize_weights = normalize_weights
        self.n_choices = self.x.shape[1]
        self.n_rankings = factorial(self.n_choices)
        self.n_attributes = self.x.shape[2]
        self.device = device
        self.parameter_dim = self.x.shape[2]

        self.x_normalization_means = self.x.reshape(-1, self.n_attributes).mean(dim=0).to(device=device)
        self.x_normalization_stds = self.x.reshape(-1, self.n_attributes).std(dim=0).to(device=device)

        self.parameter_prior = torch.distributions.Normal(torch.zeros(self.n_attributes), torch.ones(self.n_attributes) * 1.0)

        # there are (self.n_choices ^ 2 + self.n_choices) / 2 comparisons for each attribute, of which there are n_attributes, and n_choices observations of utility
        self.observation_dim = (self.n_choices * (self.n_choices - 1) // 2) * self.n_attributes + self.n_choices

        # the prior for the options is a mixture of a uniform distrbiution over the domain, and a standard normal with means representative for the options presented in the experiment and a covariance matrix that is a broader version of the covariances of the options presented in the original experiment.
        self.option_prior = torch.distributions.MultivariateNormal(loc=self.x_normalization_means.cpu(), covariance_matrix=torch.tensor([[210000.0, 150000.0], [150000.0, 180000.0]], dtype=torch.float32))
        self.option_prior_uniform = torch.distributions.Uniform(low=torch.tensor([0.0, 0.0], dtype=torch.float32), high=torch.tensor([2500.0, 2500.0], dtype=torch.float32))
        self.prior_uniform_weight = 0.2
    
    def _normalize_data(self, x):
        return (x - self.x_normalization_means) / self.x_normalization_stds
    
    def normalize_choice_batch(self, x, device = None):
        return self._normalize_data(x.reshape((-1, self.n_attributes))).reshape(x.shape).to(self.get_device(device))

    def generate_choice_batch(self, batch_size, device = None):
        batch = self.option_prior.sample((self.n_choices*batch_size,))
        options_to_reject = torch.any((batch < 0.0).bool(), dim=1) # options with < 0 values will be removed by overwriting them with a uniformly generated option.
        uniform_options = torch.logical_or(torch.bernoulli(torch.ones(batch.shape[0]) * self.prior_uniform_weight), options_to_reject) # replace some of the options with a uniformly generated option
        batch[uniform_options] = self.option_prior_uniform.sample((uniform_options.sum(),))
        return batch.reshape((batch_size, self.n_choices, -1)).to(self.get_device(device))
        
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
                    random_mask = torch.bernoulli(p_error_batch.cpu())
                    random_obs = torch.multinomial(torch.tensor([1/3, 1/3, 1/3]), batch_size, replacement=True) - 1.0
                    a_obs = a_obs * (1.0 - random_mask) + random_mask * random_obs
                    obs.append(a_obs[...,None].to(self.get_device(device)))
        return torch.cat(obs, dim=1)

class DumbalskaUserStudySimulator(AbstractElicitationStudySimulator):
    def __init__(self, task, participantID = None):
        data = loadmat(f"{task.data_location}/decoy_233_participants.mat", squeeze_me=True)['data']

        self.n_participants = data["Acost"].item().shape[0]
        if participantID is None:
            participantID = torch.randint(0, self.n_participants, (1,)).item()
        assert (participantID < self.n_participants) and (participantID >= 0)

        super().__init__(participantID, task.device)

        self.n_choices = task.n_choices
        self.n_attributes = task.n_attributes

        pAcost = data["Acost"].item()[self.participantID,:]
        not_nan_locs = np.logical_not(np.isnan(data["Acost"].item()[self.participantID,:]))
        pAcost = data["Acost"].item()[self.participantID,not_nan_locs, None, None]
        pArate = data["Arate"].item()[self.participantID,not_nan_locs, None, None]
        pBcost = data["Bcost"].item()[self.participantID,not_nan_locs, None, None]
        pBrate = data["Brate"].item()[self.participantID,not_nan_locs, None, None]
        pDcost = data["Dcost"].item()[self.participantID,not_nan_locs, None, None]
        pDrate = data["Drate"].item()[self.participantID,not_nan_locs, None, None]
        self.recorded_queries = torch.tensor(np.concatenate([np.concatenate([pAcost, pArate], 2), np.concatenate([pBcost, pBrate], 2), np.concatenate([pDcost, pDrate], 2)], 1), dtype=torch.float32, device = self.device)
        self.recorded_rankings = torch.tensor(data["choice"].item()[participantID,:,not_nan_locs], dtype=torch.int64, device = self.device)
        self.recorded_choices = self.recorded_rankings[:,0]
        self.query_available = torch.ones((self.recorded_queries.shape[0],), dtype=torch.bool, device = self.device)

        # rate and cost bin IDs for each decoy in self.recorded_queries
        self.rate_bin_IDs = np.minimum(np.argmax(np.cumsum(data["Drate"].item()[participantID,:,None] - data["ratebin"].item()[None,participantID,:], axis=1), axis=1), 9)
        self.cost_bin_IDs = np.argmax(np.cumsum(data["Dcost"].item()[participantID,:,None] - data["costbin"].item()[None,participantID,::-1], axis=1), axis=1)

        # participants that were excluded from the original analysis due to poor performance. This criterion follows line 12 of decoy_make_figures.m form the original code repository.
        self.excluded_participants = torch.arange(0,233,1,dtype=torch.int32)[data["sig_sub"].item() < 0.99]

    def get_available_queries(self, device = None):
        return self.recorded_queries[self.query_available].to(device = self.get_device(device))

    def simulate_choice(self, query_index):
        choice = self.recorded_choices[self.query_available][query_index].item()
        original_index = self.get_original_index(query_index)
        self.query_available[original_index] = False
        return choice

    def get_evaluation_data(self, device = None):
        return self.recorded_queries[self.query_available].to(device = self.get_device(device)), self.recorded_choices[self.query_available].to(device = self.get_device(device))
    
    def get_training_data(self, device = None):
        return self.recorded_queries[torch.logical_not(self.query_available)].to(device = self.get_device(device)), self.recorded_choices[torch.logical_not(self.query_available)].to(device = self.get_device(device))
    
    def get_original_index(self, train_idx):
        return torch.arange(0, self.query_available.shape[0], 1, device = self.device)[self.query_available][train_idx]