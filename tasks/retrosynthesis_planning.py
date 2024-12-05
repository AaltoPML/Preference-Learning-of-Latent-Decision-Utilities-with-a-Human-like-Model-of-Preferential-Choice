import torch
from .task import AbstractTask
from math import factorial
import json
import random

class RetrosynthesisPlanningTask(AbstractTask):
    def __init__(self, n_choices, sigma_e_prior, tau_pv_prior, p_error_prior, device = torch.device("cpu"), scalarization = "Chebyshev", data_location = "car_crash"):
        self.sigma_e_prior = sigma_e_prior  # samples expected to be size (N)
        self.tau_pv_prior = tau_pv_prior    # samples expected to be size (N)
        self.p_error_prior = p_error_prior  # samples expected to be size (N)
        self.hyperparameter_dim = 3

        self.n_choices = n_choices
        self.n_rankings = factorial(n_choices)
        self.n_attributes = 6
        self.device = device
        self.parameter_dim = 6
        self.scalarization = scalarization

        self.parameter_prior = torch.distributions.Dirichlet(torch.ones(self.parameter_dim))

        # there are (self.n_choices ^ 2 + self.n_choices) / 2 comparisons for each attribute, of which there are n_attributes, and n_choices observations of utility
        self.observation_dim = (self.n_choices * (self.n_choices - 1) // 2) * self.n_attributes + self.n_choices

        # min and max AFTER transformation with self.sign
        # self.sign = torch.tensor( [-1.0, -1.0,  1.0, 1.0,  -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0]).to(device)
        # self.f_min = torch.tensor([-25.0, -13.0,  0.0, 0.0, -13.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]).to(device)
        # self.f_max = torch.tensor([0.0,  0.0, 14.0, 6.0,  -1.0, 1.0, 1.0, 6.0, 6.0, 1.0,  0.0]).to(device)

        self.sign = torch.tensor( [-1.0, -1.0,  1.0, 1.0,   1.0, -1.0]).to(device)
        self.f_min = torch.tensor([-25.0, -13.0,  0.0, 0.0,  0.0,  -1.0]).to(device)
        self.f_max = torch.tensor([0.0,  0.0, 14.0, 6.0,   1.0,  0.0]).to(device)

        with open(f"{data_location}/route_features_small_6.json") as fp:
            self.routes_DB = json.load(fp)
        
        bad_keys = []
        for target in self.routes_DB.keys():
            if len(self.routes_DB[target]) < self.n_choices:
                bad_keys.append(target)
        for target in bad_keys:
            del self.routes_DB[target]
    
    def _normalize_outcomes(self, x):
        return (self.sign * x - self.f_min) / (self.f_max - self.f_min)

    def generate_choice_batch(self, batch_size, device = None):
        x = torch.empty((batch_size, self.n_choices, self.n_attributes), device=self.get_device(device))
        for i in range(batch_size):
            target = random.choice(list(self.routes_DB.keys()))
            route_IDs = torch.randperm(len(self.routes_DB[target]))[:self.n_choices]
            for (j,ID) in enumerate(route_IDs):
                x[i,j,:] = torch.tensor(self.routes_DB[target][ID], device=self.get_device(device))

        return self._normalize_outcomes(x.reshape((batch_size * self.n_choices, self.n_attributes))).reshape((batch_size, self.n_choices, self.n_attributes))
    
    def generate_parameter_batch(self, batch_size, device = None):
        return self.parameter_prior.sample((batch_size,)).to(device=self.get_device(device))
    
    def generate_hyperparameter_batch(self, batch_size, device = None):
        sigma_e_params = self.sigma_e_prior.sample((batch_size,1))
        p_error_params = self.p_error_prior.sample((batch_size,1))
        tau_pv_params = self.tau_pv_prior.sample((batch_size,1))

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
        x_flat = x.reshape((-1, self.n_attributes))
        if self.scalarization == "WeightedSum":
            return - torch.multiply(x_flat, w.repeat_interleave(NC, 0)).sum(dim=1).reshape(-1, NC)
        elif self.scalarization == "Chebyshev":
            return - torch.multiply(x_flat.abs(), w.repeat_interleave(NC, 0)).max(dim=1).values.reshape(-1, NC)
        else:
            raise NotImplementedError
    
    def _parse_hyperparams(self, hyperparameter_batch):
        sigma_e_prior_batch = hyperparameter_batch[:,0]
        p_error_batch = hyperparameter_batch[:,1]
        tau_pv_batch = hyperparameter_batch[:,2]
        return sigma_e_prior_batch, p_error_batch, tau_pv_batch

    def observe_batch(self, x, x_f, hyperparameter_batch, device = None):
        sigma_e_prior_batch, p_error_batch, tau_pv_batch = self._parse_hyperparams(hyperparameter_batch)

        batch_size = x.shape[0]
        e_obs = x_f + torch.multiply(torch.randn(x.shape[:2], device=self.get_device(device)), sigma_e_prior_batch[:,None].repeat([1,self.n_choices]))
        obs = [e_obs]
        for i in range(self.n_choices):
            for j in range(i+1,self.n_choices):
                for k in range(0, x.shape[2]):
                    a_obs = torch.zeros((batch_size)).to(device)
                    a_obs[x[:,i,k] < x[:,j,k] - tau_pv_batch] = -1.0
                    a_obs[x[:,i,k] > x[:,j,k] + tau_pv_batch] = 1.0
                    random_mask = torch.bernoulli(p_error_batch).to(device)
                    random_obs = torch.multinomial(torch.tensor([1/3, 1/3, 1/3]), batch_size, replacement=True) - 1.0
                    random_obs = random_obs.to(device)
                    a_obs = a_obs * (1.0 - random_mask) + random_mask * random_obs
                    obs.append(a_obs[...,None].to(self.get_device(device)))
        return torch.cat(obs, dim=1)
