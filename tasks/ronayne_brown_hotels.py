import torch
from .task import AbstractTask
from math import factorial
import pandas as pd

class RonayneBrownHotelsTask(AbstractTask):
    def __init__(self, sigma_e_prior, tau_pv_prior, p_error_prior, device = torch.device("cpu"), normalize_weights = False, data_location = "ronayne_brown_hotels"):
        self.sigma_e_prior = sigma_e_prior  # samples expected to be size (N)
        self.tau_pv_prior = tau_pv_prior    # samples expected to be size (N,2)
        self.p_error_prior = p_error_prior  # samples expected to be size (N)
        self.hyperparameter_dim = 4

        self.normalize_weights = normalize_weights
        self.n_choices = 3
        self.n_rankings = factorial(self.n_choices)
        self.n_attributes = 2
        self.device = device
        self.parameter_dim = 2

        self.hotel_data = torch.tensor(pd.read_excel(f"{data_location}/data.xlsx", sheet_name="Hotels - June hotels.com", usecols=["price","rating"]).to_numpy(), dtype=torch.float32)
        self.hotel_data_means = self.hotel_data.mean(dim=0)
        self.hotel_data_stds = self.hotel_data.std(dim=0)

        conditions = torch.tensor([[[125, 3.6], [159, 3.3], [249, 4.4]],
                                   [[125, 3.6], [249, 4.4], [278, 4.1]],
                                   [[130, 2.9], [179, 3.5], [233, 4.0]],
                                   [[179, 3.5], [233, 4.0], [287, 4.5]],
                                   [[194, 3.5], [231, 4.1], [239, 4.2]],
                                   [[194, 3.5], [199, 3.6], [239, 4.2]]])
        df = pd.read_excel(f"{data_location}/data.xlsx", sheet_name="Plotting data-export to MATLAB", usecols=["cond","choice"])
        control_group = (df.cond <= 6)
        exp_choice = torch.tensor(df.choice[control_group].to_numpy() - 1, dtype=torch.int32)
        exp_condition = torch.tensor(df.cond[control_group].to_numpy() - 1, dtype=torch.int32)
        self.x = conditions[exp_condition,:,:]
        self.y = exp_choice

        self.parameter_prior = torch.distributions.Normal(torch.zeros(self.n_attributes), torch.ones(self.n_attributes) * 5.0)

        # there are (self.n_choices ^ 2 + self.n_choices) / 2 comparisons for each attribute, of which there are n_attributes, and n_choices observations of utility
        self.observation_dim = (self.n_choices * (self.n_choices - 1) // 2) * self.n_attributes + self.n_choices
    
    def _normalize_data(self, x):
        return (x - self.hotel_data_means) / self.hotel_data_stds
    
    def normalize_choice_batch(self, x):
        return self._normalize_data(x.reshape((-1, self.n_attributes))).reshape(x.shape)

    def generate_choice_batch(self, batch_size, device = None):
        return torch.vstack([self.hotel_data[None, torch.multinomial(torch.ones(self.hotel_data.shape[0]), self.n_choices), :] for _ in range(batch_size)]).to(device=self.get_device(device))
    
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
