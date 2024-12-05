import torch
from .task import AbstractTask
from math import factorial
import pandas as pd

# original feature layout (each feature occurs 6 times in a row, once for each option in the choice set):
#  0. Price divided by ln(income)
#  1. Range
#  2. Acceleration
#  3. Top speed
#  4. Pollution
#  5. Size [categorical: 0, 1, 2, 3]
#  6. "Big enough" [binary]
#  7. Luggage space
#  8. Operating cost
#  9. Station availability
# 10. Sports utility vehicle [binary]
# 11. Sports car [binary]
# 12. Station wagon [binary]
# 13. Truck [binary]
# 14. Van [binary]
# 15. Constant for EV [binary]
# 16. Commute < 5 x EV [binary]
# 17. College x EV [binary]
# 18. Constant for CNG [binary]
# 19. Constant for methanol [binary]
# 20. College x methanol [binary]
# 21. ___dummy___
# 22. Dependent variable. 1 for chosen vehicle, 0 otherwise [binary]
# 23. ___dummy___
# 24. ___dummy___
# 25. ___dummy___

# extracted features:
#  0. Price divided by ln(income)
#  1. Range
#  2. Acceleration
#  3. Top speed
#  4. Pollution
#  5. size=mini [binary]
#  6. size=subcompact [binary]
#  7. size=compact [binary]
#  8. size=mid-size [binary]
#  9. "Big enough" [binary]
# 10. Luggage space
# 11. Operating cost
# 12. Station availability
# 13. Sports utility vehicle [binary]
# 14. Sports car [binary]
# 15. Station wagon [binary]
# 16. Truck [binary]
# 17. Van [binary]
# 18. Constant for EV [binary]
# 19. Commute < 5 x EV [binary]
# 20. College x EV [binary]
# 21. Constant for CNG [binary]
# 22. Constant for methanol [binary]
# 23. College x methanol [binary]

"""
Implements the car-alt task
There are 8 tau_pv parameters, one for each continuous feature. We do not model this for binary features.
"""
class CarAltTask(AbstractTask):
    def __init__(self, sigma_e_prior, tau_pv_prior, p_error_prior, device = torch.device("cpu"), normalize_weights = False, data_location = "car-alt"):
        self.sigma_e_prior = sigma_e_prior  # samples expected to be size (N)
        self.tau_pv_prior = tau_pv_prior    # samples expected to be size (N,8)
        self.p_error_prior = p_error_prior  # samples expected to be size (N)
        self.hyperparameter_dim = 10

        # extract data from file
        f = open(f"{data_location}/xmat.txt", "r")
        matrix = [[float(i) for i in l.split()] for l in f.readlines()]
        raw_data = torch.tensor(matrix).reshape((4654,26,6))
        choice_batch_part1 = torch.stack([raw_data[:,i,:] for i in range(5)],dim=2)
        choice_batch_part2 = torch.nn.functional.one_hot(raw_data[:,5,:].to(dtype=torch.int64), 4)
        choice_batch_part3 = torch.stack([raw_data[:,i,:] for i in range(6,21)],dim=2)
        self.x = torch.cat([choice_batch_part1, choice_batch_part2, choice_batch_part3], dim=2)
        self.y = raw_data[:, 22, :].argmax(dim=1)

        self.normalize_weights = normalize_weights
        self.n_choices = self.x.shape[1]
        self.n_rankings = factorial(self.n_choices)
        self.n_attributes = self.x.shape[2]
        self.device = device
        self.parameter_dim = self.x.shape[2]

        binary_mask = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.continuous_idxs = torch.tensor([i for i in range(self.n_attributes) if binary_mask[i] == 0])
        self.binary_idxs = torch.tensor([i for i in range(self.n_attributes) if binary_mask[i] == 1])

        self.x_normalization_means = torch.zeros(self.n_attributes)
        self.x_normalization_means[self.continuous_idxs] = self.x.reshape(-1, self.n_attributes).mean(dim=0)[self.continuous_idxs]
        self.x_normalization_stds = torch.ones(self.n_attributes)
        self.x_normalization_stds[self.continuous_idxs] = self.x.reshape(-1, self.n_attributes).std(dim=0)[self.continuous_idxs]

        self.parameter_prior = torch.distributions.Normal(torch.zeros(self.n_attributes), torch.ones(self.n_attributes) * 5.0)

        # there are (self.n_choices ^ 2 + self.n_choices) / 2 comparisons for each attribute, of which there are n_attributes, and n_choices observations of utility
        self.observation_dim = (self.n_choices * (self.n_choices - 1) // 2) * self.n_attributes + self.n_choices
    
    def _normalize_data(self, x):
        return (x - self.x_normalization_means) / self.x_normalization_stds
    
    def normalize_choice_batch(self, x):
        return self._normalize_data(x.reshape((-1, self.n_attributes))).reshape(x.shape)

    def generate_choice_batch(self, batch_size, device = None):
        return torch.cat([self.x[None,torch.randint(0, self.x.shape[0], (1,)).item(),torch.randperm(6),:] for _ in range(batch_size)], dim=0).to(device=self.get_device(device))
        
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
    def _parse_hyperparams(self, hyperparameter_batch, expand_tau_pv = False):
        sigma_e_prior_batch = hyperparameter_batch[:,0]
        p_error_batch = hyperparameter_batch[:,1]
        if expand_tau_pv:
            tau_pv_batch = torch.zeros((hyperparameter_batch.shape[0], self.n_attributes), dtype=sigma_e_prior_batch.dtype)
            tau_pv_batch[:,self.continuous_idxs] = hyperparameter_batch[:,2:]
        else:
            tau_pv_batch = hyperparameter_batch[:,2:]
        return sigma_e_prior_batch, p_error_batch, tau_pv_batch

    def observe_batch(self, x, x_f, hyperparameter_batch, device = None):
        sigma_e_prior_batch, p_error_batch, tau_pv_batch = self._parse_hyperparams(hyperparameter_batch, expand_tau_pv=True)

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
