import torch
from torch.distributions import beta, Normal, studentT
from .task import AbstractTask
from math import factorial

class StandardizedRiskyChoiceTask(AbstractTask):
    def __init__(self, n_choices, sigma_e_prior, tau_pv_prior, p_error_prior, device = torch.device("cpu"), scalarization = None):
        self.sigma_e_prior = sigma_e_prior  # samples expected to be size (N)
        self.tau_pv_prior = tau_pv_prior    # samples expected to be size (N)
        self.p_error_prior = p_error_prior  # samples expected to be size (N)
        self.hyperparameter_dim = 3

        self.n_choices = n_choices
        self.n_rankings = factorial(n_choices)
        self.n_attributes = 2
        self.device = device
        self.parameter_dim = 1

        self.p_prior = beta.Beta(1.0, 1.0)
        self.v_prior = Normal(0.0, 1.0)

        # there are (self.n_choices ^ 2 + self.n_choices) / 2 comparisons for each attribute, of which there are n_attributes, and n_choices observations of utility
        self.observation_dim = (self.n_choices * (self.n_choices - 1) // 2) * self.n_attributes + self.n_choices

        self.f_min = torch.tensor([6.3840e+04, 3.0000e+01, 2.8535e+05, 1.8375e+05, 7.2222e+00, -9.7763e+04])
        self.f_max = torch.tensor([8.3061e+04, 1.3500e+03, 2.8535e+06, 1.6028e+07, 3.5785e+05, 7.5046e+04])

        self.pareto_front = None

    def generate_choice_batch(self, batch_size, device = None):
        return torch.cat((self.p_prior.sample((batch_size,self.n_choices,1)), self.v_prior.sample((batch_size,self.n_choices,1))), dim=2).to(self.get_device(device))
    
    def generate_parameter_batch(self, batch_size, device = None):
        # there are actually no parameters, so each parameter is simply 0 (so that the rest of the code still works)
        return torch.zeros((batch_size, 1), device=self.get_device(device))
    
    def generate_hyperparameter_batch(self, batch_size, device = None):
        sigma_e_params = self.sigma_e_prior.sample((batch_size,1))
        p_error_params = self.p_error_prior.sample((batch_size,1))
        tau_pv_params = self.tau_pv_prior.sample((batch_size,1))

        return torch.cat([sigma_e_params, p_error_params, tau_pv_params], dim=1).to(self.get_device(device))
    
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
                    a_obs = torch.zeros((batch_size))
                    a_obs[x[:,i,k] < x[:,j,k] - tau_pv_batch] = -1.0
                    a_obs[x[:,i,k] > x[:,j,k] + tau_pv_batch] = 1.0
                    random_mask = torch.bernoulli(p_error_batch)
                    random_obs = torch.multinomial(torch.tensor([1/3, 1/3, 1/3]), batch_size, replacement=True) - 1.0
                    a_obs = a_obs * (1.0 - random_mask) + random_mask * random_obs
                    obs.append(a_obs[...,None].to(self.get_device(device)))
        return torch.cat(obs, dim=1)
    
    def utility(self, x, w):
        """
        Calculates the utilities of a number of choices
        input:
            x: (batch_size, n_choices, n_attributes)
            w: (batch_size, parameter_dim)
        returns: (batch_size, n_choices)
        """
        return x.prod(dim=2)

class Howes2016RiskyChoiceTask(AbstractTask):
    def __init__(self, n_choices, sigma_e_prior, tau_pv, p_error, device = torch.device("cpu")):
        self.sigma_e_prior = sigma_e_prior  # samples expected to be size (N)
        self.hyperparameter_dim = 1

        self.n_choices = n_choices
        self.n_attributes = 2
        self.tau_pv = tau_pv
        self.p_error = p_error
        self.device = device
        self.parameter_dim = 1

        self.p_prior = beta.Beta(1.0, 1.0)
        self.v_prior = studentT.StudentT(100.0, 19.6, 8.0)

        # there are (self.n_choices ^ 2 + self.n_choices) / 2 comparisons for each attribute, of which there are n_attributes, and n_choices observations of utility
        self.observation_dim = (self.n_choices * (self.n_choices - 1) // 2) * self.n_attributes + self.n_choices

    def generate_choice_batch(self, batch_size, device = None):
        return torch.cat((self.p_prior.sample((batch_size,self.n_choices,1)), self.v_prior.sample((batch_size,self.n_choices,1))), dim=2).to(self.get_device(device))
    
    def generate_decoy_choice_batch(self, batch_size, alpha = 0.2):
        """
        Generate two choices with an RF decoy close to one choice. The location of the decoy within the list of choices is shuffled.

        This was used to test if training on choices with RF decoys was better, but it turned out it wasn't.
        """
        c1 = torch.cat((self.p_prior.sample((batch_size,1,1)), self.v_prior.sample((batch_size,1,1))), dim = 2)
        c2p = self.p_prior.sample((batch_size,1,1))
        c2 = torch.cat((c2p, c1.prod(dim=2)[:, None, :] / c2p), dim=2)

        # make sure c3 is an attractor for c1; i.e. is close to c1, is dominated by c1, and dominates c2 in the same dimension as c1 dominates c2.
        c1_p_greater = (c1[:,:,:1] >= c2[:,:,:1]).to(dtype=torch.int)
        c3p_p_greater = (torch.rand(batch_size,1,1) * alpha + 1.0 - alpha) * (c1[:,:,:1] - c2[:,:,:1]) + c2[:,:,:1]
        c3v_p_greater = (torch.rand(batch_size,1,1) * alpha + 1.0 - alpha) * c1[:,:,1:]
        c3p_v_greater = (torch.rand(batch_size,1,1) * alpha + 1.0 - alpha) * c1[:,:,:1]
        c3v_v_greater = (torch.rand(batch_size,1,1) * alpha + 1.0 - alpha) * (c1[:,:,1:] - c2[:,:,1:]) + c2[:,:,1:]
        c3 = torch.zeros_like(c1)
        c3 += torch.cat((c3p_p_greater, c3v_p_greater), dim = 2) * c1_p_greater.repeat((1,1,2))
        c3 += torch.cat((c3p_v_greater, c3v_v_greater), dim = 2) * (1 - c1_p_greater.repeat((1,1,2)))

        assert self.n_choices == 3
        tensors = [c1, c2, c3]
        import random
        random.shuffle(tensors)
        return torch.cat(tensors, dim = 1).to(self.device)
    
    def generate_parameter_batch(self, batch_size, device = None):
        # there are actually no parameters, so each parameter is simply 0 (so that the rest of the code still works)
        return torch.zeros((batch_size, 1), device=self.get_device(device))
    
    def generate_hyperparameter_batch(self, batch_size, device = None):
        return self.sigma_e_prior.sample((batch_size,1))

    def observe_batch(self, x, x_f, hyperparameter_batch, device = None):
        batch_size = x.shape[0]
        e_obs = x_f + torch.multiply(torch.randn(x.shape[:2], device=self.get_device(device)), hyperparameter_batch.repeat([1,self.n_choices]))
        obs = [e_obs]
        for i in range(self.n_choices):
            for j in range(i+1,self.n_choices):
                for k in range(self.n_attributes):
                    a_obs = torch.zeros((batch_size))
                    a_obs[x[:,i,k] < x[:,j,k] - self.tau_pv[k]] = -1.0
                    a_obs[x[:,i,k] > x[:,j,k] + self.tau_pv[k]] = 1.0
                    random_mask = torch.bernoulli(torch.ones(batch_size) * self.p_error)
                    random_obs = torch.multinomial(torch.tensor([1/3, 1/3, 1/3]), batch_size, replacement=True) - 1.0
                    a_obs = a_obs * (1.0 - random_mask) + random_mask * random_obs
                    obs.append(a_obs[...,None].to(self.get_device(device)))
        return torch.cat(obs, dim=1)
    
    def utility(self, x, w):
        """
        Calculates the utilities of a number of choices
        input:
            x: (batch_size, n_choices, n_attributes)
            w: (batch_size, parameter_dim)
        returns: (batch_size, n_choices)
        """
        return x.prod(dim=2)
    
def generate_decoy_choice_batch(task, w, alpha = 0.1):
    batch_size = w.shape[0]

    c1 = torch.cat((task.p_prior.sample((batch_size,1,1)), task.v_prior.sample((batch_size,1,1))), dim = 2)
    c2p = task.p_prior.sample((batch_size,1,1))
    c2 = torch.cat((c2p, c1.prod(dim=2)[:, None, :] / c2p), dim=2)

    # make sure c3 is an attractor for c1; i.e. is close to c1, is dominated by c1, and dominates c2 in the same dimension as c1 dominates c2.
    c1_p_greater = (c1[:,:,:1] >= c2[:,:,:1]).to(dtype=torch.int)
    c3p_p_greater = (torch.rand(batch_size,1,1) * alpha + 1.0 - alpha) * (c1[:,:,:1] - c2[:,:,:1]) + c2[:,:,:1]
    c3v_p_greater = (torch.rand(batch_size,1,1) * alpha + 1.0 - alpha) * c1[:,:,1:]
    c3p_v_greater = (torch.rand(batch_size,1,1) * alpha + 1.0 - alpha) * c1[:,:,:1]
    c3v_v_greater = (torch.rand(batch_size,1,1) * alpha + 1.0 - alpha) * (c1[:,:,1:] - c2[:,:,1:]) + c2[:,:,1:]
    c3 = torch.zeros_like(c1)
    c3 += torch.cat((c3p_p_greater, c3v_p_greater), dim = 2) * c1_p_greater.repeat((1,1,2))
    c3 += torch.cat((c3p_v_greater, c3v_v_greater), dim = 2) * (1 - c1_p_greater.repeat((1,1,2)))

    c2_p_greater = (c2[:,:,:1] >= c1[:,:,:1]).to(dtype=torch.int)
    c4p_p_greater = (torch.rand(batch_size,1,1) * alpha + 1.0 - alpha) * (c2[:,:,:1] - c1[:,:,:1]) + c1[:,:,:1]
    c4v_p_greater = (torch.rand(batch_size,1,1) * alpha + 1.0 - alpha) * c2[:,:,1:]
    c4p_v_greater = (torch.rand(batch_size,1,1) * alpha + 1.0 - alpha) * c2[:,:,:1]
    c4v_v_greater = (torch.rand(batch_size,1,1) * alpha + 1.0 - alpha) * (c2[:,:,1:] - c1[:,:,1:]) + c1[:,:,1:]
    c4 = torch.zeros_like(c1)
    c4 += torch.cat((c4p_p_greater, c4v_p_greater), dim = 2) * c2_p_greater.repeat((1,1,2))
    c4 += torch.cat((c4p_v_greater, c4v_v_greater), dim = 2) * (1 - c2_p_greater.repeat((1,1,2)))

    return torch.cat((c1, c2, c3, c4), dim = 1).to(task.device)