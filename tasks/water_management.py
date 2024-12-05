import torch
from .task import AbstractTask
from math import factorial

class WaterManagementMOOTask(AbstractTask):
    def __init__(self, n_choices, sigma_e_prior, tau_pv_prior, p_error_prior, device = torch.device("cpu"), scalarization = "Chebyshev", data_location = "water_management"):
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

        self.x_prior = torch.distributions.Uniform(torch.tensor([0.01, 0.01, 0.01]), torch.tensor([0.45, 0.1, 0.1]))
        self.parameter_prior = torch.distributions.Dirichlet(torch.ones(self.parameter_dim))

        # there are (self.n_choices ^ 2 + self.n_choices) / 2 comparisons for each attribute, of which there are n_attributes, and n_choices observations of utility
        self.observation_dim = (self.n_choices * (self.n_choices - 1) // 2) * self.n_attributes + self.n_choices

        self.f_min = torch.tensor([6.3840e+04, 3.0000e+01, 2.8535e+05, 1.8375e+05, 7.2222e+00, -9.7763e+04])
        self.f_max = torch.tensor([8.3061e+04, 1.3500e+03, 2.8535e+06, 1.6028e+07, 3.5785e+05, 7.5046e+04])

        self.data_location = data_location
        try:
            self.pareto_front = torch.load(f"{self.data_location}/pareto_front.pth")
        except:
            print(f"WARNING: No file with pareto front points found!")
            self.pareto_front = None

    def _simulate_batch(self, x, device = None):
        f = torch.zeros((x.shape[0],6), device=device)

        f[:,0] = 106780.37 * (x[:,1] + x[:,2]) + 61704.67
        f[:,1] = 3000 * x[:,0]
        f[:,2] = 305700 * 2289 * x[:,1] / ((0.06*2289)**0.65)
        f[:,3] = 250 * 2289 * (-39.75*x[:,1]+9.9*x[:,2]+2.74).exp()
        f[:,4] = 25 * (1.39 /(x[:,0]*x[:,1]) + 4940*x[:,2] -80)

        # Constraint functions          
        f[:,5] += torch.maximum(torch.zeros(x.shape[0]), -(1 - 0.00139/(x[:,0]*x[:,1]) - 4.94*x[:,2] + 0.08))
        f[:,5] += torch.maximum(torch.zeros(x.shape[0]), -(1 - 0.000306/(x[:,0]*x[:,1]) - 1.082*x[:,2] + 0.0986))
        f[:,5] += torch.maximum(torch.zeros(x.shape[0]), -(50000 - 12.307/(x[:,0]*x[:,1]) - 49408.24*x[:,2] - 4051.02))
        f[:,5] += torch.maximum(torch.zeros(x.shape[0]), -(16000 - 2.098/(x[:,0]*x[:,1]) - 8046.33*x[:,2] + 696.71))
        f[:,5] += torch.maximum(torch.zeros(x.shape[0]), -(10000 - 2.138/(x[:,0]*x[:,1]) - 7883.39*x[:,2] + 705.04))
        f[:,5] += torch.maximum(torch.zeros(x.shape[0]), -(2000 - 0.417*x[:,0]*x[:,1] - 1721.26*x[:,2] + 136.54))
        f[:,5] += torch.maximum(torch.zeros(x.shape[0]), -(550 - 0.164/(x[:,0]*x[:,1]) - 631.13*x[:,2] + 54.48))

        return f
    
    def _normalize_outcomes(self, x):
        return (x - self.f_min) / (self.f_max - self.f_min)

    def generate_choice_batch(self, batch_size, device = None):
        x = self.x_prior.sample((batch_size * self.n_choices,)).to(device=self.get_device(device))
        return self._normalize_outcomes(self._simulate_batch(x)).reshape((batch_size, self.n_choices, self.n_attributes))
    
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
        
    def find_pareto_front(self, N_SAMPLES = 7_500_000):
        test_points = torch.rand((N_SAMPLES * self.n_choices, 5), device=self.get_device(self.device)) * 2.0 + 1.0
        test_points = torch.cat([test_points, self._normalize_outcomes(self._simulate_batch(test_points))], dim=1)

        # find points on efficient frontier
        is_efficient = torch.ones(test_points.shape[0], dtype = bool)
        for i in range(test_points.shape[0]):
            if is_efficient[i]:
                is_efficient[is_efficient.clone()] = torch.any(test_points[is_efficient,:] < test_points[i,:], dim=1)
                is_efficient[i] = True
        
        pareto_front = test_points[is_efficient,:]
        self.pareto_front = pareto_front
        torch.save(pareto_front, f"{self.data_location}/pareto_front.pth")
    
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
