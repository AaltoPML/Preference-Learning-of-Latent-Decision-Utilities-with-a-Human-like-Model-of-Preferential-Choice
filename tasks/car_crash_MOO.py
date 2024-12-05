import torch
from .task import AbstractTask

# This is a minimization problem, but the cognitive model assumes maximization!
# mass range: 1661.7078224999998 1704.5588675
# Ain range: 6.142799377441406 11.71241283416748
# intrusion range: 0.03939998894929886 0.26399996876716614

class CarCrashMOOTask(AbstractTask):
    def __init__(self, n_choices, sigma_e, tau_pv, p_error, observe_thicknesses = True, device = torch.device("cpu"), scalarization = "Chebyshev"):
        self.n_choices = n_choices
        self.n_attributes = 8
        self.sigma_e = sigma_e
        self.tau_pv = tau_pv
        self.p_error = p_error
        self.device = device
        self.parameter_dim = 3
        self.observe_thickness = observe_thicknesses
        self.scalarization = scalarization

        self.parameter_prior = torch.distributions.Dirichlet(torch.ones(self.parameter_dim))
        # there are (self.n_choices ^ 2 + self.n_choices) / 2 comparisons for each attribute, of which there are n_observable_attributes, and n_choices observations of utility
        n_observable_attributes = 8 if self.observe_thickness else 3
        self.observation_dim = (self.n_choices * (self.n_choices - 1) // 2) * n_observable_attributes + self.n_choices

    def _simulate_batch(self, x, device = None):
        mass = 1640.2823 + 2.3573285 * x[:,0] + 2.3220035 * x[:,1] + 4.5688768 * x[:,2] + 7.7213633 * x[:,3] + 4.4559504 * x[:,4]
        Ain = 6.5856 + 1.15*x[:,0] - 1.0427*x[:,1] + 0.9738*x[:,2] + 0.8364*x[:,3] - 0.3695*x[:,0]*x[:,3] + 0.0861*x[:,0]*x[:,4] + 0.3628*x[:,1]*x[:,3] - 0.1106*x[:,0]*x[:,0] - 0.3437*x[:,2]*x[:,2] + 0.1764*x[:,3]*x[:,3]
        intrusion = -0.0551 + 0.0181*x[:,0] + 0.1024*x[:,1] + 0.0421*x[:,2] - 0.0073*x[:,0]*x[:,1] + 0.024*x[:,1]*x[:,2] - 0.0118*x[:,1]*x[:,3] - 0.0204*x[:,2]*x[:,3] - 0.008*x[:,2]*x[:,4] - 0.0241*x[:,1]*x[:,1] + 0.0109*x[:,3]*x[:,3]
        return torch.cat([mass[:,None], Ain[:,None], intrusion[:,None]], dim=1)
    
    def _normalize_outcomes(self, x):
        return (x - torch.tensor([1661.7078224999998, 6.142799377441406, 0.03939998894929886])) / torch.tensor([1704.5588675 - 1661.7078224999998, 11.71241283416748 - 6.142799377441406, 0.26399996876716614 - 0.03939998894929886])

    def generate_choice_batch(self, batch_size, device = None):
        x = torch.rand((batch_size * self.n_choices, 5), device=self.get_device(device)) * 2.0 + 1.0
        x = torch.cat([x, self._normalize_outcomes(self._simulate_batch(x))], dim=1)
        return x.reshape((batch_size, self.n_choices, 8))
    
    def generate_parameter_batch(self, batch_size, device = None):
        return self.parameter_prior.sample((batch_size,)).to(device=self.get_device(device))

    def observe_batch(self, x, x_f, device = None):
        batch_size = x.shape[0]
        e_obs = x_f + torch.randn(x.shape[:2], device=self.get_device(device)) * self.sigma_e
        obs = [e_obs]
        for i in range(self.n_choices):
            for j in range(i+1,self.n_choices):
                for k in range(0 if self.observe_thickness else 5, x.shape[2]):
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
        NC = x.shape[1]
        x_flat = x.reshape((-1, self.n_attributes))
        if self.scalarization == "WeightedSum":
            return - torch.multiply(x_flat[:,5:], w.repeat_interleave(NC, 0)).sum(dim=1).reshape(-1, NC)
        elif self.scalarization == "Chebyshev":
            return - torch.multiply(x_flat[:,5:].abs(), w.repeat_interleave(NC, 0)).max(dim=1).values.reshape(-1, NC)
        else:
            raise NotImplementedError
    
class ParameterizedCarCrashMOOTask(CarCrashMOOTask):
    def __init__(self, n_choices, sigma_e_prior, tau_pv_prior, p_error_prior, observe_thicknesses = True, device = torch.device("cpu"), scalarization = "Chebyshev", data_location = "car_crash"):
        super().__init__(n_choices, 0.0, torch.zeros(8), 0.0, observe_thicknesses = observe_thicknesses, device = device, scalarization = scalarization)
        self.sigma_e_prior = sigma_e_prior  # samples expected to be size (N)
        self.tau_pv_prior = tau_pv_prior    # samples expected to be size (N)
        self.p_error_prior = p_error_prior  # samples expected to be size (N)
        self.hyperparameter_dim = 3

        self.data_location = data_location
        try:
            self.pareto_front = torch.load(f"{self.data_location}/pareto_front.pth")
        except:
            print(f"WARNING: No file with pareto front points found!")
            self.pareto_front = None

    def _parse_hyperparams(self, hyperparameter_batch):
        sigma_e_prior_batch = hyperparameter_batch[:,0]
        p_error_batch = hyperparameter_batch[:,1]
        tau_pv_batch = hyperparameter_batch[:,2]
        return sigma_e_prior_batch, p_error_batch, tau_pv_batch

    def generate_hyperparameter_batch(self, batch_size, device = None):
        sigma_e_params = self.sigma_e_prior.sample((batch_size,1))
        p_error_params = self.p_error_prior.sample((batch_size,1))
        tau_pv_params = self.tau_pv_prior.sample((batch_size,1))

        return torch.cat([sigma_e_params, p_error_params, tau_pv_params], dim=1).to(self.get_device(device))

    def observe_batch(self, x, x_f, hyperparameter_batch, device = None):
        sigma_e_prior_batch, p_error_batch, tau_pv_batch = self._parse_hyperparams(hyperparameter_batch)

        batch_size = x.shape[0]
        e_obs = x_f + torch.multiply(torch.randn(x.shape[:2], device=self.get_device(device)), sigma_e_prior_batch[:,None].repeat([1,self.n_choices]))
        obs = [e_obs]
        for i in range(self.n_choices):
            for j in range(i+1,self.n_choices):
                for k in range(0 if self.observe_thickness else 5, x.shape[2]):
                    a_obs = torch.zeros((batch_size))
                    a_obs[x[:,i,k] < x[:,j,k] - tau_pv_batch] = -1.0
                    a_obs[x[:,i,k] > x[:,j,k] + tau_pv_batch] = 1.0
                    random_mask = torch.bernoulli(p_error_batch)
                    random_obs = torch.multinomial(torch.tensor([1/3, 1/3, 1/3]), batch_size, replacement=True) - 1.0
                    a_obs = a_obs * (1.0 - random_mask) + random_mask * random_obs
                    obs.append(a_obs[...,None].to(self.get_device(device)))
        return torch.cat(obs, dim=1)
    
    def find_pareto_front(self, N_SAMPLES = 3_000_000):
        test_points = torch.rand((N_SAMPLES, 5), device=self.get_device(self.device)) * 2.0 + 1.0
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


if __name__ == "__main__":
    from tqdm import tqdm

    task = CarCrashMOOTask(1, 0.00, torch.ones(5) * 0.05, 0.0, observe_thicknesses=False, device = torch.device("cpu"))
    param_space = []
    EPS = 0.00
    N_STEPS = 101
    for a1 in torch.linspace(EPS, 1.0 - EPS, N_STEPS):
        for a2 in torch.linspace(EPS, 1.0 - EPS, N_STEPS):
            if a1 + a2 > 1.0:
                continue
            param_space.append(torch.tensor((a1, a2, 1 - a1 - a2)))
    param_space = torch.stack(param_space)

    argmax_util = torch.ones((param_space.shape[0], 5))
    max_util = torch.ones((param_space.shape[0])) * -1000000

    N_STEPS = 21
    for t1 in tqdm(torch.linspace(1.0, 3.0, N_STEPS)):
        for t2 in torch.linspace(1.0, 3.0, N_STEPS):
            # print(max_util)
            for t3 in torch.linspace(1.0, 3.0, N_STEPS):
                for t4 in torch.linspace(1.0, 3.0, N_STEPS):
                    for t5 in torch.linspace(1.0, 3.0, N_STEPS):
                        test_batch = torch.tensor([t1,t2,t3,t4,t5])[None,:]
                        test_result = task._normalize_outcomes(task._simulate_batch(test_batch)).repeat_interleave(param_space.shape[0], 0)
                        test_batch = test_batch.repeat_interleave(param_space.shape[0], 0)

                        utils = - torch.multiply(test_result.abs(), param_space).max(dim=1).values
                        idx_improved = (utils > max_util)
                        argmax_util[idx_improved,:] = test_batch[idx_improved,:]
                        max_util[idx_improved] = utils[idx_improved]

    torch.save((param_space, argmax_util), "car_crash_no_thickness/optima.pth")