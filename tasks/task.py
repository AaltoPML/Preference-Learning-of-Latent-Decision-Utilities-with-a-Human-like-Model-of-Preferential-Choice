from abc import ABC, abstractmethod
import torch

class AbstractTask(ABC):
    """
    Abstract class for a choie task

    Implementations must have the following attributes:
        self.n_choices
        self.n_attributes
        self.device
        self.parameter_dim
        self.observation_dim

    """   
    def get_device(self, device = None):
        if device is not None:
            return device
        return self.device

    @abstractmethod
    def generate_choice_batch(self, batch_size, device = None):
        """
        Generate a batch of choices
        input:
            batch_size: (int) the number of sets of choices
        returns: (batch_size, n_choices, n_attributes) a batch of choices
        """
        pass
    
    @abstractmethod
    def generate_parameter_batch(self, batch_size, device = None):
        """
        Generate a batch of randomly sampled parameters
        input:
            batch_size: (int) the number of parameters
        returns: (batch_size, parameter_dim) a batch of parameters
        """
        pass

    @abstractmethod
    def observe_batch(self, x, x_f, device = None):
        """
        Generate observations for a number of choices with associated utilities
        input:
            x: (batch_size, n_choices, n_attributes) the choices
            x_f: (batch_size, parameter_dim) the utilities of the choices (e.g. generated by self.utility(...))
        returns: (batch_size, n_choices, observation_dim) observations of the choices x
        """
        pass
    
    @abstractmethod
    def utility(self, x, w):
        """
        Calculates the utilities of a number of choices
        input:
            x: (batch_size, n_choices, n_attributes) a batch of choices (e.g. generated by self.generate_choice_batch(...))
            w: (batch_size, parameter_dim) a batch of parameters (e.g. generated by self.generate_parameter_batch(...))
        returns: (batch_size, n_choices) the utilties for the choices x
        """

    def find_optimum(self, w, N_STEPS = 1000):
        """
        Finds an approximately optimal point (highest expected utility under given set of particles w)
        input:
            w: (batch_size, parameter_dim)
        returns: (n_attributes,)
        """
        argmax_util = torch.ones((1, self.n_attributes))
        max_util = -1000000

        for _ in range(N_STEPS):
            x = self.x_prior.sample((1,)).to(device=self.device)
            x = self._normalize_outcomes(self._simulate_batch(x))
            u = self.utility(x.repeat_interleave(w.shape[0], 0)[:,None,:], w).mean().item()

            if u > max_util:
                argmax_util = x
                max_util = u

        return argmax_util[0,:]
    
    def find_pareto_optimum(self, w, N_STEPS = None):
        """
        Finds the best point located on the pareto frontier in self.pareto_front
        input:
            w: (batch_size, parameter_dim)
            N_STEPS: Int [this paramter exists to mainting uniformity with find_optimum]
        returns: (n_attributes,)
        """
        pareto_optimum_idx = None
        pareto_optimum_E_util = -1000000
        for i in range(self.pareto_front.shape[0]):
            E_util = self.utility(self.pareto_front[None,None,i].repeat_interleave(w.shape[0], 0), w)[:,0].mean()
            if E_util > pareto_optimum_E_util:
                pareto_optimum_E_util = E_util
                pareto_optimum_idx = i
        return self.pareto_front[pareto_optimum_idx]