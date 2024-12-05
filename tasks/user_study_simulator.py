from abc import ABC, abstractmethod
import torch

class AbstractElicitationStudySimulator(ABC):
    """
    Abstract class for a simulating a user study where we elicit a user's preferences through queries.

    """
    def __init__(self, participantID, device = torch.device("cpu")):
        self.participantID = participantID
        self.device = device

    def get_device(self, device = None):
        if device is not None:
            return device
        return self.device
    
    @abstractmethod
    def get_available_queries(self):
        pass

    """
    Simulates a choice of the user on a specific query. The query is identified by an index to 
    the list of available queries returned by the last call to get_available_queries.

    NOTE: after calling this function, the query on which the choice is simulated will generally be recorded as unavailable.

    returns:
        [int]: the index of the chosen option
    """
    @abstractmethod
    def simulate_choice(self, query_index):
        pass

    """
    Provides a set of unseen queries which can be used to evaluate the fit of an inferres set of paramters and hyperparameters.

    returns:
        [torch.tensor]: set of queries of size [batch_size, n_choices, n_attributes]
        [torch.tensor]: set of indexes of chosen options of size [batch_size]
    """
    @abstractmethod
    def get_evaluation_data(self):
        pass