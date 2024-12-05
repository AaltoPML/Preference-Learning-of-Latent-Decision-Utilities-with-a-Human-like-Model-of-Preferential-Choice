import torch
from torch import optim
import tqdm

class BradleyTerryModel:
    def __init__(self, task, beta = 1.0):
        self.task = task
        self.beta = beta

    def __call__(self, choice_sets, u):
        choice_utils = self.task.utility(choice_sets, u)
        policy = (choice_utils * self.beta).exp()
        return (policy / policy.sum(dim=1)[:, None].repeat(1, policy.shape[1])).log()
    
class ParameterizedBradleyTerryModel:
    def __init__(self, task, beta_max = torch.inf):
        self.task = task
        self.beta_max = beta_max

    def __call__(self, choice_sets, u, betas):
        choice_utils = self.task.utility(choice_sets, u)
        policy = torch.multiply(choice_utils, torch.minimum(betas, torch.ones_like(betas) * self.beta_max).repeat((1,self.task.n_choices))).exp()
        return (policy / policy.sum(dim=1)[:, None].repeat(1, policy.shape[1])).log()

class CanonicalBradleyTerryModel:
    def __init__(self, task):
        self.task = task

    def __call__(self, x, w, _):
        # data is automatically normalized by utility!
        choice_utils = self.task.utility(x, w)
        policy = choice_utils.exp()
        return (policy / policy.sum(dim=1)[:, None].repeat(1, policy.shape[1])).log()
    
    """
    Infers parameters and hyperparameters of this model
        x: batch of choice sets (batch_size, n_choices, n_parameters)
        y: batch of choices made (batch_size,)
        w_init: initial value for utility parameters (n_parameters,)
        h_init: initial value for model parameters (n_hyperparameters,)
    """
    def infer(self, x, y, w_init, _, n_iter = 100000, lr = 0.001, early_stopping = True, verbose=False, aggregation = "mean", x_test = None, y_test = None, stop_margin_steps = 50):
        w_inferred = w_init.clone()
        w_inferred.requires_grad = True
        optimizer = optim.Adam([w_inferred], lr = lr)

        if early_stopping and ((x_test is None) and (y_test is None)): # legacy support for early stopping based on x and y
            x_test, y_test = x,y
        do_validation = (x_test is not None) and (y_test is not None)
        last_best_loss = torch.inf
        iter_since_last_best_loss = 0
        for _ in (tqdm_train := tqdm.tqdm(range(n_iter), disable = not verbose)):
            optimizer.zero_grad()
            log_probs = self(x, w_inferred[None,:].repeat_interleave(x.shape[0], 0), None)
            if aggregation == "sum":
                loss = - log_probs.gather(1, y[:,None].to(dtype = torch.int64)).sum()
            elif aggregation == "mean":
                loss = - log_probs.gather(1, y[:,None].to(dtype = torch.int64)).mean()
            else:
                raise NotImplementedError
            tqdm_train.set_description('Loss: {:5f}'.format(loss.item()))
            if do_validation:
                with torch.no_grad():
                    log_probs_test = self(x_test, w_inferred[None,:].repeat_interleave(x_test.shape[0], 0), None)
                    if aggregation == "sum":
                        test_loss = -log_probs_test.gather(1, y_test[:,None].to(dtype = torch.int64)).sum().item()
                    elif aggregation == "mean":
                        test_loss = -log_probs_test.gather(1, y_test[:,None].to(dtype = torch.int64)).mean().item()
                    else:
                        raise NotImplementedError
                if test_loss < last_best_loss:
                    last_best_loss = test_loss
                    iter_since_last_best_loss = 0
                else:
                    iter_since_last_best_loss += 1
                    if iter_since_last_best_loss > stop_margin_steps:
                        break
            loss.backward()
            optimizer.step()

        return (w_inferred.clone().detach(), torch.tensor([]))
    
class BowerBalzanoBradleyTerryModel:
    """
    The hyperparameter layout for this task is as follows: (batch_size, 1) where the last dimension is the top-k parameter for each batch.
    """
    def __init__(self, task):
        self.task = task

    """
    Generates a mask for masking out the features of a set of choices
    x_norm: normalized choices of size (batch_size, n_choices, n_attributes)
    top_ks: vector of k's for top-k selection of size (batch_size,)
    """
    def generate_feature_mask(self, x_norm, top_ks):
        sort_indices = x_norm.var(dim=(1)).sort(dim=1, descending=True).indices

        feature_selection_mask = torch.zeros((x_norm.shape[0],self.task.n_attributes))
        for i in range(x_norm.shape[0]):
            feature_selection_mask[i, sort_indices[i,:top_ks[i]]] = 1.0
        return feature_selection_mask

    def __call__(self, x, w, top_ks, feature_selection_mask = None):
        if feature_selection_mask is None:
            x_norm = self.task.normalize_choice_batch(x)
            feature_selection_mask = self.generate_feature_mask(x_norm, top_ks.flatten().to(dtype=torch.int))

        choice_utils = self.task.utility(x, w * feature_selection_mask)
        policy = choice_utils.exp()
        return (policy / policy.sum(dim=1)[:, None].repeat(1, policy.shape[1])).log()
    
    """
    Infers parameters and hyperparameters of this model
        x: batch of choice sets (batch_size, n_choices, n_parameters)
        y: batch of choices made (batch_size,)
        w_init: initial value for utility parameters (n_parameters,)
        h_init: initial value for model parameters (n_hyperparameters,)
    """
    def infer(self, x, y, w_init, _, n_iter = 100000, lr = 0.001, early_stopping = True, verbose=False, aggregation = "mean", x_test = None, y_test = None, stop_margin_steps = 50):
        best_topk_loss = torch.inf
        best_topk_params = None
        if early_stopping and ((x_test is None) and (y_test is None)): # legacy support for early stopping based on x and y
            x_test, y_test = x,y
        do_validation = (x_test is not None) and (y_test is not None)
        for k in range(1,self.task.n_attributes+1):
            if verbose:
                print(f"Starting optimization with k={k}")
            top_ks = torch.ones((x.shape[0],), dtype=torch.int32) * k

            w_inferred = w_init.clone()
            w_inferred.requires_grad = True
            optimizer = optim.Adam([w_inferred], lr = lr)
            x_norm = self.task.normalize_choice_batch(x)
            feature_selection_mask = self.generate_feature_mask(x_norm, top_ks)
            if do_validation:
                feature_selection_mask_test = self.generate_feature_mask(self.task.normalize_choice_batch(x_test), torch.ones((x_test.shape[0],), dtype=torch.int32) * k)

            last_best_loss = torch.inf
            iter_since_last_best_loss = 0
            for _ in (tqdm_train := tqdm.tqdm(range(n_iter), disable = not verbose)):
                optimizer.zero_grad()
                log_probs = self(x, w_inferred[None,:].repeat_interleave(x.shape[0], 0), top_ks, feature_selection_mask=feature_selection_mask)
                if aggregation == "sum":
                    loss = - log_probs.gather(1, y[:,None].to(dtype = torch.int64)).sum()
                elif aggregation == "mean":
                    loss = - log_probs.gather(1, y[:,None].to(dtype = torch.int64)).mean()
                else:
                    raise NotImplementedError
                tqdm_train.set_description('Loss: {:5f}'.format(loss.item()))
                if do_validation:
                    with torch.no_grad():
                        log_probs_test = self(x_test, w_inferred[None,:].repeat_interleave(x_test.shape[0], 0), torch.ones((x_test.shape), dtype=torch.int32) * k, feature_selection_mask=feature_selection_mask_test)
                        if aggregation == "sum":
                            test_loss = -log_probs_test.gather(1, y_test[:,None].to(dtype = torch.int64)).sum().item()
                        elif aggregation == "mean":
                            test_loss = -log_probs_test.gather(1, y_test[:,None].to(dtype = torch.int64)).mean().item()
                        else:
                            raise NotImplementedError
                    if test_loss < last_best_loss:
                        last_best_loss = test_loss
                        iter_since_last_best_loss = 0
                    else:
                        iter_since_last_best_loss += 1
                        if iter_since_last_best_loss > stop_margin_steps:
                            break
                loss.backward()
                optimizer.step()

            if do_validation: # choose best k based on test set, if we can.
                if test_loss < best_topk_loss:
                    best_topk_loss = test_loss
                    best_topk_params = (w_inferred.clone().detach(), torch.tensor([k], dtype=torch.int32))
            else:
                if loss.item() < best_topk_loss:
                    best_topk_loss = loss.item()
                    best_topk_params = (w_inferred.clone().detach(), torch.tensor([k], dtype=torch.int32))

        return best_topk_params
        
    
class LCLBradleyTerryModel:
    """
    The hyperparameter layout for this task is as follows: (batch_size, param_dim^2)
    The last dimension contains the entries for matrix A from the original paper. To access the matrix, the hyperparameter vector should be reshaped into (batch_size, param_dim, param_dim)
    """
    def __init__(self, task):
        self.task = task

    def __call__(self, x, w, h):
        xC = self.task.normalize_choice_batch(x).mean(dim=1)[:,:,None]
        A = h.reshape(w.shape[0], self.task.parameter_dim, self.task.parameter_dim)

        choice_utils = self.task.utility(x, w + A.bmm(xC).reshape(w.shape))
        policy = choice_utils.exp()
        return (policy / policy.sum(dim=1)[:, None].repeat(1, policy.shape[1])).log()
    
    """
    Infers parameters and hyperparameters of this model
        x: batch of choice sets (batch_size, n_choices, n_parameters)
        y: batch of choices made (batch_size,)
        w_init: initial value for utility parameters (n_parameters,)
        h_init: initial value for model parameters (n_hyperparameters,)
    """
    def infer(self, x, y, w_init, h_init, n_iter = 100000, lr = 0.001, early_stopping = True, verbose = False, alpha = 0.0, aggregation = "mean", x_test = None, y_test = None, stop_margin_steps = 50):
        w_inferred, h_inferred = w_init.clone(), h_init.clone()
        w_inferred.requires_grad = True
        h_inferred.requires_grad = True
        optimizer = optim.Adam([w_inferred, h_inferred], lr=lr)

        if early_stopping and ((x_test is None) and (y_test is None)): # legacy support for early stopping based on x and y
            x_test, y_test = x,y
        do_validation = (x_test is not None) and (y_test is not None)
        last_best_loss = torch.inf
        iter_since_last_best_loss = 0
        for _ in (tqdm_train := tqdm.tqdm(range(n_iter), disable = not verbose)):
            optimizer.zero_grad()
            log_probs = self(x, w_inferred[None,:].repeat_interleave(x.shape[0], 0), h_inferred[None,:].repeat_interleave(x.shape[0], 0))
            if aggregation == "sum":
                loss = - log_probs.gather(1, y[:,None].to(dtype = torch.int64)).sum()
            elif aggregation == "mean":
                loss = - log_probs.gather(1, y[:,None].to(dtype = torch.int64)).mean()
            else:
                raise NotImplementedError
            loss += alpha * torch.linalg.matrix_norm(h_inferred.reshape((self.task.n_attributes, self.task.n_attributes)), ord=1)
            tqdm_train.set_description('Loss: {:5f} w Norm: {:5f} h norm: {:5f}'.format(loss.item(), w_inferred.norm().item(), h_inferred.norm().item()))
            if do_validation:
                with torch.no_grad():
                    log_probs_test = self(x_test, w_inferred[None,:].repeat_interleave(x_test.shape[0], 0), h_inferred[None,:].repeat_interleave(x_test.shape[0], 0))
                    if aggregation == "sum":
                        test_loss = -log_probs_test.gather(1, y_test[:,None].to(dtype = torch.int64)).sum().item()
                    elif aggregation == "mean":
                        test_loss = -log_probs_test.gather(1, y_test[:,None].to(dtype = torch.int64)).mean().item()
                    else:
                        raise NotImplementedError
                if test_loss < last_best_loss:
                    last_best_loss = test_loss
                    iter_since_last_best_loss = 0
                else:
                    iter_since_last_best_loss += 1
                    if iter_since_last_best_loss > stop_margin_steps:
                        break
            loss.backward()
            optimizer.step()

        return (w_inferred.clone().detach(), h_inferred.clone().detach())