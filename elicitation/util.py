import torch
import numpy as np
from scipy.stats import wasserstein_distance

def calculate_context_effect_min_wasserstein(task, query, model, true_w, true_h, min_option):
    assert task.n_choices == 3
    total_wass = 0

    for i, not_i in zip([0, 1, 2], [[1,2], [0,2], [0,1]]):
        new_query = query.clone()
        new_query[0,i,:] = min_option.flatten()

        with torch.no_grad():
            original_probs = model(query, true_w, true_h).exp()
            new_probs = model(new_query, true_w, true_h).exp()

        total_wass += wasserstein_distance((original_probs[0,not_i] / (1-original_probs[0,i])).numpy(), (new_probs[0,not_i] / (1-new_probs[0,i]).numpy()))
        
    return total_wass

def calculate_context_effect_mean_wasserstein(task, query, model, true_w, true_h, N = 10_000):
    assert task.n_choices == 3

    total_wass = 0

    for i, not_i in zip([0, 1, 2], [[1,2], [0,2], [0,1]]):
        new_options = task.generate_choice_batch(N)
        query_batched = query.repeat(N,1,1)
        query_batched[:,i,:] = new_options[:,i,:]

        with torch.no_grad():
            original_probs = model(query, true_w, true_h).exp()
            new_probs = model(query_batched, true_w.repeat(N,1), true_h.repeat(N,1)).exp()
        total_wass += np.mean(np.nan_to_num([wasserstein_distance((original_probs[0,not_i] / (1-original_probs[0,i])).numpy(), (new_probs[j,not_i] / (1-new_probs[j,i])).numpy()) for j in range(N)]))
        
    return total_wass

def calculate_context_effect_min_prob_ratio(task, query, model, true_w, true_h, min_option):
    assert task.n_choices == 3
    ratio_HM = 0

    for i, not_i in zip([0, 1, 2], [[1,2], [0,2], [0,1]]):
        new_query = query.clone()
        new_query[0,i,:] = min_option.flatten()

        with torch.no_grad():
            original_probs = model(query, true_w, true_h).exp()
            new_probs = model(new_query, true_w, true_h).exp()
            
        original_ratio = (original_probs[0,not_i[0]] - original_probs[0,not_i[1]])
        new_ratios = (new_probs[0,not_i[0]] - new_probs[0,not_i[1]])

        log_ratio = new_ratios - original_ratio
        ratio_HM += log_ratio.abs().item()      
    return ratio_HM

def calculate_context_effect_mean_prob_ratio(task, query, model, true_w, true_h, N = 100_000):
    assert task.n_choices == 3

    ratio_HM = torch.zeros((N,))

    for i, not_i in zip([0, 1, 2], [[1,2], [0,2], [0,1]]):
        new_options = task.generate_choice_batch(N)
        query_batched = query.repeat(N,1,1)
        query_batched[:,i,:] = new_options[:,i,:]

        with torch.no_grad():
            original_probs = model(query, true_w, true_h)
            new_probs = model(query_batched, true_w.repeat(N,1), true_h.repeat(N,1))
            
        original_ratio = (original_probs[0,not_i[0]] - original_probs[0,not_i[1]])
        new_ratios = (new_probs[:,not_i[0]] - new_probs[:,not_i[1]])

        log_ratio = new_ratios - original_ratio
        ratios = log_ratio.abs()
        ratio_HM += ratios

    return ratio_HM.mean().item()

def minimize_utility(task, true_w, N = 100_000_000):
    BATCH_SIZE = 10_000
    N_batches = N // BATCH_SIZE

    min_option = None
    min_option_u = torch.inf

    n_queries_batch = (BATCH_SIZE // task.n_choices)
    batch_size_p = n_queries_batch * task.n_choices

    for _ in range(N_batches):
        x = task.generate_choice_batch(n_queries_batch).reshape(batch_size_p,1,task.n_attributes)
        x_u = task.utility(x, true_w.repeat(batch_size_p,1))

        argmin_idx = x_u.argmin()
        x_argmin, u_min = x[argmin_idx], x_u[argmin_idx]
        if u_min < min_option_u:
            min_option = x_argmin
            min_option_u = u_min
    return min_option