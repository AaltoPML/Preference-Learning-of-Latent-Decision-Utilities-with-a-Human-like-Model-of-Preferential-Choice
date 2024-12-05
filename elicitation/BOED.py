import torch
import numpy as np
from tqdm import trange
from math import floor

from utils.eval_utils import infer_and_validate_SGD_multistart
from .util import *

def create_DATA(N_QUERY_CANDIDATES, N_UPARAM_PARTICLES, N_HPARAM_PARTICLES):
    DATA = {"N_QUERY_CANDIDATES": N_QUERY_CANDIDATES, "N_UPARAM_PARTICLES": N_UPARAM_PARTICLES, "N_HPARAM_PARTICLES": N_HPARAM_PARTICLES, "N_PARTICLES": N_UPARAM_PARTICLES * N_HPARAM_PARTICLES}
    DATA["posterior_entropy"] = []
    DATA["utility_marginal_posterior_entropy"] = []
    DATA["hyperparameter_marginal_posterior_entropy"] = []
    DATA["n_unique_particles"] = []
    DATA["utility_inference_error"] = []
    DATA["hyperparameter_inference_error"] = []
    DATA["recommendation_regret"] = []
    DATA["EIG_max"] = []
    DATA["EIG_90"] = []

    return DATA

def create_joint_partcile_sets(original_uparam_particle_set, original_hparam_particle_set):
    N_UPARAM_PARTICLES = original_uparam_particle_set.shape[0]
    N_HPARAM_PARTICLES = original_hparam_particle_set.shape[0]

    uparam_particle_set = original_uparam_particle_set.repeat_interleave(N_HPARAM_PARTICLES,0)
    hparam_particle_set = original_hparam_particle_set.repeat((N_UPARAM_PARTICLES,1))

    return uparam_particle_set, hparam_particle_set

"""
Calculates the uparam marginal distribution of a given joint distribution derived from a joint particle set.
NOTE this makes use of the specific structure of joint particle sets produced by create_joint_partcile_sets(...)! Do not apply this to any other particle sets.
"""
def calculate_marginal_uparam_probs(joint_probs, N_UPARAM_PARTICLES, N_HPARAM_PARTICLES):
    mprobs = torch.zeros(N_UPARAM_PARTICLES)
    for i in range(N_UPARAM_PARTICLES):
        mprobs[i] = joint_probs[i*N_HPARAM_PARTICLES:(i+1)*N_HPARAM_PARTICLES].sum()
    return mprobs

"""
Calculates the hparam marginal distribution of a given joint distribution derived from a joint particle set.
NOTE this makes use of the specific structure of joint particle sets produced by create_joint_partcile_sets(...)! Do not apply this to any other particle sets.
"""
def calculate_marginal_hparam_probs(joint_probs, N_UPARAM_PARTICLES, N_HPARAM_PARTICLES):
    mprobs = torch.zeros(N_HPARAM_PARTICLES)
    for i in range(joint_probs.shape[0]):
        mprobs[i % N_HPARAM_PARTICLES] += joint_probs[i]
    return mprobs

"""
This function creates a new set of particle indices which deduplicates particles based on the uparam part of a joint particle set only (Thus, it disregards the hparam part entirely).
This gives a set of particles (correponding to the particle indices) with the same distribution as the uparam marginal of the joint distribution modeled by the given particle set.
In practice, all particles (u,h_1), ...,  (u,h_n) get mapped onto (u,h_1). This allows for maximally efficient calculations involving only the marginal distribution over the utility parameters.
NOTE this makes use of the specific structure of joint particle sets produced by create_joint_partcile_sets(...)! Do not apply this to any other particle sets.
"""
def construct_uparam_marginal_dedup_particle_set_idxs(particle_set_idxs, N_UPARAM_PARTICLES, N_HPARAM_PARTICLES):
    particle_set_idxs = torch.tensor([floor(i / N_HPARAM_PARTICLES) * N_HPARAM_PARTICLES for i in particle_set_idxs])
    dedup_particle_idxs = list(set([i.item() for i in particle_set_idxs]))

    return particle_set_idxs, dedup_particle_idxs

"""
This implementation mirrors the implementation in AbstractTask, but gains additional efficiency by avoiding calculations 
for duplicate particles in uparam_particle_set.
"""
def find_pareto_optimum(ptask, uparam_particle_set, particle_set_idxs, dedup_particle_idxs, N_PARTICLES):
    pareto_optimum_idx = None
    pareto_optimum_E_util = -1000000
    for i in range(ptask.pareto_front.shape[0]):
        dedup_E_utils = torch.zeros(N_PARTICLES)
        dedup_E_utils[dedup_particle_idxs] = ptask.utility(ptask.pareto_front[None,None,i].repeat_interleave(len(dedup_particle_idxs), 0), uparam_particle_set[dedup_particle_idxs,:])[:,0]
        E_util = dedup_E_utils[particle_set_idxs].mean()
        if E_util > pareto_optimum_E_util:
            pareto_optimum_E_util = E_util
            pareto_optimum_idx = i
    return ptask.pareto_front[pareto_optimum_idx]

def calculate_regret_random_recommendation(ptask, true_w):
    true_pareto_optimum_utility = ptask.utility(ptask.find_pareto_optimum(true_w)[None,None,:], true_w).item()
    avg_regret = (true_pareto_optimum_utility - ptask.utility(ptask.pareto_front[:,None,:], true_w.repeat(ptask.pareto_front.shape[0],1)).flatten()).mean().item()
    return avg_regret

def run_BOED(ptask, model, true_model, uparam_particle_set, hparam_particle_set, true_w, true_h, N_STEPS, DATA, 
             particle_set_idxs = None, 
             calculate_regret = True, 
             step_zero_data = True, 
             calculate_hparam_error = True,
             save_posterior = False,
             calculate_context_effect = True):
    N_QUERY_CANDIDATES = DATA["N_QUERY_CANDIDATES"]
    N_PARTICLES = DATA["N_PARTICLES"]
    N_UPARAM_PARTICLES = DATA["N_UPARAM_PARTICLES"]
    N_HPARAM_PARTICLES = DATA["N_HPARAM_PARTICLES"]
    assert (uparam_particle_set.shape[0] == N_PARTICLES) and (hparam_particle_set.shape[0] == N_PARTICLES)
    if ptask.n_choices != 3:
        calculate_context_effect = False

    particle_set_idxs = torch.tensor(list(range(N_PARTICLES))) if particle_set_idxs is None else particle_set_idxs
    uparam_particle_errors = torch.pow(true_w.repeat((N_PARTICLES,1)) - uparam_particle_set, 2).sum(dim=1).sqrt()
    hparam_particle_errors = torch.pow(true_h.repeat((N_PARTICLES,1)) - hparam_particle_set, 2).sum(dim=1).sqrt() if calculate_hparam_error else torch.zeros((N_PARTICLES,))
    if calculate_regret:
        true_pareto_optimum_utility = ptask.utility(ptask.find_pareto_optimum(true_w)[None,None,:], true_w).item()
    else:
        true_pareto_optimum_utility = None

    if step_zero_data:
        implied_probs = torch.ones(N_PARTICLES) / N_PARTICLES
        uparam_marginal_probs = calculate_marginal_uparam_probs(implied_probs, N_UPARAM_PARTICLES, N_HPARAM_PARTICLES)
        hparam_marginal_probs = calculate_marginal_hparam_probs(implied_probs, N_UPARAM_PARTICLES, N_HPARAM_PARTICLES)
        DATA["posterior_entropy"].append(torch.distributions.Categorical(probs = implied_probs).entropy().item())
        DATA["utility_marginal_posterior_entropy"].append(torch.distributions.Categorical(probs = uparam_marginal_probs).entropy().item())
        DATA["hyperparameter_marginal_posterior_entropy"].append(torch.distributions.Categorical(probs = hparam_marginal_probs).entropy().item())
        DATA["utility_inference_error"].append(uparam_particle_errors.mean().item())
        DATA["hyperparameter_inference_error"].append(hparam_particle_errors.mean().item())
        DATA["n_unique_particles"].append(N_PARTICLES)
        if calculate_regret:
            marginal_particle_set_idxs, marginal_dedup_particle_idxs = construct_uparam_marginal_dedup_particle_set_idxs(particle_set_idxs, N_UPARAM_PARTICLES, N_HPARAM_PARTICLES)
            DATA["recommendation_regret"].append(true_pareto_optimum_utility - ptask.utility(find_pareto_optimum(ptask, uparam_particle_set, marginal_particle_set_idxs, marginal_dedup_particle_idxs, N_PARTICLES)[None,None,:], true_w).item())
        if save_posterior:
            DATA["utility_params_posterior"] = [uparam_marginal_probs.numpy().tolist()]
        if calculate_context_effect:
            DATA["context_effect_min_wasserstein"] = []
            DATA["context_effect_min_prob_ratio"] = []
            DATA["context_effect_mean_wasserstein"] = []
            DATA["context_effect_mean_prob_ratio"] = []
            min_option = minimize_utility(ptask, true_w)

    for _ in trange(N_STEPS):
        EIGs = []
        choice_options = ptask.generate_choice_batch(N_QUERY_CANDIDATES)

        best_query = None
        best_query_EIG = 0.0
        dedup_particle_idxs = list(set([i.item() for i in particle_set_idxs]))

        for choice_IDX in range(choice_options.shape[0]):
            with torch.no_grad():
                # the code below is a bit convoluted but does the following:
                # 1. repeat the query for each particle that is still in the particle set (all indices in dedup_particle_idxs)
                # 2. predict the probability of each choice for each of those particles
                # 3. place those predictions into dedup_log_preds, which is of size (N_PARTICLES, N_CHOICES). Indices for particles that are not in the particle set anymore stay 0
                # 4. select the proper log probabilities for each particle index in particle_set_idxs
                query_batched = choice_options[[choice_IDX],:,:].repeat_interleave(len(dedup_particle_idxs), 0)
                dedup_log_preds = torch.zeros((N_PARTICLES,ptask.n_choices))
                dedup_log_preds[dedup_particle_idxs,:] = model(query_batched, uparam_particle_set[dedup_particle_idxs,:], hparam_particle_set[dedup_particle_idxs,:])
                log_preds = dedup_log_preds[particle_set_idxs,:]
                E_choice = log_preds.exp().sum(dim=0) / N_PARTICLES
                EIG = (torch.multiply(log_preds, log_preds.exp()).sum() / N_PARTICLES) - E_choice.log() @ E_choice
                EIGs.append(EIG.item())

                if EIG.item() >= best_query_EIG:
                    best_query = choice_options[[choice_IDX],:,:]
                    best_query_EIG = EIG.item()

        with torch.no_grad():
            true_choice = torch.multinomial(true_model(best_query, true_w, true_h).exp(),1).flatten().item()

            query_batched = best_query.repeat_interleave(len(dedup_particle_idxs), 0)
            dedup_obs_probs = torch.zeros((N_PARTICLES,ptask.n_choices))
            dedup_obs_probs[dedup_particle_idxs,:] = model(query_batched, uparam_particle_set[dedup_particle_idxs,:], hparam_particle_set[dedup_particle_idxs,:])
            obs_probs = dedup_obs_probs[particle_set_idxs,true_choice].exp()
            obs_probs /= obs_probs.sum()

            idx_idxs = torch.multinomial(obs_probs, N_PARTICLES, True)
            particle_set_idxs = particle_set_idxs[idx_idxs]
            implied_probs = torch.zeros(N_PARTICLES)
            for idx in particle_set_idxs:
                implied_probs[idx.item()] += 1/N_PARTICLES
            uparam_marginal_probs = calculate_marginal_uparam_probs(implied_probs, N_UPARAM_PARTICLES, N_HPARAM_PARTICLES)
            hparam_marginal_probs = calculate_marginal_hparam_probs(implied_probs, N_UPARAM_PARTICLES, N_HPARAM_PARTICLES)
            n_unique_particles = (implied_probs != 0.0).to(dtype=torch.float).sum().item()
            posterior_entropy = torch.distributions.Categorical(probs = implied_probs).entropy().item()
            uparam_marginal_posterior_entropy = torch.distributions.Categorical(probs = uparam_marginal_probs).entropy().item()
            hparam_marginal_posterior_entropy = torch.distributions.Categorical(probs = hparam_marginal_probs).entropy().item()
            uparam_inference_error = uparam_particle_errors[particle_set_idxs].mean().item()
            hparam_inference_error = hparam_particle_errors[particle_set_idxs].mean().item()
            if calculate_regret:
                marginal_particle_set_idxs, marginal_dedup_particle_idxs = construct_uparam_marginal_dedup_particle_set_idxs(particle_set_idxs, N_UPARAM_PARTICLES, N_HPARAM_PARTICLES)
                recommendation_regret = true_pareto_optimum_utility - ptask.utility(find_pareto_optimum(ptask, uparam_particle_set, marginal_particle_set_idxs, marginal_dedup_particle_idxs, N_PARTICLES)[None,None,:], true_w).item()

        DATA["utility_inference_error"].append(uparam_inference_error)
        DATA["hyperparameter_inference_error"].append(hparam_inference_error)
        DATA["posterior_entropy"].append(posterior_entropy)
        DATA["utility_marginal_posterior_entropy"].append(uparam_marginal_posterior_entropy)
        DATA["hyperparameter_marginal_posterior_entropy"].append(hparam_marginal_posterior_entropy)
        if calculate_regret:
            DATA["recommendation_regret"].append(recommendation_regret)
        DATA['n_unique_particles'].append(n_unique_particles)
        DATA["EIG_max"].append(np.max(EIGs))
        DATA["EIG_90"].append(np.percentile(EIGs, 90))
        if save_posterior:
            DATA["utility_params_posterior"].append(uparam_marginal_probs.numpy().tolist())
        if calculate_context_effect:
            DATA["context_effect_min_wasserstein"].append(calculate_context_effect_min_wasserstein(ptask, best_query, true_model, true_w, true_h, min_option))
            DATA["context_effect_min_prob_ratio"].append(calculate_context_effect_min_prob_ratio(ptask, best_query, true_model, true_w, true_h, min_option))
            DATA["context_effect_mean_wasserstein"].append(calculate_context_effect_mean_wasserstein(ptask, best_query, true_model, true_w, true_h))
            DATA["context_effect_mean_prob_ratio"].append(calculate_context_effect_mean_prob_ratio(ptask, best_query, true_model, true_w, true_h))


    return DATA, particle_set_idxs

def create_SUS_DATA(N_UPARAM_PARTICLES, N_HPARAM_PARTICLES):
    DATA = {"N_UPARAM_PARTICLES": N_UPARAM_PARTICLES, "N_HPARAM_PARTICLES": N_HPARAM_PARTICLES, "N_PARTICLES": N_UPARAM_PARTICLES * N_HPARAM_PARTICLES}
    DATA["posterior_entropy"] = []
    DATA["utility_marginal_posterior_entropy"] = []
    DATA["hyperparameter_marginal_posterior_entropy"] = []
    DATA["n_unique_particles"] = []
    DATA["EIG_max"] = []
    DATA["EIG_90"] = []
    DATA["mean_ll_val"] = []
    DATA["mean_l_val"] = []
    DATA["query_index"] = []

    return DATA

def evaluate_inferences(simulator, model, uparam_particle_set, hparam_particle_set, particle_set_idxs, dedup_particle_idxs, particle_batch_size = 4096):
    mean_lls = torch.zeros(hparam_particle_set.shape[0], device = simulator.device)
    mean_ls = torch.zeros(hparam_particle_set.shape[0], device = simulator.device)
    x_val, y_val = simulator.get_evaluation_data()
    for val_IDX in range(x_val.shape[0]):
        with torch.no_grad():
            dedup_log_preds = torch.zeros((hparam_particle_set.shape[0],simulator.n_choices), device = simulator.device)
            for i in range(len(dedup_particle_idxs)//particle_batch_size + 1):
                batch_dedup_particle_idxs = dedup_particle_idxs[i*particle_batch_size:(i+1)*particle_batch_size]
                query_batched = x_val[[val_IDX],:,:].repeat_interleave(len(batch_dedup_particle_idxs), 0)
                dedup_log_preds[batch_dedup_particle_idxs,:] = model(query_batched, uparam_particle_set[batch_dedup_particle_idxs,:], hparam_particle_set[batch_dedup_particle_idxs,:])
            lls = dedup_log_preds[particle_set_idxs,y_val[val_IDX]]
            mean_lls += lls
            mean_ls += lls.exp()
    return (mean_lls / y_val.shape[0]).mean().item(), (mean_ls / y_val.shape[0]).mean().item()

def run_simulated_elicitation(ptask, simulator, model, uparam_particle_set, hparam_particle_set, N_STEPS, DATA, validate_inferences = True, particle_batch_size = 4096, ED_strategy = "BOED"):
    N_PARTICLES = DATA["N_PARTICLES"]
    N_UPARAM_PARTICLES = DATA["N_UPARAM_PARTICLES"]
    N_HPARAM_PARTICLES = DATA["N_HPARAM_PARTICLES"]
    assert (uparam_particle_set.shape[0] == N_PARTICLES) and (hparam_particle_set.shape[0] == N_PARTICLES)

    particle_set_idxs = torch.tensor(list(range(N_PARTICLES)))
    dedup_particle_idxs = list(set([i.item() for i in particle_set_idxs]))
    implied_probs = torch.ones(N_PARTICLES) / N_PARTICLES
    uparam_marginal_probs = calculate_marginal_uparam_probs(implied_probs, N_UPARAM_PARTICLES, N_HPARAM_PARTICLES)
    hparam_marginal_probs = calculate_marginal_hparam_probs(implied_probs, N_UPARAM_PARTICLES, N_HPARAM_PARTICLES)
    DATA["posterior_entropy"].append(torch.distributions.Categorical(probs = implied_probs).entropy().item())
    DATA["utility_marginal_posterior_entropy"].append(torch.distributions.Categorical(probs = uparam_marginal_probs).entropy().item())
    DATA["hyperparameter_marginal_posterior_entropy"].append(torch.distributions.Categorical(probs = hparam_marginal_probs).entropy().item())
    DATA["n_unique_particles"].append(N_PARTICLES)
    if validate_inferences:
        mean_ll_val, mean_l_val = evaluate_inferences(simulator, model, uparam_particle_set, hparam_particle_set, particle_set_idxs, particle_set_idxs, particle_batch_size=particle_batch_size)
        DATA["mean_ll_val"].append(mean_ll_val)
        DATA["mean_l_val"].append(mean_l_val)

    for _ in trange(N_STEPS):
        EIGs = []
        choice_options = simulator.get_available_queries()

        best_query = None
        best_query_idx = 0

        if ED_strategy == "BOED":
            best_query_EIG = 0.0
            for choice_IDX in range(choice_options.shape[0]):
                with torch.no_grad():
                    # the code below is a bit convoluted but does the following:
                    # 1. repeat the query for each particle that is still in the particle set (all indices in dedup_particle_idxs)
                    # 2. predict the probability of each choice for each of those particles
                    # 3. place those predictions into dedup_log_preds, which is of size (N_PARTICLES, N_CHOICES). Indices for particles that are not in the particle set anymore stay 0
                    # 4. select the proper log probabilities for each particle index in particle_set_idxs
                    dedup_log_preds = torch.zeros((N_PARTICLES,ptask.n_choices), device = simulator.device)
                    for i in range(len(dedup_particle_idxs)//particle_batch_size + 1):
                        batch_dedup_particle_idxs = dedup_particle_idxs[i*particle_batch_size:(i+1)*particle_batch_size]
                        query_batched = choice_options[[choice_IDX],:,:].repeat_interleave(len(batch_dedup_particle_idxs), 0)
                        dedup_log_preds[batch_dedup_particle_idxs,:] = model(query_batched, uparam_particle_set[batch_dedup_particle_idxs,:], hparam_particle_set[batch_dedup_particle_idxs,:])
                    log_preds = dedup_log_preds[particle_set_idxs,:]
                    E_choice = log_preds.exp().sum(dim=0) / N_PARTICLES
                    EIG = (torch.multiply(log_preds, log_preds.exp()).sum() / N_PARTICLES) - E_choice.log() @ E_choice
                    EIGs.append(EIG.item())

                    if EIG.item() >= best_query_EIG:
                        best_query = choice_options[[choice_IDX],:,:]
                        best_query_idx = choice_IDX
                        best_query_EIG = EIG.item()
        elif ED_strategy == "random":
            best_query_idx = torch.randint(0, choice_options.shape[0], (1,)).item()
            best_query = choice_options[[best_query_idx],:,:]
        else:
            raise NotImplementedError

        with torch.no_grad():
            original_query_index = simulator.get_original_index(best_query_idx).item()
            true_choice = simulator.simulate_choice(best_query_idx)
            
            dedup_obs_probs = torch.zeros((N_PARTICLES,ptask.n_choices), device = simulator.device)
            for i in range(len(dedup_particle_idxs)//particle_batch_size + 1):
                batch_dedup_particle_idxs = dedup_particle_idxs[i*particle_batch_size:(i+1)*particle_batch_size]
                query_batched = best_query.repeat_interleave(len(batch_dedup_particle_idxs), 0)
                dedup_obs_probs[batch_dedup_particle_idxs,:] = model(query_batched, uparam_particle_set[batch_dedup_particle_idxs,:], hparam_particle_set[batch_dedup_particle_idxs,:])
            obs_probs = dedup_obs_probs[particle_set_idxs,true_choice].exp()
            obs_probs /= obs_probs.sum()

            # resampling has to run on CPU due to memory issues with MPS
            idx_idxs = torch.multinomial(obs_probs.cpu(), N_PARTICLES, True).to(device = particle_set_idxs.device)
            particle_set_idxs = particle_set_idxs[idx_idxs]
            dedup_particle_idxs = list(set([i.item() for i in particle_set_idxs]))
            implied_probs = torch.zeros(N_PARTICLES)
            for idx in particle_set_idxs:
                implied_probs[idx.item()] += 1/N_PARTICLES
            uparam_marginal_probs = calculate_marginal_uparam_probs(implied_probs, N_UPARAM_PARTICLES, N_HPARAM_PARTICLES)
            hparam_marginal_probs = calculate_marginal_hparam_probs(implied_probs, N_UPARAM_PARTICLES, N_HPARAM_PARTICLES)
            n_unique_particles = (implied_probs != 0.0).to(dtype=torch.float).sum().item()
            posterior_entropy = torch.distributions.Categorical(probs = implied_probs).entropy().item()
            uparam_marginal_posterior_entropy = torch.distributions.Categorical(probs = uparam_marginal_probs).entropy().item()
            hparam_marginal_posterior_entropy = torch.distributions.Categorical(probs = hparam_marginal_probs).entropy().item()

        DATA["posterior_entropy"].append(posterior_entropy)
        DATA["utility_marginal_posterior_entropy"].append(uparam_marginal_posterior_entropy)
        DATA["hyperparameter_marginal_posterior_entropy"].append(hparam_marginal_posterior_entropy)
        DATA['n_unique_particles'].append(n_unique_particles)
        if ED_strategy == "BOED":
            DATA["EIG_max"].append(np.max(EIGs))
            DATA["EIG_90"].append(np.percentile(EIGs, 90))
        DATA["query_index"].append(original_query_index) # record the true index of the query within the user study data.
        if validate_inferences:
            mean_ll_val, mean_l_val = evaluate_inferences(simulator, model, uparam_particle_set, hparam_particle_set, particle_set_idxs, dedup_particle_idxs, particle_batch_size=particle_batch_size)
            DATA["mean_ll_val"].append(mean_ll_val)
            DATA["mean_l_val"].append(mean_l_val)

    return DATA