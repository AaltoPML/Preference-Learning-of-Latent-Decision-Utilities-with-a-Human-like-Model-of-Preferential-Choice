import torch
  
def infer_and_validate_SGD_multistart(model, x_train, y_train, x_val, y_val, init_generator, n_starts, test_portion = 0.0, model_kwargs = {}):
    assert n_starts >= 1

    if test_portion > 0.0:
        in_test = torch.bernoulli(torch.ones(x_train.shape[0]), test_portion).to(dtype=torch.bool)
        x_test, y_test = x_train[in_test], y_train[in_test]
        x_train, y_train = x_train[torch.logical_not(in_test)], y_train[torch.logical_not(in_test)]
        model_kwargs["x_test"] = x_test
        model_kwargs["y_test"] = y_test
    else:
        x_test, y_test = x_train, y_train

    w_init, h_init = init_generator()
    best_w, best_h = model.infer(x_train, y_train, w_init, h_init, **model_kwargs)
    best_ll = model(x_test, best_w[None,:].repeat(x_test.shape[0],1), best_h[None,:].repeat(x_test.shape[0],1)).gather(1, y_test[:,None].to(dtype = torch.int64)).sum()

    for _ in range(n_starts-1):
        w_init, h_init = init_generator()
        inferred_w, inferred_h = model.infer(x_train, y_train, w_init, h_init, **model_kwargs)            
        ll_of_inferred = model(x_test, inferred_w[None,:].repeat(x_test.shape[0],1), inferred_h[None,:].repeat(x_test.shape[0],1)).gather(1, y_test[:,None].to(dtype = torch.int64)).sum()
        if ll_of_inferred > best_ll:
            best_w, best_h, best_ll = inferred_w, inferred_h, ll_of_inferred

    with torch.no_grad():
        lls = model(x_val, best_w[None,:].repeat(x_val.shape[0],1), best_h[None,:].repeat(x_val.shape[0],1)).gather(1, y_val[:,None].to(dtype = torch.int64))
        val_ll = lls.sum()
        val_acc = lls.exp().mean()
        return val_ll, val_acc, best_w, best_h
    
def infer_and_validate_SGD(model, x_train, y_train, x_val, y_val, w_init, h_init, test_portion = 0.0, model_kwargs = {}):
    return infer_and_validate_SGD_multistart(model, x_train, y_train, x_val, y_val, lambda : (w_init, h_init), 1, test_portion = test_portion, model_kwargs = model_kwargs)
    
def inv_sigmoid(y):
    return torch.log(y / (1 - y))

def transform_hparams(h):
    return torch.concat([h[0].pow(2)[None], torch.sigmoid(h[1])[None], h[2:].pow(2)])

def inv_transform_hparams(h):
    return torch.concat([h[0].sqrt()[None], torch.log(h[1] / (1 - h[1]))[None], h[2:].sqrt()])

def transform_params_2d(w):
    return torch.hstack([torch.cos(w), torch.sin(w)])

def inv_transform_params_2d(w):
    return torch.atan2(w[1], w[0])

def transform_params_nd(w):
    return w

def inv_transform_params_nd(w):
    return w

def transform_params_nd(w):
    return torch.hstack([w[:i].sin().prod() * w[i].cos() for i in range(w.shape[0])] + [w.sin().prod()])

def inv_transform_params_nd(w):
    return torch.hstack([torch.atan2(w[i+1:].pow(2.0).sum().sqrt(), w[i]) for i in range(w.shape[0]-1)])

def transform_hparams_aux(h, n_aux):
    return torch.concat([h[0].pow(2)[None], torch.sigmoid(h[1])[None], h[2:-n_aux].pow(2), h[-n_aux:]])

def inv_transform_hparams_aux(h, n_aux):
    return torch.concat([h[0].sqrt()[None], torch.log(h[1] / (1 - h[1]))[None], h[2:-n_aux].sqrt(), h[-n_aux:]])

def CR_find_starting_point(ptask, CR_model, x_train, y_train, inv_transform_params, inv_transform_hparams, N_test = 1000):
    w_init = ptask.generate_parameter_batch(1).flatten()
    h_init = ptask.generate_hyperparameter_batch(1).flatten()
    init_log_prob = CR_model(x_train, w_init[None,:].repeat_interleave(x_train.shape[0], 0), h_init[None,:].repeat_interleave(x_train.shape[0], 0)).gather(1, y_train[:,None].to(dtype = torch.int64)).sum()

    with torch.no_grad():
        for _ in range(N_test):
            w_cand = ptask.generate_parameter_batch(1).flatten()
            h_cand = ptask.generate_hyperparameter_batch(1).flatten()
            log_prob = CR_model(x_train, w_cand[None,:].repeat_interleave(x_train.shape[0], 0), h_cand[None,:].repeat_interleave(x_train.shape[0], 0)).gather(1, y_train[:,None].to(dtype = torch.int64)).sum()
            if log_prob > init_log_prob:
                w_init = w_cand
                h_init = h_cand
                init_log_prob = log_prob

    w_init = inv_transform_params(w_init)
    h_init = inv_transform_hparams(h_init)

    return w_init, h_init