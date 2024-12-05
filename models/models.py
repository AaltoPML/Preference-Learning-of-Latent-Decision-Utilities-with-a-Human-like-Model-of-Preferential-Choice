import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import tqdm

class AmortizedConditionedExpectedUtilityModel(nn.Module):
    def __init__(self, observation_dim, n_choices, hyperparamter_dim, control_latent_dim = 64, main_sizes = [512, 256, 128], controller_sizes = [128, 64], dropout_rate = 0.01):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(observation_dim, main_sizes[0])
        self.fc2 = nn.Linear(main_sizes[0] + control_latent_dim, main_sizes[1])
        self.fc3 = nn.Linear(main_sizes[1] + control_latent_dim, main_sizes[2])
        self.fc4 = nn.Linear(main_sizes[2], n_choices)

        self.fcue1 = nn.Linear(hyperparamter_dim, controller_sizes[0])
        self.fcue2 = nn.Linear(controller_sizes[0], controller_sizes[1])
        self.fcue3 = nn.Linear(controller_sizes[1], control_latent_dim)

        self.control_latent_dim = control_latent_dim
        self.main_sizes = main_sizes
        self.controller_sizes = controller_sizes
        self.dropout_rate = dropout_rate

    def forward(self, o, u, h):
        ctrl_enc = self.fcue1(torch.cat([u, h], dim=1))
        ctrl_enc = F.relu(ctrl_enc)
        ctrl_enc = self.fcue2(ctrl_enc)
        ctrl_enc = F.relu(ctrl_enc)
        ctrl_enc = self.fcue3(ctrl_enc)
        ctrl_enc = F.relu(ctrl_enc)

        x = self.fc1(o)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(torch.cat([x, ctrl_enc], dim=1))
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(torch.cat([x, ctrl_enc], dim=1))
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
    
class AmortizedConditionalChoicePredictionModel(nn.Module):
    """
    Predict the (log!) probability of a choice being made.
    """
    def __init__(self, n_choices, n_attributes, hyperparamter_dim, control_latent_dim = 64, main_sizes = [1024, 256, 128], controller_sizes = [128, 64], dropout_rate = 0.0, u_dropout_rate = 0.0):
        super().__init__()
        self.input_dim = n_choices * n_attributes
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.input_dim, main_sizes[0])
        self.fc2 = nn.Linear(main_sizes[0] + control_latent_dim, main_sizes[1])
        self.fc3 = nn.Linear(main_sizes[1] + control_latent_dim, main_sizes[2])
        self.fc4 = nn.Linear(main_sizes[2], n_choices)

        self.u_dropout1 = nn.Dropout(u_dropout_rate)
        self.u_dropout2 = nn.Dropout(u_dropout_rate)
        self.fcue1 = nn.Linear(hyperparamter_dim, controller_sizes[0])
        self.fcue2 = nn.Linear(controller_sizes[0], controller_sizes[1])
        self.fcue3 = nn.Linear(controller_sizes[1], control_latent_dim)

        self.control_latent_dim = control_latent_dim
        self.main_sizes = main_sizes
        self.controller_sizes = controller_sizes
        self.dropout_rate = dropout_rate
        self.u_dropout_rate = u_dropout_rate

    def forward(self, o, u, h):
        ctrl_enc = self.fcue1(torch.cat([u, h], dim=1))
        ctrl_enc = F.relu(ctrl_enc)
        self.u_dropout1(ctrl_enc)
        ctrl_enc = self.fcue2(ctrl_enc)
        ctrl_enc = F.relu(ctrl_enc)
        self.u_dropout2(ctrl_enc)
        ctrl_enc = self.fcue3(ctrl_enc)
        ctrl_enc = F.relu(ctrl_enc)

        x = self.fc1(o.reshape(-1, self.input_dim))
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(torch.cat([x, ctrl_enc], dim=1))
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(torch.cat([x, ctrl_enc], dim=1))
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
    
    """
    Infers parameters and hyperparameters of the model for which this is a surrogate
        x: batch of choice sets (batch_size, n_choices, n_parameters)
        y: batch of choices made (batch_size,)
        w_init: initial value for utility parameters (n_parameters,)
        h_init: initial value for model parameters (n_hyperparameters,)
    """  
    def infer(self, x, y, w_init, h_init, n_iter = 1000, lr = 0.01, transform_params = None, transform_hparams = None, verbose = False, aggregation = "mean", x_test = None, y_test = None, stop_margin_steps = 50, hparam_regularizer = None):
        assert transform_params is not None, "transform_params must be passed as argument!"
        assert transform_hparams is not None, "transform_hparams must be passed as argument!"

        w_inferred, h_inferred = w_init.clone(), h_init.clone()
        w_inferred.requires_grad = True
        h_inferred.requires_grad = True
        optimizer = optim.Adam([w_inferred, h_inferred], lr=lr)

        do_validation = (x_test is not None) and (y_test is not None)
        last_best_loss = torch.inf
        iter_since_last_best_loss = 0
        for _ in (tqdm_train := tqdm.tqdm(range(n_iter), disable = not verbose)):
            optimizer.zero_grad()
            log_probs = self(x, transform_params(w_inferred)[None,:].repeat_interleave(x.shape[0], 0), transform_hparams(h_inferred)[None,:].repeat_interleave(x.shape[0], 0))
            if aggregation == "sum":
                loss = - log_probs.gather(1, y[:,None].to(dtype = torch.int64)).sum()
            elif aggregation == "mean":
                loss = - log_probs.gather(1, y[:,None].to(dtype = torch.int64)).mean()
            else:
                raise NotImplementedError
            if hparam_regularizer is not None:
                scale = 1.0 if (aggregation == "sum") else 1/x.shape[0]
                loss += scale * hparam_regularizer(transform_hparams(h_inferred))
            tqdm_train.set_description('Loss: {:5f}'.format(loss.item()))
            if do_validation:
                with torch.no_grad():
                    log_probs_test = self(x_test, transform_params(w_inferred)[None,:].repeat_interleave(x_test.shape[0], 0), transform_hparams(h_inferred)[None,:].repeat_interleave(x_test.shape[0], 0))
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

        return (transform_params(w_inferred).clone().detach(), transform_hparams(h_inferred).clone().detach())