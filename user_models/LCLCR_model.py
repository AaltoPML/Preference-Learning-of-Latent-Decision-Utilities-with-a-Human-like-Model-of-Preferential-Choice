import torch
from os.path import join

from models.models import AmortizedConditionalChoicePredictionModel

class LCLACCPModel(AmortizedConditionalChoicePredictionModel):
    def __init__(self, task, n_choices, n_attributes, hyperparamter_dim, control_latent_dim = 64, main_sizes = [1024, 256, 128], controller_sizes = [128, 64], dropout_rate = 0.0, u_dropout_rate = 0.0):
        super().__init__(n_choices, n_attributes, hyperparamter_dim, control_latent_dim = control_latent_dim, main_sizes = main_sizes, controller_sizes = controller_sizes, dropout_rate = dropout_rate, u_dropout_rate = u_dropout_rate)

        self.task = task
        self.weight_scaler_dim = (n_attributes, n_attributes)
        self.weight_scaler_dim_flat = n_attributes*n_attributes

    def forward(self, x, w, h):
        h_CR = h[:,:-self.weight_scaler_dim_flat]
        h_LCL = h[:,-self.weight_scaler_dim_flat:]

        xC = self.task.normalize_choice_batch(x).mean(dim=1)[:,:,None]
        A = h_LCL.reshape(w.shape[0], *self.weight_scaler_dim)
        w_p = w + A.bmm(xC).reshape(w.shape)
        return super().forward(x, w_p / w_p.norm(dim=1)[:,None], h_CR)
    
def load_LCLCR_model(task, CHECKPOINT_DIR, device=torch.device("cpu")):
    model_kwargs = torch.load(join(CHECKPOINT_DIR, "model_kwargs.pth"))

    CR_model = LCLACCPModel(task, task.n_choices, task.n_attributes, task.parameter_dim + task.hyperparameter_dim, **model_kwargs).to(device=device)
    CR_model.load_state_dict(torch.load(join(CHECKPOINT_DIR, "PC_model.pth"), map_location=device))
    CR_model.eval()
    return CR_model