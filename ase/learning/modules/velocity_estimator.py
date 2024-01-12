

import torch
from torch import nn
from rl_games.algos_torch.running_mean_std import RunningMeanStd


class VelocityEstimator(nn.Module):
    def __init__(self, config, **kwargs):
        super(VelocityEstimator, self).__init__()

        self.input_norm = RunningMeanStd(config['input_dim'])

        self.input_dim = config['input_dim']
        output_dim = config['output_dim']
        activation = get_activation('elu')
        hidden_dims = config['units']

        estimator_layers = []
        estimator_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        estimator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                estimator_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                estimator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                estimator_layers.append(activation)
        # estimator_layers.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)


    
    def forward(self, input):
        input = self.input_norm(input)
        return self.estimator(input)
    
    def inference(self, input):
        with torch.no_grad():
            input = self.input_norm(input)
            return self.estimator(input)
        
    def loss(self, preds, truth):
        return torch.nn.functional.mse_loss(preds, truth)
        
    
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
