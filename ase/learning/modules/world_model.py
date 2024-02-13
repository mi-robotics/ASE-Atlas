import torch 
from torch import nn
from rl_games.algos_torch.running_mean_std import RunningMeanStd

class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_norm = RunningMeanStd(config['input_dim'])

        self.input_dim = config['input_dim']
        output_dim = config['output_dim']
        activation = get_activation('elu')
        hidden_dims = config['units']

        model_layers = []
        model_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        model_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                model_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                model_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                model_layers.append(activation)
        # estimator_layers.append(nn.Tanh())
        self.model = nn.Sequential(*model_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        return
    
    def forward(self, x):
        return self.model(self.input_norm(x))
    
    

class InverseModel(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.input_dim = config['inverse_model']['obs_dim']
        self.output_dim = config['inverse_model']['action_dim']
        self.units = config['inverse_model']['units']
        self.net = MLP({'input_dim':self.input_dim, 'output_dim':self.output_dim, 'units':self.units})
        return
    
    def forward(self, next_state, state):
        '''
        Note we can use privilliged states for this reward as we do not 
        '''
        return self.net(torch.cat((next_state, state), dim=-1))
    
    def loss(self, action_pred, action_truth):
        """
        Mean squared error
        """
        return torch.nn.functional.mse_loss(action_pred, action_truth)

    
class ForwardModel(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.input_dim = config['forward_model']['obs_dim'] + config['forward_model']['action_dim']
        self.output_dim = config['forward_model']['obs_dim']
        self.units = config['forward_model']['units']

        self.net = MLP({'input_dim':self.input_dim, 'output_dim':self.output_dim, 'units':self.units})
        return
    
    def forward(self, state, action):
        return self.net(torch.cat((state,action), dim=-1))
    
    def loss(self, next_state, pred, reduce=True):
        """
        Mean squared error
        """
        if reduce:
            return (0.5 * torch.square(next_state - pred).sum(dom=-1)).mean()
        else:
            return (0.5 * torch.square(next_state - pred).sum(dom=-1))
    

class WorldModel(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.inverse_model:InverseModel = InverseModel(config)
        self.forward_model:ForwardModel = ForwardModel(config)

        return
    
    
    def forward(self, next_state, state, action):
        action_pred = self.inverse_model(next_state, state)
        next_state_pred = self.forward_model(state, action)
        return next_state_pred, action_pred
    

    def loss(self, next_state_pred, next_state, action_pred, action):
        inverse_loss = self.inverse_model.loss(action_pred, action)
        forward_loss = self.forward_model.loss(next_state, next_state_pred)
        return (inverse_loss+forward_loss), {
            'inverse_loss':inverse_loss,
            'forward_loss':forward_loss
        }
    

    def reward(self, next_state, state, action):
        """
        
        """
        with torch.no_grad():
            next_state_pred = self.forward_model(state, action)
            forward_loss = self.forward_model.loss(next_state, next_state_pred, reduce=False)
            reward = (torch.exp(forward_loss) - 1) / (torch.exp(forward_loss) + 1) 
            
        return reward
    


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