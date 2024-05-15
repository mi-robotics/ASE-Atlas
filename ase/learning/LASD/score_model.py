

import torch
from torch.nn import Linear
import math

class ScoreMLP(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config


        self.latent_dim = config['vae']['latent_dim']
        self.ase_latent_dim = 64
        self._use_ase_latent = True
        self.time_embd_dim = config['score_model']['time_embd_dim']      
        self.units = config['score_model']['units']
        self.activation = self._get_activation(config['score_model']['activation'])
        self.output_activation = torch.nn.Identity()

        self._used_mixed_predictions = config['lsgm']['use_mixed_predictions']
        self._mixing_logit = None
        if self._used_mixed_predictions:
            init = config['lsgm']['mixed_logit_init']
            init = init*torch.ones((1, self.latent_dim))
            self._mixing_logit = torch.nn.Parameter(init, requires_grad=True)

        self.input_dim = self.latent_dim + self.time_embd_dim + (self.ase_latent_dim if self._use_ase_latent else 0)

        self._build_network()

    
    def _build_network(self):

        input_dim = self.input_dim
        output_dim = self.latent_dim

        layers = []
        for units in self.units:
            layers.append(Linear(input_dim, units))
            layers.append(self.activation)
            input_dim = units

        layers.append(Linear(input_dim, output_dim))
        layers.append(self.output_activation)

        self.net = torch.nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)


    def _get_activation(self, activation):
        if activation == 'tanh':
            return torch.nn.Tanh()
        elif activation == 'relu':
            return torch.nn.ReLU()
        elif activation == 'sigmoid':
            return torch.nn.Sigmoid()
        

    def forward(self, noised_latent, t, ase_latent):

        #get positional embeddings
        emb_t = get_timestep_embedding(t, embed_dim=self.time_embd_dim, dtype=noised_latent.dtype)

        #concat latents with time embeddings
        if self._use_ase_latent:
            net_in = torch.cat([noised_latent, ase_latent, emb_t], dim=1)
        else:
            net_in = torch.cat([noised_latent,emb_t], dim=1)

        #get the noise predictions
        noise = self.net(net_in)

        return noise 



@torch.jit.script
def get_timestep_embedding(timesteps:torch.Tensor, embed_dim:int, dtype:torch.dtype):
    
    half_dim = embed_dim // 2
    embed = math.log(10000) / (half_dim - 1)
    embed = torch.exp(-torch.arange(half_dim, dtype=dtype, device=timesteps.device)*embed)
    embed = torch.outer(timesteps.ravel().to(dtype), embed)
    embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=1)

    if embed_dim % 2 == 1:
        embed = torch.nn.functional.pad(embed, [0,1])

    assert embed.dtype == dtype
    return embed

if __name__ == '__main__':

    pass