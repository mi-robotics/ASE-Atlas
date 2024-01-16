import torch
import numpy as np
from score_model import ScoreMLP
from vae import VAE

#NOTE: assuming single dimensional data

class VPSDE():

    def __init__(self, config):

        self.N = config['lsgm']['vpsde']['N']
        self.beta_0 = config['lsgm']['vpsde']['beta_min']
        self.beta_1 = config['lsgm']['vpsde']['beta_max']

        self.discrete_betas = torch.linspace(self.beta_0/self.N, self.beta_1/self.N, self.N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        return

    @property
    def T(self):
        return 1
    
    def sde(self, x, t):
        beta_t = self.beta_0 + t*(self.beta_1-self.beta_0) #get the continuous beta
        drift = -0.5*beta_t[:, None] * x #reshape betas for data dim
        diffusion = torch.sqrt(beta_t) #nosie scale, continuous G
        return drift, diffusion
    
    def marginal_prob(self, x,t):   
        log_mean_coeff = -0.25 * t ** 2 (self.beta_1 - self.beta_0) - 0.5 *t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std
    
    def prior_sampling(self, shape):
        return torch.randn(*shape)
        
    
    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N/2. * np.log(2*np.pi) - torch.sum(z**2, dim=1) / 2.
        return logps
    
    def discretize(self, x, t):
        timestep = (t* (self.N -1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        alpha = self.alphas.to(x.device)[timestep]

        f = torch.sqrt(alpha)[:, None] * x - x
        g = sqrt_beta
        
        return f, g

    



class LSGM(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.vae = VAE(config)
        self.score_model = ScoreMLP(config)
        self.sde = VPSDE(config)

        self.algo_version = config['lsgm']['algo_version']
        
        return

    

    def forward_algo2(self, obs, skill_latents):
        action_mu, latents, params, reconstruction = self.vae(obs, skill_latents)
        
        return

    def forward(self, x):
        return