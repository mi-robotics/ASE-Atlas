import torch
import numpy as np
from score_model import ScoreMLP
from vae import VAE
from distributions import Normal

#NOTE: assuming single dimensional data

class VPSDE():

    def __init__(self, config):

        self.N = config['lsgm']['vpsde']['N']
        self.eps_t = config['lsgm']['vpsde']['eps_t']
        self.sigma2_0 = config['lsgm']['vpsde']['sigma2_0'] #TODO: waht should this be
        self.beta_0 = config['lsgm']['vpsde']['beta_min']
        self.beta_1 = config['lsgm']['vpsde']['beta_max']

        self.discrete_betas = torch.linspace(self.beta_0/self.N, self.beta_1/self.N, self.N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.loss_constant = 0.5 * (1.0 + torch.log(2.0 * np.pi * self.var(t=torch.tensor(self.eps_t, device='cuda'))))
        
        return

    @property
    def T(self):
        return 1
    
    def sde(self, x, t):
        beta_t = self.beta_0 + t*(self.beta_1-self.beta_0) #get the continuous beta
        drift = -0.5 * beta_t[:, None] * x #reshape betas for data dim
        diffusion = torch.sqrt(beta_t) #nosie scale, continuous G
        return drift, diffusion
    
    def marginal_prob(self, x,t):   
        log_mean_coeff = -0.25 * t ** 2 (self.beta_1 - self.beta_0) - 0.5 *t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std
    
    def var(self, t):
        log_mean_coeff = -0.25 * t ** 2 (self.beta_1 - self.beta_0) - 0.5 *t * self.beta_0
        return 1. - torch.exp(2. * log_mean_coeff)
    
    def inv_var(self, var):
        c = torch.log((1 - var) / (1 - self.sigma2_0))
        a = self.beta_end - self.beta_start
        t = (-self.beta_start + torch.sqrt(np.square(self.beta_start) - 2 * a * c)) / a
        return t
       

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
    
    def sample_q_t(self, x, mean, std, noise):
        return mean * x + std * noise 
        

    def iw_logl_uniform(self, latents):
        t = torch.rand(latents.shape[0], device=latents.device) * (1. - self.eps_t) + self.eps_t
        g2 = self.sde.sde(torch.zeros_like(latents), t)[1] ** 2
        mean, std = self.marginal_prob(torch.zeros_like(latents), t)
        return t, mean, std, g2 
    
    def iw_logl_importance_sampling(self, latents):
        
        ones = torch.ones_like(latents, device=latents.device)
        sigma2_1, sigma2_eps = self.var(ones), self.var(ones*self.eps_t)
        log_sigma2_1, log_sigma2_eps = torch.log(sigma2_1), torch.log(sigma2_eps)

        r = torch.rand_like(latents, device=latents.device)
        var_t = torch.exp(r * log_sigma2_1 + (1-r) * log_sigma2_eps)
        t = self.inv_var(var_t)
        mean, std = self.marginal_prob(latents, t) 
        g2 = self.sde.sde(torch.zeros_like(latents), t)[1] ** 2

        return t, mean, std, g2
    
  



class LSGM(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self._latent_dim = config['vae']['latent_dim']
        self._used_mixed_predictions = config['lsgm']['used_mixed_predictions']

        self._mixing_logit = None
        if self._used_mixed_predictions:
            init = config['lsgm']['mixed_logit_init']
            init = init*torch.ones((1, self._latent_dim))
            self._mixing_logit = torch.nn.Parameter(init, requires_grad=True)

        self.vae = VAE(config)
        self.score_model = ScoreMLP(config)
        self.sde = VPSDE(config)

        self.algo_version = config['lsgm']['algo_version']


        
        return
    
    def get_mixed_prediction(self, noised_latents, std, pred):
        alpha = torch.sigmoid(self._mixing_logit)
        mixer = std * noised_latents
        pred = (1-alpha) * mixer + alpha * pred
        return pred

    
    def vae_loss_algo2(self, latents, params, reconstruction):
        """
        recon + sudo kl (neg_entropy)
        TODO: Ensure we do not update the diffussion network 
        TODO: Ensure the vae is getting gradients from the diffussion network
        """
        # --------------- VAE LOSS TERMS
        # get log q
        mu, log_var = params.chunk(2, dim=-1)
        dist = Normal(mu, log_var)
        log_prob_q = dist.log_p(latents)
        
        # sum for the negative log entropy

        # compute recontruction loss

        # --------------- SCORE LOSS TERMS

        # sample z_T
        noise_T = torch.randn(size=latents.size())

        #apply diffussion steps
        t, mean, std, g2 = self.sde.iw_logl_importance_sampling(latents)

        #sample noised latetents
        noised_latents = self.sde.sample_q_t(latents, mean, std, noise_T )

        #the model predictions
        pred = self.score_model.forward(noised_latents, t)
        if self._used_mixed_predictions:
            pred = self.get_mixed_prediction(noised_latents, std, pred)

        #calc diffussion based loss terms 
        l2_term = torch.square(params - noise_T)
        #TODO we need to get the wieghting term
        cross_entropy = 0 # w * l2_term
        cross_entropy += self.sde.loss_constant
        cross_entropy = torch.sum(cross_entropy, dim=1)

        #TODO: remaining_neg_log_p_total
        kl = None #vae_neg_entropy + cross_entropy
        nelbo_loss = None #kl_coeff * kl + vae_recon_loss
        regularizer = None

        q_loss = torch.mean(nelbo_loss) + regularizer
        return
    
    def score_loss_algo2(self):
        return

    def vae_forward_algo2(self, obs, skill_latents):
        return self.vae(obs, skill_latents)
    

    def forward(self, x):
        return