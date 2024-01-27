import torch
from torch.nn import Linear

class VaeMLP(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.units = config['units']
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.activation = self._get_activation(config['activation'])
        self.output_activation = config['output_activation']
        self._build_network()


    def _build_network(self):

        input_dim = self.input_dim
  
        layers = []
        for units in self.units:
            layers.append(Linear(input_dim, units))
            layers.append(self.activation)
            input_dim = units

        layers.append(Linear(input_dim, self.output_dim))
        layers.append(self.output_activation)

        self.net = torch.nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        return
    
    def _get_activation(self, activation):
        if activation == 'tanh':
            return torch.nn.Tanh()
        elif activation == 'relu':
            return torch.nn.ReLU()
        elif activation == 'sigmoid':
            return torch.nn.Sigmoid()

    def forward(self, x):
        return self.net(x)



class Encoder(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.skill_latent_dim = config['ase_latent_shape'][0]
        self.obs_dim = config['obs_shape'][0]
        self.output_dim = config['vae']['latent_dim']*2 
        self.units = config['vae']['encoder']['units']
        self.activation = config['vae']['activation']

        net_config = {
            'units':self.units,
            'input_dim': self.skill_latent_dim + self.obs_dim,
            'output_dim': self.output_dim,
            'activation':self.activation,
            'output_activation':torch.nn.Identity()
        }

        self.net = VaeMLP(net_config)
        return
    
    def forward(self, obs, skill_latent):

        net_in = torch.cat([obs, skill_latent], dim=1)
        latent_params = self.net(net_in)

        return latent_params
        
    

class Decoder(torch.nn.Module):

    def __init__(self, type, config):
        super().__init__()

        assert type in ['actor', 'recon']

        self.action_dims = config['num_actions']
        self.latent_dims = config['vae']['latent_dim']
        self.recon_skill = config['vae']['recon_skill']
        self.recon_state = config['vae']['recon_state']
        self.recon_next_state = config['vae']['recon_next_state']

        self.obs_dim = config['obs_shape'][0]        
        self.skill_latent_dim = config['ase_latent_shape'][0]

        self.units = config['vae']['decoder']['units']
        self.activation = config['vae']['activation']

        self.input_dim = self.latent_dims
        
        if type == 'actor':
            self.output_dim = self.action_dims
        elif type == 'recon':
            self.output_dim = self.skill_latent_dim + (self.obs_dim if self.recon_state or self.recon_next_state else 0)
        else:
            raise Exception('VAE Decdoer Miss configured type')
        
        net_config = {
            'units':self.units,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'activation':self.activation,
            'output_activation':torch.nn.Identity()
        }

        self.net = VaeMLP(net_config)
        return
    
    def forward(self, latents):
        return self.net(latents)
    

class MultiHeadDecoder(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.action_dims = config['num_actions']
        self.latent_dims = config['vae']['latent_dim']

        self.recon_skill = config['vae']['recon_skill']
        self.recon_state = config['vae']['recon_state']
        self.recon_next_state = config['vae']['recon_next_state']

        self.obs_dim = config['obs_shape'][0]        
        self.skill_latent_dim = config['ase_latent_shape'][0]

        self.units = config['vae']['decoder']['units']
        self.activation = config['vae']['activation']

        self.input_dim = self.latent_dims
        
        
        self.actor_output_dim = self.action_dims
        self.recon_output_dim = self.skill_latent_dim + (self.obs_dim if self.recon_state or self.recon_next_state else 0)
        
        net_config = {
            'units':self.units[:-1],
            'input_dim': self.input_dim,
            'output_dim': self.units[-1],
            'activation':self.activation,
            'output_activation':torch.nn.ReLU()
        }

        self.net_backbone = VaeMLP(net_config)

        self.actor_head = Linear(self.units[-1], self.actor_output_dim)
        self.actor_activation = torch.nn.Identity()

        self.recon_head = Linear(self.units[-1], self.recon_output_dim)
        self.recon_activation = torch.nn.Identity()

        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
    
        return
    
    def forward(self, input):

        input:torch.Tensor = self.net_backbone(input)

        if self.training:
            recon_input, actor_input = input.chunk(2)
            action_mu = self.actor_activation(self.actor_head(actor_input))
            reconstruction = self.recon_activation(self.recon_head(recon_input))
        else:
            action_mu = self.actor_activation(self.actor_head(input))
            reconstruction = None

        return action_mu, reconstruction

            

    

class VAE(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.beta = config['vae'].get('beta', 0.001)
        self.latent_dim = config['vae']['latent_dim']

        self.recon_skill = config['vae']['recon_skill']
        self.recon_state = config['vae']['recon_state']
        self.recon_next_state = config['vae']['recon_next_state']
        self.use_seperate_reconstructor = config['vae']['use_seperate_reconstructor']

        self.encoder = Encoder(config)

        if self.use_seperate_reconstructor:
            self.recon_decoder = Decoder('recon', config)
            self.actor_decoder = Decoder('actor', config)
        else:
            self.decoder = MultiHeadDecoder(config)

        return
    
    def reparmeterize(self, params):
        mu, log_var = params.chunk(2, dim=-1)
        
        if not self.training:
            #return means
            return  None, mu
        else:
            noise = torch.randn(log_var.size(), dtype=log_var.dtype, device=log_var.device)
        
        latents = mu + torch.exp(0.5*log_var) * noise

        return latents, mu
    
    def kl_loss(self, z, params):
        mu, log_var = params.chunk(2, dim=-1)
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    
    def recon_loss(self, recon, ase_latents, obs=None, next_obs=None, reduce=True):
        target = []
        
        if self.recon_state and self.recon_next_state:
            raise Exception('Incorrect vae configuration')
        if self.recon_state:
            target.append(obs)
        if self.recon_next_state:
            target.append(next_obs)
        if self.recon_skill:
            target.append(ase_latents)
        
        target = torch.cat([obs, ase_latents], dim=-1)
            
        return torch.nn.functional.mse_loss( recon, target, reduce=reduce)
       
    def forward(self, obs, skill_latents):

        params = self.encoder(obs, skill_latents)
        latents, mu = self.reparmeterize(params)
  
        if self.use_seperate_reconstructor:
            action_mu = self.actor_decoder(mu)

            if self.training:
                reconstruction = self.recon_decoder(latents)
            else:
                reconstruction = None
        else:
            if self.training:
                decoder_in = torch.cat([latents, mu], dim=0)
                action_mu, reconstruction = self.decoder(decoder_in)
            else:
                action_mu, reconstruction = self.decoder(mu)

            pass

        return action_mu, latents, params, reconstruction
    




if __name__ == '__main__':

    vae = VAE({

    })