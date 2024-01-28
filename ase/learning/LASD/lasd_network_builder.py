

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np
import enum

from learning.ase_network_builder import ASEBuilder 

from .vae import VAE
from .lsgm import LSGM

ENC_LOGIT_INIT_SCALE = 0.1

class LatentType(enum.Enum):
    uniform = 0
    sphere = 1

class LASDBuilder(ASEBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(ASEBuilder.Network):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.get('actions_num')
            input_shape = kwargs.get('input_shape')
            self.value_size = kwargs.get('value_size', 1)
            self.num_seqs = num_seqs = kwargs.get('num_seqs', 1)
            amp_input_shape = kwargs.get('amp_input_shape')
            self._ase_latent_shape = kwargs.get('ase_latent_shape')

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            
            self.load(params)

            _, critic_out_size = self._build_actor_critic_net(input_shape, self._ase_latent_shape, actions_num)

            self.value = torch.nn.Linear(critic_out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)
                       

            if self.is_continuous:
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 

                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if (not self.space_config['learn_sigma']):
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                elif self.space_config['fixed_sigma']:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    raise Exception('Method Not Implimented')
                    self.sigma = torch.nn.Linear(actor_out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():         
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

         
            self.critic_mlp.init_params()

            if self.is_continuous:
                if self.space_config['fixed_sigma']:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)

            self._build_disc(amp_input_shape)
            self._build_enc(amp_input_shape)

            return
        
        def load(self, params):
            super().load(params)
           
            self._enc_units = params['enc']['units']
            self._enc_activation = params['enc']['activation']
            self._enc_initializer = params['enc']['initializer']
            self._enc_separate = params['enc']['separate']

            self._vae_params = params.get('vae', False)
            self._lsgm_params = params.get('lsgm', False)

            assert not( self._vae_params and self._lsgm_params)
            assert self._vae_params or self._lsgm_params

            self._use_vae = self._vae_params is not False
            self._use_lsgm = self._lsgm_params is not False
           

            return

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            ase_latents = obs_dict['ase_latents']
            states = obs_dict.get('rnn_states', None)
            use_hidden_latents = obs_dict.get('use_hidden_latents', False)

            actor_dict = self.eval_actor(obs, ase_latents)
            value = self.eval_critic(obs, ase_latents, )

            actor_dict.update({'value':value})
            actor_dict.update({'states':states})

            return actor_dict


        def eval_actor(self, obs, ase_latents, use_hidden_latents=False):
            a_out = self.actor_cnn(obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            mu, latents, params, recon = self.actor(a_out, ase_latents)
                    
            if self.is_continuous:
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    raise Exception('Invalid Configuration - Assuming Fixed Sigma')

                return {
                    'mu':mu,
                    'sigma':sigma,
                    'vae_latents': latents,
                    'vae_params': params,
                    'vae_recon':recon
                }
            
            else:
                raise Exception('Invalid Configuration - Action space must be continuous')
            
           
        def _build_actor_critic_net(self, input_shape, ase_latent_shape, num_actions):

            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            
            act_fn = self.activations_factory.create(self.activation)
            initializer = self.init_factory.create(**self.initializer)

            if self._use_vae:
                vae_config = {
                    'vae':self._vae_params,
                    'num_actions':num_actions,
                    'ase_latent_shape':ase_latent_shape,
                    'obs_shape': input_shape
                }
                self.actor = VAE(vae_config)

            elif self._use_lsgm:
      
                lsgm_config = {
                    'lsgm':self._lsgm_params,
                    'vae':self._lsgm_params['vae'],
                    'score_model': self._lsgm_params['score_model'],
                    'num_actions':num_actions,
                    'ase_latent_shape':ase_latent_shape,
                    'obs_shape': input_shape
                }
                self.actor = LSGM(lsgm_config)
                pass

            else:
                raise Exception('LSGM config error')


            if self.separate:
                self.critic_mlp = AMPMLPNet(obs_size=input_shape[-1],
                                            ase_latent_size=ase_latent_shape[-1],
                                            units=self.units,
                                            activation=act_fn,
                                            initializer=initializer)

            return None, self.critic_mlp.get_out_size()


    def build(self, name, **kwargs):
        net = LASDBuilder.Network(self.params, **kwargs)
        return net


class AMPMLPNet(torch.nn.Module):
    def __init__(self, obs_size, ase_latent_size, units, activation, initializer):
        super().__init__()

        input_size = obs_size + ase_latent_size
        print('build amp mlp net:', input_size)
        
        self._units = units
        self._initializer = initializer
        self._mlp = []

        in_size = input_size
        for i in range(len(units)):
            unit = units[i]
            curr_dense = torch.nn.Linear(in_size, unit)
            self._mlp.append(curr_dense)
            self._mlp.append(activation)
            in_size = unit

        self._mlp = nn.Sequential(*self._mlp)
        self.init_params()
        return

    def forward(self, obs, latent, skip_style):
        inputs = [obs, latent]
        input = torch.cat(inputs, dim=-1)
        output = self._mlp(input)
        return output

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._initializer(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        return

    def get_out_size(self):
        out_size = self._units[-1]
        return out_size

