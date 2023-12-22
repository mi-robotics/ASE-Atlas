import torch
from learning.ase_models import ModelASEContinuous

class ModelLASDContinuous(ModelASEContinuous):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('lasd', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelLASDContinuous.Network(net)

    class Network(ModelASEContinuous.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)
            return

        
        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            
            net_dict = self.a2c_network(input_dict)
            keys = ['mu', 'sigma', 'vae_latents', 'vae_params', 'vae_recon', 'value', 'states']
            mu, logstd, vae_latents, vae_params, vae_recon, value, states = [net_dict[key] for key in keys]

            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma,
                    'vae_latents':vae_latents,
                    'vae_params':vae_params,
                    'vae_recon':vae_recon,
                }

                amp_obs = input_dict['amp_obs']
                disc_agent_logit = self.a2c_network.eval_disc(amp_obs)
                result["disc_agent_logit"] = disc_agent_logit

                amp_obs_replay = input_dict['amp_obs_replay']
                disc_agent_replay_logit = self.a2c_network.eval_disc(amp_obs_replay)
                result["disc_agent_replay_logit"] = disc_agent_replay_logit

                amp_demo_obs = input_dict['amp_obs_demo']
                disc_demo_logit = self.a2c_network.eval_disc(amp_demo_obs)
                result["disc_demo_logit"] = disc_demo_logit

                amp_obs = input_dict['amp_obs']
                enc_pred = self.a2c_network.eval_enc(amp_obs)
                result["enc_pred"] = enc_pred

                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : value,
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result