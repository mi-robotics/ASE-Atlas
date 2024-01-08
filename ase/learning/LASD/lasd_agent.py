# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from learning.ase_agent import ASEAgent

import torch
from torch import nn
from isaacgym.torch_utils import *
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from utils import torch_utils
from learning import ase_network_builder

class LASDAgent(ASEAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)

        print(config.keys())
        self._vae_latent_dim = self.model.a2c_network.actor_vae.latent_dim
        self._vae_beta_coef = self.model.a2c_network.actor_vae.beta
        return
    
    def calc_gradients(self, input_dict):

        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']

        obs_batch = self._preproc_obs(obs_batch)

        amp_obs = input_dict['amp_obs'][0:self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)
        if (self._enable_enc_grad_penalty()):
            amp_obs.requires_grad_(True)

        amp_obs_replay = input_dict['amp_obs_replay'][0:self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        amp_obs_demo = input_dict['amp_obs_demo'][0:self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)

        ase_latents = input_dict['ase_latents']
        
        rand_action_mask = input_dict['rand_action_mask']
        rand_action_sum = torch.sum(rand_action_mask)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
            'amp_obs' : amp_obs,
            'amp_obs_replay' : amp_obs_replay,
            'amp_obs_demo' : amp_obs_demo,
            'ase_latents': ase_latents,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len
            
        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            disc_agent_logit = res_dict['disc_agent_logit']
            disc_agent_replay_logit = res_dict['disc_agent_replay_logit']
            disc_demo_logit = res_dict['disc_demo_logit']
            enc_pred = res_dict['enc_pred']

            vae_latents = res_dict['vae_latents']
            vae_params = res_dict['vae_params']
            vae_recon = res_dict['vae_recon']

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']
            a_clipped = a_info['actor_clipped'].float()

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)

            c_loss = torch.mean(c_loss)
            a_loss = torch.sum(rand_action_mask * a_loss) / rand_action_sum
            entropy = torch.sum(rand_action_mask * entropy) / rand_action_sum
            b_loss = torch.sum(rand_action_mask * b_loss) / rand_action_sum
            a_clip_frac = torch.sum(rand_action_mask * a_clipped) / rand_action_sum
            
            disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
            disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo)
            disc_loss = disc_info['disc_loss']
            
            enc_latents = batch_dict['ase_latents'][0:self._amp_minibatch_size]
            enc_loss_mask = rand_action_mask[0:self._amp_minibatch_size]
            enc_info = self._enc_loss(enc_pred, enc_latents, batch_dict['amp_obs'], enc_loss_mask)
            enc_loss = enc_info['enc_loss']

            vae_loss = self._vae_loss(vae_latents, vae_params)

            vae_recon_loss = self._vae_recon_loss(vae_recon, obs_batch, batch_dict['ase_latents'])

            print('vae loss')
            print(vae_loss)
            print('vae recon')
            print(vae_recon_loss)

            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                 + self._disc_coef * disc_loss + self._enc_coef * enc_loss + vae_loss * self._vae_beta_coef + vae_recon_loss*2
            
            if (self._enable_amp_diversity_bonus()):
                diversity_loss = self._diversity_loss(batch_dict['obs'], mu, batch_dict['ase_latents'])
                diversity_loss = torch.sum(rand_action_mask * diversity_loss) / rand_action_sum
                loss += self._amp_diversity_bonus * diversity_loss
                a_info['amp_diversity_loss'] = diversity_loss
                
            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of the year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
        
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss,
            'vae_loss':vae_loss, 
            'vae_recon_loss':vae_recon_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)
        self.train_result.update(enc_info)

        return
    
    def _vae_loss(self, latents, params):
        #reconstruction loss - adversarial rewards 
        #KL loss - to be computers
        return self.model.a2c_network.actor_vae.kl_loss(latents, params)
    
    def _vae_recon_loss(self, vae_recon, obs, ase_latents):
        return self.model.a2c_network.actor_vae.recon_loss(vae_recon, obs, ase_latents)

    def _diversity_loss(self, obs, action_params, ase_latents):
        assert(self.model.a2c_network.is_continuous)

        n = obs.shape[0]
        assert(n == action_params.shape[0])
    
        new_z = self._sample_latents(n)
        net_dict = self._eval_actor(obs=obs, ase_latents=new_z)

        mu = net_dict['mu']

        clipped_action_params = torch.clamp(action_params, -1.0, 1.0)
        clipped_mu = torch.clamp(mu, -1.0, 1.0)

        a_diff = clipped_action_params - clipped_mu
        a_diff = torch.mean(torch.square(a_diff), dim=-1)

        z_diff = new_z * ase_latents
        z_diff = torch.sum(z_diff, dim=-1)
        z_diff = 0.5 - 0.5 * z_diff

        diversity_bonus = a_diff / (z_diff + 1e-5)
        diversity_loss = torch.square(self._amp_diversity_tar - diversity_bonus)

        return diversity_loss



    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['enc_rewards'] = batch_dict['enc_rewards']
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        
        self.writer.add_scalar('losses/enc_loss', torch_ext.mean_list(train_info['enc_loss']).item(), frame)
        self.writer.add_scalar('losses/vae_loss', torch_ext.mean_list(train_info['vae_loss']).item(), frame)
        self.writer.add_scalar('losses/vae_recon_loss', torch_ext.mean_list(train_info['vae_recon_loss']).item(), frame)
         
        if (self._enable_amp_diversity_bonus()):
            self.writer.add_scalar('losses/amp_diversity_loss', torch_ext.mean_list(train_info['amp_diversity_loss']).item(), frame)
        
        enc_reward_std, enc_reward_mean = torch.std_mean(train_info['enc_rewards'])
        self.writer.add_scalar('info/enc_reward_mean', enc_reward_mean.item(), frame)
        self.writer.add_scalar('info/enc_reward_std', enc_reward_std.item(), frame)

        if (self._enable_enc_grad_penalty()):
            self.writer.add_scalar('info/enc_grad_penalty', torch_ext.mean_list(train_info['enc_grad_penalty']).item(), frame)

        return

    def _change_char_color(self, env_ids):
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        self.vec_env.env.task.set_char_color(rand_col, env_ids)
        return

    def _amp_debug(self, info, ase_latents):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs
            ase_latents = ase_latents
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs, ase_latents)
            disc_reward = amp_rewards['disc_rewards']
            enc_reward = amp_rewards['enc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            enc_reward = enc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward, enc_reward)
        return