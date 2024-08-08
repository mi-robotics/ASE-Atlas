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

from learning import amp_agent 

import torch
from isaacgym.torch_utils import *
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from utils import torch_utils
from learning import ase_network_builder
from learning.modules.velocity_estimator import VelocityEstimator
from copy import deepcopy

class DIMAgent(amp_agent.AMPAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)

        self.modules = {}
        
        try:
            self._use_velocity_estimator = self.vec_env.env.task._use_velocity_observation
        except:
            self._use_velocity_estimator = False

        if self._use_velocity_estimator:
            estimator_config = config['vel_estimator']
            self._train_with_velocity_estimate = estimator_config.get('trainWithVelocityEstimate', False) #rollouts use estimate
            self._optimize_with_velocity_estimate = estimator_config.get('optimizeWithVelocityEstimate', False) #policy optimization uses estimate
            self._vel_est_use_ase_latent = estimator_config.get('use_ase_latent', False)
            self._vel_est_asymetric_train = estimator_config.get('use_asymetric', False)

            input_dim = self.vec_env.env.task._velocity_obs_buf.shape[1]
            if self._use_velocity_estimator:
                input_dim += self._latent_dim

            estimator_config.update({'input_dim':input_dim})
   
            self.vel_estimator = VelocityEstimator(estimator_config)
            self.vel_estimator.to(self.ppo_device)
            self.vel_optim = torch.optim.Adam(self.vel_estimator.parameters(), float(config['vel_estimator']['lr']) )
            self.vel_grad_norm = config['vel_estimator']['grad_norm']
            self._vel_obs_index = (7,10)

            self.modules['VelocityEstimator'] = self.vel_estimator
         



        return
    
    def save(self, fn):
        super().save(fn)

        for mod in self.modules.keys():
            #save module wieghts
            torch.save(self.modules[mod].state_dict(), f'{fn}_{mod}.pth')
            pass

    def restore(self, fn):
        super().restore(fn)

        if self._use_velocity_estimator:
            raise 'Not Implimented'

    def init_tensors(self):
        super().init_tensors()

        #enable critic observations
        self.experience_buffer.tensor_dict['critic_obs'] = torch.zeros_like(self.experience_buffer.tensor_dict['obses'])
        self.tensor_list += ['critic_obs']
        
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['ase_latents'] = torch.zeros(batch_shape + (self._latent_dim,),
                                                                dtype=torch.float32, device=self.ppo_device)
        
        self._ase_latents = torch.zeros((batch_shape[-1], self._latent_dim), dtype=torch.float32,
                                         device=self.ppo_device)
        
        self._ase_latents_prev = torch.zeros((batch_shape[-1], self._latent_dim), dtype=torch.float32,
                                         device=self.ppo_device)
        
        self.tensor_list += ['ase_latents']
    
        
        if self._use_velocity_estimator:
            self.experience_buffer.tensor_dict['velocity_obs'] = torch.zeros(batch_shape + (self.vec_env.env.task._velocity_obs_buf.shape[1],),
                                                                dtype=torch.float32, device=self.ppo_device)
            self.tensor_list += ['velocity_obs']
            self.velocity_obs = torch.zeros((batch_shape[-1], self.vec_env.env.task._velocity_obs_buf.shape[1]),
                                            dtype=torch.float32, device=self.ppo_device)

        self._latent_reset_steps = torch.zeros(batch_shape[-1], dtype=torch.int32, device=self.ppo_device)
        num_envs = self.vec_env.env.task.num_envs
        env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.ppo_device)
        self._reset_latent_step_count(env_ids)

        return
    
    def play_steps(self):
        self.set_eval()
        
        epinfos = []
        done_indices = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs = self.env_reset(done_indices)

            if self._use_velocity_estimator:
                self.velocity_obs[done_indices] = self.vec_env.env.task.get_velocity_obs(done_indices)
                self.experience_buffer.update_data('velocity_obs', n, self.velocity_obs)

            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('critic_obs', n, self.obs['critic_obs'])

            

            self._update_latents()

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, self._ase_latents, masks)
            else:
                if self._use_velocity_estimator and self._train_with_velocity_estimate:
                    
                    #use the velocity observations to predict the noise 
                    vel_est_input = self.velocity_obs
                    if self._vel_est_use_ase_latent:
                        vel_est_input = torch.cat([vel_est_input, self._ase_latents],dim=-1)

                    velocity_est = self.vel_estimator.inference(vel_est_input)

                    #replace the velocity in the observation
                    obs_est = deepcopy(self.obs)
                    obs_est['obs'][:, self._vel_obs_index[0]:self._vel_obs_index[1]] = velocity_est[:, 1:]
                    obs_est['obs'][:, 0] = velocity_est[:, 0]

                    if self._vel_est_asymetric_train:
                        #use the un-noised / no estimate critic observation to get action values
                        res_dict = self.get_action_values(obs_est, self._ase_latents, self._rand_action_probs, critic_obs=self.obs['critic_obs'])
                    else:
                        res_dict = self.get_action_values(obs_est, self._ase_latents, self._rand_action_probs)

                else:
                    res_dict = self.get_action_values(self.obs, self._ase_latents, self._rand_action_probs)

            for k in update_list:
                #default: updates values 
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            
            if self._use_velocity_estimator:
                self.velocity_obs = infos['velocity_obs']

            shaped_rewards = self.rewards_shaper(rewards)

            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['critic_obs']) #TODO: next obs is used for the value prediction, should this used the privilleged obs
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])
            self.experience_buffer.update_data('ase_latents', n, self._ase_latents)
            self.experience_buffer.update_data('rand_action_mask', n, res_dict['rand_action_mask'])

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs['critic_obs'], self._ase_latents)
            next_vals *= (1.0 - terminated) #TODO: next obs is used for the value prediction, should this used the privilleged obs
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
        
            if (self.vec_env.env.task.viewer):
                self._amp_debug(infos, self._ase_latents)

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']
        
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']
        mb_ase_latents = self.experience_buffer.tensor_dict['ase_latents']
        amp_rewards = self._calc_amp_rewards(mb_amp_obs, mb_ase_latents)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)
        
        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        return batch_dict

    def get_action_values(self, obs_dict, ase_latents, rand_action_probs, critic_obs = None):
        processed_obs = self._preproc_obs(obs_dict['obs'])
        processed_critic_obs = None
        if critic_obs is not None:
            processed_critic_obs = self._preproc_obs(critic_obs)

        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states,
            'ase_latents': ase_latents,
            'critic_obs': processed_critic_obs
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs_dict['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value

        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        
        rand_action_mask = torch.bernoulli(rand_action_probs)
        det_action_mask = rand_action_mask == 0.0
        res_dict['actions'][det_action_mask] = res_dict['mus'][det_action_mask]
        res_dict['rand_action_mask'] = rand_action_mask

        return res_dict

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        
        ase_latents = batch_dict['ase_latents']
        self.dataset.values_dict['ase_latents'] = ase_latents
        self.dataset.values_dict['critic_obs'] = batch_dict['critic_obs']
        self.dataset.values_dict['next_obses'] = batch_dict['next_obses']

        if self._use_velocity_estimator:
            self.dataset.values_dict['velocity_obs'] = batch_dict['velocity_obs']
        
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
        critic_obs = None

        if self._use_velocity_estimator:
            _obs = input_dict['obs'].clone()

        if self._optimize_with_velocity_estimate:
            vel_est_input = input_dict['velocity_obs']
            if self._vel_est_use_ase_latent:
                vel_est_input = torch.cat([vel_est_input, input_dict['ase_latents']], dim=-1)

            state_est = self.vel_estimator.inference(vel_est_input)
            obs_batch[:, self._vel_obs_index[0]:self._vel_obs_index[1]] = state_est[:, 1:]            
            obs_batch[:, 0] = state_est[:, 0]


            if self._vel_est_asymetric_train:
                critic_obs = input_dict['critic_obs']


        if self._use_velocity_estimator:
            #Note: we do this first to prevent overwriting GT velocity with using optim with estimates
            vel_est_input = input_dict['velocity_obs']
            if self._vel_est_use_ase_latent:
                vel_est_input = torch.cat([vel_est_input, input_dict['ase_latents']], dim=-1)

            h_gt = critic_obs[:, 0]
            velocity_gt = critic_obs[:, self._vel_obs_index[0]:self._vel_obs_index[1]]
            gt = torch.cat((h_gt.unsqueeze(-1), velocity_gt), dim=-1)
            
            vel_loss = self.vel_estimator.loss(self.vel_estimator(vel_est_input), gt)
            self.vel_optim.zero_grad()
            vel_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vel_estimator.parameters(), self.vel_grad_norm)
            self.vel_optim.step()

        obs_batch = self._preproc_obs(obs_batch)
        if critic_obs is not None:
            critic_obs = self._preproc_obs(critic_obs)

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
        amp_obs_latents = input_dict['ase_latents'][0:self._amp_minibatch_size]
        
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
            'critic_obs':critic_obs,
            'amp_obs_latents':amp_obs_latents
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
            # enc_pred = res_dict['enc_pred']

            dim_real_logit = res_dict['dim_real_logit']
            dim_fake_logit = res_dict['dim_fake_logit']

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
            
           
            dim_loss_mask = rand_action_mask[0:self._amp_minibatch_size]
            # enc_info = self._enc_loss(enc_pred, enc_latents, batch_dict['amp_obs'], enc_loss_mask)
          
            # enc_loss = enc_info['enc_loss']
            #TODO return infos, impliment masking
            dim_loss_info = self._dim_loss(dim_real_logit, dim_fake_logit, dim_loss_mask)
            dim_loss = dim_loss_info['dim_loss']
            

            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                 + self._disc_coef * disc_loss + self._disc_coef * dim_loss
            
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
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)
        self.train_result.update(dim_loss_info)
        if self._use_velocity_estimator:
            self.train_result.update({'velocity_est_loss': vel_loss})

        return
    
    def env_reset(self, env_ids=None):
        obs = super().env_reset(env_ids)
   
        if (env_ids is None):
            num_envs = self.vec_env.env.task.num_envs
            env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.ppo_device)

        if (len(env_ids) > 0):
            self._reset_latents(env_ids)
            self._reset_latent_step_count(env_ids)

        return obs

    def _reset_latent_step_count(self, env_ids):
        self._latent_reset_steps[env_ids] = torch.randint_like(self._latent_reset_steps[env_ids], low=self._latent_steps_min, 
                                                         high=self._latent_steps_max)
        return

    def _load_config_params(self, config):
        super()._load_config_params(config)
        
        self._latent_dim = config['latent_dim']
        self._latent_steps_min = config.get('latent_steps_min', np.inf)
        self._latent_steps_max = config.get('latent_steps_max', np.inf)
        self._latent_dim = config['latent_dim']
        self._amp_diversity_bonus = config['amp_diversity_bonus']
        self._amp_diversity_tar = config['amp_diversity_tar']
        
        self._enc_coef = config['enc_coef']
        self._enc_weight_decay = config['enc_weight_decay']
        self._enc_reward_scale = config['enc_reward_scale']
        self._enc_grad_penalty = config['enc_grad_penalty']

        self._enc_reward_w = config['enc_reward_w']

        return
    
    def _build_net_config(self):
        config = super()._build_net_config()
        config['ase_latent_shape'] = (self._latent_dim,)
        return config

    def _reset_latents(self, env_ids):
        n = len(env_ids)
        z = self._sample_latents(n)
        self._ase_latents[env_ids] = z

        if (self.vec_env.env.task.viewer):
            self._change_char_color(env_ids)

        return

    def _sample_latents(self, n):
        z = self.model.a2c_network.sample_latents(n)
        return z

    def _update_latents(self):
        new_latent_envs = self._latent_reset_steps <= self.vec_env.env.task.progress_buf

        need_update = torch.any(new_latent_envs)
        if (need_update):
            new_latent_env_ids = new_latent_envs.nonzero(as_tuple=False).flatten()
            self._reset_latents(new_latent_env_ids)
            self._latent_reset_steps[new_latent_env_ids] += torch.randint_like(self._latent_reset_steps[new_latent_env_ids],
                                                                               low=self._latent_steps_min, 
                                                                               high=self._latent_steps_max)
            if (self.vec_env.env.task.viewer):
                self._change_char_color(new_latent_env_ids)

        return

    def _eval_actor(self, obs, ase_latents):
        output = self.model.a2c_network.eval_actor(obs=obs, ase_latents=ase_latents)
        return output

    def _eval_critic(self, obs, ase_latents):
        self.model.eval()
        processed_obs = self._preproc_obs(obs)
        value = self.model.a2c_network.eval_critic(processed_obs, ase_latents)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def _calc_amp_rewards(self, amp_obs, ase_latents):
        disc_r = self._calc_disc_rewards(amp_obs)
        # enc_r = self._calc_enc_rewards(amp_obs, ase_latents)
        dim_r = self._calc_dim_rewards(amp_obs, ase_latents)
        output = {
            'disc_rewards': disc_r,
            'dim_rewards': dim_r
        }
        return output
    
    def _calc_dim_rewards(self, amp_obs, ase_latents):
        with torch.no_grad():
            disc_logits = self._eval_dim(amp_obs, ase_latents)
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
            disc_r *= self._disc_reward_scale

        return disc_r
    
    def _eval_dim(self, amp_obs, ase_latents):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_dim(proc_amp_obs, ase_latents)

    
    def _dim_loss(self, dim_real_logit, dim_fake_logit, dim_real_input):
       
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(dim_fake_logit)
        disc_loss_demo = self._disc_loss_pos(dim_real_logit)
        dim_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.a2c_network.get_dim_logit_weights()
        dim_logit_loss = torch.sum(torch.square(logit_weights))
        dim_loss += self._disc_logit_reg * dim_logit_loss

        # grad penalty
        dim_real_grad = torch.autograd.grad(dim_real_logit, dim_real_input, grad_outputs=torch.ones_like(dim_real_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        dim_real_grad = dim_real_grad[0]
        dim_real_grad = torch.sum(torch.square(dim_real_grad), dim=-1)
        dim_real_penalty = torch.mean(dim_real_grad)
        dim_loss += self._disc_grad_penalty * dim_real_penalty

        

        # weight decay
        if (self._disc_weight_decay != 0):
            dim_weights = self.model.a2c_network.get_dim_weights()
            dim_weights = torch.cat(dim_weights, dim=-1)
            dim_weight_decay = torch.sum(torch.square(dim_weights))
            dim_loss += self._disc_weight_decay * dim_weight_decay

        dim_fake_acc, dim_real_acc = self._compute_disc_acc(dim_fake_logit, dim_real_logit)

        dim_info = {
            'dim_loss': dim_loss,
            'dim_grad_penalty': dim_real_penalty.detach(),
            'dim_logit_loss': dim_logit_loss.detach(),
            'dim_fake_acc': dim_fake_acc.detach(),
            'dim_real_acc': dim_real_acc.detach(),
            'dim_fake_logit': dim_fake_logit.detach(),
            'dim_real_logit': dim_real_logit.detach()
        }
        return dim_info


    def _enc_loss(self, enc_pred, ase_latent, enc_obs, loss_mask):
        enc_err = self._calc_enc_error(enc_pred, ase_latent)
        #mask_sum = torch.sum(loss_mask)
        #enc_err = enc_err.squeeze(-1)
        #enc_loss = torch.sum(loss_mask * enc_err) / mask_sum
        enc_loss = torch.mean(enc_err)

        # weight decay
        if (self._enc_weight_decay != 0):
            enc_weights = self.model.a2c_network.get_enc_weights()
            enc_weights = torch.cat(enc_weights, dim=-1)
            enc_weight_decay = torch.sum(torch.square(enc_weights))
            enc_loss += self._enc_weight_decay * enc_weight_decay
            
        enc_info = {
            'enc_loss': enc_loss
        }

        if (self._enable_enc_grad_penalty()):
            enc_obs_grad = torch.autograd.grad(enc_err, enc_obs, grad_outputs=torch.ones_like(enc_err),
                                               create_graph=True, retain_graph=True, only_inputs=True)
            enc_obs_grad = enc_obs_grad[0]
            enc_obs_grad = torch.sum(torch.square(enc_obs_grad), dim=-1)
            #enc_grad_penalty = torch.sum(loss_mask * enc_obs_grad) / mask_sum
            enc_grad_penalty = torch.mean(enc_obs_grad)

            enc_loss += self._enc_grad_penalty * enc_grad_penalty

            enc_info['enc_grad_penalty'] = enc_grad_penalty.detach()

        return enc_info

    def _diversity_loss(self, obs, action_params, ase_latents):
        assert(self.model.a2c_network.is_continuous)

        n = obs.shape[0]
        assert(n == action_params.shape[0])

        new_z = self._sample_latents(n)
        mu, sigma,  = self._eval_actor(obs=obs, ase_latents=new_z)

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

    def _calc_enc_error(self, enc_pred, ase_latent):
        err = enc_pred * ase_latent
        err = -torch.sum(err, dim=-1, keepdim=True)
        return err

    def _enable_enc_grad_penalty(self):
        return self._enc_grad_penalty != 0

    def _enable_amp_diversity_bonus(self):
        return self._amp_diversity_bonus != 0

    def _eval_enc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_enc(proc_amp_obs)

    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards['disc_rewards']
        dim_r = amp_rewards['dim_rewards']
        combined_rewards = self._task_reward_w * task_rewards \
                         + self._disc_reward_w * disc_r \
                         + self._enc_reward_w * dim_r
        return combined_rewards

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['dim_rewards'] = batch_dict['dim_rewards']
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        
        self.writer.add_scalar('losses/enc_loss', torch_ext.mean_list(train_info['enc_loss']).item(), frame)

        if self._use_velocity_estimator:
            self.writer.add_scalar('losses/velocity_est_loss', torch_ext.mean_list(train_info['velocity_est_loss']).item(), frame)
         
        if (self._enable_amp_diversity_bonus()):
            self.writer.add_scalar('losses/amp_diversity_loss', torch_ext.mean_list(train_info['amp_diversity_loss']).item(), frame)
        
        # enc_reward_std, enc_reward_mean = torch.std_mean(train_info['enc_rewards'])
        # self.writer.add_scalar('info/enc_reward_mean', enc_reward_mean.item(), frame)
        # self.writer.add_scalar('info/enc_reward_std', enc_reward_std.item(), frame)

        # if (self._enable_enc_grad_penalty()):
        #     self.writer.add_scalar('info/enc_grad_penalty', torch_ext.mean_list(train_info['enc_grad_penalty']).item(), frame)
        self.writer.add_scalar('losses/dim_loss', torch_ext.mean_list(train_info['dim_loss']).item(), frame)

        self.writer.add_scalar('dim_info/dim_fake_acc', torch_ext.mean_list(train_info['dim_fake_acc']).item(), frame)
        self.writer.add_scalar('dim_info/dim_real_acc', torch_ext.mean_list(train_info['dim_real_acc']).item(), frame)
        self.writer.add_scalar('dim_info/dim_fake_logit', torch_ext.mean_list(train_info['dim_fake_logit']).item(), frame)
        self.writer.add_scalar('dim_info/dim_real_logit', torch_ext.mean_list(train_info['dim_real_logit']).item(), frame)
        self.writer.add_scalar('dim_info/dim_grad_penalty', torch_ext.mean_list(train_info['dim_grad_penalty']).item(), frame)
        self.writer.add_scalar('dim_info/dim_logit_loss', torch_ext.mean_list(train_info['dim_logit_loss']).item(), frame)

        disc_reward_std, disc_reward_mean = torch.std_mean(train_info['dim_rewards'])
        self.writer.add_scalar('info/dim_reward_mean', disc_reward_mean.item(), frame)
        print( disc_reward_mean.item())
        self.writer.add_scalar('info/dim_reward_std', disc_reward_std.item(), frame)

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