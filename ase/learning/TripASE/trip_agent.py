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
import torch.nn.functional as F
from utils import torch_utils
from learning import ase_network_builder
from copy import deepcopy

class TripASEAgent(ASEAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)

        self._use_privilliged_world_model = True
        self._triplet_reward_w = 1
        # self._curiase_coef = 0.5
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
                    obs_est['obs'][:, self._vel_obs_index[0]:self._vel_obs_index[1]] = velocity_est

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

        mb_obs = self.experience_buffer.tensor_dict['critic_obs']
        mb_next_obs = self.experience_buffer.tensor_dict['next_obses']
        mb_actions = self.experience_buffer.tensor_dict['actions']
        # curiase_rewards = self._calc_curiase_rewards(mb_obs, mb_next_obs, mb_actions)
        triplet_rewards, triplet_encodings,  triplet_info = self.model.a2c_network._compute_meanV2_contrastive_reward(mb_ase_latents, self._preproc_amp_obs(mb_amp_obs))
        
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards, triplet_rewards)
        
        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        
       
        # batch_dict['triplet_positive'] = a2c_common.swap_and_flatten01(triplet_encodings[0])
        # batch_dict['triplet_negative'] = a2c_common.swap_and_flatten01(triplet_encodings[1])



        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

    
        batch_dict['triplet_rewards'] = a2c_common.swap_and_flatten01(triplet_rewards)
        batch_dict['triplet_pos_magnitude'] = triplet_info['pos_mean']        
        batch_dict['triplet_neg_magnitude'] = triplet_info['neg_mean']

  
        return batch_dict
    
    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        
        # self.dataset.values_dict['triplet_anchor'] = batch_dict['triplet_anchor']
        # self.dataset.values_dict['triplet_positive'] = batch_dict['triplet_positive']
        # self.dataset.values_dict['triplet_negative'] = batch_dict['triplet_negative']
        
        
        return
    
    def _calc_curiase_rewards(self, states, next_states, actions):
        """
        
        """
        # states_ = states.reshape(states.size(0)*states.size(1), -1)
        states = self._preproc_obs(states)
        next_states = self._preproc_obs(next_states)
        return self.model.a2c_network._compute_curiase_r(states, next_states, actions)
    
    def _combine_rewards(self, task_rewards, amp_rewards, triplet_rewards):
        disc_r = amp_rewards['disc_rewards']
        enc_r = amp_rewards['enc_rewards']
      
        combined_rewards = self._task_reward_w * task_rewards \
                         + self._disc_reward_w * disc_r \
                         + self._triplet_reward_w * triplet_rewards
                        #  + self._enc_reward_w * enc_r \
        return combined_rewards
    

    def optimize_contrastive_NTXENT(self, batch_dict):

        amp_obs = batch_dict['amp_obs'] #TODO --> encode all

        ase_latents = batch_dict['ase_latents']

        latents_diffs = torch.diff(ase_latents, dim=0) # col wise changes
        latents_diffs = torch.any(latents_diffs !=0, dim=1) #check for changes on a row
        changes_idxs = torch.where(latents_diffs)[0] + 1

        grouped_amp_obs = []
        start_idx = 0

        for idx in changes_idxs:
            grouped_amp_obs.append(amp_obs[start_idx:idx])
            start_idx = idx
        grouped_amp_obs.append(amp_obs[start_idx:]) 


        grouped_positive_samples = []
        for i in range(len(grouped_amp_obs)):
            num_steps = grouped_amp_obs[i].size(0)

            idxs = torch.arange(0, num_steps)
            samples = torch.randint(-10, 10, num_steps)
            idxs = idxs + samples
            idxs = torch.clamp(idxs, 0, num_steps-1)

            positive_samples = grouped_amp_obs[i][idxs]
            grouped_positive_samples.append(positive_samples)


        # create negatives
        K = 1
        negative_batch = []
        for i in range(len(grouped_amp_obs)):
            num_steps = grouped_amp_obs[i].size(0)
            samples = torch.randint(0, num_steps, K)
            negatives = grouped_amp_obs[i][samples].unsqueeze(0)
            negative_batch.append(negatives)
        negative_batch = torch.cat(negative_batch)
        
        losses = []
        for i in range(len(grouped_amp_obs)):
            #compute positive similarity scores
            Z_i = grouped_amp_obs[i]
            Z_j = grouped_positive_samples[i]
            positive_scores = F.cosine_similarity(Z_i, Z_j, dim=1)

            #compute negative similarity score 
            mask = torch.zeros(len(grouped_amp_obs), dtype=torch.bool)
            mask[i] = True

            negatives = negative_batch[~mask] # [num unique ase, 10, enc dim]
            negatives = negatives.reshape(negatives.size(0)*negatives.size(1), negatives.size(2)).unsqueeze(0)
            Z_k = negatives.repeat([Z_i.size(0),1,1])

            Z_i_ext = Z_i.unsqueeze(1)

            negative_scores = F.cosine_similarity(Z_i_ext, Z_k, dim=2)

            denom = torch.sum(negative_scores, dim=-1)

            losses.append(-torch.log(positive_scores/denom, dim=-1).mean())

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
        next_obs_batch = input_dict['next_obses']
        critic_obs = None

        if self._use_velocity_estimator:
            _obs = input_dict['obs'].clone()

        if self._optimize_with_velocity_estimate:
            vel_est_input = input_dict['velocity_obs']
            if self._vel_est_use_ase_latent:
                vel_est_input = torch.cat([vel_est_input, input_dict['ase_latents']], dim=-1)

            obs_batch[:, self._vel_obs_index[0]:self._vel_obs_index[1]] = self.vel_estimator.inference(vel_est_input)

            if self._vel_est_asymetric_train:
                critic_obs = input_dict['critic_obs']


        if self._use_velocity_estimator:
            #Note: we do this first to prevent overwriting GT velocity with using optim with estimates
            vel_est_input = input_dict['velocity_obs']
            if self._vel_est_use_ase_latent:
                vel_est_input = torch.cat([vel_est_input, input_dict['ase_latents']], dim=-1)

            velocity_gt = critic_obs[:, self._vel_obs_index[0]:self._vel_obs_index[1]]
            vel_loss = self.vel_estimator.loss(self.vel_estimator(vel_est_input), velocity_gt)
            self.vel_optim.zero_grad()
            vel_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vel_estimator.parameters(), self.vel_grad_norm)
            self.vel_optim.step()

        obs_batch = self._preproc_obs(obs_batch)
        if critic_obs is not None:
            critic_obs = self._preproc_obs(critic_obs)

        next_obs_batch = self._preproc_obs(next_obs_batch)

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
            'critic_obs':critic_obs, 
            'next_obses':next_obs_batch,
            'actions':actions_batch
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

            # print('in eval--------------------------------') --------------- TRIPLET LOSS
            anchors = amp_obs
            positives = amp_obs + torch.randn(amp_obs.size(), device=amp_obs.device) * 0.1
            perm = torch.randperm(positives.size(0))
            negatives = anchors[perm]

            anchor_enc = self.model.a2c_network._disc_mlp(anchors)
            pos_enc = self.model.a2c_network._disc_mlp(positives)
            neg_enc = self.model.a2c_network._disc_mlp(negatives)

            # Compute the distance between the anchor and the positive
            pos_dist = torch.nn.functional.pairwise_distance(anchor_enc/torch.norm(anchor_enc, dim=-1, keepdim=True), 
                                                             pos_enc/torch.norm(pos_enc, dim=-1, keepdim=True), p=2)
            # Compute the distance between the anchor and the negative
            neg_dist = torch.nn.functional.pairwise_distance(anchor_enc/torch.norm(anchor_enc, dim=-1, keepdim=True), 
                                                             neg_enc/torch.norm(neg_enc, dim=-1, keepdim=True), p=2)
            
            # Compute the triplet loss
            contrastive_loss = torch.mean(torch.clamp(pos_dist - neg_dist + 1.0, min=0.0))
            # print('end eval--------------------------------')


            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                 + self._disc_coef * disc_loss + self._enc_coef * enc_loss \
                 + contrastive_loss
         

            #TODO --> we could put in a triplet loss here too
            
            if (self._enable_amp_diversity_bonus()):
                diversity_loss = self._diversity_loss(batch_dict['obs'], mu, batch_dict['ase_latents'])
                diversity_loss = torch.sum(rand_action_mask * diversity_loss) / rand_action_sum
                loss += self._amp_diversity_bonus * diversity_loss
                a_info['amp_diversity_loss'] = diversity_loss
                
            a_info['actor_loss'] = a_loss.detach()
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss.detach()

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
            'contrastive_loss': contrastive_loss.detach()

        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)
        self.train_result.update(enc_info)

        if self._use_velocity_estimator:
            self.train_result.update({'velocity_est_loss': vel_loss})

        return
     

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['enc_rewards'] = batch_dict['enc_rewards']
        train_info['triplet_rewards'] = batch_dict['triplet_rewards']
   
        train_info['triplet_pos_magnitude'] = batch_dict['triplet_pos_magnitude']
        train_info['triplet_neg_magnitude'] = batch_dict['triplet_neg_magnitude']

        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        
        self.writer.add_scalar('losses/enc_loss', torch_ext.mean_list(train_info['enc_loss']).item(), frame)
        self.writer.add_scalar('losses/contrastive_loss', torch_ext.mean_list(train_info['contrastive_loss']).item(), frame)
        #TODO
    
        if (self._enable_amp_diversity_bonus()):
            self.writer.add_scalar('losses/amp_diversity_loss', torch_ext.mean_list(train_info['amp_diversity_loss']).item(), frame)
        
        enc_reward_std, enc_reward_mean = torch.std_mean(train_info['enc_rewards'])
        self.writer.add_scalar('info/enc_reward_mean', enc_reward_mean.item(), frame)
        self.writer.add_scalar('info/enc_reward_std', enc_reward_std.item(), frame)

        curi_reward_std, curi_reward_mean = torch.std_mean(train_info['triplet_rewards'])
        self.writer.add_scalar('info/triplet_reward_mean', curi_reward_mean.item(), frame)
        self.writer.add_scalar('info/triplet_reward_std', curi_reward_std.item(), frame)        
        self.writer.add_scalar('info/triplet_pos_mag', train_info['triplet_pos_magnitude'].item(), frame)
        self.writer.add_scalar('info/triplet_neg_mag', train_info['triplet_neg_magnitude'].item(), frame)


        if (self._enable_enc_grad_penalty()):
            self.writer.add_scalar('info/enc_grad_penalty', torch_ext.mean_list(train_info['enc_grad_penalty']).item(), frame)

        return

