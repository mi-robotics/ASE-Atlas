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

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np
import enum
import time

from learning import amp_network_builder
from learning.modules.velocity_estimator import VelocityEstimator
import faiss 

from learning.ase_network_builder import ASEBuilder 
from learning.modules.world_model import WorldModel

ENC_LOGIT_INIT_SCALE = 0.1

class LatentType(enum.Enum):
    uniform = 0
    sphere = 1

class TripASEBuilder(ASEBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(ASEBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            actions_num = kwargs.get('actions_num')
            input_shape = kwargs.get('input_shape')

            self.reward_scale = 10.
            self.reward_scale_min = 2.
            return
        
        def load(self, params):
            super().load(params)
            self._enc_units = params['enc']['units']
            self._enc_activation = params['enc']['activation']
            self._enc_initializer = params['enc']['initializer']
            self._enc_separate = params['enc']['separate']

            return

        def _get_transition_encoding(self, transitions):
            with torch.no_grad():
                encodings = self._disc_mlp(transitions)
            return encodings
        
        def _compute_contrastive_reward(self, _ase_latents, _amp_obs):
          
            '''
            latents -> 32, num_envs, latent dim
            amp_obs -> 32, num_envs, amb_obs_dim (10,109)
            '''
            # ------------------------------------------------------------  data formatting

            ase_latents = _ase_latents
            amp_obs = _amp_obs

            ase_latents = ase_latents.transpose(1,0).reshape(
                ase_latents.size(0)*ase_latents.size(1), -1
            ) # -->  num_envs*32, latent_dims

            amp_obs = amp_obs.transpose(1,0).reshape(
                amp_obs.size(0)*amp_obs.size(1), amp_obs.size(2), -1
            ) # --> num_envs*32, 10, 109

            


            # ------------------------------------------------------------ latent mapping 
            # we need some mapping between latents and amp obs
            latents_diffs = torch.diff(ase_latents, dim=0) # col wise changes
            latents_diffs = torch.any(latents_diffs !=0, dim=1) #check for changes on a row
            changes_idxs = torch.where(latents_diffs)[0] + 1

            
            repeats = torch.cat([
                # Count for the first segment
                torch.Tensor([changes_idxs[0]]),
                # Counts for intermediate segments
                torch.diff(changes_idxs).cpu(),
                # Count for the last segment
                torch.Tensor([ase_latents.size(0) - changes_idxs[-1]])
            ])

            #groups latents by the latents
            latent_amp_obs = []
            start_idx = 0

            for idx in changes_idxs:
                latent_amp_obs.append(amp_obs[start_idx:idx])
                start_idx = idx
            latent_amp_obs.append(amp_obs[start_idx:]) # --> [num latents, latent_seq_len, amp_obs.size()]

            #pair the latents
            unique_latents = []
            start_idx = 0
        
            for idx in changes_idxs:
                unique_latents.append(ase_latents[idx-1])
                start_idx = idx
            unique_latents.append(ase_latents[-1])
            
            unique_latents = torch.stack(unique_latents)

            assert len(latent_amp_obs) == unique_latents.size(0)

            # ------------------------------------------------------------ create dataset 
            # Convert the PyTorch tensor to a NumPy array for FAISS
            dataset_latents = unique_latents.cpu().numpy()
            _cpu_latents = ase_latents.cpu().numpy()
    
            start = time.time()

            # Create a basic L2 index
            base_index = faiss.IndexFlatL2(24)

            # Wrap the base index with IndexIDMap2
            index = faiss.IndexIDMap2(base_index)

            # Assign an ID to each vector. In this case, use a range of IDs.
            ids = np.arange(unique_latents.size(0), dtype=np.int64)

            # Add vectors and their IDs to the index
            index.add_with_ids(dataset_latents, ids)

            # ------------------------------------------------------------ query dataset 
            k = 2
            D, I = index.search(dataset_latents, k) # --> return a nearest n for each time steps

            nearest_n = I[:, 1] # removes the exact match
            nearest_n = torch.repeat_interleave(torch.from_numpy(nearest_n), repeats.int(), dim=0).numpy()

            end = time.time()

            kf = 1
            Df, If = index.search(-dataset_latents, k) # --> return a nearest n for each time steps

            furthest_n = If[:, 0] # removes the exact match
            furthest_n = torch.repeat_interleave(torch.from_numpy(furthest_n), repeats.int(), dim=0).numpy()


            # ------------------------------------------------------------ colelct possitive and negative samples 
            positive_batch_amp_obs = [latent_amp_obs[ni] for ni in nearest_n]
            negative_batch_amp_obs = [latent_amp_obs[ri] for ri in furthest_n]


            positive_amp_obs = []
            negative_amp_obs = []

            for i in range(len(positive_batch_amp_obs)):
                pos_samples = positive_batch_amp_obs[i].size(0)
                neg_samples = negative_batch_amp_obs[i].size(0)
                n_pos = 15
                if pos_samples<=n_pos:
                    n_pos=pos_samples-1
                n_neg = 15
                if neg_samples<=n_neg:
                    n_neg=neg_samples-1


                

                pos_obs = positive_batch_amp_obs[i][[np.random.randint(n_pos, pos_samples)]]            
                neg_obs = negative_batch_amp_obs[i][[np.random.randint(n_neg, neg_samples)]]

                positive_amp_obs.append(pos_obs)
                negative_amp_obs.append(neg_obs)


            anchor = amp_obs
            positive_amp_obs = torch.cat(positive_amp_obs)
            negative_amp_obs = torch.cat(negative_amp_obs)

            anchor = anchor.reshape(_amp_obs.size(1), _amp_obs.size(0), _amp_obs.size(2),-1).transpose(1,0)
            positive_amp_obs = positive_amp_obs.reshape(_amp_obs.size(1), _amp_obs.size(0), _amp_obs.size(2),-1).transpose(1,0)
            negative_amp_obs = negative_amp_obs.reshape(_amp_obs.size(1), _amp_obs.size(0), _amp_obs.size(2),-1).transpose(1,0)

       
            # ------------------------------------------------------------ get samples representations

            model_inputs = torch.cat((anchor, positive_amp_obs, negative_amp_obs)).squeeze(-1)

            transition_encoding = self._get_transition_encoding(model_inputs)
  
            anchor_encoding = transition_encoding[:anchor.size(0)]
            positive_encoding = transition_encoding[anchor.size(0):anchor.size(0)+positive_amp_obs.size(0)]
            negative_encoding = transition_encoding[anchor.size(0)+positive_amp_obs.size(0): ]

            anchor_encoding = anchor_encoding / torch.norm(anchor_encoding, p=2, dim=-1, keepdim=True)            
            positive_encoding = positive_encoding / torch.norm(positive_encoding, p=2, dim=-1, keepdim=True)
            negative_encoding = negative_encoding / torch.norm(negative_encoding, p=2, dim=-1, keepdim=True)


            pos_dist = torch.norm(anchor_encoding-positive_encoding, p=2, dim=-1, keepdim=True)
            neg_dist = torch.norm(anchor_encoding-negative_encoding, p=2, dim=-1, keepdim=True)


            # print(pos_dist)
            # print(neg_dist)
            # input()
            # def phi(x):
            #     x = torch.clamp(x, max=10)
            #     return (torch.exp(x) - 1 )/ (torch.exp(x) +1)

            # rewards =  0.5 * (phi(neg_dist) + 1 - phi(pos_dist)) - 0.5
            # rewards = rewards.clamp(0)

        
            rewards = - pos_dist/2 + neg_dist/2 
            rewards *= self.reward_scale

            if rewards.mean()>0.7:
                self.reward_scale*=0.75
                self.reward_scale = max(self.reward_scale, self.reward_scale_min)

            print('REWARD SCALE -----------------------: ', self.reward_scale)
            # rewards = rewards.clamp(0)
            # ------------------------------------------------------------ reshape to input shape

          
            return (rewards,
                ( positive_amp_obs, negative_amp_obs),
                {'pos_mean':torch.norm(positive_encoding, p=2, dim=-1).mean(),
                 'neg_mean':torch.norm(negative_encoding, p=2, dim=-1).mean()})
        


        def _compute_mean_contrastive_reward(self, _ase_latents, _amp_obs):
          
            '''
            latents -> 32, num_envs, latent dim
            amp_obs -> 32, num_envs, amb_obs_dim (10,109)
            '''
            # ------------------------------------------------------------  data formatting

            ase_latents = _ase_latents
            amp_obs = _amp_obs

            ase_latents = ase_latents.transpose(1,0).reshape(
                ase_latents.size(0)*ase_latents.size(1), -1
            ) # -->  num_envs*32, latent_dims

            amp_obs = amp_obs.transpose(1,0).reshape(
                amp_obs.size(0)*amp_obs.size(1), amp_obs.size(2)
            ) # --> num_envs*32, 10, 109

            with torch.no_grad():
                amp_obs_encoding = self._enc_mlp(amp_obs)

            
            # ------------------------------------------------------------ latent mapping 
            # we need some mapping between latents and amp obs
            latents_diffs = torch.diff(ase_latents, dim=0) # col wise changes
            latents_diffs = torch.any(latents_diffs !=0, dim=1) #check for changes on a row
            changes_idxs = torch.where(latents_diffs)[0] + 1

            
            repeats = torch.cat([
                # Count for the first segment
                torch.Tensor([changes_idxs[0]]).cuda(),
                # Counts for intermediate segments
                torch.diff(changes_idxs),
                # Count for the last segment
                torch.Tensor([ase_latents.size(0) - changes_idxs[-1]]).cuda()
            ])

            #groups latents by the latents
            latent_amp_obs_enc = []
            start_idx = 0

            for idx in changes_idxs:
                latent_amp_obs_enc.append(amp_obs_encoding[start_idx:idx])
                start_idx = idx
            latent_amp_obs_enc.append(amp_obs_encoding[start_idx:]) # --> [num latents, latent_seq_len, amp_obs.size()]

            #pair the latents
            unique_latents = []
            start_idx = 0
        
            for idx in changes_idxs:
                unique_latents.append(ase_latents[idx-1])
                start_idx = idx
            unique_latents.append(ase_latents[-1])
            
            unique_latents = torch.stack(unique_latents)

            assert len(latent_amp_obs_enc) == unique_latents.size(0)

            # ------------------------------------------------------------ create dataset 
            # Convert the PyTorch tensor to a NumPy array for FAISS
            dataset_latents = unique_latents.cpu().numpy()
            _cpu_latents = ase_latents.cpu().numpy()
    
            start = time.time()

            # Create a basic L2 index
            base_index = faiss.IndexFlatL2(24)

            # Wrap the base index with IndexIDMap2
            index = faiss.IndexIDMap2(base_index)

            # Assign an ID to each vector. In this case, use a range of IDs.
            ids = np.arange(unique_latents.size(0), dtype=np.int64)

            # Add vectors and their IDs to the index
            index.add_with_ids(dataset_latents, ids)

            # ------------------------------------------------------------ query dataset 
            k = 2
            D, I = index.search(dataset_latents, k) # --> return a nearest n for each time steps

            nearest_n = I[:, 1] # removes the exact match
            # nearest_n = torch.repeat_interleave(torch.from_numpy(nearest_n), repeats.int(), dim=0).numpy()

            end = time.time()

            # kf = 1
            # Df, If = index.search(-dataset_latents, k) # --> return a nearest n for each time steps

            # furthest_n = If[:, 0] # removes the exact match
            furthest_n = np.random.randint(0, len(dataset_latents), len(dataset_latents))

            assert len(furthest_n) == len(nearest_n)
            # furthest_n = torch.repeat_interleave(torch.from_numpy(furthest_n), repeats.int(), dim=0).numpy()


            # ------------------------------------------------------------ colelct possitive and negative samples 
            positive_batch_amp_obs = [latent_amp_obs_enc[ni] for ni in nearest_n]
            negative_batch_amp_obs = [latent_amp_obs_enc[ri] for ri in furthest_n]


            anchor_amp_obs_enc = []
            positive_amp_obs_enc = []
            negative_amp_obs_enc = []

            for i in range(len(positive_batch_amp_obs)):
                anc_samples = latent_amp_obs_enc[i].size(0)
                pos_samples = positive_batch_amp_obs[i].size(0)
                neg_samples = negative_batch_amp_obs[i].size(0)

                n_anc = 50
                if anc_samples<=n_anc:
                    n_anc=anc_samples-1
                n_pos = 50
                if pos_samples<=n_pos:
                    n_pos=pos_samples-1
                n_neg = 50
                if neg_samples<=n_neg:
                    n_neg=neg_samples-1
                
                
                anc_obs_enc= torch.mean(latent_amp_obs_enc[i][n_anc:], dim=0, keepdim=True)
                pos_obs_enc= torch.mean(positive_batch_amp_obs[i][n_pos:], dim=0, keepdim=True)          
                neg_obs_enc= torch.mean(negative_batch_amp_obs[i][n_neg:], dim=0, keepdim=True)


                anchor_amp_obs_enc.append(anc_obs_enc)
                positive_amp_obs_enc.append(pos_obs_enc)
                negative_amp_obs_enc.append(neg_obs_enc)


            anchor_amp_obs_enc = torch.cat(anchor_amp_obs_enc)            
            positive_amp_obs_enc = torch.cat(positive_amp_obs_enc)
            negative_amp_obs_enc = torch.cat(negative_amp_obs_enc)

      

            anchor_encoding = anchor_amp_obs_enc / torch.norm(anchor_amp_obs_enc, p=2, dim=-1, keepdim=True)            
            positive_encoding = positive_amp_obs_enc / torch.norm(positive_amp_obs_enc, p=2, dim=-1, keepdim=True)
            negative_encoding = negative_amp_obs_enc / torch.norm(negative_amp_obs_enc, p=2, dim=-1, keepdim=True)


            pos_dist = torch.norm(anchor_encoding-positive_encoding, p=2, dim=-1, keepdim=True)
            neg_dist = torch.norm(anchor_encoding-negative_encoding, p=2, dim=-1, keepdim=True)

    
            rewards = - pos_dist/2 + neg_dist/2 # # --> unique latents, 1
            rewards = torch.repeat_interleave(rewards, repeats.int(), dim=0)
            rewards = rewards.reshape(_amp_obs.size(1), _amp_obs.size(0),-1).transpose(1,0)
            rewards *= self.reward_scale

            if rewards.mean()>0.7:
                self.reward_scale*=0.75
                self.reward_scale = max(self.reward_scale, self.reward_scale_min)

            print('REWARD SCALE -----------------------: ', self.reward_scale)
       
            return (rewards,
                None,
                {'pos_mean':torch.norm(positive_encoding, p=2, dim=-1).mean(),
                 'neg_mean':torch.norm(negative_encoding, p=2, dim=-1).mean()})
        


        def _compute_meanV2_contrastive_reward(self, _ase_latents, _amp_obs):
          
            '''
            latents -> 32, num_envs, latent dim
            amp_obs -> 32, num_envs, amb_obs_dim (10,109)
            '''
            # ------------------------------------------------------------  data formatting

            ase_latents = _ase_latents
            amp_obs = _amp_obs

            ase_latents = ase_latents.transpose(1,0).reshape(
                ase_latents.size(0)*ase_latents.size(1), -1
            ) # -->  num_envs*32, latent_dims

            amp_obs = amp_obs.transpose(1,0).reshape(
                amp_obs.size(0)*amp_obs.size(1), amp_obs.size(2)
            ) # --> num_envs*32, 10, 109

            with torch.no_grad():
                amp_obs_encoding = self._enc_mlp(amp_obs)

            
            # ------------------------------------------------------------ latent mapping 
            # we need some mapping between latents and amp obs
            latents_diffs = torch.diff(ase_latents, dim=0) # col wise changes
            latents_diffs = torch.any(latents_diffs !=0, dim=1) #check for changes on a row
            changes_idxs = torch.where(latents_diffs)[0] + 1

            
            repeats = torch.cat([
                # Count for the first segment
                torch.Tensor([changes_idxs[0]]).cuda(),
                # Counts for intermediate segments
                torch.diff(changes_idxs),
                # Count for the last segment
                torch.Tensor([ase_latents.size(0) - changes_idxs[-1]]).cuda()
            ])

            #groups latents by the latents
            latent_amp_obs_enc = []
            start_idx = 0

            for idx in changes_idxs:
                latent_amp_obs_enc.append(amp_obs_encoding[start_idx:idx])
                start_idx = idx
            latent_amp_obs_enc.append(amp_obs_encoding[start_idx:]) # --> [num latents, latent_seq_len, amp_obs.size()]

            #pair the latents
            unique_latents = []
            start_idx = 0
        
            for idx in changes_idxs:
                unique_latents.append(ase_latents[idx-1])
                start_idx = idx
            unique_latents.append(ase_latents[-1])
            
            unique_latents = torch.stack(unique_latents)

            assert len(latent_amp_obs_enc) == unique_latents.size(0)

            # ------------------------------------------------------------ create dataset 
            # Convert the PyTorch tensor to a NumPy array for FAISS
            dataset_latents = unique_latents.cpu().numpy()
            _cpu_latents = ase_latents.cpu().numpy()
    
            start = time.time()

            # Create a basic L2 index
            base_index = faiss.IndexFlatL2(24)

            # Wrap the base index with IndexIDMap2
            index = faiss.IndexIDMap2(base_index)

            # Assign an ID to each vector. In this case, use a range of IDs.
            ids = np.arange(unique_latents.size(0), dtype=np.int64)

            # Add vectors and their IDs to the index
            index.add_with_ids(dataset_latents, ids)

            # ------------------------------------------------------------ query dataset 
            k = 2
            D, I = index.search(dataset_latents, k) # --> return a nearest n for each time steps

            nearest_n = I[:, 1] # removes the exact match
            # nearest_n = torch.repeat_interleave(torch.from_numpy(nearest_n), repeats.int(), dim=0).numpy()

            end = time.time()

            # kf = 1
            # Df, If = index.search(-dataset_latents, k) # --> return a nearest n for each time steps

            # furthest_n = If[:, 0] # removes the exact match
            furthest_n = np.random.randint(0, len(dataset_latents), len(dataset_latents))

            assert len(furthest_n) == len(nearest_n)
            # furthest_n = torch.repeat_interleave(torch.from_numpy(furthest_n), repeats.int(), dim=0).numpy()


            # ------------------------------------------------------------ colelct possitive and negative samples 
            positive_batch_amp_obs = [latent_amp_obs_enc[ni] for ni in nearest_n]
            negative_batch_amp_obs = [latent_amp_obs_enc[ri] for ri in furthest_n]


            anchor_amp_obs_enc = []
            positive_amp_obs_enc = []
            negative_amp_obs_enc = []

            for i in range(len(positive_batch_amp_obs)):
                anc_samples = latent_amp_obs_enc[i].size(0)
                pos_samples = positive_batch_amp_obs[i].size(0)
                neg_samples = negative_batch_amp_obs[i].size(0)

                n_anc = 50
                if anc_samples<=n_anc:
                    n_anc=anc_samples-1
                n_pos = 50
                if pos_samples<=n_pos:
                    n_pos=pos_samples-1
                n_neg = 50
                if neg_samples<=n_neg:
                    n_neg=neg_samples-1
                
                
                # anc_obs_enc= torch.mean(latent_amp_obs_enc[i][n_anc:], dim=0, keepdim=True)
                pos_obs_enc= torch.mean(positive_batch_amp_obs[i][n_pos:], dim=0, keepdim=True)          
                neg_obs_enc= torch.mean(negative_batch_amp_obs[i][n_neg:], dim=0, keepdim=True)


                # anchor_amp_obs_enc.append(anc_obs_enc)
                positive_amp_obs_enc.append(pos_obs_enc)
                negative_amp_obs_enc.append(neg_obs_enc)


            anchor_amp_obs_enc = amp_obs_encoding#torch.cat(anchor_amp_obs_enc) 

            positive_amp_obs_enc = torch.cat(positive_amp_obs_enc)
            positive_amp_obs_enc = torch.repeat_interleave(positive_amp_obs_enc, repeats.int(), dim=0)

            negative_amp_obs_enc = torch.cat(negative_amp_obs_enc)
            negative_amp_obs_enc =torch.repeat_interleave(negative_amp_obs_enc, repeats.int(), dim=0)


            anchor_encoding = anchor_amp_obs_enc / torch.norm(anchor_amp_obs_enc, p=2, dim=-1, keepdim=True)            
            positive_encoding = positive_amp_obs_enc / torch.norm(positive_amp_obs_enc, p=2, dim=-1, keepdim=True)
            negative_encoding = negative_amp_obs_enc / torch.norm(negative_amp_obs_enc, p=2, dim=-1, keepdim=True)


            # pos_dist = torch.norm(anchor_encoding-positive_encoding, p=2, dim=-1, keepdim=True)
            # neg_dist = torch.norm(anchor_encoding-negative_encoding, p=2, dim=-1, keepdim=True)
            pos_dist = torch.norm(anchor_encoding-positive_encoding, p=2, dim=-1, keepdim=True)
            neg_dist = torch.norm(anchor_encoding-negative_encoding, p=2, dim=-1, keepdim=True)

    
            rewards = - pos_dist/2 + neg_dist/2 # # --> unique latents, 1
            # rewards = torch.repeat_interleave(rewards, repeats.int(), dim=0)
            rewards = rewards.reshape(_amp_obs.size(1), _amp_obs.size(0),-1).transpose(1,0)
            rewards *= self.reward_scale

            if rewards.mean()>0.7:
                self.reward_scale*=0.75
                self.reward_scale = max(self.reward_scale, self.reward_scale_min)

            print('REWARD SCALE -----------------------: ', self.reward_scale)
       
            return (rewards,
                None,
                {'pos_mean':torch.norm(positive_encoding, p=2, dim=-1).mean(),
                 'neg_mean':torch.norm(negative_encoding, p=2, dim=-1).mean()})



    
    def build(self, name, **kwargs):
        net = TripASEBuilder.Network(self.params, **kwargs)
        return net
