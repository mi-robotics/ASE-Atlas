

import torch 
import pandas as pd
import numpy as np


from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import players
import learning.ase_players as ase_player

'''
c -> (64, )
z -> (8,)
s -> (109,)
pi(a|s,c,z)

1 = walking 
2 = jump
3 = bound
4 = handstand
5 = mixed walking - slow
6 = mixed walking / gallop
7 = mixed walking / leap
8 = fast wlaking 
9 = handstand
10 = pace / turn (bit dodgey)
11 = natural walking
12 = HS
13 = natural walking - faster
14 = natrual walking v fast
15 = slow mixed walking
16 = natural walking slower
17 = biped
18 = mixed
19 = HS
20 = slow 
21 = HS
22 = bow
23 = natural walking
24 = BP 
25 = mixed / turning 
26 = turning
27 = natty slow
28 = mivery mixed
29 = turn
'''


CLS_PATH = '/home/mcarroll/Documents/cdt-1/ASE-Atlas/ase/utils/unique_classes_all_motions.npy'#_50_seq_len
CLS = 4
COLLECT_DATA = False

class DataItem:
    def __init__(self):
        self.obs = None
        self.next_obs = None
        self.ase_latents = None
        self.skill_index = None

class DataCollection:
    def __init__(self):
        self.obs = []
        self.next_obs = []
        self.ase_latents = []
        self.skill_index = []

        self.num_transitions = 0
        return 
    
    def add_step(self, data:DataItem):
        self.obs.append(data.obs)
        self.next_obs.append(data.next_obs)
        self.ase_latents.append(data.ase_latents)
        self.skill_index.append(data.skill_index)

        self.num_transitions += 1

        if self.num_transitions > 1e5:
            self.save_and_quit()

        if self.num_transitions % 1000:
            print(self.num_transitions, '--------------------------------------------------------------------------')

    def save_and_quit(self):
        obs = torch.stack(self.obs)
        next_obs = torch.stack(self.next_obs)
        ase_latents = torch.stack(self.ase_latents)
        skill_index = torch.stack(self.skill_index)

        print(obs.size())        
        print(next_obs.size())
        print(ase_latents.size())
        print(skill_index.size())


        torch.save(obs, '/home/mcarroll/Documents/cdt-1/ASE-Atlas/ase/learning/CASE/obs.pt')
        torch.save(next_obs, '/home/mcarroll/Documents/cdt-1/ASE-Atlas/ase/learning/CASE/next-obs.pt')
        torch.save(ase_latents, '/home/mcarroll/Documents/cdt-1/ASE-Atlas/ase/learning/CASE/ase_latents.pt')
        torch.save(skill_index, '/home/mcarroll/Documents/cdt-1/ASE-Atlas/ase/learning/CASE/skill_index.pt')

        quit()
        return 

class CASEPlayerContinuous(ase_player.ASEPlayer):
    def __init__(self, config):
        super().__init__(config)
        self.classes = np.load(CLS_PATH)
        self.cls = self.classes[CLS]
        self.cls_index = 0
        self._use_rand_skills = False

        self.data_collector = DataCollection()

        self.data_item = None

  


    def get_action(self, obs_dict, is_determenistic=False):
        self._update_latents()
        

        # print(self._ase_latents)
        obs = obs_dict['obs']
        if len(obs.size()) == len(self.obs_shape):
            obs = obs.unsqueeze(0)

        self.update_observation_buffer(obs)

        obs = self._obs_buffer[:, self._obs_delay]

        # current_action = self.base_policy(obs, self._ase_latents)
        obs = self._preproc_obs(obs)
        ase_latents = self._ase_latents


        # print(ase_latents.shape)
        # print(obs.shape)
        # print(torch.tensor(self.cls, device=self.device).unsqueeze(0).shape)
        # input()

        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states,
            'ase_latents': ase_latents,
            'skill_conditions':torch.tensor(self.cls, device=self.device).unsqueeze(0)
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']

        if is_determenistic:
            current_action = mu
        else:
            current_action = action


        current_action = current_action.detach()

        # print(current_action)
        # input()

        # self.save_policy()

        self.data_item = DataItem()
        self.data_item.obs = obs[0].detach().cpu()
        self.data_item.ase_latents = ase_latents[0].detach().cpu()
        self.data_item.skill_index = torch.tensor([self.cls_index])
    
        return  players.rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        

    def _post_step(self, info, obs):
        self.data_item.next_obs = self._preproc_obs(obs)[0].detach().cpu()
        self.data_collector.add_step(self.data_item)
        self.data_item = None
        return

    def save_policy(self):

        # Velocity estimator ------------------------------------------------------------


        self.vel_estimator.eval()
        # Example input
        example_input = torch.rand(1, 129)

        # Trace the model
        traced_model = torch.jit.trace(self.vel_estimator.cpu().eval(), example_input)

        # Save the scripted model
        traced_model.save("./case_velocity_estimator.pt")

        # Policy ----------------------------------------------------------------------
        
        class CombinedModel(torch.nn.Module):
            def __init__(self, normalizer, policy):
                super(CombinedModel, self).__init__()
                self.normalizer = normalizer.eval().cpu()
                self.policy = policy.eval().cpu()

            def forward(self, input, latent, skill):
                # Apply normalization
                normalized_input = self.normalizer(input)
                # Evaluate the policy's actor with the normalized input and latent vector
                mu, sigma = self.policy.eval_actor(normalized_input, latent, skill)
                return mu#, sigma
            
        cmod = CombinedModel(self.running_mean_std,  self.model.a2c_network)
        cmod.eval()
    
        example_input = torch.rand(1, 109)
        example_latent = torch.rand(1,8)
        example_skill = torch.rand(1,64)

        # Trace the combined model
        traced_combined_model = torch.jit.trace(cmod, (example_input, example_latent, example_skill))

        # Save the traced model
        traced_combined_model.save("./case_policy.pt")

        # Discriminator --------------------------------------------------------------
        class CombinedDisc(torch.nn.Module):
            def __init__(self, normalizer, policy):
                super(CombinedDisc, self).__init__()
                self.normalizer = normalizer.eval().cpu()
                self.policy = policy.eval().cpu()

            def forward(self, amp_obs, skill):
                # Apply normalization
                normalized_input = self.normalizer(amp_obs)
                # Evaluate the policy's actor with the normalized input and latent vector
                input = torch.cat((normalized_input, skill), dim=-1)
                logits = self.policy.eval_disc(input)
                return logits
            
        disc = CombinedDisc(self._amp_input_mean_std,  self.model.a2c_network)
        disc.eval()

        example_amp_obs = torch.rand(1, 10*109)
        example_skill = torch.rand(1,64)

        # Trace the combined model
        traced_combined_disc = torch.jit.trace(disc, (example_amp_obs, example_skill))

        # Save the traced model
        traced_combined_disc.save("./case_discriminator.pt")

    

        input()

        return 
    
    def _update_latents(self):
        if (self._latent_step_count <= 0):
            self._reset_skill()

        super()._update_latents()
        return
    

    def _reset_skill(self):

        if self._use_rand_skills:
       
            self.cls_index = torch.randint(low=0, high=29, size=(1,)).item()

            self.cls = self.classes[self.cls_index]

  

        return 