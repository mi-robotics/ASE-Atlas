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

import torch

from isaacgym.torch_utils import *
from rl_games.algos_torch import players
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from learning import amp_players
from learning import ase_network_builder
from learning.modules.velocity_estimator import VelocityEstimator

# ASE_LATENT_FIXING = torch.Tensor([[-0.0583, -0.2628, -0.0631, -0.0972,  0.1531,  0.1208,  0.2005, -0.3900,
#          -0.0791, -0.2718, -0.5174, -0.1158,  0.2162,  0.0113,  0.1427,  0.0029,
#          -0.2820,  0.0700,  0.2520,  0.0654, -0.1332,  0.1135, -0.0670, -0.2607]]).cuda()

ASE_LATENT_FIXING = None

PLOT_MEASUREMENTS = False

class ASEPlayer(amp_players.AMPPlayerContinuous):
    def __init__(self, config):
        self._latent_dim = config['latent_dim']
        self._latent_steps_min = config.get('latent_steps_min', np.inf)
        self._latent_steps_max = config.get('latent_steps_max', np.inf)
        self._obs_delay = config.get('player_obs_delay', 0)

        self._enc_reward_scale = config['enc_reward_scale']
        
        super().__init__(config)

        
        
        if (hasattr(self, 'env')):
            batch_size = self.env.task.num_envs
        else:
            batch_size = self.env_info['num_envs']
        self._ase_latents = torch.zeros((batch_size, self._latent_dim), dtype=torch.float32,
                                         device=self.device)
        
        print(self.env.observation_space.shape)
        self._obs_buffer = torch.zeros((batch_size, 3, self.env.observation_space.shape[0]), dtype=torch.float32,
                                         device=self.device)
        
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

        if PLOT_MEASUREMENTS:
            # Initialize data structures for each plot
            self.data_streams = [([], []) for _ in range(9)]
            self.init_plots()
            # Animate
            ani = FuncAnimation(self.fig, self.update_plots, blit=False, interval=1000)
            # Non-blocking show
            plt.show(block=False)

        return
    
    def init_plots(self):
        self.fig, self.axes = plt.subplots(3, 3, figsize=(15, 10))
        return
    
    def update_data(self, obs):
        #dof pos: one leg: FL (2)
        #dof vel: one leg: FL (2)
        #root_ang_vel: (3)
        #foot pos: one leg: FL (3)
        #total: (10)
        # new_data should be a list of tuples/lists with 9 elements, each containing the new data for the respective plot
        
        
        for i, data in enumerate(obs):
            x, y = self.data_streams[i]
            x.append(data[0])
            y.append(data[1])
            self.data_streams[i] = (x, y)

    def update_plots(self, frame):
        for i, ax in enumerate(self.axes.flatten()):
            x, y = self.data_streams[i]
            ax.clear()
            ax.plot(x, y)
        plt.tight_layout()
        
    
    def restore(self, fn):
        super().restore(fn)
        
        path = fn.split('.pth')[0]

        for mod in self.modules.keys():
            mod_path = f'{path}_{mod}.pth'
            self.modules[mod].load_state_dict(torch.load(mod_path))
        return

    def run(self):
        self._reset_latent_step_count()
        super().run()
        return

    def get_action(self, obs_dict, is_determenistic=False):
        self._update_latents()

        # print(self._ase_latents)

        obs = obs_dict['obs']
        if len(obs.size()) == len(self.obs_shape):
            obs = obs.unsqueeze(0)

        self.update_observation_buffer(obs)

        obs = self._obs_buffer[:, self._obs_delay]

        if PLOT_MEASUREMENTS:
            self.update_data(obs)

        obs = self._preproc_obs(obs)
        ase_latents = self._ase_latents

        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states,
            'ase_latents': ase_latents
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
        return  players.rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))

    def update_observation_buffer(self, obs):
        self._obs_buffer[:, [1,2], :] = self._obs_buffer[:, [0,1], :]
        self._obs_buffer[:, [0], :] = obs

    def env_reset(self, env_ids=None):
        obs = super().env_reset(env_ids)
        self._reset_latents(env_ids)
        return obs
    
    def _build_net_config(self):
        config = super()._build_net_config()
        config['ase_latent_shape'] = (self._latent_dim,)
        return config
    
    def _reset_latents(self, done_env_ids=None):

        if ASE_LATENT_FIXING is not None:
            self._ase_latents = ASE_LATENT_FIXING
        else:
            if (done_env_ids is None):
                num_envs = self.env.task.num_envs
                done_env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.device)

            rand_vals = self.model.a2c_network.sample_latents(len(done_env_ids))
            self._ase_latents[done_env_ids] = rand_vals
            self._change_char_color(done_env_ids)

        return

    def _update_latents(self):
        if (self._latent_step_count <= 0):
            self._reset_latents()
            self._reset_latent_step_count()

            if (self.env.task.viewer):
                # print("Sampling new amp latents------------------------------")
                num_envs = self.env.task.num_envs
                env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.device)
                self._change_char_color(env_ids)
        else:
            self._latent_step_count -= 1
        return
    
    def _reset_latent_step_count(self):
        self._latent_step_count = np.random.randint(self._latent_steps_min, self._latent_steps_max)
        return

    def _calc_amp_rewards(self, amp_obs, ase_latents):
        disc_r = self._calc_disc_rewards(amp_obs)
        enc_r = self._calc_enc_rewards(amp_obs, ase_latents)
        output = {
            'disc_rewards': disc_r,
            'enc_rewards': enc_r
        }
        return output
    
    def _calc_enc_rewards(self, amp_obs, ase_latents):
        with torch.no_grad():
            enc_pred = self._eval_enc(amp_obs)
            err = self._calc_enc_error(enc_pred, ase_latents)
            enc_r = torch.clamp_min(-err, 0.0)
            enc_r *= self._enc_reward_scale

        return enc_r
    
    def _calc_enc_error(self, enc_pred, ase_latent):
        err = enc_pred * ase_latent
        err = -torch.sum(err, dim=-1, keepdim=True)
        return err
    
    def _eval_enc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_enc(proc_amp_obs)

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs
            ase_latents = self._ase_latents
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs, ase_latents)
            disc_reward = amp_rewards['disc_rewards']
            enc_reward = amp_rewards['enc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            enc_reward = enc_reward.cpu().numpy()[0, 0]
            # print("disc_pred: ", disc_pred, disc_reward, enc_reward)
        return

    def _change_char_color(self, env_ids):
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        self.env.task.set_char_color(rand_col, env_ids)
        return
