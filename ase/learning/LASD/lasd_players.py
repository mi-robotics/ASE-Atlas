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

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import players
import learning.ase_players as ase_player

class LASDPlayerContinuous(ase_player.ASEPlayer):
    def __init__(self, config):
        super().__init__(config)
        return

    # def get_action(self, obs_dict, is_determenistic=False):
    #     self._update_latents()

    #     obs = obs_dict['obs']
    #     if len(obs.size()) == len(self.obs_shape):
    #         obs = obs.unsqueeze(0)
    #     obs = self._preproc_obs(obs)
    #     ase_latents = self._ase_latents

    #     input_dict = {
    #         'is_train': False,
    #         'prev_actions': None, 
    #         'obs' : obs,
    #         'rnn_states' : self.states,
    #         'ase_latents': ase_latents
    #     }
    #     with torch.no_grad():
    #         res_dict = self.model(input_dict)
    #     mu = res_dict['mus']
    #     action = res_dict['actions']
    #     self.states = res_dict['rnn_states']
    #     if is_determenistic:
    #         current_action = mu
    #     else:
    #         current_action = action
    #     current_action = current_action.detach()
    #     return  players.rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
    
    # def _build_net_config(self):
    #     config = super()._build_net_config()
    #     config['ase_latent_shape'] = (self._latent_dim,)
    #     return config
    
