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

from learning import amp_network_builder
from learning.modules.velocity_estimator import VelocityEstimator


from learning.ase_network_builder import ASEBuilder 
from learning.modules.world_model import WorldModel

ENC_LOGIT_INIT_SCALE = 0.1

class LatentType(enum.Enum):
    uniform = 0
    sphere = 1

class CuriASEBuilder(ASEBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(ASEBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__()
            self._build_world_model(self)
            return
        
        def load(self, params):
            super().load(params)
           
            self._enc_units = params['enc']['units']
            self._enc_activation = params['enc']['activation']
            self._enc_initializer = params['enc']['initializer']
            self._enc_separate = params['enc']['separate']

            return
        
        def _build_world_model(self):
            config = {

            }
            self.world_model = WorldModel(config)
            return
        
        def _compute_curiase_r(self, states, next_states, actions):
            return self.world_model.reward(next_states, states, actions)
        
        
        
        



    def build(self, name, **kwargs):
        net = CuriASEBuilder.Network(self.params, **kwargs)
        return net
