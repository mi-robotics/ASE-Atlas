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

from learning import amp_models
import torch

class ModelDIMContinuous(amp_models.ModelAMPContinuous):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('dim', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelDIMContinuous.Network(net)

    class Network(amp_models.ModelAMPContinuous.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            result = super().forward(input_dict)

            if (is_train):
                amp_obs = input_dict['amp_obs']
                amp_obs_latents = input_dict['amp_obs_latents']

                amp_obs_replay = input_dict['amp_obs_replay']
                ase_latents_replay = input_dict['ase_latents_replay']

                #TODO condition this with config
                amp_obs = torch.cat((amp_obs, amp_obs_replay))
                amp_obs_latents = torch.cat((amp_obs_latents, ase_latents_replay))
                
                result['dim_real_logit'], result['dim_fake_logits'] = self.dim_forward(amp_obs, amp_obs_latents)

            return result
        
        def dim_forward(self, amp_obs, amp_obs_latents):

            real_samples = torch.cat((amp_obs, amp_obs_latents), dim=-1)

            print('CHECKING LATENT SHAPES---')
            print(amp_obs_latents.size())
            fake_latents = amp_obs_latents[torch.randperm(amp_obs_latents.size(0))]
            fake_samples = torch.cat((amp_obs, fake_latents), dim=-1)

            all_samples = torch.cat((real_samples, fake_samples))

            logits = self.a2c_network.eval_dim(all_samples)

            real_logits = logits[:real_samples.size(0)]
            fake_logits = logits[real_samples.size(0):]
            return real_logits, fake_logits 
