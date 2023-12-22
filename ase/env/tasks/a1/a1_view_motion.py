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

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *
from utils import torch_utils
import math

from env.tasks.a1.a1_ase import A1ASE
import time

class A1ViewMotion(A1ASE):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        control_freq_inv = cfg["env"]["controlFrequencyInv"]
        self._motion_dt = control_freq_inv * sim_params.dt


        cfg["env"]["controlFrequencyInv"] = 1
        cfg["env"]["pdControl"] = False

       

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        num_motions = self._motion_lib.num_motions()
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)

        return

    def pre_physics_step(self, actions):
        # self.actions = actions.to(self.device).clone()
        # forces = torch.zeros_like(self.actions)
        # force_tensor = gymtorch.unwrap_tensor(forces)
        # self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        
        return

    def post_physics_step(self):
        super().post_physics_step()
        self._motion_sync()
        self._create_sphere()
        time.sleep(0.1)

        print('SIM FOOT POS')
        print(self.feet_names)
        print(self._rigid_body_pos[0, self._key_body_ids,:])

        input()
            

        #     pass



        # input()
 

        return
    
    def _get_humanoid_collision_filter(self):
        return 1 # disable self collisions

    def _motion_sync(self):
        num_motions = self._motion_lib.num_motions()
        motion_ids = self._motion_ids
        motion_times = self.progress_buf * self._motion_dt

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
           = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        print('MOTION FOOT POS')
        print(key_pos)
        
     
        
        root_vel = torch.zeros_like(root_vel)
        root_ang_vel = torch.zeros_like(root_ang_vel)
        dof_vel = torch.zeros_like(dof_vel)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        env_ids_int32 = self._robot_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return
    
    def _create_sphere(self):
        foot_contacts = torch.where(torch.sum(self._contact_forces[:, self.feet_indices, :], dim=-1) > 0, True, False)
        contact_positions = self._rigid_body_pos[:, self.feet_indices]
        
    
        self.gym.fetch_results(self.sim, True)
        self.gym.clear_lines(self.viewer)

        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 12, 12, sphere_pose, color=(1, 1, 0))
        
        for i in range(self.num_envs):
    
            pos = contact_positions[i][foot_contacts[i]]
    
            if pos.size(0)>0:
                for p in contact_positions[i][foot_contacts[i]]:
                    t = gymapi.Transform(gymapi.Vec3(*p))
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], t)

        # # # Update the viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        # # Wait for dt to elapse in real time.
        # # This synchronizes the physics simulation with the rendering rate.
        self.gym.sync_frame_time(self.sim)
        
        return

    def _compute_reset(self):
        motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        self.reset_buf[:], self._terminate_buf[:] = compute_view_motion_reset(self.reset_buf, motion_lengths, self.progress_buf, self._motion_dt)
        return

    def _reset_actors(self, env_ids):
        return

    def _reset_env_tensors(self, env_ids):
        num_motions = self._motion_lib.num_motions()
        self._motion_ids[env_ids] = torch.remainder(self._motion_ids[env_ids] + self.num_envs, num_motions)
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

@torch.jit.script
def compute_view_motion_reset(reset_buf, motion_lengths, progress_buf, dt):
    # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    motion_times = progress_buf * dt
    reset = torch.where(motion_times > motion_lengths, torch.ones_like(reset_buf), terminated)
    return reset, terminated