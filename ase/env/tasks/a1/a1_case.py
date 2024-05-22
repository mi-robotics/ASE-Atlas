import numpy as np
import torch

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.a1.a1_ase import A1ASE, build_amp_observations
from utils import gym_util
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *

from utils import torch_utils

class A1CASE(A1ASE):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        self._skill_conditions = torch.zeros((self.num_envs, 64), device=self.device, dtype=torch.float32)
        self._is_continual = False #TODO
        return
    
    def _load_motion(self, motion_file):
        #TODO use the version with motion labels 
        #Load local skill labels
        assert(self._dof_offsets[-1] == self.num_dof)
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device,
                                     dof_frames=self._dof_frames,
                                     use_classes=True,
                                     class_file=self.cfg['env']['class_files']
                                     )
        
        self._unique_skill_labels = self._motion_lib._unique_skill_labels
        self._unique_skill_labels = self._unique_skill_labels.to(self.device)

        return
    
    
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        self._reset_skill_labels(env_ids)
        super()._reset_envs(env_ids)
        #TODO -> reset env skill label
        self._init_amp_obs(env_ids)

        return
    
    def get_skill_conditions(self, env_ids):
        # print(self._skill_conditions)
        # input()
        return self._skill_conditions[env_ids]
    
    def _reset_skill_labels(self, env_ids):
        if self._is_continual:
            new_skill = self._motion_lib.sample_skill_labels(len(env_ids))
        else:
            new_skill = self._unique_skill_labels[torch.randint(0, len(self._unique_skill_labels), (len(env_ids),), device=self.device)]

        self._skill_conditions[env_ids] = new_skill.clone()
        return 
    
    
    def fetch_amp_obs_demo(self, num_samples):

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        motion_ids = self._motion_lib.sample_motions(num_samples)
        
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        motion_skill_conditions = self._motion_lib._motion_skill_labels[motion_ids]

  

        return amp_obs_demo_flat, motion_skill_conditions
    
    def _compute_amp_observations(self, env_ids=None):

        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if (env_ids is None):
            self._curr_amp_obs_buf[:] = build_amp_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._dof_offsets, self._dof_frames)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._rigid_body_pos[env_ids][:, 0, :],
                                                                   self._rigid_body_rot[env_ids][:, 0, :],
                                                                   self._rigid_body_vel[env_ids][:, 0, :],
                                                                   self._rigid_body_ang_vel[env_ids][:, 0, :],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._dof_offsets, self._dof_frames)
            #TODO: Need to check that dof vel is based on its axis of rotation in the parent
        return
    
    def post_physics_step(self):
        super().post_physics_step()
    
        self.extras["amp_skill_conditions"] = self._skill_conditions

        return
    
