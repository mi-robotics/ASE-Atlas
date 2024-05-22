import numpy as np
import torch

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.a1.a1_case import A1CASE, build_amp_observations
from utils import gym_util
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *

from utils import torch_utils

class A1CASEGetUp(A1CASE):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)


        return
    
    def _reset_ref_state_init(self, env_ids):
        '''
        - If skill label is recovery skill label then we set the initial state to that clip
        - we want to ensure when reset the state we do not reset in recover state for all other skills
        '''

        recovery_skill_labels = torch.tensor(self._motion_lib.recovery_skill_label, device=self.device)
        revocery_moiton_ids = torch.tensor(self._motion_lib.recovery_motion_ids, device=self.device)
    

        new_skill_labels = self._skill_conditions[env_ids]
        recovery_mask = new_skill_labels == recovery_skill_labels
        recovery_mask = torch.all(recovery_mask, dim=-1) # 
        num_recovery = len(torch.nonzero(recovery_mask, as_tuple=True)[0])
       
        revocery_moiton_ids_choice = revocery_moiton_ids[torch.randint(0, len(revocery_moiton_ids), (num_recovery,), device=self.device)]
        
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs, exclude_index=revocery_moiton_ids)
        motion_ids[recovery_mask] = revocery_moiton_ids_choice

    
        if (self._state_init == A1CASE.StateInit.Random
            or self._state_init == A1CASE.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == A1CASE.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return
    
    def _compute_reset(self):
        '''
        - if the skill label is the recovery do not terminate on base ground contact
        '''

        recovery_skill_labels = torch.tensor(self._motion_lib.recovery_skill_label, device=self.device)
 
        recovery_mask = self._skill_conditions == recovery_skill_labels
        recovery_mask = torch.all(recovery_mask, dim=-1) # 
        

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights,
                                                   recovery_mask)
        
       
        return
    
@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, recovery_indexs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, 1:, :] = 0        
        masked_contact_buf[recovery_indexs, 0, :] = 0

        fall_contact = torch.any(torch.abs(masked_contact_buf) > 1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_or(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        # print(fall_contact, fall_height)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

    

 




    
