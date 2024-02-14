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

import numpy as np
import os
import torch
from tqdm import tqdm

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils

from env.tasks.base_task import BaseTask

from .a1_config import A1RoughCfg

class A1(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.a1_cfg = A1RoughCfg()

        self.sim_params = sim_params
        self.physics_engine = physics_engine

        #TODO: remove
        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]

        self._observation_method = self.cfg['env']['obsMethod']

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
        self._use_velocity_observation = self.cfg["env"].get("useVelocityObs", None)
        self._use_noisey_measurement = self.cfg["env"].get("useNoiseyMeasurements", False)
        self._noise_level = self.cfg["env"].get("noiseLevel", 1.0)

        print('ERALY TERMINATION',self._enable_early_termination)
        
        key_bodies = self.cfg["env"]["keyBodies"]
        contact_bodies = self.cfg["env"]["contactBodies"]
        self._setup_character_props(key_bodies)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()
    

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
         
        super().__init__(cfg=self.cfg)
 
        self.dt = self.control_freq_inv * sim_params.dt
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
    
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 4 #Number contact sensors - 4 feet
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()
    
        self._robot_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]
        self._initial_robot_root_states = self._robot_root_states.clone()
        self._initial_robot_root_states[:, 7:13] = 0

        initial_pos_tensor = torch.Tensor(self.a1_cfg.init_state.pos)
        initial_pos_tensor = initial_pos_tensor.to(self._initial_robot_root_states.device)
        self._initial_robot_root_states[:, :3] = initial_pos_tensor


        self._robot_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]
        self.base_quat = self._root_states[:, 3:7]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs

        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        self._rigid_states = rigid_body_state_reshaped
        self._rigid_body_pos = rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., :self.num_bodies, 10:13]
        
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., :self.num_bodies, :]
        
        #TODO: must be initializes and reset
        self._action_history_buf = torch.zeros((self.num_envs, 2, self.num_actions), device=self.device, dtype=torch.float)
        self._contact_filter = torch.zeros((self.num_envs, 2, 4), device=self.device, dtype=torch.float)
        self._velocity_obs_buf = torch.zeros((self.num_envs, 121), device=self.device, dtype=torch.float)



        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        self._build_termination_heights()

        # Adding stuff here -----------------------------------------------------------------
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
         
            angle = self.a1_cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.a1_cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.a1_cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.a1_cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.a1_cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        #TODO: hack init position
        self.default_dof_pos_all[:] = self._initial_dof_pos[0]#self.default_dof_pos[0]

        # Adding stuff here -----------------------------------------------------------------
        
        self._key_body_ids = self._build_key_body_ids_tensor(self.feet_names)

       
        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)

        if self.viewer != None:
            self._init_camera()

        
        

        return

    def get_obs_size(self):
        if self._observation_method == 'max':
            return 253
        else:
            return 109
        

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        return

    def reset(self, env_ids=None):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)
        return
    
    def get_velocity_obs(self, env_ids):
        return self._velocity_obs_buf[env_ids]

    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.actor_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
        return

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._robot_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self._velocity_obs_buf[env_ids] = 0
        self._action_history_buf[env_ids] = 0
        self._contact_filter[env_ids] = 0
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
     
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _setup_character_props(self, key_bodies):
        
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        if  (asset_file == "parkour/a1/urdf/a1.urdf"):
            self._dof_body_ids = [1,2,3,5,6,7,9,10,11,13,14,15] 
            self._dof_offsets = list(np.arange(start=0, stop=(13), step=1))
            self._dof_obs_size = 12*6
            self._num_actions = 12
            self._num_obs = 1 + 17 * (3 + 6 + 3 + 3) - 3
            #[height, ]

        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        return

    def _build_termination_heights(self):
        #NOTE: we will not need to use termination heights
        termination_height = -1
        self._termination_heights = np.array([termination_height] * self.num_bodies)
        
        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return

    
    def _create_envs(self, num_envs, spacing, num_per_row):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        asset_options = gymapi.AssetOptions()
        # asset_options.default_dof_drive_mode = self.a1_cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.a1_cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.a1_cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.a1_cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.a1_cfg.asset.fix_base_link
        asset_options.density = self.a1_cfg.asset.density 
        asset_options.angular_damping = self.a1_cfg.asset.angular_damping
        asset_options.linear_damping = self.a1_cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.a1_cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.a1_cfg.asset.max_linear_velocity
        asset_options.armature = self.a1_cfg.asset.armature
        asset_options.thickness = self.a1_cfg.asset.thickness
        asset_options.disable_gravity = self.a1_cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
 

        self.torso_index = 0
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
  
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        self.num_joints = self.gym.get_asset_joint_count(robot_asset)
        self.feet_names = [s for s in body_names if self.a1_cfg.asset.foot_name in s]
    
        #Add foot force sensor to the robot 
        for s in self.feet_names:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)
        
         
        # penalized_contact_names = []
        # for name in self.a1_cfg.asset.penalize_contacts_on:
        #     penalized_contact_names.extend([s for s in body_names if name in s])
        #TODO: we can do something smart here
        termination_contact_names = []
        for name in self.a1_cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])


        base_init_state_list = self.a1_cfg.init_state.pos + self.a1_cfg.init_state.rot + self.a1_cfg.init_state.lin_vel + self.a1_cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.actor_handles = [] 
        self.envs = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, 
                            env_ptr, 
                            robot_asset, 
                            rigid_shape_props_asset,
                            dof_props_asset,
                            start_pose)
            self.envs.append(env_ptr)

        dofFrames = self.gym.get_actor_dof_frames(self.envs[0], self.actor_handles[0])
        self._dof_frames = torch.zeros((self.num_envs, self.num_dofs, 3), dtype=torch.float, device=self.device)
        for i, axis in enumerate(dofFrames['axis']):
            a = torch.Tensor(axis) 
            a[torch.isclose(a, torch.tensor(0.0), atol=1e-6 )] = 0
            self._dof_frames[:,i] = a

     
        
        self._process_dof_props(dof_props_asset)

        #I dont this we want randomization for train ase low level
        # if self.a1_cfg.domain_rand.randomize_friction:
        #     self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)

        self.feet_indices = torch.zeros(len(self.feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.feet_names[i])

        # self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        # for i in range(len(penalized_contact_names)):
        #     self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
        thigh_names = ["FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint"]
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(thigh_names):
            self.thigh_indices[i] = self.dof_names.index(name)
        calf_names = ["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"]
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(calf_names):
            self.calf_indices[i] = self.dof_names.index(name)

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.actor_handles[0])
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()

    def _process_dof_props(self, props):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset

        Returns:
            [numpy.array]: Modified DOF properties
        """
     
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(len(props)):
            self.dof_pos_limits[i, 0] = props["lower"][i].item()
            self.dof_pos_limits[i, 1] = props["upper"][i].item()
            self.dof_vel_limits[i] = props["velocity"][i].item()
            self.torque_limits[i] = props["effort"][i].item()
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.a1_cfg.rewards.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.a1_cfg.rewards.soft_dof_pos_limit
        return props
    
    def _build_env(self, env_id, env_ptr, robot_asset, rigid_shape_props_asset, dof_props_asset, start_pose):
        col_group = env_id
        segmentation_id = 0
 
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        #NOTE i have enabled self collision this may cause issues 
        actor_handle = self.gym.create_actor(env_ptr, robot_asset, start_pose, "a1_robot", col_group, self.a1_cfg.asset.self_collisions, segmentation_id)

        #TODO: UNSURE keep or remove
        self.gym.enable_actor_dof_force_sensors(env_ptr, actor_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, actor_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        if (self._pd_control and True):
            dof_prop = self.gym.get_asset_dof_properties(robot_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            dof_prop["stiffness"].fill(self.a1_cfg.control.stiffness['joint'])
            dof_prop["damping"].fill(self.a1_cfg.control.damping['joint'])
            self.gym.set_actor_dof_properties(env_ptr, actor_handle, dof_prop)

        self.actor_handles.append(actor_handle)

        return 

    def _build_pd_action_offset_scale(self):
        num_joints = len(self._dof_offsets) - 1
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = self._dof_offsets[j]
            dof_size = self._dof_offsets[j + 1] - self._dof_offsets[j]

            if (dof_size == 3):
                curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
                curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
                curr_low = np.max(np.abs(curr_low))
                curr_high = np.max(np.abs(curr_high))
                curr_scale = max([curr_low, curr_high])
                curr_scale = 1.2 * curr_scale
                curr_scale = min([curr_scale, np.pi])

                lim_low[dof_offset:(dof_offset + dof_size)] = -curr_scale
                lim_high[dof_offset:(dof_offset + dof_size)] = curr_scale
                
                #lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                #lim_high[dof_offset:(dof_offset + dof_size)] = np.pi


            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
               
                curr_mid = 0.5 * (curr_high + curr_low)
                
                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

   

        
                
        # mid range of the limits
        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        # max direction either side of the limits from the mid
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
 
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _get_humanoid_collision_filter(self):
        return 0

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights)
        
       
        return
    
    def _get_ground_penetration(self):
     
        foot_contacts_bool = self._contact_forces[:, self.feet_indices, 2] > 10
        if self.cfg.env.include_foot_contacts:
            
            return foot_contacts_bool
        else:
            return torch.zeros_like(foot_contacts_bool).to(self.device)

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        obs, noised_obs = self._compute_humanoid_obs(env_ids)

        if (env_ids is None):
            self.critic_obs_buf[:] = obs
            self.obs_buf[:] = noised_obs
        else:
            self.critic_obs_buf[env_ids] = obs
            self.obs_buf[env_ids] = noised_obs

        #TODO: i need a better way of getting velocity obs, also this should be noised
        if self._use_velocity_observation is not None:
            vel_obs = self._compute_velocity_obs(env_ids)
            if (env_ids is None):
                self._velocity_obs_buf[:] = vel_obs
            else:
                self._velocity_obs_buf[env_ids] = vel_obs

        return
    

    def _get_noised_measurements(self, dof_pos, dof_vel, ang_vel, root_rot, foot_pos ):
        """
        class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        quantize_height = True
        class noise_scales:
            rotation = 0.0
            dof_pos = 0.01 - 0
            dof_vel = 0.05
            lin_vel = 0.05
            ang_vel = 0.05
            gravity = 0.02 / rotation
            height_measurements = 0.02

        """
        def sample_noise(x, scale):
            return (2.0 * torch.rand_like(x) - 1) * scale * self._noise_level

        _dof_pos = dof_pos + sample_noise(dof_pos, self.a1_cfg.noise.noise_scales.dof_pos)
        _dof_vel = dof_vel + sample_noise(dof_vel, self.a1_cfg.noise.noise_scales.dof_vel)
        # _lin_vel = lin_vel + sample_noise(lin_vel, self.a1_cfg.noise.noise_scales.lin_vel)
        _ang_vel = ang_vel + sample_noise(ang_vel, self.a1_cfg.noise.noise_scales.ang_vel)
        _foot_pos = foot_pos + sample_noise(foot_pos, self.a1_cfg.noise.noise_scales.feet_pos)

        # Extract components
        w, v = root_rot[:, 0], root_rot[:, 1:]
        angles = 2 * torch.acos(w)
        axis = v / torch.sin(angles/2).unsqueeze(1)

        _angles = angles + sample_noise(angles, self.a1_cfg.noise.noise_scales.gravity)
        
        #TODO: Replace this with func - Q form angle axis
        new_w = torch.cos(_angles / 2)
        new_v = axis * torch.sin(_angles / 2).unsqueeze(1)
        _root_rot = torch.cat((new_w.unsqueeze(1), new_v), dim=1)

        return _dof_pos, _dof_vel, _ang_vel, _root_rot, _foot_pos
    
 
    def _compute_humanoid_obs(self, env_ids=None):

        if self._observation_method == 'max':
            if (env_ids is None):
                body_pos = self._rigid_body_pos
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
            else:
                body_pos = self._rigid_body_pos[env_ids]
                body_rot = self._rigid_body_rot[env_ids]
                body_vel = self._rigid_body_vel[env_ids]
                body_ang_vel = self._rigid_body_ang_vel[env_ids]

            obs = compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
                                                    self._root_height_obs)

        else:
            if  (env_ids is None):
                env_ids = torch.arange(self.num_envs)
            
            root_pos = self._rigid_body_pos[env_ids][:,0,:]
            root_rot = self._rigid_body_rot[env_ids][:,0,:]
            root_vel = self._rigid_body_vel[env_ids][:,0,:]
            root_ang_vel = self._rigid_body_ang_vel[env_ids][:,0,:]
            dof_pos = self._dof_pos[env_ids][:,:]
            dof_vel = self._dof_vel[env_ids][:,:]
            key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
          
            obs = compute_humanoid_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos,
                                                self._local_root_obs, self._root_height_obs, self._dof_obs_size, self._dof_offsets, self._dof_frames)
            
            


            if self._use_noisey_measurement:
                _dof_pos, _dof_vel, _root_ang_vel, _root_rot, _key_body_pos = self._get_noised_measurements(dof_pos, dof_vel, root_ang_vel, root_rot, key_body_pos)
                #Note we are not noising the feet positions, if we want to do this we have to use a kinematic solver
                noise_obs = compute_humanoid_observations(root_pos, _root_rot, root_vel, _root_ang_vel, _dof_pos, _dof_vel, _key_body_pos,
                                                    self._local_root_obs, self._root_height_obs, self._dof_obs_size, self._dof_offsets, self._dof_frames)
            else:
                noise_obs = obs.clone()
            
        return obs, noise_obs

    def _compute_velocity_obs(self, env_ids=None):
        """
        - base_angular_vel
        - base_rpy
        - dof_positions
        - dof_vel
        - previous_actions
        - foot_contacts
        root_pos, root_rot, root_ang_vel, dof_pos, dof_vel, key_body_pos,
                                 action_hist, contact_filter, dof_obs_size, dof_offsets, dof_frames
        """

        if (env_ids is None):
            root_pos = self._rigid_body_pos[:, 0, :]
            root_rot = self._rigid_body_rot[:, 0, :]
            root_ang_vel = self._rigid_body_ang_vel[:, 0, :]
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            key_body_pos = self._rigid_body_pos[:][:, self._key_body_ids, :]
            action_hist = self._action_history_buf[:, 0, :]
            contact_filter = self._contact_filter[:, 1, :]
        else:
            root_pos = self._rigid_body_pos[env_ids, 0, :]
            root_rot = self._rigid_body_rot[env_ids, 0, :]
            root_ang_vel = self._rigid_body_ang_vel[env_ids, 0, :]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids,:]
            action_hist = self._action_history_buf[env_ids][:, 0, :]
            contact_filter = self._contact_filter[env_ids][:, 1,:]

        if self._use_noisey_measurement:
            dof_pos, dof_vel, root_ang_vel, root_rot, key_body_pos = self._get_noised_measurements(dof_pos, dof_vel, root_ang_vel, root_rot, key_body_pos)

        velocity_obs = compute_velocity_observation(
            root_pos, root_rot, root_ang_vel, dof_pos, dof_vel, key_body_pos,
            action_hist, contact_filter, self._dof_obs_size, self._dof_offsets, self._dof_frames
        )

        return velocity_obs

    def _reset_actors(self, env_ids):
        self._robot_root_states[env_ids] = self._initial_robot_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        return
    

    def reindex_dof(self, vec):
        return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.a1_cfg.control.action_scale
        control_type = self.a1_cfg.control.control_type

        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos_all - self._dof_pos) - self.d_gains*self._dof_vel
      
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def pre_physics_step(self, actions):
        # print(actions)
        if True:
            self.actions = actions.to(self.device).clone()
            #store action in buffer and update action history
            self._action_history_buf[:, 0, :] = self._action_history_buf[:, 1, :]
            self._action_history_buf[:, 1, :] = actions
            if (self._pd_control):
                pd_tar = self._action_to_pd_targets(self.actions)
                pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
             
                self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
               
        else:

            self.actions = actions.to(self.device).clone()
            self.actions = self.reindex_dof(actions)

            clip_actions = self.a1_cfg.normalization.clip_actions / self.a1_cfg.control.action_scale
            self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        return
    
    def _physics_step(self):
        if True:
            for i in range(self.control_freq_inv):
                self.render()
                self.gym.simulate(self.sim)
        else:

            # TODO: currently this is pd:60fps, control:30fps -> typically control:200/250, pd:800/1000
            for i in range(self.control_freq_inv):
                self.render()
                self.torques = self._compute_torques(self.actions).view(self.torques.shape)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self.gym.refresh_dof_state_tensor(self.sim)

        return

    def post_physics_step(self):
        self.progress_buf += 1

        contact = torch.norm(self._contact_forces[:, self.feet_indices], dim=-1) > 2.
        self._contact_filter[:,1,:] = torch.logical_or(contact, self._contact_filter[:,0,:]) 
        self._contact_filter[:,0,:] = contact
        
        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()

        
        self.extras["terminate"] = self._terminate_buf
        self.extras["velocity_obs"] = self._velocity_obs_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.actor_handles[0]
        body_ids = []
        

        for body_name in key_body_names:

            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.actor_handles[0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
         
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
      
        return body_ids

    def _action_to_pd_targets(self, action):
        #TODO: PD 
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def render(self, sync_frame_time=False):
        if self.viewer:
            self._update_camera()

        super().render(sync_frame_time)
        return
    
    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._robot_root_states[0, 0:3].cpu().numpy()
        
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], 
                              self._cam_prev_char_pos[1] - 3.0, 
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._robot_root_states[0, 0:3].cpu().numpy()
        
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
                                  char_root_pos[1] + cam_delta[1], 
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_obs(pose, dof_obs_size, dof_offsets, dof_frames):
    # type: (Tensor, int, List[int], Tensor) -> Tensor
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)

    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose)
        elif (dof_size == 1):
      
            axis = dof_frames[0,j]
            joint_pose_q = quat_from_angle_axis(joint_pose[...,0], axis)

        else:
            joint_pose_q = None
            assert(False), "Unsupported joint type="

        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q)
        dof_obs[:, (j * joint_obs_size):((j + 1) * joint_obs_size)] = joint_dof_obs

    assert((num_joints * joint_obs_size) == dof_obs_size)

    return dof_obs

@torch.jit.script
def compute_humanoid_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos,
                                  local_root_obs, root_height_obs, dof_obs_size, dof_offsets, dof_frames):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int], Tensor) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets, dof_frames)

    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs

@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    #inject noise to the velocity reading
    # noise = torch.randn((1,3), device='cuda:0')*0.01
    # local_body_vel[:, :3] += noise

    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel), dim=-1)
    return obs

@torch.jit.script
def compute_velocity_observation(root_pos, root_rot, root_ang_vel, dof_pos, dof_vel, key_body_pos,
                                 action_hist, contact_filter, dof_obs_size, dof_offsets, dof_frames):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, List[int], Tensor) -> Tensor

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    roll, pitch, _ = get_euler_xyz(root_rot)
    yaw = torch.zeros_like(roll, device=roll.device, dtype=roll.dtype)
 
    imu = quat_from_euler_xyz(roll, pitch, yaw)
    root_rot_obs = torch_utils.quat_to_tan_norm(imu)
    
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets, dof_frames)

    obs = torch.cat((root_rot_obs, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos, action_hist, contact_filter), dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
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
