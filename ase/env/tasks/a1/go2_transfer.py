from enum import Enum
import numpy as np
import torch

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.a1.a1_ase import A1ASE
from utils import gym_util
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *

from utils import torch_utils

from .go2_config import Go2Config
from ..base_task import BaseTask

class Go2Transfer(A1ASE):
    """
    Assuming that we have trained a policy for the A1 Unitree - converting this to the Go1 which has different joint freedoms, mopstly

    A1 Limits:
        0
        -1.123992 1.123992
        1
        -2.094395 5.2359877
        2
        -3.0525808 -0.56025076
        3
        -1.123992 1.123992
        4
        -2.094395 5.2359877
        5
        -3.0525808 -0.56025076
        6
        -1.123992 1.123992
        7
        -2.094395 5.2359877
        8
        -3.0525808 -0.56025076
        9
        -1.123992 1.123992
        10
        -2.094395 5.2359877
        11
        -3.0525808 -0.56025076

    Go2 Limits:
        0
        -1.46608 1.46608
        1
        -2.5831 4.5030003
        2
        -3.099688 -0.46077195
        3
        -1.46608 1.46608
        4
        -2.5831 4.5030003
        5
        -3.099688 -0.46077195
        6
        -1.46608 1.46608
        7
        -1.5359001 5.5502
        8
        -3.099688 -0.46077195
        9
        -1.46608 1.46608
        10
        -1.5359001 5.5502
        11
        -3.099688 -0.46077195
    """
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        state_init = cfg["env"]["stateInit"]
        self._state_init = A1ASE.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        self.cfg = cfg
        self.a1_cfg = Go2Config()

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
         
        BaseTask.__init__(self, cfg=self.cfg)
 
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
        
        # Joint positions offsets and PD gains
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

        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None

        self._set_up_limit_matching()

        return
        

    def _set_up_limit_matching(self):
        '''
        given some target proportional target to actual joint target angle should be the same
        '''
        a1 = [[-1.123992, 1.123992],
        [-2.094395, 5.2359877],
        [-3.0525808, -0.56025076],
        [-1.123992, 1.123992],
        [-2.094395, 5.2359877],
        [-3.0525808, -0.56025076],
        [-1.123992, 1.123992],
        [-2.094395, 5.2359877],
        [-3.0525808, -0.56025076],
        [-1.123992, 1.123992],
        [-2.094395, 5.2359877],
        [-3.0525808, -0.56025076]]

        go2 = [
        [-1.46608, 1.46608],
        [-2.5831, 4.5030003],
        [-3.099688, -0.46077195],
        [-1.46608, 1.46608],
        [-2.5831, 4.5030003],
        [-3.099688, -0.46077195],
        [-1.46608, 1.46608],
        [-1.5359001, 5.5502],
        [-3.099688, -0.46077195],
        [-1.46608, 1.46608],
        [-1.5359001, 5.5502],
        [-3.099688, -0.46077195]]

        """
        given 0.2
        jarget = mean-0.2*(diff)
        
        - scale diff, delta mean
        """
        print('/////////////////// TESTING REMAPPING /////////////////////////')

        scale_diffs = []
        for i in range(len(a1)):
            a1_delta = a1[i][1] - a1[i][0]  
            go_delta = go2[i][1] - go2[i][0]
            scale_diffs.append(a1_delta/go_delta)

        mean_deltas = []
        for i in range(len(a1)):
            a1_mean = (a1[i][1] + a1[i][0])/2
            go_mean = (go2[i][1] + go2[i][0])/2
            mean_deltas.append(a1_mean-go_mean)

        _a1_scale_diffs = np.array(scale_diffs)
        _a1_mean_deltas = np.array(mean_deltas)

        # target = 0.4

        # for i in range(len(a1)):
        #     a1_offset = (a1[i][1] + a1[i][0])/2
        #     a1_scale = (a1[i][1] - a1[i][0])/2
        #     a1_target = 0.4 * (a1_scale) + a1_offset

        #     go_offset = (go2[i][1] + go2[i][0])/2
        #     go_scale = (go2[i][1] - go2[i][0])/2
        #     go_target = 0.4 * (go_scale) + go_offset

        #     go_remapped = 0.4 * (go_scale*scale_diffs[i]) + mean_deltas[i] + go_offset
            
        #     print('targets')
        #     print(a1_target)
        #     print(go_target)
        #     print(go_remapped)

        #     input()

        return _a1_scale_diffs, _a1_mean_deltas
    

    def _setup_character_props(self, key_bodies):
  

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)
        print(asset_file)
        if  (asset_file == "unitree_go2/go2_description/urdf/go2_description.urdf"):
            self._dof_body_ids = [1,2,3,5,6,7,9,10,11,13,14,15] 
            self._dof_offsets = list(np.arange(start=0, stop=(13), step=1))
            self._dof_obs_size = 12*6
            self._num_actions = 12
            self._num_obs = 1 + 17 * (3 + 6 + 3 + 3) - 3
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + self._num_actions + 3 * num_key_bodies 
            # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        return
    


    def _action_to_pd_targets(self, action):
        #TODO: PD 
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar


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

        scale_diff, mean_delata = self._set_up_limit_matching()

        self._pd_action_offset += mean_delata
        self._pd_action_scale *= scale_diff

        print(self._pd_action_offset.shape)
        print(self._pd_action_scale.shape)
        print(scale_diff.shape)
        print(mean_delata.shape)
        input()
 
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return
    

    # def pre_physics_step(self, actions):
    #     # print(actions)
    #     if True:
    #         self.actions = actions.to(self.device).clone()
    #         #store action in buffer and update action history
    #         self._action_history_buf[:, 0, :] = self._action_history_buf[:, 1, :]
    #         self._action_history_buf[:, 1, :] = actions
    #         if (self._pd_control):
    #             pd_tar = self._action_to_pd_targets(self.actions)
    #             pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
             
    #             self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
               
    #     else:

    #         self.actions = actions.to(self.device).clone()
    #         self.actions = self.reindex_dof(actions)

    #         clip_actions = self.a1_cfg.normalization.clip_actions / self.a1_cfg.control.action_scale
    #         self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

    #     return
    
    # def _physics_step(self):
    #     if True:
    #         for i in range(self.control_freq_inv):
    #             self.render()
    #             self.gym.simulate(self.sim)
    #     else:

    #         # TODO: currently this is pd:60fps, control:30fps -> typically control:200/250, pd:800/1000
    #         for i in range(self.control_freq_inv):
    #             self.render()
    #             self.torques = self._compute_torques(self.actions).view(self.torques.shape)
    #             self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
    #             self.gym.simulate(self.sim)
    #             self.gym.fetch_results(self.sim, True)
    #             self.gym.refresh_dof_state_tensor(self.sim)

    #     return
    

    # def _action_to_pd_targets(self, action):
    #     #TODO: PD 
    #     pd_tar = self._pd_action_offset + self._pd_action_scale * action
    #     return pd_tar
    


