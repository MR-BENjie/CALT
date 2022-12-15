# Copyright (c) 2021-2022, NVIDIA Corporation
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

"""Factory: Class for nut-bolt pick task.

Inherits nut-bolt environment class and abstract task class (not enforced). Can be executed with
python train.py task=FactoryTaskNutBoltPick
"""

import hydra
import omegaconf
import os
import torch
import math

from isaacgym import gymapi, gymtorch, torch_utils
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_env_nut_bolt import FactoryEnvNutBolt
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask
from isaacgymenvs.utils import torch_jit_utils


class FactoryTaskNutBoltPick(FactoryEnvNutBolt, FactoryABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.cfg = cfg
        self.task = "FactoryTaskNutBoltPick"

        self._get_task_yaml_params()
        self._acquire_task_tensors()
        self.parse_controller_spec()

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        if self.viewer is not None:
            self._set_viewer_params()
        self._last_subtask = -1
        self.subtask = 0
    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_nut_bolt.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_nut_bolt = hydra.compose(config_name=asset_info_path)
        self.asset_info_nut_bolt = self.asset_info_nut_bolt['']['']['']['']['']['']['assets']['factory']['yaml']  # strip superfluous nesting

        ppo_path = 'train/'+self.task+'PPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        # Grasp pose tensors
        if self.task == "FactoryTaskNutBoltPick":
            nut_grasp_heights = self.bolt_head_heights + self.nut_heights * 0.5  # nut COM
            self.nut_grasp_pos_local = nut_grasp_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
                (self.num_envs, 1))
            self.nut_grasp_quat_local = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(
                self.num_envs, 1)

            # Keypoint tensors
            self.keypoint_offsets = self._get_keypoint_offsets(
                self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
            self.keypoints_gripper = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3),
                                                 dtype=torch.float32,
                                                 device=self.device)
            self.keypoints_nut = torch.zeros_like(self.keypoints_gripper, device=self.device)

            self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs,
                                                                                                            1)
        elif self.task == "FactoryTaskNutBoltPlace":
            self.nut_base_pos_local = \
                self.bolt_head_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
            bolt_heights = self.bolt_head_heights + self.bolt_shank_lengths
            self.bolt_tip_pos_local = \
                bolt_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))

            # Keypoint tensors
            self.keypoint_offsets = \
                self._get_keypoint_offsets(self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
            self.keypoints_nut = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3),
                                             dtype=torch.float32,
                                             device=self.device)
            self.keypoints_bolt = torch.zeros_like(self.keypoints_nut, device=self.device)

            self.identity_quat = \
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

            self.actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
        elif self.task == "FactoryTaskNutBoltScrew":
            target_heights = self.cfg_base.env.table_height + self.bolt_head_heights + self.nut_heights * 0.5
            self.target_pos = target_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
                (self.num_envs, 1))
        else :
            assert False
    def _refresh_task_tensors(self):
        """Refresh tensors."""

        # Compute pose of nut grasping frame
        if self.task == "FactoryTaskNutBoltPick":
            self.nut_grasp_quat, self.nut_grasp_pos = torch_jit_utils.tf_combine(self.nut_quat,
                                                                                 self.nut_pos,
                                                                                 self.nut_grasp_quat_local,
                                                                                 self.nut_grasp_pos_local)

            # Compute pos of keypoints on gripper and nut in world frame
            for idx, keypoint_offset in enumerate(self.keypoint_offsets):
                self.keypoints_gripper[:, idx] = torch_jit_utils.tf_combine(self.fingertip_midpoint_quat,
                                                                            self.fingertip_midpoint_pos,
                                                                            self.identity_quat,
                                                                            keypoint_offset.repeat(self.num_envs, 1))[1]
                self.keypoints_nut[:, idx] = torch_jit_utils.tf_combine(self.nut_grasp_quat,
                                                                        self.nut_grasp_pos,
                                                                        self.identity_quat,
                                                                        keypoint_offset.repeat(self.num_envs, 1))[1]
        elif self.task == "FactoryTaskNutBoltPlace":
            for idx, keypoint_offset in enumerate(self.keypoint_offsets):
                self.keypoints_nut[:, idx] = torch_jit_utils.tf_combine(self.nut_quat,
                                                                        self.nut_pos,
                                                                        self.identity_quat,
                                                                        (keypoint_offset + self.nut_base_pos_local))[1]
                self.keypoints_bolt[:, idx] = torch_jit_utils.tf_combine(self.bolt_quat,
                                                                         self.bolt_pos,
                                                                         self.identity_quat,
                                                                         (keypoint_offset + self.bolt_tip_pos_local))[1]
        elif self.task == "FactoryTaskNutBoltScrew":
            self.fingerpad_midpoint_pos = fc.translate_along_local_z(pos=self.finger_midpoint_pos,
                                                                     quat=self.hand_quat,
                                                                     offset=self.asset_info_franka_table.franka_finger_length - self.asset_info_franka_table.franka_fingerpad_length * 0.5,
                                                                     device=self.device)
            self.finger_nut_keypoint_dist = self._get_keypoint_dist(body='finger_nut')
            self.nut_keypoint_dist = self._get_keypoint_dist(body='nut')
            self.nut_dist_to_target = torch.norm(self.target_pos - self.nut_com_pos, p=2,
                                                 dim=-1)  # distance between nut COM and target
            self.nut_dist_to_fingerpads = torch.norm(self.fingerpad_midpoint_pos - self.nut_com_pos, p=2,
                                                     dim=-1)  # distance between nut COM and midpoint between centers of fingerpads
        else :
            assert False
    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

        if self.task == "FactoryTaskNutBoltPick":
            self._apply_actions_as_ctrl_targets(actions=self.actions,
                                            ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                                            do_scale=True)
        else:
            self._apply_actions_as_ctrl_targets(actions=self.actions,
                                                ctrl_target_gripper_dof_pos=0.0,
                                                do_scale=True)

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        # In this policy, episode length is constant
        if self.task == "FactoryTaskNutBoltPick":
            is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

            if self.cfg_task.env.close_and_lift:
                # At this point, robot has executed RL policy. Now close gripper and lift (open-loop)
                if is_last_step:
                    self._close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
                    self._lift_gripper(sim_steps=self.cfg_task.env.num_gripper_lift_sim_steps)

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        """Compute observations."""

        # Shallow copies of tensors
        if self.task == "FactoryTaskNutBoltPick":
            obs_tensors = [self.fingertip_midpoint_pos,
                           self.fingertip_midpoint_quat,
                           self.fingertip_midpoint_linvel,
                           self.fingertip_midpoint_angvel,
                           self.nut_grasp_pos,
                           self.nut_grasp_quat]

            self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)
        elif self.task == "FactoryTaskNutBoltPlace":
            obs_tensors = [self.fingertip_midpoint_pos,
                           self.fingertip_midpoint_quat,
                           self.fingertip_midpoint_linvel,
                           self.fingertip_midpoint_angvel,
                           self.nut_pos,
                           self.nut_quat,
                           self.bolt_pos,
                           self.bolt_quat]

            if self.cfg_task.rl.add_obs_bolt_tip_pos:
                obs_tensors += [self.bolt_tip_pos_local]

            self.obs_buf = torch.cat(obs_tensors, dim=-1)
        elif self.task == "FactoryTaskNutBoltScrew":

            obs_tensors = [self.fingertip_midpoint_pos,
                           self.fingertip_midpoint_quat,
                           self.fingertip_midpoint_linvel,
                           self.fingertip_midpoint_angvel,
                           self.nut_com_pos,
                           self.nut_com_quat,
                           self.nut_com_linvel,
                           self.nut_com_angvel]

            if self.cfg_task.rl.add_obs_finger_force:
                obs_tensors += [self.left_finger_force, self.right_finger_force]

            obs_tensors = torch.cat(obs_tensors, dim=-1)
            self.obs_buf[:, :obs_tensors.shape[-1]] = obs_tensors  # shape = (num_envs, num_observations)

        else:
            assert False

        return self.obs_buf

    def get_taks_observations(self, subtask):
        # Shallow copies of tensors
        if subtask == 0:
            obs_tensors = [self.fingertip_midpoint_pos,
                           self.fingertip_midpoint_quat,
                           self.fingertip_midpoint_linvel,
                           self.fingertip_midpoint_angvel,
                           self.nut_grasp_pos,
                           self.nut_grasp_quat]

            obs = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)
        elif subtask == 1:
            obs_tensors = [self.fingertip_midpoint_pos,
                           self.fingertip_midpoint_quat,
                           self.fingertip_midpoint_linvel,
                           self.fingertip_midpoint_angvel,
                           self.nut_pos,
                           self.nut_quat,
                           self.bolt_pos,
                           self.bolt_quat]
            obs = torch.cat(obs_tensors, dim=-1)
        elif subtask ==2:

            obs_tensors = [self.fingertip_midpoint_pos,
                           self.fingertip_midpoint_quat,
                           self.fingertip_midpoint_linvel,
                           self.fingertip_midpoint_angvel,
                           self.nut_com_pos,
                           self.nut_com_quat,
                           self.nut_com_linvel,
                           self.nut_com_angvel]

            obs_tensors = torch.cat(obs_tensors, dim=-1)
            obs = torch.zeros((obs_tensors.shape[0],32))
            obs[:, :obs_tensors.shape[-1]] = obs_tensors  # shape = (num_envs, num_observations)

        else:
            assert False

        return obs
    def compute_reward(self):
        """Update reward and reset buffers."""
        if self.task == "FactoryTaskNutBoltScrew":
            curr_successes = self._get_curr_successes()
            curr_failures = self._get_curr_failures(curr_successes)
            self._update_reset_buf(curr_failures)
            self._update_rew_buf(curr_successes)
        else:
            self._update_reset_buf()
            self._update_rew_buf()

    def _update_reset_buf(self, curr_failures = None):
        """Assign environments for reset if successful or failed."""

        # If max episode length has been reached
        if self.task == "FactoryTaskNutBoltPick":
            self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.max_episode_length - 1,
                                            torch.ones_like(self.reset_buf),
                                            self.reset_buf)
        elif self.task == "FactoryTaskNutBoltPlace":
            self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1,
                                            torch.ones_like(self.reset_buf),
                                            self.reset_buf)
        elif self.task == "FactoryTaskNutBoltScrew":
            self.reset_buf[:] = curr_failures
        else :
            assert False

    def _update_rew_buf(self, curr_successes = None):
        """Compute reward at current timestep."""
        if self.task == "FactoryTaskNutBoltPick":
            keypoint_reward = -self._get_keypoint_dist()
            action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale

            self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                              - action_penalty * self.cfg_task.rl.action_penalty_scale

            # In this policy, episode length is constant across all envs
            is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

            if is_last_step:
                # Check if nut is picked up and above table
                lift_success = self._check_lift_success(height_multiple=3.0)
                self.rew_buf[:] += lift_success * self.cfg_task.rl.success_bonus
                self.extras['successes_'] = lift_success
                self.extras['successes'] = torch.mean(lift_success.float())
        elif self.task == "FactoryTaskNutBoltPlace":
            keypoint_reward = -self._get_keypoint_dist()
            action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale

            self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                              - action_penalty * self.cfg_task.rl.action_penalty_scale

            # In this policy, episode length is constant across all envs
            is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

            if is_last_step:
                # Check if nut is close enough to bolt
                is_nut_close_to_bolt = self._check_nut_close_to_bolt()
                self.rew_buf[:] += is_nut_close_to_bolt * self.cfg_task.rl.success_bonus
                self.extras['successes_'] = is_nut_close_to_bolt
                self.extras['successes'] = torch.mean(is_nut_close_to_bolt.float())
        elif self.task == "FactoryTaskNutBoltScrew":
            keypoint_reward = -(self.nut_keypoint_dist + self.finger_nut_keypoint_dist)
            action_penalty = torch.norm(self.actions, p=2, dim=-1)

            self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                              - action_penalty * self.cfg_task.rl.action_penalty_scale \
                              + curr_successes * self.cfg_task.rl.success_bonus

            is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
            successs = self._get_curr_successes()
            if is_last_step or torch.logical_or(successs, self._get_curr_failures(successs)).all():
                # Check if nut is close enough to bolt
                self.extras['successes_'] = successs
                self.extras['successes'] = torch.mean(successs.float())
        else:
             assert False
    def reset_idx(self, env_ids):
        """Reset specified environments."""

        if self.task == "FactoryTaskNutBoltPick":
            self._reset_franka(env_ids)
            self._reset_object(env_ids)
            self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps)
        """
        elif self.task == "FactoryTaskNutBoltPlace":
            # Close gripper onto nut
            self._reset_franka(env_ids)
            self._reset_object(env_ids)

            # Close gripper onto nut
            self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps)

            self._reset_buffers(env_ids)
        elif self.task == "FactoryTaskNutBoltScrew":
            self._reset_franka(env_ids)
            self._reset_object(env_ids)
            self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps)

            self._reset_buffers(env_ids)
        else :
            assert False
        """
        self._reset_buffers(env_ids)

    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""
        if self.task == "FactoryTaskNutBoltPick":
            self.dof_pos[env_ids] = torch.cat((torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos,
                                                            device=self.device).repeat((len(env_ids), 1)),
                                               (self.nut_widths_max[env_ids] * 0.5) * 1.1,
                                               # buffer on gripper DOF pos to prevent initial contact
                                               (self.nut_widths_max[env_ids] * 0.5) * 1.1),
                                              # buffer on gripper DOF pos to prevent initial contact
                                              dim=-1)
            """
            self.dof_pos[env_ids] = torch.cat(
                (torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device),
                 torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device),
                 torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device)),
                dim=-1).unsqueeze(0).repeat((self.num_envs, 1))  # shape = (num_envs, num_dofs)
            """
            self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
            self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

            multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                  len(multi_env_ids_int32))
        elif self.task == "FactoryTaskNutBoltPlace":
            multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                  len(multi_env_ids_int32))
        elif self.task == "FactoryTaskNutBoltScrew":
            multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                  len(multi_env_ids_int32))
        else:
            assert False


    def _reset_object(self, env_ids):
        """Reset root states of nut and bolt."""

        # shape of root_pos = (num_envs, num_actors, 3)
        # shape of root_quat = (num_envs, num_actors, 4)
        # shape of root_linvel = (num_envs, num_actors, 3)
        # shape of root_angvel = (num_envs, num_actors, 3)

        # Randomize root state of nut
        if self.task == "FactoryTaskNutBoltPick":
            nut_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
            nut_noise_xy = nut_noise_xy @ torch.diag(
                torch.tensor(self.cfg_task.randomize.nut_pos_xy_initial_noise, device=self.device))
            self.root_pos[env_ids, self.nut_actor_id_env, 0] = self.cfg_task.randomize.nut_pos_xy_initial[0] + nut_noise_xy[
                env_ids, 0]
            self.root_pos[env_ids, self.nut_actor_id_env, 1] = self.cfg_task.randomize.nut_pos_xy_initial[1] + nut_noise_xy[
                env_ids, 1]
            self.root_pos[
                env_ids, self.nut_actor_id_env, 2] = self.cfg_base.env.table_height - self.bolt_head_heights.squeeze(-1)
            self.root_quat[env_ids, self.nut_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32,
                                                                          device=self.device).repeat(len(env_ids), 1)

            self.root_linvel[env_ids, self.nut_actor_id_env] = 0.0
            self.root_angvel[env_ids, self.nut_actor_id_env] = 0.0

            # Randomize root state of bolt
            bolt_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
            bolt_noise_xy = bolt_noise_xy @ torch.diag(
                torch.tensor(self.cfg_task.randomize.bolt_pos_xy_noise, device=self.device))
            self.root_pos[env_ids, self.bolt_actor_id_env, 0] = self.cfg_task.randomize.bolt_pos_xy_initial[0] + \
                                                                bolt_noise_xy[env_ids, 0]
            self.root_pos[env_ids, self.bolt_actor_id_env, 1] = self.cfg_task.randomize.bolt_pos_xy_initial[1] + \
                                                                bolt_noise_xy[env_ids, 1]
            self.root_pos[env_ids, self.bolt_actor_id_env, 2] = self.cfg_base.env.table_height
            self.root_quat[env_ids, self.bolt_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32,
                                                                           device=self.device).repeat(len(env_ids), 1)

            self.root_linvel[env_ids, self.bolt_actor_id_env] = 0.0
            self.root_angvel[env_ids, self.bolt_actor_id_env] = 0.0

            nut_bolt_actor_ids_sim = torch.cat((self.nut_actor_ids_sim[env_ids],
                                                self.bolt_actor_ids_sim[env_ids]),
                                               dim=0)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state),
                                                         gymtorch.unwrap_tensor(nut_bolt_actor_ids_sim),
                                                         len(nut_bolt_actor_ids_sim))
        elif self.task == "FactoryTaskNutBoltPlace":

            """Reset root states of nut and bolt.

            # shape of root_pos = (num_envs, num_actors, 3)
            # shape of root_quat = (num_envs, num_actors, 4)
            # shape of root_linvel = (num_envs, num_actors, 3)
            # shape of root_angvel = (num_envs, num_actors, 3)

            # Randomize root state of nut within gripper
            self.root_pos[env_ids, self.nut_actor_id_env, 0] = 0.0
            self.root_pos[env_ids, self.nut_actor_id_env, 1] = 0.0
            fingertip_midpoint_pos_reset = 0.58781  # self.fingertip_midpoint_pos at reset
            nut_base_pos_local = self.bolt_head_heights.squeeze(-1)
            self.root_pos[env_ids, self.nut_actor_id_env, 2] = fingertip_midpoint_pos_reset - nut_base_pos_local

            nut_noise_pos_in_gripper = \
                2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
            nut_noise_pos_in_gripper = nut_noise_pos_in_gripper @ torch.diag(
                torch.tensor(self.cfg_task.randomize.nut_noise_pos_in_gripper, device=self.device))
            self.root_pos[env_ids, self.nut_actor_id_env, :] += nut_noise_pos_in_gripper[env_ids]

            nut_rot_euler = torch.tensor([0.0, 0.0, math.pi * 0.5], device=self.device).repeat(len(env_ids), 1)
            nut_noise_rot_in_gripper = \
                2 * (torch.rand(self.num_envs, dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
            nut_noise_rot_in_gripper *= self.cfg_task.randomize.nut_noise_rot_in_gripper
            nut_rot_euler[:, 2] += nut_noise_rot_in_gripper
            nut_rot_quat = torch_utils.quat_from_euler_xyz(nut_rot_euler[:, 0], nut_rot_euler[:, 1],
                                                           nut_rot_euler[:, 2])
            self.root_quat[env_ids, self.nut_actor_id_env] = nut_rot_quat

            # Randomize root state of bolt
            bolt_noise_xy = 2 * (
                        torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
            bolt_noise_xy = bolt_noise_xy @ torch.diag(
                torch.tensor(self.cfg_task.randomize.bolt_pos_xy_noise, dtype=torch.float32, device=self.device))
            self.root_pos[env_ids, self.bolt_actor_id_env, 0] = self.cfg_task.randomize.bolt_pos_xy_initial[0] + \
                                                                bolt_noise_xy[env_ids, 0]
            self.root_pos[env_ids, self.bolt_actor_id_env, 1] = self.cfg_task.randomize.bolt_pos_xy_initial[1] + \
                                                                bolt_noise_xy[env_ids, 1]
            self.root_pos[env_ids, self.bolt_actor_id_env, 2] = self.cfg_base.env.table_height
            self.root_quat[env_ids, self.bolt_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32,
                                                                           device=self.device).repeat(len(env_ids), 1)

            self.root_linvel[env_ids, self.bolt_actor_id_env] = 0.0
            self.root_angvel[env_ids, self.bolt_actor_id_env] = 0.0
            """
            nut_bolt_actor_ids_sim = torch.cat((self.nut_actor_ids_sim[env_ids],
                                                self.bolt_actor_ids_sim[env_ids]),
                                               dim=0)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state),
                                                         gymtorch.unwrap_tensor(nut_bolt_actor_ids_sim),
                                                         len(nut_bolt_actor_ids_sim))

        elif self.task == "FactoryTaskNutBoltScrew":

            """Reset root state of nut.

            # shape of root_pos = (num_envs, num_actors, 3)
            # shape of root_quat = (num_envs, num_actors, 4)
            # shape of root_linvel = (num_envs, num_actors, 3)
            # shape of root_angvel = (num_envs, num_actors, 3)

            nut_pos = self.cfg_base.env.table_height + self.bolt_shank_lengths[env_ids]
            self.root_pos[env_ids, self.nut_actor_id_env] = \
                nut_pos * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(len(env_ids), 1)

            nut_rot = self.cfg_task.randomize.nut_rot_initial * torch.ones((len(env_ids), 1),
                                                                           device=self.device) * math.pi / 180.0
            self.root_quat[env_ids, self.nut_actor_id_env] = torch.cat(
                (torch.zeros((len(env_ids), 1), device=self.device),
                 torch.zeros((len(env_ids), 1), device=self.device),
                 torch.sin(nut_rot * 0.5),
                 torch.cos(nut_rot * 0.5)),
                dim=-1)

            self.root_linvel[env_ids, self.nut_actor_id_env] = 0.0
            self.root_angvel[env_ids, self.nut_actor_id_env] = 0.0
            """
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state),
                                                         gymtorch.unwrap_tensor(self.nut_actor_ids_sim),
                                                         len(self.nut_actor_ids_sim))
        else :
            assert False
    def _reset_buffers(self, env_ids):
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """Apply actions from policy as position/rotation targets."""

        if self.task == "FactoryTaskNutBoltScrew":
            """Apply actions from policy as position/rotation targets or force/torque targets."""

            # Interpret actions as target pos displacements and set pos target
            pos_actions = actions[:, 0:3]
            if do_scale:
                pos_actions = pos_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
            self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
            self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

            # Interpret actions as target rot (axis-angle) displacements
            rot_actions = actions[:, 3:6]
            if self.cfg_task.rl.unidirectional_rot:
                rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
                # print(rot_actions[:, 2])
                rot_actions[:, 2] = -1
                #rot_actions[:, 2] = torch.where(rot_actions[:, 2] < -1, -1 * torch.ones_like(rot_actions[:, 2]),
                #                                rot_actions[:, 2])
                rot_actions[:, 2] = torch.where(rot_actions[:, 2] > -0.3, -1 * torch.ones_like(rot_actions[:, 2]),
                                                rot_actions[:, 2])

            if do_scale:
                rot_actions = rot_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

            # Convert to quat and set rot target
            angle = torch.norm(rot_actions, p=2, dim=-1)
            axis = rot_actions / angle.unsqueeze(-1)
            rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
            if self.cfg_task.rl.clamp_rot:
                rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                               rot_actions_quat,
                                               torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(
                                                   self.num_envs,
                                                   1))
            self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat,
                                                                            self.fingertip_midpoint_quat)

            if self.cfg_ctrl['do_force_ctrl']:
                # Interpret actions as target forces and target torques
                force_actions = actions[:, 6:9]
                if self.cfg_task.rl.unidirectional_force:
                    force_actions[:, 2] = -(force_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
                if do_scale:
                    force_actions = force_actions @ torch.diag(
                        torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

                torque_actions = actions[:, 9:12]
                if do_scale:
                    torque_actions = torque_actions @ torch.diag(
                        torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

                self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

            self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

            self.generate_ctrl_signals()
        else:
            # Interpret actions as target pos displacements and set pos target
            pos_actions = actions[:, 0:3]
            if do_scale:
                pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
            self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

            # Interpret actions as target rot (axis-angle) displacements
            rot_actions = actions[:, 3:6]
            if self.task == "FactoryTaskNutBoltScrew":
                if self.cfg_task.rl.unidirectional_rot:
                    rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
                    #rot_actions[:, 2] = -(rot_actions[:, 2]+2)*0.5
                    #rot_actions[:, 2] = torch.ones_like(rot_actions[:, 2])*-1
                    #print(rot_actions[:, 2])
            if do_scale:
                rot_actions = rot_actions @ torch.diag(
                        torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

            # Convert to quat and set rot target
            angle = torch.norm(rot_actions, p=2, dim=-1)
            axis = rot_actions / angle.unsqueeze(-1)
            rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
            if self.cfg_task.rl.clamp_rot:
                rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                               rot_actions_quat,
                                               torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,1))
            self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

            if self.cfg_ctrl['do_force_ctrl']:
                # Interpret actions as target forces and target torques
                force_actions = actions[:, 6:9]
                if self.task == "FactoryTaskNutBoltScrew":
                    if self.cfg_task.rl.unidirectional_force:
                        force_actions[:, 2] = -(force_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
                if do_scale:
                    force_actions = force_actions @ torch.diag(
                        torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

                torque_actions = actions[:, 9:12]
                if do_scale:
                    torque_actions = torque_actions @ torch.diag(
                        torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

                self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

            self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

            self.generate_ctrl_signals()

    def _open_gripper(self, sim_steps=20):
        """Fully open gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self._move_gripper_to_dof_pos(gripper_dof_pos=0.1, sim_steps=sim_steps)


    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

    def _get_keypoint_dist(self, body = None):
        """Get keypoint distance."""
        if self.task == "FactoryTaskNutBoltPick":
            keypoint_dist = torch.sum(torch.norm(self.keypoints_nut - self.keypoints_gripper, p=2, dim=-1), dim=-1)
        elif self.task == "FactoryTaskNutBoltPlace":
            keypoint_dist = torch.sum(torch.norm(self.keypoints_bolt - self.keypoints_nut, p=2, dim=-1), dim=-1)
        elif self.task == "FactoryTaskNutBoltScrew":

            """Get keypoint distances."""

            axis_length = self.asset_info_franka_table.franka_hand_length + self.asset_info_franka_table.franka_finger_length

            if body == 'finger' or body == 'nut':
                # Keypoint distance between finger/nut and target
                if body == 'finger':
                    self.keypoint1 = self.fingertip_midpoint_pos
                    self.keypoint2 = fc.translate_along_local_z(pos=self.keypoint1,
                                                                quat=self.fingertip_midpoint_quat,
                                                                offset=-axis_length,
                                                                device=self.device)

                elif body == 'nut':
                    self.keypoint1 = self.nut_com_pos
                    self.keypoint2 = fc.translate_along_local_z(pos=self.nut_com_pos,
                                                                quat=self.nut_com_quat,
                                                                offset=axis_length,
                                                                device=self.device)

                self.keypoint1_targ = self.target_pos
                self.keypoint2_targ = self.keypoint1_targ + torch.tensor([0.0, 0.0, axis_length], device=self.device)

            elif body == 'finger_nut':
                # Keypoint distance between finger and nut
                self.keypoint1 = self.fingerpad_midpoint_pos
                self.keypoint2 = fc.translate_along_local_z(pos=self.keypoint1,
                                                            quat=self.fingertip_midpoint_quat,
                                                            offset=-axis_length,
                                                            device=self.device)

                self.keypoint1_targ = self.nut_com_pos
                self.keypoint2_targ = fc.translate_along_local_z(pos=self.nut_com_pos,
                                                                 quat=self.nut_com_quat,
                                                                 offset=axis_length,
                                                                 device=self.device)

            self.keypoint3 = self.keypoint1 + (self.keypoint2 - self.keypoint1) * 1.0 / 3.0
            self.keypoint4 = self.keypoint1 + (self.keypoint2 - self.keypoint1) * 2.0 / 3.0
            self.keypoint3_targ = self.keypoint1_targ + (self.keypoint2_targ - self.keypoint1_targ) * 1.0 / 3.0
            self.keypoint4_targ = self.keypoint1_targ + (self.keypoint2_targ - self.keypoint1_targ) * 2.0 / 3.0
            keypoint_dist = torch.norm(self.keypoint1_targ - self.keypoint1, p=2, dim=-1) \
                            + torch.norm(self.keypoint2_targ - self.keypoint2, p=2, dim=-1) \
                            + torch.norm(self.keypoint3_targ - self.keypoint3, p=2, dim=-1) \
                            + torch.norm(self.keypoint4_targ - self.keypoint4, p=2, dim=-1)
        else:
            assert False
        return keypoint_dist
    def _check_nut_close_to_bolt(self):
        """Check if nut is close to bolt."""

        keypoint_dist = torch.norm(self.keypoints_bolt - self.keypoints_nut, p=2, dim=-1)

        is_nut_close_to_bolt = torch.where(torch.sum(keypoint_dist, dim=-1) < self.cfg_task.rl.close_error_thresh,
                                           torch.ones_like(self.progress_buf),
                                           torch.zeros_like(self.progress_buf))

        return is_nut_close_to_bolt
    def _get_curr_successes(self):
        """Get success mask at current timestep."""

        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        thresh = torch.ones_like(self.nut_dist_to_target,dtype=torch.float)*0.02
        # If nut is close enough to target pos
        is_close = torch.where(self.nut_dist_to_target < thresh,
                               torch.ones_like(curr_successes),
                               torch.zeros_like(curr_successes))
        #is_close = torch.where(self.nut_keypoint_dist < self.thread_pitches.squeeze(-1) * 8,
        #                       torch.ones_like(curr_successes),
        #                       torch.zeros_like(curr_successes))

        curr_successes = torch.logical_or(curr_successes, is_close)

        return curr_successes

    def _get_curr_failures(self, curr_successes):
        """Get failure mask at current timestep."""

        curr_failures = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # If nut is too far from target pos
        self.is_far = torch.where(self.nut_dist_to_target > self.cfg_task.rl.far_error_thresh,
                                  torch.ones_like(curr_failures),
                                  curr_failures)

        self.is_expired = torch.where(self.progress_buf[:] >= self.cfg_task.rl.max_episode_length,
                                      torch.ones_like(curr_failures),
                                      curr_failures)
        # If nut has slipped (distance-based definition)
        self.is_slipped = \
            torch.where(
                self.nut_dist_to_fingerpads > self.asset_info_franka_table.franka_fingerpad_length + self.nut_heights.squeeze(-1) ,
                torch.ones_like(curr_failures),
                curr_failures)
        self.is_slipped = torch.logical_and(self.is_slipped, torch.logical_not(curr_successes))  # ignore slip if successful

        # If nut has fallen (i.e., if nut XY pos has drifted from center of bolt and nut Z pos has drifted below top of bolt)
        self.is_fallen = torch.logical_and(
            torch.norm(self.nut_com_pos[:, 0:2], p=2, dim=-1) > self.bolt_widths.squeeze(-1),
            self.nut_com_pos[:, 2] < self.cfg_base.env.table_height + self.bolt_head_heights.squeeze(
                -1) + self.bolt_shank_lengths.squeeze(-1) + self.nut_heights.squeeze(-1) * 0.5)

        curr_failures = torch.logical_or(curr_failures, self.is_far)
        curr_failures = torch.logical_or(curr_failures, self.is_expired)
        #curr_failures = torch.logical_or(curr_failures, self.is_slipped)
        curr_failures = torch.logical_or(curr_failures, self.is_fallen)

        return curr_failures
    def _close_gripper(self, sim_steps=20):
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self._move_gripper_to_dof_pos(gripper_dof_pos=0.0, sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""

        delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
                                      device=self.device)  # No hand motion
        self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)

        # Step sim
        for _ in range(sim_steps):
            self.render()
            self.gym.simulate(self.sim)

    def _lift_gripper(self, franka_gripper_width=0.0, lift_distance=0.3, sim_steps=20):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        delta_hand_pose = torch.zeros([self.num_envs, 6], device=self.device)
        delta_hand_pose[:, 2] = lift_distance

        # Step sim

        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(delta_hand_pose, franka_gripper_width, do_scale=False)
            self.render()
            self.gym.simulate(self.sim)

    def _check_lift_success(self, height_multiple):
        """Check if nut is above table by more than specified multiple times height of nut."""

        lift_success = torch.where(
            self.nut_pos[:, 2] > self.cfg_base.env.table_height + self.nut_heights.squeeze(-1) * height_multiple,
            torch.ones((self.num_envs,), device=self.device),
            torch.zeros((self.num_envs,), device=self.device))

        return lift_success


    def _randomize_gripper_pose(self, env_ids, sim_steps):
        """Move gripper to random pose."""

        # Set target pos above table
        if self.task == "FactoryTaskNutBoltPick":
            self.ctrl_target_fingertip_midpoint_pos = \
                torch.tensor([0.0, 0.0, self.cfg_base.env.table_height], device=self.device) \
                + torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device)
            self.ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(self.num_envs, 1)

            fingertip_midpoint_pos_noise = \
                2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
            fingertip_midpoint_pos_noise = \
                fingertip_midpoint_pos_noise @ torch.diag(torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_noise,
                                                                       device=self.device))
            self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

            # Set target rot
            ctrl_target_fingertip_midpoint_euler = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                                                                device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

            fingertip_midpoint_rot_noise = \
                2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
            fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
                torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device))
            ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
            self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
                ctrl_target_fingertip_midpoint_euler[:, 0],
                ctrl_target_fingertip_midpoint_euler[:, 1],
                ctrl_target_fingertip_midpoint_euler[:, 2])

        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[:, :6] = delta_hand_pose

            if self.task == "FactoryTaskNutBoltPick":
                self._apply_actions_as_ctrl_targets(actions=actions,
                                                    ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                                                    do_scale=False)
            else:
                self._apply_actions_as_ctrl_targets(actions=actions,
                                                    ctrl_target_gripper_dof_pos=0.0,
                                                    do_scale=False)
            if self.task == "FactoryTaskNutBoltPick":
                self.gym.simulate(self.sim)
                self.render()
        if self.task == "FactoryTaskNutBoltPick":
            self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        # Set DOF state
        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

    def check_success(self):
        if self.task == "FactoryTaskNutBoltPick":
                # Check if nut is picked up and above table
            lift_success = self._check_lift_success(height_multiple=3.0)
            return (lift_success)
        elif self.task == "FactoryTaskNutBoltPlace":
                # Check if nut is close enough to bolt
            is_nut_close_to_bolt = self._check_nut_close_to_bolt()
            return (is_nut_close_to_bolt)
        elif self.task == "FactoryTaskNutBoltScrew":
            curr_successes = self._get_curr_successes()
            # Check if nut is close enough to bolt
            return (curr_successes)
        else :
            assert  False
