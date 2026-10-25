import isaacgym
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float, to_torch, get_axis_params, quat_apply
from getup_gym.common.math_utils import torch_rand_float

from getup_gym.envs.legged_robot import LeggedRobot
from getup_gym.envs.Tita_wheel.config import TitaWheelGetupCfg, TitaWheelFFCfg
from getup_gym.common.rewards_init import reward_register
from getup_gym.utils.utils import random_quat


class TitaWheel(LeggedRobot):
    """Tita: 8-DOF bipedal wheeled robot for fall recovery."""

    def __init__(self, cfg: TitaWheelGetupCfg, sim_params, physics_engine, sim_device, headless):
        # Ensure config compatibility with LeggedRobot which expects observation_stack
        if not hasattr(cfg.env, "observation_stack"):
            class ObservationStack:
                add_time_number = getattr(getattr(cfg.env, "myself_setting", None), "add_time_number", 5)
                max_change_angel = getattr(getattr(cfg.env, "myself_setting", None), "max_change_angel", 0.1)
                pitch_theta_setting = getattr(getattr(cfg.env, "myself_setting", None), "pitch_theta_setting", 0.0)
            cfg.env.observation_stack = ObservationStack()
        if not hasattr(cfg, "observation_stack"):
            cfg.observation_stack = cfg.env.observation_stack

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.unactuated_time = self.cfg.env.unactuated_timesteps * 0.02 / self.dt

    def compute_observations_without_noise(self):
        """Computes observations"""
        left_leg_pos = self.dof_pos[:, 0:3]
        right_leg_pos = self.dof_pos[:, 4:7]

        commands = self.commands[:, :3] * self.commands_scale[:3]
        if self.cfg.commands.heading_command:
            commands = self.commands[:, :5] * self.commands_scale[:5]
        if self.cfg.commands.base_height_command:
            base_height_dim = self.cfg.commands.num_commands - 1
            commands = self.commands[:, :base_height_dim] * self.commands_scale[:base_height_dim]
        # print(f"lin_x: {self.commands[0,0]:.2g}, ang_z: {self.commands[0,2]:.2g}")

        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)

        self.obs_buf = torch.cat(
            (
                # self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_lin_vel * self.obs_scales.lin_vel * 0,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                commands,
                (left_leg_pos - self.default_dof_pos[:, 0:3]) * self.obs_scales.dof_pos,
                (right_leg_pos - self.default_dof_pos[:, 4:7]) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

    def compute_observations(self):
        """add noise to observation"""
        self.compute_observations_without_noise()
        self.compute_privileged_observations()
        self.compute_teacher_encoder_observations()
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        if (
            (self.common_step_counter - 1) / 24 - 1 > self.cfg.rewards.delays.delay_start_epochs
        ) & self.cfg.rewards.delays.using_delay_action:
            self.obs_buf *= self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
        self.last_observations_save()
        self.compute_student_encoder_observations()
        return self.obs_buf

    def compute_privileged_observations(self):
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        ).unsqueeze(1)
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1.0,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            self.privileged_obs_buf = torch.cat(
                (
                    self.obs_buf,
                    heights,
                    base_height,
                ),
                dim=1,
            )
            self.privileged_obs_buf[:, 0:3] = self.base_lin_vel * self.obs_scales.lin_vel
            if (
                (self.common_step_counter - 1) / 24 - 1 > self.cfg.rewards.delays.delay_start_epochs
            ) & self.cfg.rewards.delays.using_delay_action:
                self.privileged_obs_buf *= (
                    self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
                )

    def compute_teacher_encoder_observations(self):
        "computer the teacher encoder's observations "
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1.0,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )

        ContactForce_lwheel = self.contact_forces[:, 4, :3].squeeze(1)
        ContactForce_rwheel = self.contact_forces[:, 8, :3].squeeze(1)

        Torque = self.torques
        JointAccelerate = (self.last_dof_vel - self.dof_vel) / self.dt

        states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        body_states = gymtorch.wrap_tensor(states)
        body_states_reshape = body_states.reshape(self.num_envs, 9 * 13)

        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        ).unsqueeze(1)

        teacher_encoder_observations = torch.cat(
            [
                ContactForce_lwheel,
                ContactForce_rwheel,
                base_height * 0.4,
                self.root_states,
            ],
            dim=-1,
        )
        self.teacher_encoder_obs_buf = teacher_encoder_observations
        if (
            (self.common_step_counter - 1) / 24 - 1 > self.cfg.rewards.delays.delay_start_epochs
        ) & self.cfg.rewards.delays.using_delay_action:
            self.teacher_encoder_obs_buf *= (
                self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
            )

    def compute_student_encoder_observations(self):
        "compute the student encoder's observations"
        self.lotstime_obs = self.observations_stack[0]
        for i in range(self.cfg.env.myself_setting.add_time_number - 1):
            self.lotstime_obs = torch.cat(
                (
                    self.lotstime_obs,
                    self.observations_stack[i + 1],
                ),
                dim=1,
            )
        student_encoder_observations = self.lotstime_obs
        self.student_encoder_obs_buf = student_encoder_observations
        if (
            (self.common_step_counter - 1) / 24 - 1 > self.cfg.rewards.delays.delay_start_epochs
        ) & self.cfg.rewards.delays.using_delay_action:
            self.student_encoder_obs_buf *= (
                self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
            )

    def _reward_update(self):
        # 第一段：整体只有轮子受力
        reward_1 = torch.norm(self.contact_forces[:, 0, :], dim=-1) > 10
        # 第二段：站立到0.27m
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
        stand_idx = base_height < 0.27
        # 第三段：0.3m到目标高度
        walk_idx = base_height >= 0.3
        walk_idx_num = torch.sum(1.0 * walk_idx)
        # 第四段：开始行走
        moving_idx = self.force_scale < 80.0

        # reward update
        self.reward_group = self.reward_group_list["reward_group_2"]
        if torch.sum(1.0 * stand_idx) > self.num_envs * 2 / 3:
            self.reward_group = self.reward_group_list["reward_group_2"]
            # print("第二阶段")
        elif torch.sum(1.0 * walk_idx) > self.num_envs * 2 / 3:
            self.reward_group = self.reward_group_list["reward_group_3"]
            # print("由二转三")
            if torch.sum(1.0 * moving_idx) > self.num_envs * 2 / 3:
                self.reward_group = self.reward_group_list["reward_group_4"]
                # print("进入第四阶段")
        # self.reward_group = self.reward_group_list["reward_group_4"]
            

    def _reward_register(self):
        """reward register function"""
        reward_register(self.RewardManager)

    def _update_base_height_target(self):
        target_height_list = self.cfg.rewards.curriculum.target_height
        base_height = torch.mean(
            torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1), dim=0
        )
        self.target_height = target_height_list[0] * torch.ones(self.num_envs, device=self.device)
        idx = (base_height >= target_height_list[0] * 0.8) & (base_height < target_height_list[1])
        self.target_height[idx] = target_height_list[1]
        idx = (base_height >= target_height_list[1] * 0.8) & (base_height < target_height_list[2])
        self.target_height[idx] = target_height_list[2]
        idx = (base_height >= target_height_list[2] * 0.8) & (base_height < target_height_list[3])
        self.target_height[idx] = target_height_list[3]
        idx = (base_height >= target_height_list[3] * 0.8)
        self.target_height[idx] = self.cfg.rewards.base_height_target

    def _pre_reward_callback(self):
        self.rigid_body_parameter_calculation()

    def rigid_body_parameter_calculation(self):
        """calaculate some robot's rigid body parameters"""
        states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        body_states = gymtorch.wrap_tensor(states)
        self.body_states_reshape = body_states.reshape(self.num_envs, 9, 13)
        self.feet_distance_z = torch.abs(
            self.body_states_reshape[:, 4, 2] - self.body_states_reshape[:, 8, 2]
        ).squeeze(-1)
        self.feet_distance_x = torch.abs(
            self.body_states_reshape[:, 4, 0] - self.body_states_reshape[:, 8, 0]
        ).squeeze(-1)

        self.feet_distance_y = torch.abs(
            self.body_states_reshape[:, 4, 1] - self.body_states_reshape[:, 8, 1]
        ).squeeze(-1)
        self.feet_distance_z_l2r = (
            self.body_states_reshape[:, 4, 2] - self.body_states_reshape[:, 8, 2]
        ).squeeze(-1)
        self.feet_distance_z_r2l = (
            self.body_states_reshape[:, 8, 2] - self.body_states_reshape[:, 4, 2]
        ).squeeze(-1)

        self.lfeet_vel_y = torch.abs(self.body_states_reshape[:, 4, 8]).squeeze(-1)
        self.rfeet_vel_y = torch.abs(self.body_states_reshape[:, 8, 8]).squeeze(-1)

        self.contacts_x = torch.abs(self.contact_forces[:, self.feet_indices, 0]) > 1.0
        self.lcontacts_x = (
            torch.abs(self.contact_forces[:, self.feet_indices[0], 0]) > 1.0
        )
        self.rcontacts_x = (
            torch.abs(self.contact_forces[:, self.feet_indices[1], 0]) > 1.0
        )
        self.contacts_z = torch.abs(self.contact_forces[:, self.feet_indices, 2]) > 1.0
        self.single_contact_x = torch.sum(1.0 * self.contacts_x, dim=1) == 1
        self.two_contact_x = torch.sum(1.0 * self.contacts_x, dim=1) == 2

        self.single_contact_z = torch.sum(1.0 * self.contacts_z, dim=1) == 1
        self.two_contact_z = torch.sum(1.0 * self.contacts_z, dim=1) == 2

        self.lfeet_fem_distance_x = (
            self.body_states_reshape[:, 2, 0] - self.body_states_reshape[:, 4, 0]
        )
        self.rfeet_fem_distance_x = (
            self.body_states_reshape[:, 6, 0] - self.body_states_reshape[:, 8, 0]
        )

    def _compute_torques(self, actions):
        """Compute torques from actions."""
        self.action_scale = self.cfg.control.action_scale
        if self.cfg.control.using_actions_scale_changing & (
            (self.common_step_counter - 1) / 24
            == self.cfg.control.action_scale_changing_start_epochs
        ):
            self._action_scale_update()

        actions_scaled = actions * torch.tensor(self.action_scale, device=self.device)
        control_type = self.cfg.control.control_type
        torques = torch.zeros_like(actions_scaled)
        joint_control_name = list(control_type.keys())

        self.target_dof_pos = actions_scaled + self.default_dof_pos
        for j in range(self.num_dofs):
            joint_control_j = (
                j if j < self.num_dofs / 2 else j - int(self.num_dofs / 2)
            )
            if control_type[joint_control_name[joint_control_j]] == "P":
                torques[:, j] = (
                    self.p_gains[j]
                    * (
                        actions_scaled[:, j]
                        + self.default_dof_pos[:, j]
                        - self.dof_pos[:, j]
                    )
                    - self.d_gains[j] * self.dof_vel[:, j]
                )
            elif control_type[joint_control_name[joint_control_j]] == "V":
                torques[:, j] = (
                    self.p_gains[j] * (actions_scaled[:, j] - self.dof_vel[:, j])
                    - self.d_gains[j]
                    * (self.dof_vel[:, j] - self.last_dof_vel[:, j])
                    / self.sim_params.dt
                )
            elif control_type[joint_control_name[joint_control_j]] == "T":
                torques = actions_scaled
            else:
                raise NameError(
                    f"Unknown controller type: {control_type[joint_control_name[joint_control_j]]}"
                )
        if self.cfg.domain_rand.randomize_motor_strength:
            self.torques[:, self.motor_rigid_idx] = (
                self.motor_strength[:, :] * self.torques[:, self.motor_rigid_idx]
            )
        

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _action_scale_update(self):
        """update the action scale value"""
        update_end_epoch = self.cfg.control.action_scale_changing_end_epochs
        update_using_epoch = (
            update_end_epoch - self.cfg.control.action_scale_changing_start_epochs
        )
        action_scale_final = self.cfg.control.action_scale_final
        time_coeff = (
            update_end_epoch - (self.common_step_counter - 1) / 24
        ) / update_using_epoch
        for i in range(len(self.action_scale)):
            self.action_scale[i] = action_scale_final[i] + (
                self.action_scale[i] - action_scale_final[i]
            ) * time_coeff

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environments"""
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            0.1, 1.5, (len(env_ids), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = 0.0

        if (self.common_step_counter - 1) / 24 - 1 > 3000:
            lie_idx = env_ids > 2000
            self.dof_pos[env_ids[lie_idx], 0] = 0
            self.dof_pos[env_ids[lie_idx], 4] = 0
            self.dof_pos[env_ids[lie_idx], 1] = 1.0
            self.dof_pos[env_ids[lie_idx], 5] = 1.0
            self.dof_pos[env_ids[lie_idx], 2] = 0.0
            self.dof_pos[env_ids[lie_idx], 6] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environments"""
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )

        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 3:7] = random_quat(torch_rand_float(0, 1, (len(env_ids), 4), device=self.device))

        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )
        back_idx = torch.where(env_ids < 2000)[0]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        back_idx = back_idx.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _get_heights_below_foot(self):
        """ Samples heights of the terrain at required points around each foot.

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = self.feet_state[:, :, :2]

        heights = self._get_terrain_heights_from_points(points)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_terrain_heights_from_points(self, points):
        points = points + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        return heights

# ============================================================================
# Force-guided fall-recovery framework (FTSR x FR-Net) for Tita_wheel.
# ============================================================================
from isaacgym import gymapi

from getup_gym.common.reward_manager import RewardManagerCfg
from getup_gym.common import reward_functions as rf


class TitaWheelFF(TitaWheel):
    """Tita wheeled bipedal fall recovery with:
    - an encoder estimating per-link contact forces from an H-frame observation
      history (FR-Net style), no teacher-student;
    - an external guidance force driven by relu(g_proj + 0.9), whose margin is
      mixed into the advantage as an optimizable constraint (FTSR);
    - staged rewards flagged per-env by
      (g_proj < thr) AND (estimated base-link contact force < thr):
      stage 1 = base flip recovery, stage 2 = stable standing."""

    def __init__(self, cfg: TitaWheelFFCfg, sim_params, physics_engine, sim_device, headless):
        # Ensure config compatibility with LeggedRobot which expects observation_stack
        # (mirrors TitaWheel.__init__, done here so this class owns the FF config typing)
        if not hasattr(cfg.env, "observation_stack"):
            class ObservationStack:
                add_time_number = getattr(getattr(cfg.env, "myself_setting", None), "add_time_number", 5)
                max_change_angel = getattr(getattr(cfg.env, "myself_setting", None), "max_change_angel", 0.1)
                pitch_theta_setting = getattr(getattr(cfg.env, "myself_setting", None), "pitch_theta_setting", 0.0)
            cfg.env.observation_stack = ObservationStack()
        if not hasattr(cfg, "observation_stack"):
            cfg.observation_stack = cfg.env.observation_stack

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.num_contact_links = getattr(cfg.env, "num_contact_links", self.num_bodies)
        self._est_contact_forces = None
        self.force_margin = torch.zeros(self.num_envs, device=self.device)
        self.stage_flip_mask = torch.ones(self.num_envs, device=self.device)
        self.stage_stand_mask = torch.zeros(self.num_envs, device=self.device)
        self.normalized_link_contact_forces = torch.zeros(
            self.num_envs, self.num_contact_links, device=self.device
        )
        self.encoder_obs_buf = None

    def step(self, actions, est_contact_forces=None):
        """Run one policy step.

        Args:
            actions: policy actions.
            est_contact_forces: encoder's per-link contact-force estimate
                (normalized, (num_envs, num_contact_links)); used for the
                staged-reward flag. None defaults to the flip stage.
        """
        self._est_contact_forces = est_contact_forces
        obs, privileged_obs, _, _, rewards, dones, extras = super().step(actions)
        extras["force"] = self.force_margin.unsqueeze(-1)
        extras["contact_force_targets"] = self.normalized_link_contact_forces
        return (
            obs,
            privileged_obs,
            self.encoder_obs_buf,
            self.normalized_link_contact_forces,
            rewards,
            dones,
            extras,
        )

    def compute_observations(self):
        """add noise to observation, then build the H-frame history stack used
        as the contact-force encoder input"""
        self.compute_observations_without_noise()
        self.compute_privileged_observations()
        self.compute_teacher_encoder_observations()
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        if (
            (self.common_step_counter - 1) / 24 - 1 > self.cfg.rewards.delays.delay_start_epochs
        ) & self.cfg.rewards.delays.using_delay_action:
            self.obs_buf *= self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
        self.last_observations_save()
        self.compute_student_encoder_observations()
        # the H-frame history stack is exactly the encoder's input
        self.encoder_obs_buf = self.student_encoder_obs_buf
        return self.obs_buf

    def compute_privileged_observations(self):
        """Critic sees proprio obs + terrain heights + base height + true normalized
        per-link contact forces (asymmetric actor-critic, FR-Net style)."""
        super().compute_privileged_observations()
        self.privileged_obs_buf = torch.cat(
            (self.privileged_obs_buf, self.normalized_link_contact_forces), dim=-1
        )

    def _pre_reward_callback(self):
        self.rigid_body_parameter_calculation()

        # ---- force-guided + staged-reward bookkeeping (FTSR x FR-Net) ----
        g_proj = self.projected_gravity[:, 2]
        offset = self.cfg.rewards.force_guidance.proj_gravity_offset
        # margin = relu(g_proj + offset): zero once the robot is (nearly) upright
        self.force_margin = torch.clamp(g_proj + offset, min=0.0)

        weight = self.total_mass * 9.81
        self.normalized_link_contact_forces = torch.norm(
            self.contact_forces.view(self.num_envs, self.num_bodies, 3), dim=-1
        )[:, : self.num_contact_links] / weight

        thr = self.cfg.rewards.stage.proj_gravity_threshold
        est = self._est_contact_forces
        if est is not None:
            # stand stage: upright AND base no longer in large-area ground contact
            self.stage_stand_mask = (
                (g_proj < thr)
                & (est[:, 0] < self.cfg.rewards.stage.base_contact_force_threshold)
            ).float()
        else:
            self.stage_stand_mask = torch.zeros_like(g_proj)
        self.stage_flip_mask = 1.0 - self.stage_stand_mask

    def _apply_force_guidance_hook(self):
        """FTSR force-guided assistance: upward base force proportional to
        relu(g_proj + offset), phased out over training iterations."""
        fg_cfg = self.cfg.rewards.force_guidance
        if not fg_cfg.using_force_guidance:
            return
        epoch = (self.common_step_counter - 1) / 24 - 1
        if epoch < 1 or epoch >= fg_cfg.force_end_epochs:
            return
        time_coeff = torch.tensor(
            (fg_cfg.force_end_epochs - epoch) / fg_cfg.force_end_epochs,
            device=self.device,
        )
        time_coeff = torch.clamp(time_coeff, 0.0, 1.0)
        # cap the assist below the robot weight so it can never lift the robot
        # off the ground (the margin has no height feedback, unlike the old pull-up)
        force_cap = min(fg_cfg.force_max, fg_cfg.force_weight_ratio * self.total_mass * 9.81)
        force = force_cap * (1.0 - torch.exp(-fg_cfg.mu * self.force_margin)) * time_coeff

        forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        body_idx = int(getattr(self, "base_indice", 0))
        forces[:, body_idx, 2] = force
        torques = torch.zeros_like(forces)
        if hasattr(self, "real_episode_length_buf") and hasattr(self, "unactuated_time"):
            mask = (
                self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
            ).unsqueeze(1)
            forces = forces * mask
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(forces),
            gymtorch.unwrap_tensor(torques),
            gymapi.ENV_SPACE,
        )

    def _reward_update(self):
        pass  # single reward group; staging is per-env inside the reward functions

    def _reward_register(self):
        """register only the reward terms used by the FF framework"""
        funcs = [
            rf.pen_action_rate_l2,
            rf.pen_base_orientation_l2,
            rf.pen_base_orientation_z_l2,
            rf.pen_dof_acc_l2,
            rf.pen_dof_pos_bias_l2,
            rf.pen_termination,
            rf.pen_torques_l2,
            rf.pen_two_leg_bias_l2,
            rf.track_base_height_exp,
            rf.rew_wheel_contact_force,
            rf.rew_proj_gravity_gaussian,
            rf.rew_stand_posture_exp,
            rf.pen_base_contact_force_l2,
            rf.rew_wheel_support_exp,
        ]
        for f in funcs:
            self.RewardManager.register_reward(RewardManagerCfg(func=f, weight=0.0, params={}))
