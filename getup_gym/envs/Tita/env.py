import isaacgym
import torch
import os
import csv
from isaacgym import gymtorch, gymapi, gymutil, gymutil
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float, to_torch, get_axis_params, quat_apply
from getup_gym.common.math_utils import torch_rand_float

from getup_gym.envs.legged_robot import LeggedRobot
from getup_gym.envs.Tita.config import TitaGetupCfg
from getup_gym.common.rewards_init import reward_register
from getup_gym.utils.utils import random_quat


class Tita(LeggedRobot):
    """Tita: 8-DOF bipedal wheeled robot for fall recovery."""

    def __init__(self, cfg: TitaGetupCfg, sim_params, physics_engine, sim_device, headless):
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

        # 初始化接触载荷和辅助目标刚体索引
        self._init_force_target_indices()

        self._init_recovery_state()
        self._init_deadpoint_state()
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

        # 1. 机身坐标系线速度
        base_lin_vel = self.base_lin_vel
        # =====================================================
        # 2. 新增五维归一化接触载荷
        # =====================================================
        all_contact_loads = self._get_five_contact_loads() 
        contact_loads = all_contact_loads[:, 0:3]
        # [base, left_calf, right_calf, left_wheel, right_wheel]
        # 3. 五维接触状态
        contact_states = self._get_five_contact_states()
        # 4. 左右轮滑移速度
        slip_velocity = self._get_wheel_slip_velocity()

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
                base_lin_vel,
                ContactForce_lwheel,
                ContactForce_rwheel,
                base_height * 0.4,
                contact_states,
                contact_loads,
                slip_velocity,
            ],
            dim=-1,
        )
        assert teacher_encoder_observations.shape[-1] == 20, (
            f"Teacher observation dimension error: "
            f"{teacher_encoder_observations.shape[-1]}"
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
        self._update_recovery_stage()
        self._update_deadpoint_detector()
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
                    self.p_gains[:, j]
                    * (
                        actions_scaled[:, j]
                        + self.default_dof_pos[:, j]
                        - self.dof_pos[:, j]
                    )
                    - self.d_gains[:, j] * self.dof_vel[:, j]
                )
            elif control_type[joint_control_name[joint_control_j]] == "V":
                torques[:, j] = (
                    self.p_gains[:, j] * (actions_scaled[:, j] - self.dof_vel[:, j])
                    - self.d_gains[:, j]
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
        
        # print("不限幅力矩:", torques[:, [2, 3]])
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
            -0.5, 0.5, (len(env_ids), 6), device=self.device)
        back_idx = torch.where(env_ids < 2000)[0]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        back_idx = back_idx.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_pd_params(self, env_ids):

        kp_scale = torch.rand(len(env_ids), self.num_actions, device=self.device) * (
            self.cfg.domain_rand.kp_range[1] - self.cfg.domain_rand.kp_range[0]
        ) + self.cfg.domain_rand.kp_range[0]
        kd_scale = torch.rand(len(env_ids), self.num_actions, device=self.device) * (
            self.cfg.domain_rand.kd_range[1] - self.cfg.domain_rand.kd_range[0]
        ) + self.cfg.domain_rand.kd_range[0]

        self.p_gains[env_ids] = self.kp_origin * kp_scale
        self.d_gains[env_ids] = self.kd_origin * kd_scale

    def reset_idx(self, env_ids):
        """B3:重置环境:并以一定概率重采样历史死点状态。"""

        if env_ids.numel() == 0:
            return

        cfg = self.cfg.rewards.recovery
        # --------------------------------------------------
        # A. 先记录刚结束的 episode 是否出现过死点 / 是否恢复站立成功
        # --------------------------------------------------
        if hasattr(self, "deadpoint_ever"):
            deadpoint_done = (self.deadpoint_ever[env_ids].detach().clone())
        else:
            deadpoint_done = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)

        if hasattr(self, "recovery_succeeded"):
            stand_success_done = (self.recovery_succeeded[env_ids].detach().clone())
        else:
            stand_success_done = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)

        # --------------------------------------------------
        # B. 决定哪些重置环境使用困难状态回放
        # --------------------------------------------------
        replay_ids = env_ids[:0]
        replay_enabled = (
            getattr(cfg, "deadpoint_replay_enabled", False)
            and hasattr(self, "deadpoint_replay_size")
            and self.deadpoint_replay_size
            >= cfg.deadpoint_replay_min_size
        )

        if replay_enabled and cfg.deadpoint_replay_probability > 0:
            use_replay = (torch.rand(env_ids.numel(), device=self.device) < cfg.deadpoint_replay_probability)
            replay_ids = env_ids[use_replay]

        # --------------------------------------------------
        # C. 执行原框架的重置
        #    包含 DOF、root state、命令和历史缓存等
        # --------------------------------------------------
        super().reset_idx(env_ids)
        if self.cfg.domain_rand.randomize_pd:
            self._reset_pd_params(env_ids)
        # --------------------------------------------------
        # D. 覆盖部分环境的初始状态
        # --------------------------------------------------
        if replay_ids.numel() > 0:
            slots = torch.randint(
                low=0, high=self.deadpoint_replay_size, size=(replay_ids.numel(),), device=self.device)

            # 恢复相对环境原点保存的基座状态
            self.root_states[replay_ids] = (self.deadpoint_replay_root[slots])
            self.root_states[replay_ids, :3] += (self.env_origins[replay_ids])

            # 恢复关节位置和速度
            dof_state_view = self.dof_state.view(self.num_envs, self.num_dof, 2)
            dof_state_view[replay_ids] = (self.deadpoint_replay_dof[slots])
            replay_ids_int32 = replay_ids.to(torch.int32)

            # 写回 Isaac Gym
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(replay_ids_int32),
                replay_ids_int32.numel(),
            )

            self.gym.set_dof_state_tensor_indexed(self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(replay_ids_int32),
                replay_ids_int32.numel(),
            )

            # 刷新仿真状态
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)

            # 同步基座观测缓存
            self.base_quat[replay_ids] = (self.root_states[replay_ids, 3:7])

            self.base_lin_vel[replay_ids] = quat_rotate_inverse(
                self.base_quat[replay_ids], self.root_states[replay_ids, 7:10])

            self.base_ang_vel[replay_ids] = quat_rotate_inverse(
                self.base_quat[replay_ids], self.root_states[replay_ids, 10:13])

            self.projected_gravity[replay_ids] = quat_rotate_inverse(
                self.base_quat[replay_ids], self.gravity_vec[replay_ids])

            # 同步轮子位置缓存
            self.feet_state = self.rigid_body_states[:, self.feet_indices, :]
            self.foot_positions = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
            self.last_foot_positions[replay_ids] = (self.foot_positions[replay_ids])

            # 清除回放环境的观测历史
            if hasattr(self, "observations_stack"):
                for history in self.observations_stack:
                    if (isinstance(history, torch.Tensor) and history.shape[0] == self.num_envs):
                        history[replay_ids] = 0.0

            # 回放的起点就是一个已知困难状态，
            # 因此直接开启脱困奖励，不必再次等待死点检测。
            self.commands[replay_ids] = 0.0

        # --------------------------------------------------
        # E. 清空 B2 阶段状态
        # --------------------------------------------------
        if hasattr(self, "recovery_stage"):
            self.recovery_stage[env_ids] = 0
            self.upright_count[env_ids] = 0
            self.wheel_ready_count[env_ids] = 0
            self.stand_success_count[env_ids] = 0
            self.fall_count[env_ids] = 0
            self.recovery_succeeded[env_ids] = False

        # --------------------------------------------------
        # F. 清空 B3 当前 episode 的检测状态
        # --------------------------------------------------
        if hasattr(self, "deadpoint_candidate_steps"):
            self.deadpoint_candidate_steps[env_ids] = 0
            self.deadpoint_no_progress_steps[env_ids] = 0
            self.deadpoint_best_height[env_ids] = -1e6
            self.deadpoint_active[env_ids] = False
            self.deadpoint_ever[env_ids] = False
            self.deadpoint_prev_height[env_ids] = 0.0
            self.deadpoint_height_delta[env_ids] = 0.0

        # 对于回放环境，保留“该 episode 从困难状态开始”的标志
        if replay_ids.numel() > 0:
            self.deadpoint_ever[replay_ids] = True
            replay_base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
            self.deadpoint_prev_height[replay_ids] = (replay_base_height[replay_ids])

        # --------------------------------------------------
        # G. 将死点统计写入训练日志
        # --------------------------------------------------
        if (hasattr(self, "extras")
            and "episode" in self.extras
            and deadpoint_done.numel() > 0):
            self.extras["episode"]["deadpoint_episode_rate"] = (deadpoint_done.float().mean())
            self.extras["episode"]["deadpoint_episode_count"] = (deadpoint_done.float().sum())
            self.extras["episode"]["deadpoint_replay_rate"] = (replay_ids.numel() / env_ids.numel())
            self.extras["episode"]["stand_success_rate"] = (stand_success_done.float().mean())

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
    
    def _init_recovery_state(self):
        n = self.num_envs
        device = self.device

        # 0: 姿态调整
        # 1: 建立双轮支撑
        # 2: 膝盖卸载
        # 3: 稳定站立
        self.recovery_stage = torch.zeros(n, dtype=torch.long, device=device)

        # 阶段转换计数器
        self.upright_count = torch.zeros(n, dtype=torch.long, device=device)
        self.wheel_ready_count = torch.zeros(n, dtype=torch.long, device=device)
        self.stand_success_count = torch.zeros(n, dtype=torch.long, device=device)
        self.fall_count = torch.zeros(n, dtype=torch.long, device=device)

        # 当前回合是否曾经成功恢复
        self.recovery_succeeded = torch.zeros(n, dtype=torch.bool, device=device)

    def _init_deadpoint_state(self):
        """B3:初始化死点检测器和困难状态回放缓冲区。"""

        n = self.num_envs
        device = self.device
        cfg = self.cfg.rewards.recovery

        # 当前连续处于候选死点状态的步数
        self.deadpoint_candidate_steps = torch.zeros(n, dtype=torch.long, device=device)

        # 当前没有明显高度进展的连续步数
        self.deadpoint_no_progress_steps = torch.zeros(n, dtype=torch.long, device=device)

        # 候选状态期间达到过的最高基座高度
        self.deadpoint_best_height = torch.full((n,), -1e6, dtype=torch.float, device=device)

        # 当前是否满足死点条件
        self.deadpoint_active = torch.zeros(n, dtype=torch.bool, device=device)

        # 本回合是否曾经检测到死点
        self.deadpoint_ever = torch.zeros(n, dtype=torch.bool, device=device)

        # 死点事件累计次数，供调试使用
        self.deadpoint_events = torch.zeros(n, dtype=torch.long, device=device)

        # 高度变化奖励需要的历史状态
        self.deadpoint_prev_height = torch.zeros(n, dtype=torch.float, device=device)
        self.deadpoint_height_delta = torch.zeros(n, dtype=torch.float, device=device)

        # 困难状态回放缓冲区
        capacity = int(cfg.deadpoint_replay_capacity)
        self.deadpoint_replay_root = torch.zeros((capacity, 13), dtype=torch.float, device=device)
        self.deadpoint_replay_dof = torch.zeros((capacity, self.num_dof, 2), dtype=torch.float, device=device)

        # Python 整数记录环形缓冲区的写入位置与有效样本数
        self.deadpoint_replay_ptr = 0
        self.deadpoint_replay_size = 0

    def _store_deadpoint_states(self, env_ids):
        """
        将检测到的死点状态保存到回放缓冲区。
        保存内容：
        1. 基座完整根状态(13 维）
        2. 全部关节的位置和速度

        基座位置改为相对环境原点的坐标，
        这样可以在其他并行环境中重放。
        """
        if env_ids.numel() == 0:
            return

        capacity = self.deadpoint_replay_root.shape[0]

        # 单次事件过多时，仅保留最后 capacity 个状态
        if env_ids.numel() > capacity:
            env_ids = env_ids[-capacity:]

        # 保存根状态
        root_snapshot = self.root_states[env_ids].detach().clone()

        # 转成相对当前环境原点的位置
        root_snapshot[:, :3] -= self.env_origins[env_ids]

        # 初始化时清零基座线速度和角速度，减少回放瞬间的速度干扰
        root_snapshot[:, 7:13] = 0.0

        # 保存 DOF 状态，兼容展平或视图形式的 dof_state
        dof_state_view = self.dof_state.view(self.num_envs, self.num_dof, 2)
        dof_snapshot = (dof_state_view[env_ids].detach().clone())

        # 保留关节位置，清零关节速度
        dof_snapshot[:, :, 1] = 0.0
        count = env_ids.numel()
        slots = (torch.arange(count, device=self.device) + self.deadpoint_replay_ptr) % capacity

        self.deadpoint_replay_root[slots] = root_snapshot
        self.deadpoint_replay_dof[slots] = dof_snapshot

        self.deadpoint_replay_ptr = (self.deadpoint_replay_ptr + count) % capacity
        self.deadpoint_replay_size = min(capacity, self.deadpoint_replay_size + count)

    def _update_deadpoint_detector(self):
        """B3:检测缺乏恢复进展的局部最优。"""

        cfg = self.cfg.rewards.recovery
        # 1. 基座高度
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # 2. 基座接近直立
        upright = (self.projected_gravity[:, 2] < cfg.upright_z)
        # 3. 左右轮均有足够的垂直接触力
        wheel_fz = self.contact_forces[:, self.feet_indices, 2]
        both_wheels = torch.any(wheel_fz > cfg.wheel_contact_force, dim=1)
        # 4. 至少一侧膝盖或基座有明显接触力
        link_indices = torch.cat((self.calf_indices, self.base_indices), dim=-1)
        link_force = torch.norm(self.contact_forces[:, link_indices, :], dim=-1)
        any_link = torch.any(link_force > cfg.knee_contact_force, dim=1)
        # 5. 排除已经完成恢复的环境
        height_in_deadpoint_range = (
            (base_height >= cfg.deadpoint_height_min)
            & (base_height < cfg.height_stand_target - cfg.height_stand_tolerance)
        )

        candidate = (
            # both_wheels
            any_link
            # & upright
            & height_in_deadpoint_range
            & (self.recovery_stage < 3)
        )

        # 6. 记录已触发死点后的基座高度变化
        self.deadpoint_height_delta.zero_()
        tracked = self.deadpoint_ever
        self.deadpoint_height_delta[tracked] = (base_height[tracked] - self.deadpoint_prev_height[tracked])
        self.deadpoint_prev_height[:] = base_height

        # 7. 进入候选状态的第一步
        start = candidate & (self.deadpoint_candidate_steps == 0)
        best_before = torch.where(start, base_height, self.deadpoint_best_height)

        # 只有高度至少提高指定幅度，才算有效进展
        improved = candidate & (base_height >= best_before + cfg.deadpoint_min_height_progress)

        # 8. 更新候选死点持续时间
        self.deadpoint_candidate_steps = torch.where(
            candidate,
            self.deadpoint_candidate_steps + 1,
            torch.zeros_like(self.deadpoint_candidate_steps),
        )

        # 9. 更新无进展计数
        reset_no_progress = start | improved
        next_no_progress = torch.where(
            reset_no_progress,
            torch.zeros_like(self.deadpoint_no_progress_steps),
            self.deadpoint_no_progress_steps + 1,
        )

        self.deadpoint_no_progress_steps = torch.where(
            candidate,
            next_no_progress,
            torch.zeros_like(self.deadpoint_no_progress_steps),
        )

        # 10. 更新候选状态期间的最高高度
        self.deadpoint_best_height = torch.where(candidate,
            torch.maximum(best_before, base_height),
            torch.full_like(base_height, -1e6),
        )

        # 11. 将秒数转换成控制步数
        hold_steps = max(1, int(round(cfg.deadpoint_hold_s / self.dt)))
        no_progress_steps = max(1, int(round(cfg.deadpoint_no_progress_s / self.dt)))

        # 12. 持续接触 + 长时间没有进展，才确认死点
        self.deadpoint_active = (candidate
            & (self.deadpoint_candidate_steps >= hold_steps)
            & (self.deadpoint_no_progress_steps >= no_progress_steps)
        )

        # 13. 记录本回合第一次死点事件
        new_event = (self.deadpoint_active & (~self.deadpoint_ever))
        self.deadpoint_ever |= self.deadpoint_active
        self.deadpoint_events[new_event] += 1

        if torch.any(new_event):
            event_ids = torch.where(new_event)[0]

            # 目录与文件路径（只计算一次，缓存到实例属性）
            if not hasattr(self, "_deadpoint_md_path"):
                import os
                log_dir = "/home/gyj/rsl_rl/getup_gym/logs/B3"
                os.makedirs(log_dir, exist_ok=True)
                self._deadpoint_md_path = os.path.join(log_dir, "deadpoint_events1.md")

                # 文件不存在则写表头
                if not os.path.exists(self._deadpoint_md_path):
                    with open(self._deadpoint_md_path, "w", encoding="utf-8") as f:
                        f.write("# B3 Deadpoint Events\n\n")
                        f.write("| 序号 | common_step_counter | env_id | base_height | candidate_steps | no_progress_steps |\n")
                        f.write("|---|---|---|---|---|---|\n")
                    self._deadpoint_md_counter = 0

            if not hasattr(self, "_deadpoint_md_counter"):
                self._deadpoint_md_counter = 0

            # 逐 env 写一行
            with open(self._deadpoint_md_path, "a", encoding="utf-8") as f:
                for env_id in event_ids.tolist():
                    self._deadpoint_md_counter += 1
                    line = (
                        f"| {self._deadpoint_md_counter} "
                        f"| {self.common_step_counter} "
                        f"| {env_id} "
                        f"| {base_height[env_id].item():.4f} "
                        f"| {self.deadpoint_candidate_steps[env_id].item()} "
                        f"| {self.deadpoint_no_progress_steps[env_id].item()} |\n"
                    )
                    f.write(line)
                    print(
                        f"[Deadpoint] env_id={env_id}  "
                        f"common_step_counter={self.common_step_counter}  "
                        f"base_height={base_height[env_id].item():.4f}"
                    )

        # 第一次触发时不把之前的高度变化算作脱困进展
        self.deadpoint_prev_height[new_event] = (base_height[new_event])
        self.deadpoint_height_delta[new_event] = 0.0
        # 14. 保存真实死点状态
        if (cfg.deadpoint_replay_enabled and torch.any(new_event)):
            event_ids = torch.where(new_event)[0]
            self._store_deadpoint_states(event_ids)
            


    def _update_recovery_stage(self):
        """根据每个环境自身的状态更新恢复阶段。"""
        cfg = self.cfg.rewards.recovery
    
        # 1. 基座高度
        base_height = torch.mean( self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)

        # 2. 基座是否接近直立
        upright = (self.projected_gravity[:, 2] < cfg.upright_z)

        # 3. 左右轮子的垂直接触力
        wheel_fz = self.contact_forces[:, self.feet_indices, 2]
        both_wheels = torch.any(wheel_fz > cfg.wheel_contact_force, dim=1)

        # 4. 左右膝接触力
        knee_force = torch.norm(self.contact_forces[:, self.calf_indices, :], dim=-1)
        knees_clear = torch.all(knee_force < cfg.knee_contact_force, dim=1)

        # 5. 默认关节姿态误差，不约束轮子的转动角度
        leg_ids = [0, 1, 2, 4, 5, 6]
        pose_error = torch.mean(torch.abs(self.dof_pos[:, leg_ids] - self.default_dof_pos[:, leg_ids]), dim=1)

        # 6. 基座角速度
        base_ang_speed = torch.norm(self.base_ang_vel[:, :2], dim=1)

        # 7. 基座接触检测
        # 当前项目已有 base_indice 的使用方式；
        # 运行前仍需确认该索引确实指向基座碰撞刚体。
        base_contact_force = torch.norm(self.contact_forces[:, self.base_indice, :], dim=-1)
        base_clear = (base_contact_force < cfg.base_contact_force)

        # 8. 将秒数换算成策略控制步数
        transition_steps = max(1, int(cfg.transition_hold_s / self.dt))
        stand_steps = max(1, int(cfg.stand_hold_s / self.dt))
        fall_steps = max(1, int(cfg.fall_hold_s / self.dt))

        # -------------------------------------------------
        # 阶段 0 -> 阶段 1：姿态调整 -> 建立双轮支撑
        # -------------------------------------------------
        cond_01 = ((self.recovery_stage == 0)
                    & upright
                    & (base_height >= cfg.height_p0_to_p1))

        self.upright_count = torch.where(cond_01,
                            self.upright_count + 1,
                            torch.zeros_like(self.upright_count))

        enter_p1 = ((self.recovery_stage == 0) & (self.upright_count >= transition_steps))

        self.recovery_stage[enter_p1] = 1
        self.upright_count[enter_p1] = 0

        # -------------------------------------------------
        # 阶段 1 -> 阶段 2：建立双轮支撑 -> 膝盖卸载
        # -------------------------------------------------
        cond_12 = ((self.recovery_stage == 1)
                    & upright
                    & (base_height >= cfg.height_p1_to_p2)
                    & both_wheels)

        self.wheel_ready_count = torch.where(cond_12,
                                    self.wheel_ready_count + 1,
                                    torch.zeros_like(self.wheel_ready_count))

        enter_p2 = ((self.recovery_stage == 1) & (self.wheel_ready_count >= transition_steps))
        self.recovery_stage[enter_p2] = 2
        self.wheel_ready_count[enter_p2] = 0

        # -------------------------------------------------
        # 阶段 2 -> 阶段 3：膝盖卸载 -> 稳定站立
        # -------------------------------------------------
        height_stable = (torch.abs(base_height - cfg.height_stand_target) <= cfg.height_stand_tolerance)

        stand_condition = (
            upright
            & height_stable
            & both_wheels
            & knees_clear
            & base_clear
            & (pose_error < cfg.stand_pose_tolerance)
            & (base_ang_speed < cfg.stand_ang_vel_tolerance)
        )

        cond_23 = ((self.recovery_stage == 2) & stand_condition)

        self.stand_success_count = torch.where( cond_23, self.stand_success_count + 1,
            torch.zeros_like(self.stand_success_count))

        enter_p3 = ((self.recovery_stage == 2)& (self.stand_success_count >= stand_steps))

        self.recovery_stage[enter_p3] = 3
        self.recovery_succeeded[enter_p3] = True

        # -------------------------------------------------
        # 严重跌倒保护：
        # 不因短暂失去接触就退回前期，但持续严重跌倒时允许重新恢复。
        # -------------------------------------------------
        severe_fall = ((self.recovery_stage >= 2) & (~upright) & (base_height < cfg.height_p1_to_p2))
        self.fall_count = torch.where(severe_fall, self.fall_count + 1, torch.zeros_like(self.fall_count))
        regress = ((self.recovery_stage >= 2) & (self.fall_count >= fall_steps))

        self.recovery_stage[regress] = 1
        self.fall_count[regress] = 0
        self.wheel_ready_count[regress] = 0
        self.stand_success_count[regress] = 0

    def _process_rigid_body_props(self, props, env_id):
        """记录每个并行环境经过质量随机化后的机器人总质量。"""

        props = super()._process_rigid_body_props(props, env_id)

        if not hasattr(self, "robot_mass_per_env"):
            self.robot_mass_per_env = torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device
            )

        self.robot_mass_per_env[env_id] = sum(
            prop.mass for prop in props
        )

        return props
    
    def _init_force_target_indices(self):
        """初始化五个接触载荷刚体及三个辅助目标刚体的索引。"""

        # 五维接触载荷：
        # [base, left_calf, right_calf, left_wheel, right_wheel]

        self.contact_load_body_indices = torch.tensor(
            [
                self.base_indice,
                int(self.calf_indices[0].item()),
                int(self.calf_indices[1].item()),
                int(self.feet_indices[0].item()),
                int(self.feet_indices[1].item()),
            ],
            dtype=torch.long,
            device=self.device
        )

        # 三个候选辅助目标：
        # [base, left_thigh, right_thigh]

        thigh_indices = self.find_rigid_body_id(
            ["left_leg_2", "right_leg_2"]
        )

        self.pull_up_target_indices = torch.tensor(
            [
                self.base_indice,
                thigh_indices[0],
                thigh_indices[1],
            ],
            dtype=torch.long,
            device=self.device
        )

        if torch.any(self.contact_load_body_indices < 0):
            raise RuntimeError("接触载荷刚体索引无效，请检查 URDF 刚体名称")

        if torch.any(self.pull_up_target_indices < 0):
            raise RuntimeError("辅助目标刚体索引无效，请检查 URDF 刚体名称")
        
    def _get_five_contact_loads(self):
        """
        获取五个刚体的归一化接触载荷。

        返回：
            [num_envs, 5]

        顺序：
            base, left_calf, right_calf,
            left_wheel, right_wheel
        """

        # 获取五个刚体的三维接触力
        contact_force = self.contact_forces[
            :, self.contact_load_body_indices, :
        ]

        # 接触力模长
        contact_force_norm = torch.linalg.vector_norm(
            contact_force,
            dim=-1
        )

        # 实际机器人总重
        gravity = torch.linalg.vector_norm(
            torch.tensor(
                self.cfg.sim.gravity,
                dtype=torch.float,
                device=self.device
            )
        )

        robot_weight = self.robot_mass_per_env * gravity

        # 归一化接触载荷
        contact_loads = contact_force_norm / (
            robot_weight.unsqueeze(1).clamp_min(1e-6)
        )

        return contact_loads
    
    def _get_five_contact_states(self):
        """
        五维接触状态：
        [base, left_calf, right_calf, left_wheel, right_wheel]

        返回：[num_envs, 5]
        0 = 未接触
        1 = 接触
        """

        forces = self.contact_forces[
            :, self.contact_load_body_indices, :
        ]

        force_norm = torch.linalg.vector_norm(forces, dim=-1)

        # 与现有恢复阶段接触力判据保持一致
        threshold = torch.tensor(
            [
                self.cfg.rewards.recovery.base_contact_force,
                self.cfg.rewards.recovery.knee_contact_force,
                self.cfg.rewards.recovery.knee_contact_force,
                self.cfg.rewards.recovery.wheel_contact_force,
                self.cfg.rewards.recovery.wheel_contact_force,
            ],
            device=self.device,
            dtype=torch.float,
        )

        contact_states = (force_norm > threshold.unsqueeze(0)).float()

        return contact_states
    
    def _get_wheel_slip_velocity(self):
        """
        计算左右轮沿滚动方向的接触点速度。

        返回：[num_envs, 2]
        单位:m/s

        当前适用于水平地面。
        """

        wheel_indices = self.feet_indices

        # 刚体状态：[position(3), quaternion(4),
        #            linear_velocity(3), angular_velocity(3)]
        wheel_states = self.rigid_body_states[:, wheel_indices, :]

        wheel_quat = wheel_states[:, :, 3:7]
        wheel_lin_vel = wheel_states[:, :, 7:10]
        wheel_ang_vel = wheel_states[:, :, 10:13]

        # URDF轮子半径
        wheel_radius = 0.0925

        # 轮子局部旋转轴为Z轴，转换到世界坐标系
        local_axis = torch.zeros(
            self.num_envs, 2, 3,
            device=self.device
        )
        local_axis[..., 2] = 1.0

        wheel_axis = quat_apply(
            wheel_quat.reshape(-1, 4),
            local_axis.reshape(-1, 3)
        ).reshape(self.num_envs, 2, 3)

        # 水平地面法向
        world_up = torch.zeros_like(wheel_axis)
        world_up[..., 2] = 1.0

        # 轮子滚动方向
        rolling_dir = torch.cross(
            wheel_axis, world_up, dim=-1
        )

        rolling_dir = torch.nn.functional.normalize(
            rolling_dir, dim=-1, eps=1e-6
        )

        # 轮心到假设接触点的向量
        r_contact = torch.zeros_like(wheel_lin_vel)
        r_contact[..., 2] = -wheel_radius

        # 接触点速度
        contact_point_vel = (
            wheel_lin_vel
            + torch.cross(
                wheel_ang_vel,
                r_contact,
                dim=-1
            )
        )

        # 沿滚动方向投影
        slip_velocity = torch.sum(
            contact_point_vel * rolling_dir,
            dim=-1
        )

        # 非接触状态下不提供虚假的地面滑移监督
        contact_states = self._get_five_contact_states()
        wheel_contact = contact_states[:, 3:5]

        slip_velocity = slip_velocity * wheel_contact

        return slip_velocity
    
    def _base_force_pull_up(self):
        """
        基于接触载荷选择辅助目标刚体。

        目标选择：
            1. 基座接触载荷最大 -> 基座
            2. 左小腿接触载荷最大 -> 左大腿
            3. 右小腿接触载荷最大 -> 右大腿

        其余辅助力、辅助力矩、时间衰减、
        高度控制和延迟屏蔽逻辑保持原样。
        """

        # =====================================================
        # 1. 保留原有辅助力计算逻辑
        # =====================================================

        force_max = self.cfg.rewards.curriculum.base_pull_up_max

        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights,
            dim=1
        )

        time_coeff = torch.tensor(
            (
                self.cfg.rewards.curriculum.pull_up_end_epochs
                - ((self.common_step_counter - 1) / 24)
            ) / self.cfg.rewards.curriculum.pull_up_end_epochs,
            device=self.device
        )

        time_coeff = torch.clamp(time_coeff, 0.0, 1.0)

        if not getattr(
            self.cfg.rewards.curriculum,
            "using_pull_up_end",
            True
        ):
            time_coeff = torch.ones_like(time_coeff)

        base_height_target = getattr(
            self.cfg.rewards.curriculum,
            "base_height_target",
            0.5
        )

        force_scale = (
            (base_height_target - base_height) / base_height_target
        ) * force_max * time_coeff

        self.force_scale = torch.clamp(
            force_scale,
            0.0,
            force_max
        )

        # =====================================================
        # 2. 获取五维归一化接触载荷
        # =====================================================

        contact_loads = self._get_five_contact_loads()

        # 仅使用基座、左右小腿的接触载荷选择目标
        candidate_loads = contact_loads[:, 0:3]

        # 每个环境分别选择接触载荷最大的刚体
        target_slot = torch.argmax(
            candidate_loads,
            dim=1
        )

        # 映射为实际辅助目标刚体索引
        target_body_indices = self.pull_up_target_indices[
            target_slot
        ]
        
        # =====================================================
        # 3. 初始化外力和力矩张量
        # =====================================================

        forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3),
            device=self.device,
            dtype=torch.float
        )

        torques = torch.zeros_like(forces)

        env_indices = torch.arange(
            self.num_envs,
            device=self.device
        )

        # =====================================================
        # 4. 对每个环境的目标刚体施加 Z 方向辅助力
        # =====================================================

        forces[
            env_indices,
            target_body_indices,
            2
        ] = self.force_scale

        # =====================================================
        # 5. 保留原有辅助力矩逻辑
        # =====================================================

        env_start = getattr(self.cfg.rewards, "pull_up_env_start", 0)
        env_end = getattr(self.cfg.rewards, "pull_up_env_end", self.num_envs)
        torque_scale = getattr(self.cfg.rewards, "pull_up_torque_scale", -10.0)

        if env_end > env_start:

            env_range = torch.arange(env_start, env_end, device=self.device)

            selected_body_indices = target_body_indices[env_start:env_end]

            rotate_scale = torch.abs(self.root_states[env_start:env_end, 4] + 1)

            height_scale = (
                (base_height_target - base_height)
                / base_height_target
            )[env_start:env_end]

            torques[env_range, selected_body_indices, 0] = (
                torque_scale
                * time_coeff
                * rotate_scale
                * height_scale
            )

        # =====================================================
        # 6. 保留原有延迟动作屏蔽逻辑
        # =====================================================

        if (
            hasattr(self, "real_episode_length_buf")
            and hasattr(self, "unactuated_time")
        ):

            mask = (
                self.real_episode_length_buf.unsqueeze(1)
                > self.unactuated_time
            ).unsqueeze(1)

            forces = forces * mask
            torques = torques * mask

        # =====================================================
        # 7. 保留原有 Isaac Gym 外力施加逻辑
        # =====================================================

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(forces),
            gymtorch.unwrap_tensor(torques),
            gymapi.ENV_SPACE
        )