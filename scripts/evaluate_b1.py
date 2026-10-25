import os
import sys
import math
import csv
import types
import numpy as np

# 确保从项目根目录运行时可以导入 getup_gym
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

from getup_gym import GETUP_GYM_ROOT_DIR
from getup_gym.utils.registry import task_registry
from getup_gym.common.helpers import get_args
import torch


# ============================================================
# 1. 六类初始姿态模板
# ============================================================
# 格式：(名称, roll, pitch, yaw, root_height)
#
# 注意：
# 1) 角度和高度是初始调试值，不是经过验证的 Tita 标准姿态。
# 2) 需通过 Isaac Gym 可视化确认前趴/后躺和翻转姿态的实际方向。
# 3) root_height 是机器人根刚体原点高度，不是基座底面高度。
# 4) 默认腿部关节角不一定代表真实跌倒时的腿部构型。

POSES = {
    0: ("front_prone",  0.0,  math.pi / 2, 0.0, 0.31),
    1: ("back_supine",  0.0, -math.pi / 2, 0.0, 0.31),
    2: ("left_side",   -math.pi / 2, 0.0, 0.0, 0.27),
    3: ("right_side",   math.pi / 2, 0.0, 0.0, 0.27),
    4: ("base_flip_a", 0.0, 0.0, 0.0, 0.38),
    5: ("base_flip_b", 0.0, math.pi, 0.0, 0.12),
}


def quat_from_rpy(roll, pitch, yaw):
    """欧拉角转 Isaac Gym 四元数 [x, y, z, w]。"""

    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)

    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)

    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)

    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


# ============================================================
# 2. 环境与策略加载
# ============================================================

def _prepare_env_and_runner(args):
    """复用项目 play.py 的环境创建和策略加载方式。"""

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # 评估只使用一个环境
    env_cfg.env.num_envs = 1
    env_cfg.env.num_envs_teacher = 0

    # 固定地面
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 5

    # 关闭观测噪声
    env_cfg.noise.add_noise = False

    # 关闭随机外部推力、辅助起身和动力学随机化
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_force_push = False
    env_cfg.rewards.curriculum.using_pull_up = False

    for name in (
        "randomize_friction",
        "randomize_base_mass",
        "randomize_restitution",
        "add_mass_ranges",
        "randomize_motor_strength",
        "randomize_joint_pos",
        "randomize_torso_size",
        "randomize_pd",
    ):
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)

    # 将运动指令固定为零
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    # 创建环境
    env, _ = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
    )

    obs = env.get_observations()
    student_obs = env.get_student_encoder_observations()

    # 与原 play.py 保持一致，先执行一次零动作初始化
    zero_actions = torch.zeros(
        (env.num_envs, env.num_actions),
        device=env.device,
    )

    obs, _, _, student_obs, _, _, _ = env.step(
        zero_actions
    )

    # 加载 checkpoint
    train_cfg.runner.resume = True

    ppo_runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
    )

    policy = ppo_runner.get_inference_policy(
        device=env.device
    )

    encoder_policy, inference_mode = (
        ppo_runner.get_encoder_inference_policy(
            device=env.device
        )
    )

    return (
        env,
        obs,
        student_obs,
        ppo_runner,
        policy,
        encoder_policy,
        inference_mode,
    )


# ============================================================
# 3. B1 统计与回调采集器
# ============================================================

class B1Collector:
    """
    不修改 env.py 的 B1 采集器。

    通过包装以下原有方法采集数据：
      - _post_physics_step_callback
      - check_termination
      - reset_idx

    在原始 reset_idx 重置机器人之前保存最后一帧。
    """

    def __init__(self, env):
        self.env = env

        self.device = env.device
        self.dt = float(env.dt)

        self.contact_threshold = 0.1

        self.target_height = float(
            env.cfg.rewards.base_height_target
        )
        self.height_tolerance = 0.04

        self.upright_threshold = -0.85
        self.ang_speed_tolerance = 0.50
        self.pose_tolerance = 0.12

        self.stand_hold_steps = max(
            1, int(round(0.50 / self.dt))
        )

        self.wheel_indices = (
            env.feet_indices.detach().cpu().tolist()
        )
        self.knee_indices = (
            env.calf_indices.detach().cpu().tolist()
        )

        if len(self.wheel_indices) != 2:
            raise RuntimeError(
                "feet_indices 不是两个刚体。"
                "请检查 config.py 中的 foot_name。"
            )

        if len(self.knee_indices) != 2:
            raise RuntimeError(
                "calf_indices 不是两个刚体。"
                "请检查 config.py 中的 calf_name。"
            )

        # 项目中的 body_indices 第一个元素对应 torso。
        # 这里只记录 torso 接触，不将其等同于所有基座碰撞体。
        self.torso_index = int(env.body_indices[0])

        # Tita 的 8 个 DOF 中，0、1、2、4、5、6 为腿部关节。
        # 两个轮子的转动 DOF 不参与默认腿部姿态误差。
        self.leg_ids = torch.tensor(
            [0, 1, 2, 4, 5, 6],
            dtype=torch.long,
            device=self.device,
        )

        self.active = False
        self.pending_rows = []

        self.pose_id = -1
        self.trial_id = -1
        self.seed = -1

        self._clear_accumulators()
        self._install_hooks()

    def _clear_accumulators(self):
        """清空单次 episode 的统计数据。"""

        self.steps = 0

        self.knee_contact_time = np.zeros(2)
        self.wheel_contact_time = np.zeros(2)

        self.wheel_fz_abs_sum = np.zeros(2)
        self.knee_force_peak = np.zeros(2)

        self.base_contact_time = 0.0
        self.both_wheels_time = 0.0
        self.triple_support_time = 0.0

        self.height_sum = 0.0
        self.height_max = -float("inf")

        self.ang_speed_sum = 0.0
        self.pose_error_sum = 0.0

        self.stable_count = 0
        self.success = False
        self.success_step = -1

        self.last = None

    def _install_hooks(self):
        """包装环境回调，不直接改写原始环境文件。"""

        env = self.env

        original_callback = env._post_physics_step_callback
        original_check = env.check_termination
        original_reset = env.reset_idx

        def wrapped_callback(_env):
            # 保留原有的命令更新、地形高度更新等操作
            original_callback()

            # 原始 callback 运行后，读取当前控制步状态
            self.sample_step()

        def wrapped_check(_env):
            # 先执行项目自己的终止判断
            original_check()

            # 仅在 B1 评估时，稳定站立才提前结束 episode
            if self.active and self.success:
                _env.reset_buf[0] = True

        def wrapped_reset(_env, env_ids):
            # 必须在父类重置前记录最后一帧
            if (
                self.active
                and env_ids is not None
                and len(env_ids) > 0
            ):
                for idx in env_ids.tolist():
                    if int(idx) == 0:
                        self.finalize_episode()
                        break

            # 保留原始 reset_idx 的全部逻辑
            original_reset(env_ids)

        env._post_physics_step_callback = types.MethodType(
            wrapped_callback, env
        )

        env.check_termination = types.MethodType(
            wrapped_check, env
        )

        env.reset_idx = types.MethodType(
            wrapped_reset, env
        )

    def sample_step(self):
        """在每个控制步累计当前状态。"""

        if not self.active:
            return

        env = self.env
        dt = self.dt
        threshold = self.contact_threshold

        # 基座根刚体高度相对地面的高度
        root_height = float(
            env.root_states[0, 2].item()
        )

        if isinstance(env.measured_heights, torch.Tensor):
            terrain_height = float(
                env.measured_heights[0].mean().item()
            )
        else:
            terrain_height = 0.0

        base_height = root_height - terrain_height

        # 当前左右轮的三维接触力
        wheel_force = (env.contact_forces[0, env.feet_indices, :].detach().cpu().numpy())

        wheel_fz = wheel_force[:, 2]
        wheel_force_norm = np.linalg.norm(
            wheel_force, axis=1
        )

        wheel_contact = (
            wheel_force_norm > threshold
        )

        both_wheels = bool(np.any(wheel_contact))

        # 当前左右膝接触力
        knee_force_tensor = env.contact_forces[
            0, env.calf_indices, :
        ]

        knee_force = (
            torch.norm(knee_force_tensor, dim=-1)
            .detach()
            .cpu()
            .numpy()
        )

        knee_contact = knee_force > threshold
        any_knee = bool(np.any(knee_contact))

        # torso 接触地面
        torso_force = torch.norm(
            env.contact_forces[0, self.torso_index, :]
        )

        torso_contact = (
            float(torso_force.item()) > threshold
        )

        # 直立判据：机身坐标系下的重力 z 分量接近 -1
        upright = (
            float(env.projected_gravity[0, 2].item())
            < self.upright_threshold
        )

        # 默认腿部姿态误差，单位 rad
        pose_error = float(
            torch.mean(
                torch.abs(
                    env.dof_pos[0, self.leg_ids]
                    - env.default_dof_pos[0, self.leg_ids]
                )
            ).item()
        )

        # 基座角速度大小，单位 rad/s
        ang_speed = float(
            torch.norm(env.base_ang_vel[0]).item()
        )

        height_ok = (
            abs(base_height - self.target_height)
            <= self.height_tolerance
        )

        # 严格的稳定站立条件
        stand_now = (
            upright
            and height_ok
            and both_wheels
            and (not any_knee)
            and (not torso_contact)
            and pose_error < self.pose_tolerance
            and ang_speed < self.ang_speed_tolerance
        )

        # 累计时间和状态
        self.steps += 1

        self.knee_contact_time += (
            knee_contact.astype(float) * dt
        )

        self.wheel_contact_time += (
            wheel_contact.astype(float) * dt
        )

        self.wheel_fz_abs_sum += np.abs(wheel_fz)

        self.knee_force_peak = np.maximum(
            self.knee_force_peak,
            knee_force,
        )

        self.height_sum += base_height
        self.height_max = max(
            self.height_max, base_height
        )

        self.ang_speed_sum += ang_speed
        self.pose_error_sum += pose_error

        self.base_contact_time += (
            float(torso_contact) * dt
        )

        self.both_wheels_time += (
            float(both_wheels) * dt
        )

        self.triple_support_time += (
            float(both_wheels and any_knee) * dt
        )

        # 连续站立计数
        if stand_now:
            self.stable_count += 1
        else:
            self.stable_count = 0

        # 第一次连续站立达到 0.5 秒时判定成功
        if (
            not self.success
            and self.stable_count >= self.stand_hold_steps
        ):
            self.success = True
            self.success_step = self.steps

        # 保存最后一个控制步的指标，供 finalize 使用
        self.last = {
            "final_base_height_m": base_height,
            "final_pose_error_rad": pose_error,
            "final_base_ang_speed_rad_s": ang_speed,

            "final_left_wheel_fz_n": float(wheel_fz[0]),
            "final_right_wheel_fz_n": float(wheel_fz[1]),

            "final_left_knee_contact": int(knee_contact[0]),
            "final_right_knee_contact": int(knee_contact[1]),

            "final_base_contact": int(torso_contact),
            "final_both_wheels_contact": int(both_wheels),

            "triple_support_at_end": int(
                both_wheels and any_knee
            ),
        }

    def finalize_episode(self):
        """保存一条 episode 记录。"""

        if not self.active or self.steps <= 0:
            self.active = False
            return

        last = self.last or {}

        env = self.env

        if self.success:
            end_reason = "success"
        elif bool(env.time_out_buf[0].item()):
            end_reason = "timeout"
        else:
            end_reason = "termination"

        triple_support_at_end = int(
            last.get("triple_support_at_end", 0)
        )

        row = {
            "pose_id": self.pose_id,
            "pose_name": POSES[self.pose_id][0],
            "trial_id": self.trial_id,
            "random_seed": self.seed,

            "episode_steps": self.steps,
            "episode_time_s": self.steps * self.dt,
            "termination_reason": end_reason,

            "success": int(self.success),
            "success_time_s": (
                self.success_step * self.dt
                if self.success else -1.0
            ),

            # 失败时最终仍然双轮+膝盖支撑
            "triple_support_at_end": triple_support_at_end,

            "deadpoint_at_end": int(
                triple_support_at_end and not self.success
            ),

            "left_knee_contact_time_s": float(
                self.knee_contact_time[0]
            ),
            "right_knee_contact_time_s": float(
                self.knee_contact_time[1]
            ),

            "left_knee_peak_force_n": float(
                self.knee_force_peak[0]
            ),
            "right_knee_peak_force_n": float(
                self.knee_force_peak[1]
            ),

            "left_wheel_contact_time_s": float(
                self.wheel_contact_time[0]
            ),
            "right_wheel_contact_time_s": float(
                self.wheel_contact_time[1]
            ),

            "both_wheels_contact_time_s": float(
                self.both_wheels_time
            ),

            "triple_support_time_s": float(
                self.triple_support_time
            ),

            "base_contact_time_s": float(
                self.base_contact_time
            ),

            "mean_left_wheel_abs_fz_n": float(
                self.wheel_fz_abs_sum[0] / self.steps
            ),
            "mean_right_wheel_abs_fz_n": float(
                self.wheel_fz_abs_sum[1] / self.steps
            ),

            "mean_base_height_m": float(
                self.height_sum / self.steps
            ),
            "max_base_height_m": float(
                self.height_max
            ),

            "mean_pose_error_rad": float(
                self.pose_error_sum / self.steps
            ),
            "mean_base_ang_speed_rad_s": float(
                self.ang_speed_sum / self.steps
            ),
        }

        row.update(last)

        self.pending_rows.append(row)

        # 结束本回合采集
        self.active = False

    def set_pose(self, pose_id, trial_id):
        """设置下一次测试的初始状态，并清空上一回合数据。"""

        env = self.env

        name, roll, pitch, yaw, root_height = POSES[pose_id]

        # 每个 trial 使用可复现的随机种子
        seed = 20261024 + pose_id * 1000 + trial_id
        rng = np.random.default_rng(seed)

        # 角度扰动：用于构建同类姿态附近的不同测试状态
        angle_jitter = math.radians(3.0)

        roll += rng.uniform(-angle_jitter, angle_jitter)
        pitch += rng.uniform(-angle_jitter, angle_jitter)
        yaw += rng.uniform(-angle_jitter, angle_jitter)

        q = torch.tensor(
            quat_from_rpy(roll, pitch, yaw),
            dtype=torch.float,
            device=self.device,
        )

        ids = torch.tensor(
            [0], dtype=torch.long, device=self.device
        )

        ids_int32 = ids.to(torch.int32)

        # ----------------------------------------------------
        # A. 重置当前 episode 的软件状态
        # ----------------------------------------------------
        self._clear_accumulators()

        self.pose_id = pose_id
        self.trial_id = trial_id
        self.seed = seed

        env.episode_length_buf[ids] = 0
        env.reset_buf[ids] = False
        env.time_out_buf[ids] = False

        if hasattr(env, "real_episode_length_buf"):
            env.real_episode_length_buf[ids] = 0

        for name in (
            "actions",
            "last_actions",
            "last_last_action",
            "last_dof_vel",
            "last_root_vel",
            "last_base_velocity",
            "last_last_base_velocity",
            "last_base_ang_velocity",
            "last_last_base_ang_velocity",
            "last_dof_pos",
            "last_last_dof_pos",
        ):
            value = getattr(env, name, None)
            if isinstance(value, torch.Tensor):
                if value.shape[0] == env.num_envs:
                    value[ids] = 0.0

        if hasattr(env, "time_counter"):
            env.time_counter[ids] = 0.0

        if hasattr(env, "feet_air_time"):
            env.feet_air_time[ids] = 0.0

        if hasattr(env, "last_contacts"):
            env.last_contacts[ids] = False

        # ----------------------------------------------------
        # B. 写入基座状态
        # ----------------------------------------------------
        env.root_states[ids, :3] = env.env_origins[ids]

        env.root_states[ids, 2] = (
            env.env_origins[ids, 2] + root_height
        )

        env.root_states[ids, 3:7] = q.unsqueeze(0)

        # 初始线速度、角速度设为零
        env.root_states[ids, 7:13] = 0.0

        # ----------------------------------------------------
        # C. 写入关节状态
        # ----------------------------------------------------
        # 初始关节姿态使用默认姿态，后续可替换为真实跌倒姿态。
        env.dof_pos[ids] = env.default_dof_pos[ids]
        env.dof_vel[ids] = 0.0

        # 腿部关节小幅扰动，不扰动轮子转动 DOF
        leg_noise = torch.tensor(
            rng.uniform(-0.03, 0.03, size=6),
            dtype=torch.float,
            device=self.device,
        )

        env.dof_pos[0, self.leg_ids] += leg_noise

        # 命令固定为零
        if hasattr(env, "commands"):
            env.commands[ids] = 0.0

        # ----------------------------------------------------
        # D. 将状态写回 Isaac Gym
        # ----------------------------------------------------
        env.gym.set_actor_root_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.root_states),
            gymtorch.unwrap_tensor(ids_int32),
            len(ids_int32),
        )

        env.gym.set_dof_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.dof_state),
            gymtorch.unwrap_tensor(ids_int32),
            len(ids_int32),
        )

        # 刷新仿真状态张量
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)

        # ----------------------------------------------------
        # E. 同步策略观测所需的缓存
        # ----------------------------------------------------
        env.base_quat[:] = env.root_states[:, 3:7]

        env.base_lin_vel[:] = quat_rotate_inverse(
            env.base_quat,
            env.root_states[:, 7:10],
        )

        env.base_ang_vel[:] = quat_rotate_inverse(
            env.base_quat,
            env.root_states[:, 10:13],
        )

        env.projected_gravity[:] = quat_rotate_inverse(
            env.base_quat,
            env.gravity_vec,
        )

        env.feet_state = env.rigid_body_states[
            :, env.feet_indices, :
        ]

        env.foot_positions = env.rigid_body_states[
            :, env.feet_indices, 0:3
        ]

        env.last_foot_positions[:] = env.foot_positions

        # ----------------------------------------------------
        # F. 清空时间堆叠观测，避免沿用上一回合的历史
        # ----------------------------------------------------
        if hasattr(env, "observations_stack"):
            env.observations_stack.clear()

            stack_size = int(
                getattr(
                    env.cfg.env.myself_setting,
                    "add_time_number",
                    5,
                )
            )

            for _ in range(stack_size):
                env.observations_stack.append(
                    torch.zeros_like(env.obs_buf)
                )

        # 重新计算观测，最后一个观测会进入历史堆叠
        env.compute_observations()

        obs = env.get_observations()
        student_obs = env.get_student_encoder_observations()

        # 从这里开始才统计当前 trial
        self.active = True

        return obs, student_obs


# ============================================================
# 4. 评估主流程
# ============================================================

def evaluate_b1():
    args = get_args()

    pose_id = int(os.environ.get("B1_POSE_ID", "0"))
    num_trials = int(os.environ.get("B1_EPISODES", "100"))

    if pose_id not in POSES:
        raise ValueError(
            f"B1_POSE_ID 必须是 0-5，当前值为 {pose_id}"
        )

    if num_trials <= 0:
        raise ValueError("B1_EPISODES 必须大于 0")

    (
        env,
        obs,
        student_obs,
        ppo_runner,
        policy,
        encoder_policy,
        inference_mode,
    ) = _prepare_env_and_runner(args)

    collector = B1Collector(env)

    output_dir = os.path.join(
        GETUP_GYM_ROOT_DIR,
        "logs",
        "B1",
    )

    os.makedirs(output_dir, exist_ok=True)

    pose_name = POSES[pose_id][0]

    csv_path = os.path.join(
        output_dir,
        f"b1_{pose_name}.csv",
    )

    print("=" * 65)
    print("B1 Tita Fall-Recovery Evaluation")
    print(f"Task: {args.task}")
    print(f"Pose: {pose_name} (ID={pose_id})")
    print(f"Trials: {num_trials}")
    print(f"Checkpoint run: {getattr(args, 'load_run', None)}")
    print(f"Checkpoint: {getattr(args, 'checkpoint', None)}")
    print(f"Output: {csv_path}")
    print("=" * 65)

    completed = 0
    writer = None

    with open(csv_path, "w", newline="") as f:
        while completed < num_trials:

            # 第一次测试，以及每次 episode 结束后的下一次测试
            obs, student_obs = collector.set_pose(
                pose_id,
                completed,
            )

            # 当前 trial 的推理循环
            while collector.active:

                with torch.inference_mode():

                    if (
                        inference_mode == "CTS"
                        and encoder_policy is not None
                    ):
                        encoder_output = encoder_policy(
                            student_obs
                        )

                        zt = encoder_output["zt"]

                        obs_input = torch.cat(
                            (zt, obs),
                            dim=-1,
                        )
                    else:
                        obs_input = obs

                    actions = policy(obs_input.detach())

                (
                    obs,
                    _,
                    _,
                    student_obs,
                    _,
                    _,
                    _,
                ) = env.step(actions.detach())

                # reset_idx hook 会在重置前保存数据。
                # 若发生终止，collector.active 会变为 False。
                if env.viewer is not None:
                    base_pos = (
                        env.root_states[0, :3]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    cam_offset = np.array([0.0, 3.0, 0.2])
                    look_offset = np.array([0.0, 0.0, 0.1])

                    env.set_camera(
                        base_pos + cam_offset,
                        base_pos + look_offset,
                    )

            # 本回合结束，写入 CSV
            if not collector.pending_rows:
                raise RuntimeError(
                    "episode 已结束，但没有产生 B1 记录。"
                    "请检查 reset_idx hook 和 episode 终止逻辑。"
                )

            row = collector.pending_rows.pop(0)

            if writer is None:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(row.keys()),
                )
                writer.writeheader()

            writer.writerow(row)
            f.flush()

            completed += 1

            print(
                f"[B1] {completed}/{num_trials} "
                f"pose={row['pose_name']} "
                f"success={row['success']} "
                f"triple_support="
                f"{row['triple_support_at_end']} "
                f"deadpoint={row['deadpoint_at_end']} "
                f"time={row['episode_time_s']:.2f}s"
            )

    print("=" * 65)
    print("B1 evaluation finished.")
    print(f"Pose: {pose_name}")
    print(f"Trials completed: {completed}")
    print(f"CSV: {csv_path}")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_b1()
