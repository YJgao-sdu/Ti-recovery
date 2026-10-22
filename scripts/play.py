"""Play / evaluation script for getup_gym."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from getup_gym import GETUP_GYM_ROOT_DIR
from getup_gym.utils.registry import task_registry
from getup_gym.common.helpers import get_args
import torch
import numpy as np


class StudentPolicy(torch.nn.Module):
    """Student encoder + actor composed into a single deployable module.

    The deployed student policy consumes only proprioceptive inputs:
    zt = student_encoder(student_obs), actions = actor(cat(zt, obs)).
    """

    def __init__(self, student_encoder, actor):
        super().__init__()
        self.student_encoder = student_encoder
        self.actor = actor

    def forward(self, student_obs, obs):
        zt = self.student_encoder(student_obs)["zt"]
        obs_input = torch.cat((zt, obs), dim=-1)
        return self.actor(obs_input)


def _prepare_env_and_runner(args):
    """Shared setup for play and export: single evaluation env + loaded runner."""
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Override for evaluation
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.num_envs_teacher = 0
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.rewards.curriculum.using_pull_up = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    teacher_obs = env.get_teacher_encoder_observations()
    student_obs = env.get_student_encoder_observations()
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    obs, _, teacher_obs, student_obs, rews, dones, infos = env.step(actions.detach())

    train_cfg.runner.resume = True
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder_policy, inference_mode = ppo_runner.get_encoder_inference_policy(device=env.device)
    return env, obs, student_obs, ppo_runner, policy, encoder_policy, inference_mode


def play(args):
    env, obs, student_obs, ppo_runner, policy, encoder_policy, inference_mode = _prepare_env_and_runner(args)

    print("Running policy...")
    # Camera follows the robot from the side, framing the whole body
    cam_pos_offset = np.array([0.0, 3.0, 0.2])  # [m] lateral offset + height above robot base
    cam_look_offset = np.array([0.0, 0.0, 0.1])  # [m] aim at mid-body height above robot base
    for _ in range(10 * int(env.max_episode_length)):
        if inference_mode == "CTS" and encoder_policy is not None:
            zt = encoder_policy(student_obs)["zt"]
            obs_input = torch.cat((zt, obs), dim=-1)
        else:
            obs_input = obs
        actions = policy(obs_input.detach())
        obs, _, teacher_obs, student_obs, rews, dones, infos = env.step(actions.detach())
        if env.viewer is not None:
            base_pos = env.root_states[0, :3].cpu().numpy()
            env.set_camera(base_pos + cam_pos_offset, base_pos + cam_look_offset)


def export_policy_as_onnx(args):
    """Export the student policy (student encoder + actor) to ONNX for deployment."""
    env, obs, student_obs, ppo_runner, policy, encoder_policy, inference_mode = _prepare_env_and_runner(args)

    export_dir = os.path.join(GETUP_GYM_ROOT_DIR, "logs", args.task, "exported", "policies")
    os.makedirs(export_dir, exist_ok=True)

    if inference_mode == "CTS" and encoder_policy is not None and student_obs is not None:
        model = StudentPolicy(ppo_runner.student_encoder, ppo_runner.alg.actor_critic.actor)
        onnx_path = os.path.join(export_dir, "student_policy.onnx")
        torch.onnx.export(
            model,
            (student_obs, obs),
            onnx_path,
            input_names=["student_obs", "obs"],
            output_names=["actions"],
            dynamic_axes={
                "student_obs": {0: "batch_size"},
                "obs": {0: "batch_size"},
                "actions": {0: "batch_size"},
            },
            opset_version=11,
            do_constant_folding=True,
        )
        print(f"Exported student policy (student encoder + actor) to: {onnx_path}")
    else:
        model = ppo_runner.alg.actor_critic.actor
        onnx_path = os.path.join(export_dir, "policy.onnx")
        torch.onnx.export(
            model,
            obs,
            onnx_path,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch_size"}, "actions": {0: "batch_size"}},
            opset_version=11,
            do_constant_folding=True,
        )
        print(f"Exported policy (actor only) to: {onnx_path}")


if __name__ == "__main__":
    args = get_args()
    if args.export_onnx:
        export_policy_as_onnx(args)
    else:
        play(args)
