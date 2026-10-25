"""On-policy runner for force-guided PPO with contact-force estimation (no teacher-student)."""

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from getup_gym.algorithms.ff_ppo import FFPPO
from getup_gym.modules.actor_critic_ff import ActorCriticFF


class FFOnPolicyRunner:
    """Runner for the FTSR x FR-Net force-guided fall-recovery training."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = self.cfg["algorithm_config"]
        self.policy_cfg = self.cfg["policy_config"]
        self.device = device
        self.env = env

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Initialize environment to get observation shapes
        actions_zeros = torch.zeros(
            self.env.num_envs, self.env.num_actions, device=self.device, requires_grad=False
        )
        obs, privileged_obs, encoder_obs, contact_force_targets, _, _, self.extras = self.env.step(
            actions_zeros, None
        )

        self.num_obs = obs.size(1)
        self.num_critic_obs = privileged_obs.size(1) if privileged_obs is not None else self.num_obs
        self.num_encoder_obs = encoder_obs.size(1)
        self.num_contact_links = getattr(self.env, "num_contact_links", contact_force_targets.size(1))

        # Actor-Critic with contact-force encoder
        self.actor_critic = ActorCriticFF(
            num_actor_obs=self.num_obs + self.num_contact_links,
            num_critic_obs=self.num_critic_obs,
            num_actions=self.env.num_actions,
            num_contact_links=self.num_contact_links,
            encoder_input_size=self.num_encoder_obs,
            **self.policy_cfg,
        ).to(self.device)

        # Algorithm
        self.alg = FFPPO(self.actor_critic, device=self.device, **self.alg_cfg)
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.num_obs + self.num_contact_links],
            [self.num_critic_obs],
            [self.env.num_actions],
        )

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        encoder_obs = self.env.get_student_encoder_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, encoder_obs = obs.to(self.device), critic_obs.to(self.device), encoder_obs.to(self.device)
        self.alg.actor_critic.train()

        force_buf = self.extras.get("force") if self.extras else None
        contact_force_targets = self.extras.get("contact_force_targets") if self.extras else None

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            encoder_obs_buffer = []
            targets_buffer = []

            # Rollout. no_grad (rather than inference_mode) keeps the env
            # observation tensors usable in the learning step's autograd.
            with torch.no_grad():
                for i in range(self.num_steps_per_env):
                    est = self.actor_critic.encode(encoder_obs)
                    act_input = torch.cat((obs, est), dim=-1)
                    actions = self.alg.act(act_input, critic_obs, force=force_buf)

                    # (encoder_obs at time i, targets at time i) pair for the aux loss
                    encoder_obs_buffer.append(encoder_obs)
                    targets_buffer.append(contact_force_targets)

                    obs, privileged_obs, encoder_obs, contact_force_targets, rewards, dones, infos = self.env.step(
                        actions, est
                    )
                    force_buf = infos.get("force")
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, encoder_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        encoder_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop
            self.alg.compute_returns(critic_obs, force_buf)
            encoder_obs_batch = torch.stack(encoder_obs_buffer)
            targets_batch = torch.stack(targets_buffer)
            loss_dict = self.alg.update(encoder_obs_batch, targets_batch)

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(it, num_learning_iterations, collection_time, learn_time, loss_dict,
                         ep_infos, rewbuffer, lenbuffer, force_buf)
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, it, num_learning_iterations, collection_time, learn_time, loss_dict,
            ep_infos, rewbuffer, lenbuffer, force_buf, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += collection_time + learn_time
        iteration_time = collection_time + learn_time

        ep_string = ""
        if ep_infos:
            for key in ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in ep_infos:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar(f"Episode/{key}", value, it)
                ep_string += f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (collection_time + learn_time))

        for key, value in loss_dict.items():
            self.writer.add_scalar(f"Loss/{key}", value, it)
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, it)
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), it)
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection_time", collection_time, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)

        if len(rewbuffer) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(rewbuffer), it)
            if force_buf is not None:
                self.writer.add_scalar("Train/mean_force_margin", force_buf.mean().item(), it)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(lenbuffer), it)
        if hasattr(self.env, "stage_stand_mask"):
            self.writer.add_scalar(
                "Train/stage_stand_fraction", self.env.stage_stand_mask.mean().item(), it
            )

        log_string = (
            f"{'#' * width}\n"
            f"{'Learning iteration':>{pad}} {it}/{self.current_learning_iteration + num_learning_iterations}\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning: {learn_time:.3f}s)\n"
        )
        for key, value in loss_dict.items():
            log_string += f"{key:>{pad}} {value:.4f}\n"
        log_string += (
            f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
            f"{ep_string}"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
            f"{'#' * width}\n"
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        """Return a policy callable (obs, encoder_obs) -> actions, where
        encoder_obs is the H-frame proprioceptive observation history."""
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)

        def policy(obs, encoder_obs):
            est = self.alg.actor_critic.encode(encoder_obs)
            act_input = torch.cat((obs, est), dim=-1)
            return self.alg.actor_critic.act_inference(act_input)

        return policy
