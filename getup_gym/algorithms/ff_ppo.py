"""Force-guided PPO with a contact-force estimation encoder (FTSR x FR-Net).

The external guidance force is formulated as an optimizable constraint:
its margin (relu(g_proj + 0.9)) is stored in the rollout buffer and mixed
into the advantage function in the FTSR penalty-function form

    A_mixed = A_reward - beta * A_constraint,

where A_constraint is the GAE of the constraint margin with a zero value
baseline. The contact-force encoder is trained by an auxiliary MSE loss
against the true per-link contact forces (FR-Net style, no teacher-student).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FFPPO:
    """Proximal Policy Optimization with force-guided (constraint) advantages."""

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        encoder_mse_coef=1.0,
        constraint_beta=0.0,
        constraint_gamma=0.99,
        constraint_lam=0.95,
        **kwargs,
    ):
        if kwargs:
            print(f"FFPPO: ignoring unexpected kwargs {list(kwargs.keys())}")
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), learning_rate)

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.normalize_advantage_per_mini_batch = False

        # encoder auxiliary loss
        self.encoder_mse_coef = encoder_mse_coef
        # force-guided (constraint) advantage parameters
        self.constraint_beta = constraint_beta
        self.constraint_gamma = constraint_gamma
        self.constraint_lam = constraint_lam

        self.storage = None
        self.transition = None

    def init_storage(self, num_envs, num_transitions_per_env, obs_shape, critic_obs_shape, actions_shape):
        from getup_gym.storage.rollout_storage import RolloutStorage

        self.num_envs = num_envs
        self.transition = RolloutStorage.Transition()
        # the storage "force" buffer holds the external-force constraint margin
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            obs_shape,
            critic_obs_shape,
            actions_shape,
            teacher_encoder_input_size=None,
            using_force_guidance=True,
            device=self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, force=None):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.force = force
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_force=None):
        """FTSR mixed advantage: A_mixed = A_rew - beta * A_constraint.

        - A_rew: standard GAE of the rewards (value targets stay reward-based)
        - A_constraint: GAE of the external-force constraint margin stored in
          the storage force buffer, with a zero value baseline
        """
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        if last_force is None:
            last_force = torch.zeros(self.num_envs, 1, device=self.device)

        num_steps = self.storage.num_transitions_per_env
        # reward GAE (unnormalized)
        advantage = 0
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = self.storage.values[step + 1]
            next_is_not_terminal = 1.0 - self.storage.dones[step].float()
            delta = (
                self.storage.rewards[step]
                + next_is_not_terminal * self.gamma * next_values
                - self.storage.values[step]
            )
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            self.storage.returns[step] = advantage + self.storage.values[step]
            self.storage.advantages[step] = advantage

        # constraint GAE with zero value baseline (FTSR penalty-function form)
        if self.constraint_beta != 0.0:
            adv_constraint = 0
            for step in reversed(range(num_steps)):
                next_is_not_terminal = 1.0 - self.storage.dones[step].float()
                adv_constraint = (
                    self.storage.force[step]
                    + next_is_not_terminal
                    * self.constraint_gamma
                    * self.constraint_lam
                    * adv_constraint
                )
                self.storage.advantages[step] -= self.constraint_beta * adv_constraint

        # normalize the mixed advantages
        advantages = self.storage.advantages
        self.storage.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def update(self, encoder_obs_buf=None, contact_force_targets=None) -> dict:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0

        generator = self.storage.mini_batch_generator_together(
            self.num_mini_batches, self.num_learning_epochs
        )
        for (
            obs_batch,
            critic_obs_batch,
            _,
            _,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            _,
            _,
        ) in generator:
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL adaptive learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        # auxiliary encoder update: MSE between estimated and true per-link
        # contact forces, over the whole rollout buffer (FR-Net estimation loss)
        encoder_loss = 0.0
        if encoder_obs_buf is not None and contact_force_targets is not None:
            est = self.actor_critic.encode(encoder_obs_buf.flatten(0, 1))
            encoder_loss = self.encoder_mse_coef * F.mse_loss(
                est, contact_force_targets.flatten(0, 1)
            )
            self.optimizer.zero_grad()
            encoder_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            encoder_loss = encoder_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()

        return {
            "mean_value_loss": mean_value_loss / num_updates,
            "mean_surrogate_loss": mean_surrogate_loss / num_updates,
            "encoder_loss": encoder_loss,
        }
