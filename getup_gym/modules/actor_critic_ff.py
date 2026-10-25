"""Actor-Critic with a contact-force estimation encoder (FR-Net style, no teacher-student)."""

import torch
import torch.nn as nn
from torch.distributions import Normal

from getup_gym.modules.actor_critic import get_activation


class ActorCriticFF(nn.Module):
    """Actor-Critic with an encoder that estimates per-link contact forces from
    a proprioceptive observation history (FR-Net Mass-Contact-Predictor analogue,
    predicting contact forces only).

    - encoder: obs history (H frames) -> estimated per-link contact forces
               (normalized magnitudes, non-negative via softplus)
    - actor:   [proprioceptive obs, estimated contact forces] -> action distribution
    - critic:  privileged observations -> value (asymmetric actor-critic)
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_contact_links=9,
        encoder_input_size=170,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[256, 128],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(f"ActorCriticFF: ignoring unexpected args {list(kwargs.keys())}")
        super().__init__()

        act = get_activation(activation)

        self.encoder_input_size = encoder_input_size
        self.num_contact_links = num_contact_links
        self.mlp_input_dim_a = num_actor_obs
        self.mlp_input_dim_c = num_critic_obs
        self.num_actions = num_actions

        # Contact-force encoder: obs history -> per-link contact force magnitudes
        enc_layers = [nn.Linear(encoder_input_size, encoder_hidden_dims[0]), act]
        for l in range(len(encoder_hidden_dims) - 1):
            enc_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
            enc_layers.append(act)
        enc_layers.append(nn.Linear(encoder_hidden_dims[-1], num_contact_links))
        enc_layers.append(nn.Softplus())
        self.encoder = nn.Sequential(*enc_layers)

        # Policy: [obs, estimated contact forces] -> actions
        actor_layers = []
        actor_layers.append(nn.Linear(self.mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(act)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(act)
        self.actor = nn.Sequential(*actor_layers)

        # Value function: privileged observations -> value
        critic_layers = []
        critic_layers.append(nn.Linear(self.mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(act)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(act)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Contact-Force Encoder MLP: {self.encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def encode(self, encoder_obs):
        """Estimate per-link contact forces (normalized magnitudes) from the
        H-frame observation history."""
        return self.encoder(encoder_obs)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        mean = mean.clamp(-1000, 1000)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def enforce_minimum_std(self, min_std):
        self.std.data = torch.max(self.std.detach(), min_std.detach())
