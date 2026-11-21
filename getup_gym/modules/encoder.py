"""Encoder modules for teacher-student distillation."""

import torch
import torch.nn as nn
from torch.distributions import Normal


class TeacherEncoder(nn.Module):
    """Privileged teacher encoder processing full state observations."""

    def __init__(
        self,
        encoder_input_size,
        layer_size,
        encoder_hidden_dims=[256, 128],
        activation="elu",
    ):
        super().__init__()
        self.encoder_input_size = encoder_input_size
        self.layer_size = layer_size
        act = get_activation(activation)

        layers = []
        layers.append(nn.Linear(encoder_input_size, encoder_hidden_dims[0]))
        layers.append(act)
        for i in range(len(encoder_hidden_dims) - 1):
            layers.append(nn.Linear(encoder_hidden_dims[i], encoder_hidden_dims[i + 1]))
            layers.append(act)
        layers.append(nn.Linear(encoder_hidden_dims[-1], layer_size))
        self.encoder = nn.Sequential(*layers)
        Normal.set_default_validate_args = False
        print(f"Teacher Encoder MLP: {self.encoder}")

    def forward(self, obs: torch.Tensor) -> dict:
        zt = self.encoder(obs)
        return {"zt": zt}

    def backward(self, obs: torch.Tensor) -> dict:
        zt = self.encoder(obs)
        return {"zt": zt}



class StudentEncoder(nn.Module):
    """Student encoder with shared backbone and physical prediction heads."""

    def __init__(
        self,
        encoder_input_size,
        layer_size,
        encoder_hidden_dims=[256, 128],
        activation="elu",
    ):
        super().__init__()

        self.encoder_input_size = encoder_input_size
        self.layer_size = layer_size 

        # Shared Backbone: 170 -> 256 -> 128
        hidden_size_1 = encoder_hidden_dims[0]
        hidden_size_2 = encoder_hidden_dims[1]

        self.backbone = nn.Sequential(
            nn.Linear(encoder_input_size, hidden_size_1),
            get_activation(activation),

            nn.Linear(hidden_size_1, hidden_size_2),
            get_activation(activation),
        )

        # Head 1: Teacher latent distillation
        self.latent_head = nn.Linear(
            hidden_size_2,
            layer_size
        )
        
        # ==========================================
        # Physical Prediction Heads
        # Each task has an independent nonlinear MLP.
        # ==========================================

        # Head 2: Base linear velocity
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_size_2, 64),
            get_activation(activation),

            nn.Linear(64, 3)
        )

        # Head 3: Five-dimensional contact probability
        self.contact_head = nn.Sequential(
            nn.Linear(hidden_size_2, 64),
            get_activation(activation),

            nn.Linear(64, 5)
        )

        # Head 4: Left/right wheel slip velocity
        self.slip_head = nn.Sequential(
            nn.Linear(hidden_size_2, 64),
            get_activation(activation),

            nn.Linear(64, 2)
        )

        print("Student Backbone:", self.backbone)
        print("Latent Head:", self.latent_head)
        print("Velocity Head:", self.velocity_head)
        print("Contact Head:", self.contact_head)
        print("Slip Head:", self.slip_head)

    def forward(self, obs: torch.Tensor) -> dict:

        if obs.size(1) != self.encoder_input_size:
            raise ValueError(
                f"Input obs size mismatch: "
                f"expected {self.encoder_input_size}, "
                f"got {obs.size(1)}"
            )

        # Shared feature
        features = self.backbone(obs)

        # Four prediction heads
        zt = self.latent_head(features)

        velocity = self.velocity_head(features)

        contact_logits = self.contact_head(features)

        slip = self.slip_head(features)

        return {
            "zt": zt,
            "features": features,
            "velocity": velocity,
            "contact_logits": contact_logits,
            "slip": slip,
        }

    def backward(self, obs: torch.Tensor) -> dict:
        return self.forward(obs)



def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Invalid activation function: {act_name}")
