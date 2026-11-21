
import torch

from getup_gym.modules.encoder import (
    StudentEncoder,
    TeacherEncoder,
)

from getup_gym.algorithms.physics_assisted_mse import (
    PhysicsAssistedMSE,
)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    num_envs = 32
    num_steps = 24

    # Student: 170 -> 256 -> 128 -> 12
    student = StudentEncoder(
        encoder_input_size=170,
        layer_size=12,
        encoder_hidden_dims=[256, 128],
        activation="elu",
    ).to(device)

    # Teacher: 20 -> 256 -> 128 -> 12
    teacher = TeacherEncoder(
        encoder_input_size=20,
        layer_size=12,
        encoder_hidden_dims=[256, 128],
        activation="elu",
    ).to(device)

    alg = PhysicsAssistedMSE(
        num_transition_per_env=num_steps,
        num_envs=num_envs,
        encoder=student,
        teacher_encoder=teacher,
        device=device,

        learning_rate=1e-3,
        num_mini_batches=4,
        num_learning_epochs=2,
        max_grad_norm=0.8,
        mse_loss_coef=1.0,

        velocity_loss_coef=1.0,
        contact_loss_coef=1.0,
        slip_loss_coef=1.0,
    )

    # Fill EncoderStorage with synthetic samples.
    for _ in range(num_steps):

        student_obs = torch.randn(
            num_envs, 170, device=device
        )

        teacher_obs = torch.randn(
            num_envs, 20, device=device
        )

        # Contact labels must be binary.
        teacher_obs[:, 10:15] = torch.randint(
            0, 2, (num_envs, 5), device=device
        ).float()

        zt = teacher(teacher_obs)["zt"]

        alg.act(
            zt,
            student_obs,
            teacher_obs,
        )

    # Execute one complete B1 update.
    result = alg.update()

    print("\n===== B1 Algorithm Test =====")

    for key, value in result.items():
        print(f"{key}: {value:.6f}")

    expected_keys = [
        "mean_mse_loss",
        "mean_velocity_loss",
        "mean_contact_loss",
        "mean_slip_loss",
        "grad_norm_latent",
        "grad_norm_velocity",
        "grad_norm_contact",
        "grad_norm_slip",
        "cos_latent_velocity",
        "cos_latent_contact",
        "cos_latent_slip",
        "cos_velocity_contact",
        "cos_velocity_slip",
        "cos_contact_slip",
    ]

    for key in expected_keys:
        assert key in result, f"Missing metric: {key}"

        assert torch.isfinite(
            torch.tensor(result[key])
        ), f"Invalid metric: {key}"

    print("\nAll B1 algorithm tests passed!")


if __name__ == "__main__":
    main()
