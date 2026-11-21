
import torch
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from getup_gym.modules.encoder import StudentEncoder


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # A组当前基线输入维度
    input_dim = 170

    # Teacher latent dimension
    latent_dim = 12

    student_encoder = StudentEncoder(
        encoder_input_size=input_dim,
        layer_size=latent_dim,
        encoder_hidden_dims=[256, 128],
        activation="elu",
    ).to(device)

    # Simulate a batch of observations
    x = torch.randn(
        32,
        input_dim,
        device=device
    )

    # Forward
    output = student_encoder(x)

    print("\n===== B1 Student Encoder Test =====")

    for key, value in output.items():
        print(key, value.shape)

    # Dimension checks
    assert output["features"].shape == (32, 128)
    assert output["zt"].shape == (32, 12)
    assert output["velocity"].shape == (32, 3)
    assert output["contact_logits"].shape == (32, 5)
    assert output["slip"].shape == (32, 2)

    # Gradient test
    loss = (
        output["zt"].pow(2).mean()
        + output["velocity"].pow(2).mean()
        + output["contact_logits"].pow(2).mean()
        + output["slip"].pow(2).mean()
    )

    loss.backward()

    assert student_encoder.backbone[0].weight.grad is not None
    assert student_encoder.latent_head.weight.grad is not None
    assert student_encoder.velocity_head.weight.grad is not None
    assert student_encoder.contact_head.weight.grad is not None
    assert student_encoder.slip_head.weight.grad is not None

    print("\nAll dimension and gradient tests passed!")


if __name__ == "__main__":
    main()
