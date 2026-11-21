
"""
Physics-assisted Student Encoder training for B1.

Four supervised tasks:
1. Teacher latent distillation
2. Base linear velocity prediction
3. Five-dimensional contact prediction
4. Left/right wheel slip velocity prediction

The original EncoderMSE and EncoderStorage are reused.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from getup_gym.algorithms.encoder_mse import EncoderMSE


class PhysicsAssistedMSE(EncoderMSE):

    def __init__(
        self,
        *args,
        velocity_loss_coef=1.0,
        contact_loss_coef=1.0,
        slip_loss_coef=1.0,
        velocity_mean=None,
        velocity_std=None,
        slip_mean=None,
        slip_std=None,
        grad_log_interval=1,
        **kwargs,
    ):

        # Reuse original EncoderMSE initialization:
        # EncoderStorage, optimizer, Teacher, Student and PPO-independent settings.
        super().__init__(*args, **kwargs)

        self.velocity_loss_coef = float(velocity_loss_coef)
        self.contact_loss_coef = float(contact_loss_coef)
        self.slip_loss_coef = float(slip_loss_coef)

        self.grad_log_interval = int(grad_log_interval)

        if self.grad_log_interval < 1:
            raise ValueError("grad_log_interval must be >= 1")

        # Fixed normalization statistics.
        # These must be calculated from an independent calibration dataset.
        self.velocity_mean = self._make_stat(
            velocity_mean, 3
        )
        self.velocity_std = self._make_stat(
            velocity_std, 3
        )

        self.slip_mean = self._make_stat(
            slip_mean, 2
        )
        self.slip_std = self._make_stat(
            slip_std, 2
        )

        if torch.any(self.velocity_std <= 0):
            raise ValueError("velocity_std must be positive")

        if torch.any(self.slip_std <= 0):
            raise ValueError("slip_std must be positive")

        self.velocity_loss_fn = nn.SmoothL1Loss()
        self.contact_loss_fn = nn.BCEWithLogitsLoss()
        self.slip_loss_fn = nn.SmoothL1Loss()

        self.update_count = 0

        print("PhysicsAssistedMSE initialized")
        print("Velocity loss coef:", self.velocity_loss_coef)
        print("Contact loss coef:", self.contact_loss_coef)
        print("Slip loss coef:", self.slip_loss_coef)
        print("Gradient log interval:", self.grad_log_interval)

        print("Velocity mean:", self.velocity_mean.tolist())
        print("Velocity std:", self.velocity_std.tolist())
        print("Slip mean:", self.slip_mean.tolist())
        print("Slip std:", self.slip_std.tolist())

    def _make_stat(self, value, dim):

        if value is None:
            raise ValueError(
                "B1 normalization statistics cannot be None."
            )

        tensor = torch.as_tensor(
            value,
            dtype=torch.float32,
            device=self.device,
        )

        if tensor.shape != (dim,):
            raise ValueError(
                f"Expected shape ({dim},), "
                f"got {tuple(tensor.shape)}"
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                "Normalization statistics contain NaN or Inf."
            )
        if torch.any(self.velocity_std <= 0):
            raise ValueError("velocity_std must be positive")

        if torch.any(self.slip_std <= 0):
            raise ValueError("slip_std must be positive")

        return tensor


    # ---------------------------------------------------------
    # 1. Compute four task losses
    # ---------------------------------------------------------

    def compute_losses(
        self,
        student_obs_batch,
        teacher_obs_batch,
        zt_t_batch,
    ):

        # Student forward.
        output = self.encoder(student_obs_batch)

        required_keys = [
            "zt",
            "features",
            "velocity",
            "contact_logits",
            "slip",
        ]

        for key in required_keys:
            if key not in output:
                raise KeyError(
                    f"StudentEncoder output missing key: {key}"
                )

        # Teacher latent target.
        with torch.no_grad():

            if teacher_obs_batch is not None:
                zt_target = self.teacher_encoder.backward(
                    teacher_obs_batch
                )["zt"]
            else:
                zt_target = zt_t_batch

            zt_target = zt_target.detach()

        # Physical labels from the 20-dimensional Teacher observation.
        #
        # 0:3   base linear velocity
        # 3:6   left wheel contact force
        # 6:9   right wheel contact force
        # 9:10  base height
        # 10:15 five contact states
        # 15:18 contact loads
        # 18:20 left/right wheel slip velocity

        if teacher_obs_batch is None:
            raise ValueError(
                "B1 physical supervision requires teacher observations."
            )

        velocity_target = teacher_obs_batch[:, 0:3]
        contact_target = teacher_obs_batch[:, 10:15]
        slip_target = teacher_obs_batch[:, 18:20]

        # Identify invalid all-zero Teacher observations.
        # The environment can zero the entire Teacher observation
        # during the configured delayed-action/unactuated period.
        valid_mask = (
            teacher_obs_batch.abs().sum(dim=-1) > 1e-8
        )

        if not valid_mask.any():
            raise RuntimeError(
                "No valid Teacher samples in this mini-batch."
            )

        # Validate physical targets.
        if not torch.isfinite(teacher_obs_batch).all():
            raise ValueError(
                "Teacher observation contains NaN or Inf."
            )

        if not torch.all(
            (contact_target >= 0.0) &
            (contact_target <= 1.0)
        ):
            raise ValueError(
                "Contact targets must be in [0, 1]. "
                "Check _get_five_contact_states()."
            )

        # Fixed normalization.
        velocity_target = (
            velocity_target - self.velocity_mean
        ) / self.velocity_std

        slip_target = (
            slip_target - self.slip_mean
        ) / self.slip_std

        # Apply the same validity mask to all four tasks.
        zt_pred = output["zt"][valid_mask]
        zt_target = zt_target[valid_mask]

        velocity_pred = output["velocity"][valid_mask]
        velocity_target = velocity_target[valid_mask]

        contact_logits = output["contact_logits"][valid_mask]
        contact_target = contact_target[valid_mask]

        slip_pred = output["slip"][valid_mask]
        slip_target = slip_target[valid_mask]
        # 只在对应轮子实际接触地面时
        # 计算该轮子的滑移速度损失

        wheel_contact_mask = (contact_target[:, 3:5] > 0.5)

        slip_loss_element = F.smooth_l1_loss(
            slip_pred,
            slip_target,
            reduction="none"
        )

        slip_mask_float = wheel_contact_mask.float()

        # Four losses.
        loss_z = F.mse_loss(
            zt_pred,
            zt_target,
        )

        loss_v = self.velocity_loss_fn(
            velocity_pred,
            velocity_target,
        )

        loss_c = self.contact_loss_fn(
            contact_logits,
            contact_target,
        )

        loss_s = (
            (slip_loss_element * slip_mask_float).sum()
            / slip_mask_float.sum().clamp_min(1.0)
        )

        valid_slip_count = slip_mask_float.sum()

        slip_valid_ratio = (
            valid_slip_count
            / slip_mask_float.numel()
        )
        return {
            "loss_z": loss_z,
            "loss_v": loss_v,
            "loss_c": loss_c,
            "loss_s": loss_s,

            "slip_valid_count": valid_slip_count.detach(),
            "slip_valid_ratio": slip_valid_ratio.detach(),
        }

    # ---------------------------------------------------------
    # 2. Extract shared Backbone gradients for each task
    # ---------------------------------------------------------

    def compute_task_gradients(self, losses):

        backbone_params = tuple(
            p for p in self.encoder.backbone.parameters()
            if p.requires_grad
        )

        if len(backbone_params) == 0:
            raise RuntimeError(
                "No trainable parameters found in Student Backbone."
            )

        task_names = [
            "latent",
            "velocity",
            "contact",
            "slip",
        ]

        task_grads = {}

        for index, name in enumerate(task_names):

            loss = losses[
                ["loss_z", "loss_v", "loss_c", "loss_s"][index]
            ]

            grads = torch.autograd.grad(
                loss,
                backbone_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

            flat_grads = []

            for grad, param in zip(grads, backbone_params):

                if grad is None:
                    flat_grads.append(
                        torch.zeros_like(param).reshape(-1)
                    )
                else:
                    flat_grads.append(
                        grad.detach().reshape(-1)
                    )

            task_grads[name] = torch.cat(flat_grads)

        return task_grads

    # ---------------------------------------------------------
    # 3. Calculate gradient norms and cosine similarities
    # ---------------------------------------------------------

    def compute_gradient_metrics(self, task_grads):

        names = [
            "latent",
            "velocity",
            "contact",
            "slip",
        ]

        metrics = {}

        # Four gradient norms.
        for name in names:

            grad = task_grads[name]

            metrics[
                f"grad_norm_{name}"
            ] = torch.norm(
                grad,
                p=2,
            ).item()

        # Six pairwise cosine similarities.
        pairs = [
            ("latent", "velocity"),
            ("latent", "contact"),
            ("latent", "slip"),
            ("velocity", "contact"),
            ("velocity", "slip"),
            ("contact", "slip"),
        ]

        for name_a, name_b in pairs:

            grad_a = task_grads[name_a]
            grad_b = task_grads[name_b]

            cosine = F.cosine_similarity(
                grad_a.unsqueeze(0),
                grad_b.unsqueeze(0),
                dim=1,
                eps=1e-8,
            )

            metrics[
                f"cos_{name_a}_{name_b}"
            ] = cosine.item()

        return metrics

    # ---------------------------------------------------------
    # 4. Joint training update
    # ---------------------------------------------------------

    def update(self):

        self.update_count += 1

        loss_sums = {
            "mean_mse_loss": 0.0,
            "mean_velocity_loss": 0.0,
            "mean_contact_loss": 0.0,
            "mean_slip_loss": 0.0,

            "mean_slip_valid_ratio": 0.0,
            "mean_slip_valid_count": 0.0,
        }

        gradient_sums = {}
        gradient_count = 0

        generator = self.encoder_storage.mini_batch_generator(
            self.num_mini_batch,
            self.num_learning_epochs,
        )

        num_updates = 0

        for batch_index, batch in enumerate(generator):

            zt_t_batch, student_obs_batch, teacher_obs_batch = batch

            # Compute all four losses.
            losses = self.compute_losses(
                student_obs_batch,
                teacher_obs_batch,
                zt_t_batch,
            )

            loss_z = losses["loss_z"]
            loss_v = losses["loss_v"]
            loss_c = losses["loss_c"]
            loss_s = losses["loss_s"]

            # Gradient diagnostics:
            # First mini-batch of this iteration only.
            if (
                batch_index == 0
                and self.update_count % self.grad_log_interval == 0
            ):

                task_grads = self.compute_task_gradients(
                    losses
                )

                metrics = self.compute_gradient_metrics(
                    task_grads
                )

                for key, value in metrics.items():
                    gradient_sums[key] = (
                        gradient_sums.get(key, 0.0) + value
                    )

                gradient_count += 1

            # Joint loss.
            total_loss = (
                self.mse_loss_coef * loss_z
                + self.velocity_loss_coef * loss_v
                + self.contact_loss_coef * loss_c
                + self.slip_loss_coef * loss_s
            )

            # Optimize Student only.
            self.student_optimizer.zero_grad()

            total_loss.backward()

            nn.utils.clip_grad_norm_(
                self.encoder.parameters(),
                self.max_grad_norm,
            )

            self.student_optimizer.step()

            # Statistics.
            loss_sums["mean_mse_loss"] += loss_z.detach().item()
            loss_sums["mean_velocity_loss"] += loss_v.detach().item()
            loss_sums["mean_contact_loss"] += loss_c.detach().item()
            loss_sums["mean_slip_loss"] += loss_s.detach().item()
            loss_sums["mean_slip_valid_ratio"] += (losses["slip_valid_ratio"].item())
            loss_sums["mean_slip_valid_count"] += (losses["slip_valid_count"].item())

            num_updates += 1

        if num_updates == 0:
            raise RuntimeError(
                "No mini-batches were generated. "
                "Check EncoderStorage and mini-batch configuration."
            )

        # Average losses.
        for key in loss_sums:
            loss_sums[key] /= num_updates

        # Average gradient metrics.
        if gradient_count > 0:
            for key, value in gradient_sums.items():
                gradient_sums[key] = value / gradient_count

        # Clear the original EncoderStorage.
        self.encoder_storage.clear()

        result = dict(loss_sums)
        result.update(gradient_sums)

        return result
