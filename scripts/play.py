"""Play / evaluation script for getup_gym."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from getup_gym import GETUP_GYM_ROOT_DIR
from getup_gym.utils.registry import task_registry
from getup_gym.common.helpers import get_args, get_load_path, get_student_encoder_load_path
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import atexit
import traceback

# Shared categorical palette: fixed hue order
_PLOT_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#6250d6", "#e34948",
]

# Toggle: record and plot the two wheel raw actions + body state
# (ang_vel_y, projected_gravity_x). Set to False to run without them.
PLOT_WHEEL_STATE = False


class _LatentProjection(torch.nn.Module):
    """Student encoder's backbone + latent head composed as one module.

    Equivalent to StudentEncoder.forward(obs)["zt"] = latent_head(backbone(obs)),
    but built from the two submodules directly so the traced ONNX graph contains
    no Python control flow: StudentEncoder.forward's input-size check and shape
    asserts would otherwise be baked in during export.
    """

    def __init__(self, student_encoder):
        super().__init__()
        self.backbone = student_encoder.backbone
        self.latent_head = student_encoder.latent_head

    def forward(self, obs):
        return self.latent_head(self.backbone(obs))


class StudentPolicy(torch.nn.Module):
    """Student encoder + actor composed into a single deployable module.

    Mirrors the play() CTS inference path exactly:
    zt = student_encoder(student_obs), actions = act_inference(cat(zt, obs)).

    The encoder's backbone + latent head are used directly (via
    _LatentProjection) so the traced ONNX graph contains no Python control
    flow: StudentEncoder.forward's input-size check would otherwise be baked
    in as a constant and emit a TracerWarning during export.
    """

    def __init__(self, student_encoder, actor_critic):
        super().__init__()
        if hasattr(student_encoder, "encoder"):
            self.encoder = student_encoder.encoder
        else:
            self.encoder = _LatentProjection(student_encoder)
        self.actor_critic = actor_critic

    def forward(self, student_obs, obs):
        zt = self.encoder(student_obs)
        obs_input = torch.cat((zt, obs), dim=-1)
        return self.actor_critic.act_inference(obs_input)


class ActorPolicy(torch.nn.Module):
    """Actor-critic wrapped to its act_inference callable (deterministic mean
    actions), for tasks without a student encoder."""

    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, obs):
        return self.actor_critic.act_inference(obs)


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
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.rewards.curriculum.using_pull_up = False
    env_cfg.terrain.mesh_type = "plane"
    # env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    # env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
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
    return env, obs, student_obs, ppo_runner, policy, encoder_policy, inference_mode, train_cfg


def play(args):
    env, obs, student_obs, ppo_runner, policy, encoder_policy, inference_mode, train_cfg = _prepare_env_and_runner(args)

    print("Running policy...")
    # Camera follows the robot from the side, framing the whole body
    cam_pos_offset = np.array([0.0, 1.5, 0.2])  # [m] lateral offset + height above robot base
    cam_look_offset = np.array([0.0, 0.0, 0.1])  # [m] aim at mid-body height above robot base

    # ==== Real-time plotting setup ====
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 4), sharex=True)
    max_points = 600
    redraw_interval = 5  # render the wheel figure every N steps (~10 Hz)

    # Plot 1: linear velocity (command vs actual)
    cmd_history = deque(maxlen=max_points)
    vel_mean_history = deque(maxlen=max_points)
    step_history = deque(maxlen=max_points)
    line_cmd, = ax1.plot([], [], "r-", label="commands[0,0]")
    line_vel, = ax1.plot([], [], "b-", label="mean base_lin_vel[:,0]")
    ax1.set_ylabel("Linear vel")
    ax1.legend()

    # Plot 2: angular velocity (command vs actual)
    cmd2_history = deque(maxlen=max_points)
    vel2_mean_history = deque(maxlen=max_points)
    line_cmd2, = ax2.plot([], [], "g-", label="commands[0,2]")
    line_vel2, = ax2.plot([], [], "m-", label="mean base_ang_vel[:,2]")
    ax2.set_ylabel("Angular vel")
    ax2.legend()

    # Plot 3: base height (target vs actual)
    cmd_height_history = deque(maxlen=max_points)
    base_height_history = deque(maxlen=max_points)
    line_cmd_h, = ax3.plot([], [], "c-", label="target base_height")
    line_base_h, = ax3.plot([], [], "k-", label="actual base_height")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Height [m]")
    ax3.legend()

    if PLOT_WHEEL_STATE:
        # ==== Two figures: wheel actions + pitch rate, and wheel actions + attitude ====
        # DOF order (tita_description.urdf): left leg 0-2, left wheel 3, right leg 4-6, right wheel 7
        wheel_left_idx, wheel_right_idx = 3, env.num_dof - 1
        if env.num_dof != 8:
            print(
                f"WARNING: expected 8 DOFs (Tita layout), got {env.num_dof}; "
                "wheel action indices may be wrong"
            )
        fig_wheel, ax_wheel = plt.subplots(figsize=(10, 5))
        try:
            fig_wheel.canvas.manager.set_window_title("Wheel actions / ang_vel_y")
        except AttributeError:
            pass
        wheel_t_full = []  # full histories, used for the saved figure and raw data
        wheel_data_full = []
        wheel_histories = [deque(maxlen=max_points) for _ in range(3)]
        wheel_names = ["wheel_L", "wheel_R", "ang_vel_y"]
        wheel_lines = [
            ax_wheel.plot(
                [], [], color=_PLOT_COLORS[k % len(_PLOT_COLORS)], label=wheel_names[k], linewidth=1.2
            )[0]
            for k in range(3)
        ]
        ax_wheel.set_xlabel("Time [s]")
        ax_wheel.legend(ncol=3, fontsize=8)
        ax_wheel.grid(alpha=0.3)
        wheel_axes = [ax_wheel]

        fig_grav, ax_grav = plt.subplots(figsize=(10, 5))
        try:
            fig_grav.canvas.manager.set_window_title("Wheel actions / projected_gravity_x")
        except AttributeError:
            pass
        grav_t_full = []
        grav_data_full = []
        grav_histories = [deque(maxlen=max_points) for _ in range(3)]
        grav_names = ["wheel_L", "wheel_R", "projected_gravity_x"]
        grav_lines = [
            ax_grav.plot(
                [], [], color=_PLOT_COLORS[k % len(_PLOT_COLORS)], label=grav_names[k], linewidth=1.2
            )[0]
            for k in range(3)
        ]
        ax_grav.set_xlabel("Time [s]")
        ax_grav.legend(ncol=3, fontsize=8)
        ax_grav.grid(alpha=0.3)
        grav_axes = [ax_grav]

        # Track manual zooming so autoscale does not override it while inspecting
        wheel_user_zoomed = False
        wheel_last_xlim = None
        grav_user_zoomed = False
        grav_last_xlim = None

        fig_wheel.tight_layout()
        fig_grav.tight_layout()

        # Resolve the loaded run folder up front and register the save hooks, so
        # the plots are written even if play is interrupted (Ctrl+C / exception)
        log_root = os.path.join(GETUP_GYM_ROOT_DIR, "logs", args.task)
        try:
            run_dir = os.path.dirname(
                get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            )
        except Exception:
            print("WARNING: could not resolve the load_run folder; saving to the task log root instead")
            run_dir = log_root
        wheel_saved_flag = [False]
        grav_saved_flag = [False]
        atexit.register(
            _save_play_figure, fig_wheel, wheel_axes, wheel_lines, wheel_t_full, wheel_data_full, run_dir, wheel_saved_flag, "play_wheel"
        )
        atexit.register(
            _save_play_figure, fig_grav, grav_axes, grav_lines, grav_t_full, grav_data_full, run_dir, grav_saved_flag, "play_gravity"
        )

    fig.tight_layout()

    for i in range(10 * int(env.max_episode_length)):
        if inference_mode == "CTS" and encoder_policy is not None:
            zt = encoder_policy(student_obs)["zt"]
            obs_input = torch.cat((zt, obs), dim=-1)
        else:
            obs_input = obs
        actions = policy(obs_input.detach())
        obs, _, teacher_obs, student_obs, rews, dones, infos = env.step(actions.detach())

        # ==== Data collection ====
        cmd_val = env.commands[0, 0].item()
        vel_mean = env.base_lin_vel[:, 0].mean().item()
        cmd2_val = env.commands[0, 2].item()
        vel2_mean = env.base_ang_vel[:, 2].mean().item()
        cmd_height = (
            env.target_height[0].item()
            if hasattr(env, "target_height")
            else env.commands[0, 4].item()
        )
        base_height = env.root_states[0, 2].item()

        cmd_history.append(cmd_val)
        vel_mean_history.append(vel_mean)
        cmd2_history.append(cmd2_val)
        vel2_mean_history.append(vel2_mean)
        cmd_height_history.append(cmd_height)
        base_height_history.append(base_height)
        step_history.append(i)

        # ==== Update plot 1 ====
        line_cmd.set_data(step_history, cmd_history)
        line_vel.set_data(step_history, vel_mean_history)
        ax1.relim()
        ax1.autoscale_view()

        # ==== Update plot 2 ====
        line_cmd2.set_data(step_history, cmd2_history)
        line_vel2.set_data(step_history, vel2_mean_history)
        ax2.relim()
        ax2.autoscale_view()

        # ==== Update plot 3 ====
        line_cmd_h.set_data(step_history, cmd_height_history)
        line_base_h.set_data(step_history, base_height_history)
        ax3.relim()
        ax3.autoscale_view()

        if PLOT_WHEEL_STATE:
            # ==== Record wheel actions and body state (every step) ====
            t_now = i * env.dt
            action_l = actions[0, wheel_left_idx].item()  # raw left wheel action (policy output)
            action_r = actions[0, wheel_right_idx].item()  # raw right wheel action
            ang_vel_y = env.base_ang_vel[0, 1].item()  # body pitch rate
            grav_x = env.projected_gravity[0, 0].item()  # body attitude (gravity x)

            wheel_t_full.append(t_now)
            wheel_data_full.append([action_l, action_r, ang_vel_y])
            grav_t_full.append(t_now)
            grav_data_full.append([action_l, action_r, grav_x])
            for k, val in enumerate(wheel_data_full[-1]):
                wheel_histories[k].append(val)
            for k, val in enumerate(grav_data_full[-1]):
                grav_histories[k].append(val)

            # Track manual zoom so autoscale does not override it while inspecting
            wheel_user_zoomed, _ = _track_zoom(
                wheel_axes, wheel_t_full, max_points, wheel_user_zoomed, wheel_last_xlim
            )
            grav_user_zoomed, _ = _track_zoom(
                grav_axes, grav_t_full, max_points, grav_user_zoomed, grav_last_xlim
            )

            # Render the two figures only every redraw_interval steps, and skip
            # each while zoomed, so box-select zoom interactions stay responsive
            if i % redraw_interval == 0:
                if not wheel_user_zoomed:
                    wheel_last_xlim = _redraw_series(
                        fig_wheel, wheel_axes, wheel_lines, wheel_t_full, wheel_histories, max_points, wheel_last_xlim
                    )
                if not grav_user_zoomed:
                    grav_last_xlim = _redraw_series(
                        fig_grav, grav_axes, grav_lines, grav_t_full, grav_histories, max_points, grav_last_xlim
                    )

        fig.canvas.draw_idle()
        plt.pause(0.001)

        if env.viewer is not None:
            base_pos = env.root_states[0, :3].cpu().numpy()
            env.set_camera(base_pos + cam_pos_offset, base_pos + cam_look_offset)

    # ==== Save the two figures and their raw data to the loaded run folder ====
    if PLOT_WHEEL_STATE:
        _save_play_figure(fig_wheel, wheel_axes, wheel_lines, wheel_t_full, wheel_data_full, run_dir, wheel_saved_flag, "play_wheel")
        _save_play_figure(fig_grav, grav_axes, grav_lines, grav_t_full, grav_data_full, run_dir, grav_saved_flag, "play_gravity")


def _checkpoint_iter(path):
    """Extract the training iteration from a checkpoint filename (model_<iter>.pt)."""
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def _save_play_figure(fig, axes, lines, t_full, data_full, run_dir, saved_flag, name):
    """Save one recorded history (interactive HTML + raster preview + raw data)
    to the loaded run folder. Called after the play loop, and registered with
    atexit as a fallback for interrupted runs."""
    if saved_flag[0] or not t_full:
        return
    try:
        save_ts = time.strftime("%Y%m%d_%H%M%S")
        html_path = os.path.join(run_dir, f"{name}_{save_ts}.html")
        png_path = os.path.join(run_dir, f"{name}_{save_ts}.png")
        data_path = os.path.join(run_dir, f"{name}_data_{save_ts}.npz")

        # Re-plot from the full history so the saved figure covers the whole run
        data_arr = np.asarray(data_full)
        for k, line in enumerate(lines):
            line.set_data(t_full, data_arr[:, k])
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
            ax.minorticks_on()
            ax.grid(which="minor", alpha=0.15)

        # PNG is the quick raster preview; the raw data holds the exact values
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        np.savez(
            data_path,
            t=np.asarray(t_full),
            **{line.get_label(): data_arr[:, k] for k, line in enumerate(lines)},
        )
        saved_flag[0] = True
        print(f"Saved '{name}' plot (raster preview) to: {png_path}")
        print(f"Saved '{name}' raw data to: {data_path}")
    except Exception:
        print(f"Failed to save the '{name}' plot:")
        traceback.print_exc()

    # The interactive HTML needs plotly, so keep it separate from the saves above
    try:
        _write_interactive_html(html_path, t_full, data_full, axes)
        print(f"Saved '{name}' interactive plot (box-select zoom) to: {html_path}")
    except Exception:
        print("Failed to save the interactive HTML plot (is plotly installed?):")
        traceback.print_exc()


def _write_interactive_html(html_path, t_full, data_full, axes):
    """Write a self-contained interactive HTML of one recorded history.

    Open in a browser: drag a box around a curve segment to zoom the chart to it
    (the x-range is shared across the panels, the y-range follows the selection),
    hover a curve for its exact value, double-click to reset the view."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = np.asarray(t_full)
    data = np.asarray(data_full)
    groups = [(ax.get_ylabel(), len(ax.get_lines())) for ax in axes]

    fig = make_subplots(
        rows=len(groups),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[name for name, _ in groups] if len(groups) > 1 else None,
        vertical_spacing=0.08,
    )
    dim = 0
    for row, (name, n) in enumerate(groups, start=1):
        for k in range(n):
            label = axes[row - 1].get_lines()[k].get_label()
            fig.add_trace(
                go.Scattergl(
                    x=t,
                    y=data[:, dim],
                    mode="lines",
                    name=label,
                    line=dict(width=1.2, color=_PLOT_COLORS[dim % len(_PLOT_COLORS)]),
                ),
                row=row,
                col=1,
            )
            dim += 1
        fig.update_yaxes(title_text=name, row=row, col=1)
    fig.update_xaxes(title_text="Time [s]", row=len(groups), col=1)
    fig.update_layout(
        height=320 * len(groups) + 150,
        dragmode="zoom",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=70, r=20, t=40, b=50),
        showlegend=False,
    )
    fig.write_html(html_path, include_plotlyjs=True)


def _track_zoom(axes, t_full, max_points, user_zoomed, last_xlim):
    """Detect manual zoom on a shared-x figure: set user_zoomed when the view
    changes and reset it when the user zooms back out to the live window, so
    autoscale does not override the view while inspecting."""
    disp_t0 = t_full[-max_points] if len(t_full) > max_points else t_full[0]
    if not user_zoomed:
        if last_xlim is not None and axes[0].get_xlim() != last_xlim:
            user_zoomed = True  # user panned/zoomed the view
    elif (
        axes[0].get_xlim()[0] <= disp_t0
        and axes[0].get_xlim()[1] >= t_full[-1]
    ):
        user_zoomed = False  # zoomed back out: resume autoscale
    return user_zoomed, last_xlim


def _redraw_series(fig, axes, lines, t_full, histories, max_points, last_xlim):
    """Refresh a live figure from the rolling window and return the new xlim."""
    for k, line in enumerate(lines):
        line.set_data(t_full[-max_points:], histories[k])
    for ax in axes:
        ax.relim()
        ax.autoscale_view()
    last_xlim = axes[0].get_xlim()
    fig.canvas.draw_idle()
    return last_xlim


def export_policy_as_onnx(args):
    """Export the deployed policy to ONNX, mirroring the play() inference path.

    CTS tasks export StudentPolicy (student encoder + actor, two inputs), other
    tasks export the actor-only policy (one input). The composed modules are the
    same ones returned by get_inference_policy / get_encoder_inference_policy,
    set to eval() and traced on CPU with no Python control flow.
    """
    env, obs, student_obs, ppo_runner, policy, encoder_policy, inference_mode, train_cfg = _prepare_env_and_runner(args)

    # Report the exact checkpoints that were loaded, so mixed versions don't go unnoticed
    log_root = os.path.join(GETUP_GYM_ROOT_DIR, "logs", args.task)
    actor_path = get_load_path(
        log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint
    )
    actor_iter = _checkpoint_iter(actor_path)
    print(f"Actor loaded from: {actor_path} (iter {actor_iter})")
    if inference_mode == "CTS" and ppo_runner.student_encoder is not None:
        student_path = get_student_encoder_load_path(
            log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint
        )
        student_iter = _checkpoint_iter(student_path)
        print(f"Student encoder loaded from: {student_path} (iter {student_iter})")
        if actor_iter != student_iter:
            print(
                f"WARNING: actor (iter {actor_iter}) and student encoder (iter {student_iter}) "
                "checkpoints differ; the exported model mixes two training iterations."
            )

    export_dir = os.path.join(GETUP_GYM_ROOT_DIR, "logs", args.task, "exported", "policies")
    os.makedirs(export_dir, exist_ok=True)

    if inference_mode == "CTS" and encoder_policy is not None and student_obs is not None:
        model = StudentPolicy(ppo_runner.student_encoder, ppo_runner.alg.actor_critic)
        onnx_path = os.path.join(export_dir, "student_policy.onnx")
        dummy_inputs = (student_obs.detach().cpu(), obs.detach().cpu())
        input_names = ["student_obs", "obs"]
        dynamic_axes = {
            "student_obs": {0: "batch_size"},
            "obs": {0: "batch_size"},
            "actions": {0: "batch_size"},
        }
    else:
        model = ActorPolicy(ppo_runner.alg.actor_critic)
        onnx_path = os.path.join(export_dir, "policy.onnx")
        dummy_inputs = (obs.detach().cpu(),)
        input_names = ["obs"]
        dynamic_axes = {"obs": {0: "batch_size"}, "actions": {0: "batch_size"}}

    model.eval()
    model.cpu()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            onnx_path,
            input_names=input_names,
            output_names=["actions"],
            dynamic_axes=dynamic_axes,
            opset_version=11,
            do_constant_folding=True,
        )
    print(f"Exported policy (inference_mode={inference_mode}) to: {onnx_path}")


if __name__ == "__main__":
    args = get_args()
    if args.export_onnx:
        export_policy_as_onnx(args)
    else:
        play(args) 