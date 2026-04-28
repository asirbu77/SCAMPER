"""Wandb logging utilities for arena forage task transfer training.

This module provides functions for logging training metrics, rollout videos,
scent heatmap animations, and reward breakdowns to Weights & Biases.
"""

import logging
from typing import Any, Callable

import imageio
import jax
import matplotlib
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import wandb
from jax import numpy as jnp
from matplotlib import cm as mpl_cm

matplotlib.use("Agg")


def rollout_logging_fn(
    env: Any,
    jit_reset: Callable,
    jit_step: Callable,
    jit_inference_fn: Callable,
    params: Any,
    current_step: int,
    model_path: str,
    episode_length: int,
    render_camera: str = "close_profile-rodent",
    render_fps: int = 50,
    render_height: int = 480,
    render_width: int = 640,
    render_every: int = 1,
    eval_count: int = 0,
    scent_grid_resolution: int = 60,
) -> None:
    """Log evaluation rollout video and metrics to wandb.

    Runs a full episode using the current policy, logging per-step reward
    breakdowns and a rendered video of the rollout.

    Args:
        env: Environment instance with render() method.
        jit_reset: JIT-compiled environment reset function.
        jit_step: JIT-compiled environment step function.
        jit_inference_fn: JIT-compiled inference function with signature
            (params, obs, rng) -> (action, extras).
        params: Policy parameters for inference.
        current_step: Current training step (used for video filename).
        model_path: Directory path for saving video files.
        episode_length: Number of steps to run in the rollout.
        render_camera: Camera name for rendering.
        render_fps: Frames per second for the output video.
        render_height: Height in pixels for the output video.
        render_width: Width in pixels for the output video.
        render_every: Render video every N evaluation steps.
        eval_count: Current evaluation count (0-indexed).

    Note:
        All metrics are logged with commit=False to batch with other logs.
    """
    rng = jax.random.PRNGKey(current_step)
    reset_rng, act_rng = jax.random.split(rng)

    state = jit_reset(reset_rng)
    rollout = [state]

    for _ in range(episode_length):
        _, act_rng = jax.random.split(act_rng)
        action, _ = jit_inference_fn(params, state.obs, act_rng)
        state = jit_step(state, action)
        rollout.append(state)
        if state.done:
            break

    # Log reward breakdown per step
    _log_reward_metrics(rollout, current_step)
    _log_gate_metrics(rollout, current_step)

    # Render video on first eval and every render_every evals thereafter
    should_render = eval_count == 0 or eval_count % render_every == 0
    if should_render:
        _log_rollout_video(
            env,
            model_path,
            current_step,
            rollout,
            render_camera,
            render_fps,
            render_height,
            render_width,
        )
        _log_scent_heatmap_video(
            env,
            model_path,
            current_step,
            rollout,
            render_fps,
            grid_resolution=scent_grid_resolution,
        )


def _log_reward_metrics(rollout: list[Any], current_step: int) -> None:
    """Log per-step reward metrics as line plots with step-based versioning.

    Creates a combined table with all reward metrics that can be viewed
    with a slider in WandB to compare different evaluation steps.

    Args:
        rollout: List of environment states from the episode.
        current_step: Current training step for versioning.
    """
    # Get available reward metrics from first state
    if not rollout or not hasattr(rollout[0], "metrics"):
        return

    first_metrics = rollout[0].metrics
    reward_keys = sorted([k for k in first_metrics.keys() if k.startswith("rewards/")])

    if not reward_keys:
        return

    # Build combined table with all reward metrics
    columns = ["frame"] + reward_keys
    data = []
    for i, state in enumerate(rollout):
        row = [i] + [float(state.metrics.get(k, 0.0)) for k in reward_keys]
        data.append(row)

    try:
        table = wandb.Table(data=data, columns=columns)
        # Log as versioned table that WandB can display with a step slider
        wandb.log(
            {"eval/reward_curves": table},
            commit=False,
        )
        # Also log individual line plots for each metric
        for metric_name in reward_keys:
            wandb.log(
                {
                    f"eval/{metric_name}": wandb.plot.line(
                        table,
                        "frame",
                        metric_name,
                        title=f"{metric_name} (step {current_step})",
                    )
                },
                commit=False,
            )
    except Exception as e:
        logging.warning(f"Failed to log reward metrics: {e}")


def _log_gate_metrics(rollout: list[Any], current_step: int) -> None:
    """Log per-step gate metrics as line plots.

    Creates plots showing gate value and gate logit over the episode,
    useful for visualizing how quickly the gate switches between
    RGM and policy+prior during evaluation.

    Args:
        rollout: List of environment states from the episode.
        current_step: Current training step for versioning.
    """
    if not rollout or not hasattr(rollout[0], "metrics"):
        return

    first_metrics = rollout[0].metrics
    gate_keys = sorted([k for k in first_metrics.keys() if k.startswith("gate")])

    if not gate_keys:
        return

    columns = ["frame"] + gate_keys
    data = []
    for i, state in enumerate(rollout):
        row = [i] + [float(state.metrics.get(k, 0.0)) for k in gate_keys]
        data.append(row)

    try:
        table = wandb.Table(data=data, columns=columns)
        for metric_name in gate_keys:
            wandb.log(
                {
                    f"eval/{metric_name}_curve": wandb.plot.line(
                        table,
                        "frame",
                        metric_name,
                        title=f"{metric_name} (step {current_step})",
                    )
                },
                commit=False,
            )
    except Exception as e:
        logging.warning(f"Failed to log gate metrics: {e}")


def _log_rollout_video(
    env: Any,
    model_path: str,
    current_step: int,
    rollout: list[Any],
    render_camera: str,
    render_fps: int,
    render_height: int = 480,
    render_width: int = 640,
) -> None:
    """Render rollout video and log to wandb.

    Args:
        env: Environment with render() method.
        model_path: Directory to save video file.
        current_step: Training step for filename.
        rollout: List of environment states from the episode.
        render_camera: Camera name for rendering.
        render_fps: Frames per second for output video.
        render_height: Height in pixels for output video.
        render_width: Width in pixels for output video.
    """
    video_path = f"{model_path}/{current_step}.mp4"

    try:
        # Use render_camera directly - it should already include any suffix
        # (e.g., "close_profile-rodent" from config)
        with imageio.get_writer(video_path, fps=render_fps) as writer:
            frames = env.render(
                rollout,
                camera=render_camera,
                height=render_height,
                width=render_width,
            )
            for frame in frames:
                writer.append_data(frame)

        wandb.log(
            {"videos/rollout": wandb.Video(video_path, format="mp4")},
            commit=False,
        )
    except mujoco.FatalError as e:
        logging.warning(f"Rendering video failed with MuJoCo error: {e}")
    except Exception as e:
        logging.warning(f"Failed to log rollout video: {e}")


def _log_scent_heatmap_video(
    env: Any,
    model_path: str,
    current_step: int,
    rollout: list[Any],
    render_fps: int,
    grid_resolution: int = 60,
) -> None:
    """Render a 2D scent heatmap animation with rodent trajectory and log to wandb.

    Each frame shows the current scent field (accounting for collected treats),
    treat positions (blue=active, gray=collected), and the rodent's root
    position as a red dot with a trailing path.

    Args:
        env: ArenaForage environment (possibly wrapped).
        model_path: Directory to save video file.
        current_step: Training step for filename.
        rollout: List of environment states from the episode.
        render_fps: Frames per second for output video.
        grid_resolution: Number of grid cells per axis for the heatmap.
    """
    video_path = f"{model_path}/{current_step}_heatmap.mp4"

    try:
        # Get env config through wrapper chain
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        cfg = base_env._config

        num_treats = cfg.num_treats
        sigma = cfg.scent_sigma
        spawn_radius = cfg.spawn_radius
        extent = spawn_radius * 1.5

        # Pre-compute scent grids per treat
        grid_1d = np.linspace(-extent, extent, grid_resolution)
        gx, gy = np.meshgrid(grid_1d, grid_1d)

        treat_pos = np.array(rollout[0].info["treat_positions"])
        scent_per_treat = np.zeros((num_treats, grid_resolution, grid_resolution))
        for t in range(num_treats):
            dist2 = (gx - treat_pos[t, 0]) ** 2 + (gy - treat_pos[t, 1]) ** 2
            scent_per_treat[t] = np.exp(-dist2 / (2 * sigma**2))

        # Extract rodent root xy positions from qpos[0:2]
        rodent_xy = np.array([np.array(s.data.qpos[:2]) for s in rollout])

        # Determine global vmax for consistent colorbar
        vmax = max(float(scent_per_treat.sum(axis=0).max()), 1.0)

        fig, ax = plt.subplots(figsize=(4, 4), dpi=128)

        with imageio.get_writer(video_path, fps=render_fps) as writer:
            for i, state in enumerate(rollout):
                ax.clear()

                # Compute scent field for active treats
                collected = np.array(state.info["treat_collected"])
                active = ~collected
                if active.any():
                    scent_grid = np.sum(scent_per_treat[active], axis=0)
                else:
                    scent_grid = np.zeros_like(gx)

                ax.imshow(
                    scent_grid,
                    extent=[-extent, extent, -extent, extent],
                    origin="lower",
                    cmap="viridis",
                    vmin=0,
                    vmax=vmax,
                    aspect="equal",
                )

                # Plot treat positions
                for t in range(num_treats):
                    color = "gray" if collected[t] else "dodgerblue"
                    marker = "x" if collected[t] else "o"
                    ax.plot(
                        treat_pos[t, 0],
                        treat_pos[t, 1],
                        marker=marker,
                        color=color,
                        markersize=7,
                        markeredgewidth=2,
                    )

                # Spawn radius circle
                theta = np.linspace(0, 2 * np.pi, 100)
                ax.plot(
                    spawn_radius * np.cos(theta),
                    spawn_radius * np.sin(theta),
                    "w--",
                    alpha=0.4,
                    linewidth=0.8,
                )

                # Rodent trajectory trail and current position
                trail = rodent_xy[: i + 1]
                ax.plot(trail[:, 0], trail[:, 1], "r-", alpha=0.4, linewidth=0.8)
                ax.plot(
                    rodent_xy[i, 0],
                    rodent_xy[i, 1],
                    "ro",
                    markersize=6,
                )

                ax.set_xlim(-extent, extent)
                ax.set_ylim(-extent, extent)
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.set_title(f"Scent Field — frame {i}")

                fig.tight_layout()
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                frame = np.asarray(buf)[:, :, :3]
                writer.append_data(frame)

        plt.close(fig)

        wandb.log(
            {"videos/scent_heatmap": wandb.Video(video_path, format="mp4")},
            commit=False,
        )
    except Exception as e:
        logging.warning(f"Failed to log scent heatmap video: {e}")


def wandb_progress(num_steps: int, metrics: dict[str, Any]) -> None:
    """Log training progress metrics to Weights & Biases.

    Args:
        num_steps: Current training step count.
        metrics: Dictionary of metric names to values to log.
    """
    metrics["num_steps"] = num_steps
    wandb.log(metrics)


def make_wandb_progress_with_offset(
    step_offset: int = 0,
    episode_length: int | None = None,
) -> Callable[[int, dict[str, Any]], None]:
    """Create a wandb progress function with step offset for resume support.

    When resuming from a checkpoint, Brax's internal step counter restarts
    from 0. This wrapper adds the offset so W&B shows correct cumulative
    step counts.

    Brax's EpisodeWrapper sums state.metrics across timesteps. If
    ``episode_length`` is provided, any metric whose key starts with
    ``_gate`` (e.g. ``episode_gate``) is divided by the episode length
    so the logged value reflects the per-step average (0-1 range).

    Args:
        step_offset: Number of timesteps completed in previous runs.
        episode_length: Episode length for normalizing accumulated metrics.

    Returns:
        Progress function compatible with Brax's ``progress_fn`` callback.
    """

    def progress_fn(num_steps: int, metrics: dict[str, Any]) -> None:
        # Skip the initial eval on resume — it duplicates the last pre-preemption eval
        if num_steps == 0 and step_offset > 0:
            return
        adjusted_step = num_steps + step_offset
        metrics["num_steps"] = adjusted_step

        # Normalize accumulated per-step metrics to per-step averages
        if episode_length is not None:
            for key in list(metrics.keys()):
                if "_gate" in key:
                    metrics[key] = metrics[key] / episode_length

        wandb.log(metrics)

    return progress_fn
