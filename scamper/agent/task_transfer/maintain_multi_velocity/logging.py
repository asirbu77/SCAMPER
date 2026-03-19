"""Wandb logging utilities for maintain_multi_velocity task transfer training.

This module provides functions for logging training metrics, rollout videos,
and reward breakdowns to Weights & Biases. Supports multi-speed evaluation
where rollouts are run at each configured eval speed.
"""

import logging
from typing import Any, Callable, Sequence

import imageio
import jax
import mujoco
import wandb
from jax import numpy as jnp


def set_eval_target_speed(
    state: Any,
    target_speed: float,
    proprio_size: int,
) -> Any:
    """Override target speed in state for evaluation rollouts.

    Sets both state.info["target_speed"] and the corresponding element
    in the flat observation array. The target speed is the last element
    of task_obs, which comes right before proprioception in the flat obs.

    Args:
        state: Environment state (after reset).
        target_speed: Desired target speed.
        proprio_size: Size of proprioceptive observations.

    Returns:
        Updated state with overridden target speed.
    """
    new_info = {**state.info}
    new_info["target_speed"] = jnp.array(target_speed)
    # Target speed is the last element of task_obs, right before proprioception
    new_obs = state.obs.at[-(proprio_size + 1)].set(target_speed)
    return state.replace(info=new_info, obs=new_obs)


def multi_speed_rollout_logging_fn(
    env: Any,
    jit_reset: Callable,
    jit_step: Callable,
    jit_inference_fn: Callable,
    params: Any,
    current_step: int,
    model_path: str,
    episode_length: int,
    eval_speeds: Sequence[float],
    proprio_size: int,
    render_camera: str = "close_profile-rodent",
    render_fps: int = 50,
    render_height: int = 480,
    render_width: int = 640,
    render_every: int = 1,
    eval_count: int = 0,
) -> None:
    """Log evaluation rollouts at multiple target speeds to wandb.

    Runs a full episode at each configured eval speed, logging per-step
    reward breakdowns and rendered videos.

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
        eval_speeds: List of target speeds to evaluate at.
        proprio_size: Size of proprioceptive observations.
        render_camera: Camera name for rendering.
        render_fps: Frames per second for the output video.
        render_height: Height in pixels for the output video.
        render_width: Width in pixels for the output video.
        render_every: Render video every N evaluation steps.
        eval_count: Current evaluation count (0-indexed).
    """
    for speed in eval_speeds:
        speed_label = f"speed_{speed}"

        rng = jax.random.PRNGKey(current_step)
        reset_rng, act_rng = jax.random.split(rng)

        state = jit_reset(reset_rng)
        state = set_eval_target_speed(state, speed, proprio_size)

        rollout = [state]

        for _ in range(episode_length):
            _, act_rng = jax.random.split(act_rng)
            action, _ = jit_inference_fn(params, state.obs, act_rng)
            state = jit_step(state, action)
            rollout.append(state)

        _log_reward_metrics(rollout, current_step, speed_label)

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
                speed_label,
            )


def _log_reward_metrics(
    rollout: list[Any],
    current_step: int,
    speed_label: str,
) -> None:
    """Log per-step reward metrics as line plots for a single speed.

    Args:
        rollout: List of environment states from the episode.
        current_step: Current training step for versioning.
        speed_label: Label for the speed (e.g., "speed_0.5").
    """
    if not rollout or not hasattr(rollout[0], "metrics"):
        return

    first_metrics = rollout[0].metrics
    reward_keys = sorted([k for k in first_metrics.keys() if k.startswith("rewards/")])

    if not reward_keys:
        return

    columns = ["frame"] + reward_keys
    data = []
    for i, state in enumerate(rollout):
        row = [i] + [float(state.metrics.get(k, 0.0)) for k in reward_keys]
        data.append(row)

    try:
        table = wandb.Table(data=data, columns=columns)
        wandb.log(
            {f"eval/{speed_label}/reward_curves": table},
            commit=False,
        )
        for metric_name in reward_keys:
            wandb.log(
                {
                    f"eval/{speed_label}/{metric_name}": wandb.plot.line(
                        table,
                        "frame",
                        metric_name,
                        title=f"{metric_name} ({speed_label}, step {current_step})",
                    )
                },
                commit=False,
            )
    except Exception as e:
        logging.warning(f"Failed to log reward metrics for {speed_label}: {e}")


def _log_rollout_video(
    env: Any,
    model_path: str,
    current_step: int,
    rollout: list[Any],
    render_camera: str,
    render_fps: int,
    render_height: int = 480,
    render_width: int = 640,
    speed_label: str = "",
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
        speed_label: Label for the speed (e.g., "speed_0.5").
    """
    video_path = f"{model_path}/{current_step}_{speed_label}.mp4"

    try:
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
            {f"videos/{speed_label}/rollout": wandb.Video(video_path, format="mp4")},
            commit=False,
        )
    except mujoco.FatalError as e:
        logging.warning(f"Rendering video failed with MuJoCo error: {e}")
    except Exception as e:
        logging.warning(f"Failed to log rollout video for {speed_label}: {e}")


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
) -> Callable[[int, dict[str, Any]], None]:
    """Create a wandb progress function with step offset for resume support.

    When resuming from a checkpoint, Brax's internal step counter restarts
    from 0. This wrapper adds the offset so W&B shows correct cumulative
    step counts.

    Args:
        step_offset: Number of timesteps completed in previous runs.

    Returns:
        Progress function compatible with Brax's ``progress_fn`` callback.
    """

    def progress_fn(num_steps: int, metrics: dict[str, Any]) -> None:
        # Skip the initial eval on resume — it duplicates the last pre-preemption eval
        if num_steps == 0 and step_offset > 0:
            return
        adjusted_step = num_steps + step_offset
        metrics["num_steps"] = adjusted_step
        wandb.log(metrics)

    return progress_fn
