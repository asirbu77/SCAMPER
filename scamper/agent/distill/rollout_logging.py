"""Rollout logging for DAgger distillation.

Logs to wandb:
- Teacher vs student side-by-side videos with ghost overlay
"""

import logging

import imageio
import jax
import jax.numpy as jnp
import mujoco
import wandb
from omegaconf import DictConfig

from scamper.agent.distill.distill_networks import (
    DistillNetworkParams,
)
from scamper.agent.observation_utils import (
    DictRunningStatisticsState,
    flatten_obs_dict,
    normalize_dict_obs,
)


def distill_rollout_logging_fn(
    env,
    jit_reset,
    jit_step,
    cfg: DictConfig,
    model_path: str,
    current_step: int,
    params,
    normalizer_params: DictRunningStatisticsState,
    network_params: DistillNetworkParams,
    policy_params_fn_key: jax.Array,
    reset_key: jax.Array,
    render_video: bool = False,
) -> None:
    """Rollout logging for DAgger distillation.

    Renders teacher and student rollout videos on the same clip for comparison.

    Args:
        env: Rollout environment.
        jit_reset: JIT-compiled reset function.
        jit_step: JIT-compiled step function.
        cfg: Full configuration.
        model_path: Path for saving videos.
        current_step: Current training step.
        params: Compatible checkpoint params (normalizer, combined_params).
        normalizer_params: Student's observation normalizer.
        network_params: DistillNetworkParams (encoder, decoder, prior).
        policy_params_fn_key: Random key for rollout.
        reset_key: Random key for environment reset.
        render_video: Whether to render videos.
    """
    if not render_video:
        return

    from scamper.agent.distill.distill_networks import (
        make_distill_networks,
        make_logging_inference_fn,
    )

    # Compute episode length
    physics_step_per_control_step = cfg.env_config.ctrl_dt / cfg.env_config.sim_dt
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_step_per_control_step
    )
    episode_length = int(cfg.env_config.clip_length * steps_per_frame)

    # Get network config
    latent_size = cfg.network_config.latent_size
    obs_sizes = dict(cfg.network_config.obs_sizes)
    action_size = cfg.network_config.action_size

    # Create network modules
    encoder, decoder, prior, action_dist = make_distill_networks(
        obs_sizes=obs_sizes,
        action_size=action_size,
        latent_size=latent_size,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        prior_hidden_layer_sizes=tuple(cfg.network_config.prior_layer_sizes),
    )

    # Create student inference function
    make_logging_policy = make_logging_inference_fn(
        encoder, decoder, action_dist, obs_sizes, latent_size
    )
    jit_student_policy = jax.jit(make_logging_policy(deterministic=True))

    # --- Student rollout ---
    state = jit_reset(reset_key)
    rollout = [state]
    latent_means = []
    latent_logvars = []

    _, act_rng = jax.random.split(policy_params_fn_key)

    for i in range(episode_length):
        _, act_rng = jax.random.split(act_rng)

        ctrl, extras = jit_student_policy(
            (normalizer_params, network_params), state.obs, act_rng
        )
        ctrl = jnp.squeeze(ctrl, axis=0) if ctrl.shape[0] == 1 else ctrl

        latent_means.append(extras["latent_mean"])
        latent_logvars.append(extras["latent_logvar"])

        state = jit_step(state, ctrl)
        rollout.append(state)

    # --- Log encoder latent statistics per dimension ---
    latent_means = jnp.stack(latent_means)
    latent_logvars = jnp.stack(latent_logvars)

    means_mean = jnp.mean(latent_means, axis=0)
    means_std = jnp.std(latent_means, axis=0)
    logvars_mean = jnp.mean(latent_logvars, axis=0)
    logvars_std = jnp.std(latent_logvars, axis=0)

    for i in range(means_mean.shape[-1]):
        wandb.log(
            {
                f"latents/latent_means_mean{i}": float(means_mean[i]),
                f"latents/latent_means_std{i}": float(means_std[i]),
                f"latents/latent_logvars_mean{i}": float(logvars_mean[i]),
                f"latents/latent_logvars_std{i}": float(logvars_std[i]),
            },
            commit=False,
        )

    # --- Render student video ---
    _render_student_video(env, cfg, model_path, current_step, rollout)

    # --- Render teacher video on same clip ---
    if cfg.teacher_config.checkpoint_path is not None:
        _render_teacher_video(
            env, jit_reset, jit_step, cfg, model_path, current_step, reset_key
        )


def _render_student_video(env, cfg, model_path, current_step, rollout):
    """Render student (encoder+decoder) rollout video with ghost."""
    render_fps = cfg.render_config.render_fps
    video_path = f"{model_path}/{current_step}_student.mp4"
    camera_name = f"{cfg.render_config.render_camera_name}{env._suffix}"

    try:
        with imageio.get_writer(video_path, fps=render_fps) as writer:
            video = env.render(rollout, camera=camera_name, height=480, width=640)
            for frame in video:
                writer.append_data(frame)

        wandb.log(
            {"videos/student_rollout": wandb.Video(video_path, format="mp4")},
            commit=False,
        )
    except mujoco.FatalError as e:
        logging.warning(f"Student video rendering failed: {e}")


def _render_teacher_video(
    env, jit_reset, jit_step, cfg, model_path, current_step, reset_key
):
    """Render teacher rollout on the same clip for comparison."""
    from scamper.agent.distill.teacher import load_frozen_teacher
    from brax.training import distribution

    try:
        teacher_fn, teacher_normalizer, _, _ = load_frozen_teacher(
            cfg.teacher_config.checkpoint_path,
            step=cfg.teacher_config.checkpoint_step,
        )
    except Exception as e:
        logging.warning(f"Failed to load teacher for video: {e}")
        return

    action_size = cfg.network_config.action_size
    action_dist = distribution.NormalTanhDistribution(event_size=action_size)

    # Run teacher rollout on same clip
    physics_step_per_control_step = cfg.env_config.ctrl_dt / cfg.env_config.sim_dt
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_step_per_control_step
    )
    episode_length = int(cfg.env_config.clip_length * steps_per_frame)

    state = jit_reset(reset_key)
    teacher_rollout = [state]

    for _ in range(episode_length):
        obs = state.obs
        flat_obs = flatten_obs_dict(obs)
        teacher_norm = normalize_dict_obs(flat_obs, teacher_normalizer)
        logits = teacher_fn(teacher_norm)
        ctrl = action_dist.mode(logits)
        ctrl = jnp.squeeze(ctrl, axis=0) if ctrl.shape[0] == 1 else ctrl
        state = jit_step(state, ctrl)
        teacher_rollout.append(state)

    render_fps = cfg.render_config.render_fps
    video_path = f"{model_path}/{current_step}_teacher.mp4"
    camera_name = f"{cfg.render_config.render_camera_name}{env._suffix}"

    try:
        with imageio.get_writer(video_path, fps=render_fps) as writer:
            video = env.render(
                teacher_rollout, camera=camera_name, height=480, width=640
            )
            for frame in video:
                writer.append_data(frame)

        wandb.log(
            {"videos/teacher_rollout": wandb.Video(video_path, format="mp4")},
            commit=False,
        )
    except mujoco.FatalError as e:
        logging.warning(f"Teacher video rendering failed: {e}")
