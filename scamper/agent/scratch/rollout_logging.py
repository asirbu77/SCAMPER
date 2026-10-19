"""Rollout logging for from-scratch three-network PPO training.

Logs to wandb:
- Per-dimension encoder and motion-prior latent statistics
- A student rollout video with the reference-clip ghost overlay

Unlike ``distill_expert.rollout_logging`` there is no teacher to compare
against, and the policy is not rebuilt from the config: the trainer passes in
the jitted logging policy it already constructed, because ``obs_sizes`` /
``action_size`` are only known after the first environment reset (there is no
teacher checkpoint to read them from up front).
"""

import logging

import imageio
import jax
import jax.numpy as jnp
import mujoco
import wandb
from omegaconf import DictConfig


def scratch_rollout_logging_fn(
    env,
    jit_reset,
    jit_step,
    cfg: DictConfig,
    model_path: str,
    current_step: int,
    jit_logging_inference_fn,
    params,
    policy_params_fn_key: jax.Array,
    render_video: bool = False,
    ppo_network=None,
) -> None:
    """Roll out the current policy, log latent stats and render a video.

    Args:
        env: Rollout environment.
        jit_reset: JIT-compiled reset function.
        jit_step: JIT-compiled step function.
        cfg: Full configuration.
        model_path: Directory for saving videos.
        current_step: Current eval iteration.
        jit_logging_inference_fn: Jitted deterministic logging policy with
            signature ``(params, obs, key) -> (action, extras)``.
        params: ``(normalizer_params, policy_params)`` for the live policy.
        policy_params_fn_key: Random key for the rollout.
        render_video: Whether to render and upload the video.
        ppo_network: Unused; accepted for callback-signature parity.
    """
    del ppo_network

    if not render_video:
        return

    physics_step_per_control_step = cfg.env_config.ctrl_dt / cfg.env_config.sim_dt
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_step_per_control_step
    )
    episode_length = int(cfg.env_config.clip_length * steps_per_frame)

    policy_params_fn_key, reset_key = jax.random.split(policy_params_fn_key)
    state = jit_reset(reset_key)
    rollout = [state]
    latent_means = []
    latent_logvars = []
    motion_prior_means = []
    motion_prior_logvars = []

    _, act_rng = jax.random.split(policy_params_fn_key)

    for _ in range(episode_length):
        _, act_rng = jax.random.split(act_rng)

        ctrl, extras = jit_logging_inference_fn(params, state.obs, act_rng)
        ctrl = jnp.squeeze(ctrl, axis=0) if ctrl.shape[0] == 1 else ctrl

        latent_means.append(extras["latent_mean"])
        latent_logvars.append(extras["latent_logvar"])
        motion_prior_means.append(extras["motion_prior_mean"])
        motion_prior_logvars.append(extras["motion_prior_logvar"])

        state = jit_step(state, ctrl)
        rollout.append(state)

    _log_latent_stats("encoder", latent_means, latent_logvars)
    _log_latent_stats("prior", motion_prior_means, motion_prior_logvars)

    _render_student_video(env, cfg, model_path, current_step, rollout)


def _log_latent_stats(name: str, means: list, logvars: list) -> None:
    """Log per-dimension mean/std of a latent distribution over a rollout."""
    means = jnp.stack(means)
    logvars = jnp.stack(logvars)

    means_mean = jnp.mean(means, axis=0)
    means_std = jnp.std(means, axis=0)
    logvars_mean = jnp.mean(logvars, axis=0)
    logvars_std = jnp.std(logvars, axis=0)

    for i in range(means_mean.shape[-1]):
        wandb.log(
            {
                f"latents/{name}_means_mean{i}": float(means_mean[i]),
                f"latents/{name}_means_std{i}": float(means_std[i]),
                f"latents/{name}_logvars_mean{i}": float(logvars_mean[i]),
                f"latents/{name}_logvars_std{i}": float(logvars_std[i]),
            },
            commit=False,
        )


def _render_student_video(env, cfg, model_path, current_step, rollout):
    """Render the encoder+decoder rollout video with the reference ghost."""
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
