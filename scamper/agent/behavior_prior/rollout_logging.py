"""Rollout logging for behavior-prior training.

Logs latent statistics for monitoring how well the behavior prior is learning
to match the distill encoder, including:
- Distill encoder latent statistics (mean/logvar)
- Behavior-prior latent statistics (mean/logvar)
- Frozen motion-prior latent statistics (for comparison)
- KL divergence between encoder and behavior prior over rollout
- Video rendering (optional)
"""

import logging

import imageio
import jax
import jax.numpy as jnp
import mujoco
import wandb
from brax.training.acme import running_statistics
from omegaconf import DictConfig

from scamper.agent.distill_expert.distill_networks import (
    DistillEncoder,
    DistillMotionPrior,
)
from scamper.agent.behavior_prior import behavior_prior_networks
from scamper.agent.imitation.prior import compute_encoder_prior_kl_loss
from scamper.agent.imitation.prior import Prior as BehaviorPrior
from scamper.agent.observation_utils import (
    DictRunningStatisticsState,
    flatten_obs_dict,
    normalize_dict_obs,
)


def behavior_prior_training_rollout_logging_fn(
    env,
    jit_reset,
    jit_step,
    cfg: DictConfig,
    model_path: str,
    current_step: int,
    params,
    policy_params_fn_key,
    reset_key: jax.Array,
    render_video: bool = False,
) -> None:
    """Rollout logging for behavior-prior training — logs latent statistics.

    Runs a rollout with the frozen distill encoder+decoder, then compares
    the distill encoder, behavior prior, and frozen motion prior latent
    distributions.

    Args:
        env: Environment for rollout.
        jit_reset: JIT-compiled reset function.
        jit_step: JIT-compiled step function.
        cfg: Configuration.
        model_path: Path for saving videos.
        current_step: Current training step.
        params: Policy parameters; the saved tree is
            (normalizer, {encoder, decoder, prior, rgm}). The "prior" entry
            holds the motion-prior weights and "rgm" the behavior-prior
            weights — dict keys kept for checkpoint compat.
        policy_params_fn_key: Random key for rollout.
        reset_key: Random key for environment reset.
        render_video: Whether to render video.
    """
    physics_step_per_control_step = cfg.env_config.ctrl_dt / cfg.env_config.sim_dt
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_step_per_control_step
    )
    episode_length = int(cfg.env_config.clip_length * steps_per_frame)

    # Extract network config. cfg attributes `prior_layer_sizes` and
    # `rgm_layer_sizes` kept for compat with saved configs.
    latent_size = cfg.network_config.intention_size
    encoder_hidden_layer_sizes = tuple(cfg.network_config.encoder_layer_sizes)
    decoder_hidden_layer_sizes = tuple(cfg.network_config.decoder_layer_sizes)
    motion_prior_hidden_layer_sizes = tuple(cfg.network_config.prior_layer_sizes)
    behavior_prior_hidden_layer_sizes = tuple(
        cfg.network_config.get("rgm_layer_sizes", [1536, 1024, 512])
    )
    action_size = cfg.network_config.action_size
    obs_sizes = dict(cfg.network_config.obs_sizes)

    # Setup normalization
    normalize = lambda x, y: x
    if cfg.train_setup.train_config.normalize_observations:
        normalize = running_statistics.normalize

    # Extract params. Dict keys 'prior' and 'rgm' kept for checkpoint compat.
    normalizer_params, network_params = params
    encoder_params = network_params["params"]["encoder"]
    decoder_params = network_params["params"]["decoder"]
    motion_prior_params = network_params["params"]["prior"]
    behavior_prior_params = network_params["params"]["rgm"]

    # Create encoder+decoder policy for rollout
    policy = behavior_prior_networks.make_distill_encoder_decoder_inference_fn(
        encoder_params={"params": encoder_params},
        decoder_params={"params": decoder_params},
        normalizer_params=normalizer_params,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        latent_size=latent_size,
        action_size=action_size,
        obs_sizes=obs_sizes,
        deterministic=True,
    )
    jit_policy = jax.jit(policy)

    # Create encoder apply function for getting distributions
    encoder_module = DistillEncoder(
        layer_sizes=list(encoder_hidden_layer_sizes),
        latents=latent_size,
    )

    # Create behavior-prior and motion-prior modules for comparison
    behavior_prior_module = BehaviorPrior(
        layer_sizes=list(behavior_prior_hidden_layer_sizes),
        latents=latent_size,
    )

    motion_prior_module = DistillMotionPrior(
        layer_sizes=list(motion_prior_hidden_layer_sizes),
        latents=latent_size,
    )

    # Run rollout and collect latent statistics
    _, act_rng = jax.random.split(policy_params_fn_key)

    state = jit_reset(reset_key)
    rollout = [state]
    encoder_means = []
    encoder_logvars = []
    behavior_prior_means = []
    behavior_prior_logvars = []
    motion_prior_means = []
    motion_prior_logvars = []

    for i in range(episode_length):
        _, act_rng = jax.random.split(act_rng)
        obs = state.obs

        # Get action and encoder latents from policy
        ctrl, extras = jit_policy(obs, act_rng)
        ctrl = jnp.squeeze(ctrl, axis=0) if ctrl.shape[0] == 1 else ctrl

        # Collect encoder latent statistics from extras
        encoder_means.append(extras["latent_mean"])
        encoder_logvars.append(extras["latent_logvar"])

        # Get behavior-prior and motion-prior latent statistics
        flat_obs = flatten_obs_dict(obs)
        normalized_obs = normalize_dict_obs(flat_obs, normalizer_params)
        proprio_obs = normalized_obs["proprioception"]

        # Behavior-prior output
        behavior_prior_mean, behavior_prior_logvar = behavior_prior_module.apply(
            {"params": behavior_prior_params}, proprio_obs
        )
        behavior_prior_means.append(behavior_prior_mean)
        behavior_prior_logvars.append(behavior_prior_logvar)

        # Frozen motion-prior output (for comparison)
        motion_prior_mean, motion_prior_logvar = motion_prior_module.apply(
            {"params": motion_prior_params}, proprio_obs
        )
        motion_prior_means.append(motion_prior_mean)
        motion_prior_logvars.append(motion_prior_logvar)

        state = jit_step(state, ctrl)
        rollout.append(state)

    # Stack latents
    encoder_means = jnp.stack(encoder_means)
    encoder_logvars = jnp.stack(encoder_logvars)
    behavior_prior_means = jnp.stack(behavior_prior_means)
    behavior_prior_logvars = jnp.stack(behavior_prior_logvars)
    motion_prior_means = jnp.stack(motion_prior_means)
    motion_prior_logvars = jnp.stack(motion_prior_logvars)

    # Compute per-dimension statistics
    encoder_means_mean = jnp.mean(encoder_means, axis=0)
    encoder_means_std = jnp.std(encoder_means, axis=0)
    encoder_logvars_mean = jnp.mean(encoder_logvars, axis=0)
    encoder_logvars_std = jnp.std(encoder_logvars, axis=0)

    behavior_prior_means_mean = jnp.mean(behavior_prior_means, axis=0)
    behavior_prior_means_std = jnp.std(behavior_prior_means, axis=0)
    behavior_prior_logvars_mean = jnp.mean(behavior_prior_logvars, axis=0)
    behavior_prior_logvars_std = jnp.std(behavior_prior_logvars, axis=0)

    motion_prior_means_mean = jnp.mean(motion_prior_means, axis=0)
    motion_prior_means_std = jnp.std(motion_prior_means, axis=0)
    motion_prior_logvars_mean = jnp.mean(motion_prior_logvars, axis=0)
    motion_prior_logvars_std = jnp.std(motion_prior_logvars, axis=0)

    # Compute KL divergence over rollout
    kl_per_timestep = []
    for t in range(len(encoder_means)):
        kl = compute_encoder_prior_kl_loss(
            encoder_means[t : t + 1],
            encoder_logvars[t : t + 1],
            behavior_prior_means[t : t + 1],
            behavior_prior_logvars[t : t + 1],
        )
        kl_per_timestep.append(float(kl))

    # Log per-dimension latent statistics
    for i in range(encoder_means_mean.shape[-1]):
        wandb.log(
            {
                # Encoder latent statistics
                f"latents/encoder_means_mean{i}": float(encoder_means_mean[i]),
                f"latents/encoder_means_std{i}": float(encoder_means_std[i]),
                f"latents/encoder_logvars_mean{i}": float(encoder_logvars_mean[i]),
                f"latents/encoder_logvars_std{i}": float(encoder_logvars_std[i]),
                # Behavior-prior latent statistics
                f"latents/behavior_prior_means_mean{i}": float(
                    behavior_prior_means_mean[i]
                ),
                f"latents/behavior_prior_means_std{i}": float(
                    behavior_prior_means_std[i]
                ),
                f"latents/behavior_prior_logvars_mean{i}": float(
                    behavior_prior_logvars_mean[i]
                ),
                f"latents/behavior_prior_logvars_std{i}": float(
                    behavior_prior_logvars_std[i]
                ),
                # Frozen motion-prior latent statistics (for comparison)
                f"latents/motion_prior_means_mean{i}": float(
                    motion_prior_means_mean[i]
                ),
                f"latents/motion_prior_means_std{i}": float(motion_prior_means_std[i]),
                f"latents/motion_prior_logvars_mean{i}": float(
                    motion_prior_logvars_mean[i]
                ),
                f"latents/motion_prior_logvars_std{i}": float(
                    motion_prior_logvars_std[i]
                ),
            },
            commit=False,
        )

    # Optionally render video (encoder+decoder rollout)
    if render_video:
        render_fps = cfg.render_config.render_fps
        camera_name = f"{cfg.render_config.render_camera_name}-ghost"

        video_path = f"{model_path}/{current_step}_encoder_decoder.mp4"
        try:
            with imageio.get_writer(video_path, fps=render_fps) as writer:
                video = env.render(
                    rollout,
                    camera=camera_name,
                    height=480,
                    width=640,
                )
                for frame in video:
                    writer.append_data(frame)

            wandb.log(
                {
                    "videos/encoder_decoder_rollout": wandb.Video(
                        video_path, format="mp4"
                    )
                },
                commit=False,
            )
        except mujoco.FatalError as e:
            logging.warning(
                f"Video rendering failed due to MuJoCo error: {e}. "
                f"Skipping video for this iteration."
            )
