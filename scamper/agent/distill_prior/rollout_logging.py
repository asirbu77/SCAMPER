"""
Rollout logging for distill prior (RGM) training.

Logs latent statistics for monitoring how well the RGM is learning to match
the distill encoder, including:
- Distill encoder latent statistics (mean/logvar)
- RGM latent statistics (mean/logvar)
- Frozen distill prior latent statistics (for comparison)
- KL divergence between encoder and RGM over rollout
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
from scamper.agent.distill_prior import distill_prior_networks
from scamper.agent.mlp_prior.losses import compute_encoder_prior_kl_loss
from scamper.agent.mlp_prior.prior_networks import Prior as RGM
from scamper.agent.observation_utils import (
    DictRunningStatisticsState,
    flatten_obs_dict,
    normalize_dict_obs,
)


def distill_prior_training_rollout_logging_fn(
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
    """Rollout logging for RGM training — logs latent statistics.

    Runs a rollout with the frozen distill encoder+decoder, then compares
    the distill encoder, RGM, and frozen distill prior latent distributions.

    Args:
        env: Environment for rollout.
        jit_reset: JIT-compiled reset function.
        jit_step: JIT-compiled step function.
        cfg: Configuration.
        model_path: Path for saving videos.
        current_step: Current training step.
        params: Policy parameters (normalizer, {encoder, decoder, prior, rgm}).
        policy_params_fn_key: Random key for rollout.
        reset_key: Random key for environment reset.
        render_video: Whether to render video.
    """
    physics_step_per_control_step = cfg.env_config.ctrl_dt / cfg.env_config.sim_dt
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_step_per_control_step
    )
    episode_length = int(cfg.env_config.clip_length * steps_per_frame)

    # Extract network config
    latent_size = cfg.network_config.intention_size
    encoder_hidden_layer_sizes = tuple(cfg.network_config.encoder_layer_sizes)
    decoder_hidden_layer_sizes = tuple(cfg.network_config.decoder_layer_sizes)
    prior_hidden_layer_sizes = tuple(cfg.network_config.prior_layer_sizes)
    rgm_hidden_layer_sizes = tuple(
        cfg.network_config.get("rgm_layer_sizes", [1536, 1024, 512])
    )
    action_size = cfg.network_config.action_size
    obs_sizes = dict(cfg.network_config.obs_sizes)

    # Setup normalization
    normalize = lambda x, y: x
    if cfg.train_setup.train_config.normalize_observations:
        normalize = running_statistics.normalize

    # Extract params
    normalizer_params, network_params = params
    encoder_params = network_params["params"]["encoder"]
    decoder_params = network_params["params"]["decoder"]
    prior_params = network_params["params"]["prior"]
    rgm_params = network_params["params"]["rgm"]

    # Create encoder+decoder policy for rollout
    policy = distill_prior_networks.make_distill_encoder_decoder_inference_fn(
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

    # Create RGM and distill prior modules for comparison
    rgm_module = RGM(
        layer_sizes=list(rgm_hidden_layer_sizes),
        latents=latent_size,
    )

    prior_module = DistillMotionPrior(
        layer_sizes=list(prior_hidden_layer_sizes),
        latents=latent_size,
    )

    # Run rollout and collect latent statistics
    _, act_rng = jax.random.split(policy_params_fn_key)

    state = jit_reset(reset_key)
    rollout = [state]
    encoder_means = []
    encoder_logvars = []
    rgm_means = []
    rgm_logvars = []
    prior_means = []
    prior_logvars = []

    for i in range(episode_length):
        _, act_rng = jax.random.split(act_rng)
        obs = state.obs

        # Get action and encoder latents from policy
        ctrl, extras = jit_policy(obs, act_rng)
        ctrl = jnp.squeeze(ctrl, axis=0) if ctrl.shape[0] == 1 else ctrl

        # Collect encoder latent statistics from extras
        encoder_means.append(extras["latent_mean"])
        encoder_logvars.append(extras["latent_logvar"])

        # Get RGM and prior latent statistics
        flat_obs = flatten_obs_dict(obs)
        normalized_obs = normalize_dict_obs(flat_obs, normalizer_params)
        proprio_obs = normalized_obs["proprioception"]

        # RGM output
        rgm_mean, rgm_logvar = rgm_module.apply({"params": rgm_params}, proprio_obs)
        rgm_means.append(rgm_mean)
        rgm_logvars.append(rgm_logvar)

        # Frozen distill prior output (for comparison)
        prior_mean, prior_logvar = prior_module.apply(
            {"params": prior_params}, proprio_obs
        )
        prior_means.append(prior_mean)
        prior_logvars.append(prior_logvar)

        state = jit_step(state, ctrl)
        rollout.append(state)

    # Stack latents
    encoder_means = jnp.stack(encoder_means)
    encoder_logvars = jnp.stack(encoder_logvars)
    rgm_means = jnp.stack(rgm_means)
    rgm_logvars = jnp.stack(rgm_logvars)
    prior_means = jnp.stack(prior_means)
    prior_logvars = jnp.stack(prior_logvars)

    # Compute per-dimension statistics
    encoder_means_mean = jnp.mean(encoder_means, axis=0)
    encoder_means_std = jnp.std(encoder_means, axis=0)
    encoder_logvars_mean = jnp.mean(encoder_logvars, axis=0)
    encoder_logvars_std = jnp.std(encoder_logvars, axis=0)

    rgm_means_mean = jnp.mean(rgm_means, axis=0)
    rgm_means_std = jnp.std(rgm_means, axis=0)
    rgm_logvars_mean = jnp.mean(rgm_logvars, axis=0)
    rgm_logvars_std = jnp.std(rgm_logvars, axis=0)

    prior_means_mean = jnp.mean(prior_means, axis=0)
    prior_means_std = jnp.std(prior_means, axis=0)
    prior_logvars_mean = jnp.mean(prior_logvars, axis=0)
    prior_logvars_std = jnp.std(prior_logvars, axis=0)

    # Compute KL divergence over rollout
    kl_per_timestep = []
    for t in range(len(encoder_means)):
        kl = compute_encoder_prior_kl_loss(
            encoder_means[t : t + 1],
            encoder_logvars[t : t + 1],
            rgm_means[t : t + 1],
            rgm_logvars[t : t + 1],
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
                # RGM latent statistics
                f"latents/rgm_means_mean{i}": float(rgm_means_mean[i]),
                f"latents/rgm_means_std{i}": float(rgm_means_std[i]),
                f"latents/rgm_logvars_mean{i}": float(rgm_logvars_mean[i]),
                f"latents/rgm_logvars_std{i}": float(rgm_logvars_std[i]),
                # Frozen distill prior latent statistics (for comparison)
                f"latents/distill_prior_means_mean{i}": float(prior_means_mean[i]),
                f"latents/distill_prior_means_std{i}": float(prior_means_std[i]),
                f"latents/distill_prior_logvars_mean{i}": float(prior_logvars_mean[i]),
                f"latents/distill_prior_logvars_std{i}": float(prior_logvars_std[i]),
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
