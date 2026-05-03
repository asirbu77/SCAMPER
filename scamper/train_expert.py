"""Entry point for SCAMPER expert (MLP-policy) PPO training."""

import os

# Must set rendering backend before importing MuJoCo
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import logging

import hydra
import jax
import orbax.checkpoint as ocp
import wandb
from jax import numpy as jnp
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig
from scamper.environment import registry

from scamper.utils import config_utils as utils
from scamper.utils import checkpointing, wandb_logging
from scamper.agent.expert import (
    ppo as expert_ppo,
    ppo_networks as expert_networks,
)
from scamper.utils.domain_randomization import domain_randomization_maker


def _setup_environment() -> None:
    """Configure environment variables for JAX."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def mlp_rollout_logging_fn(
    env,
    jit_reset,
    jit_step,
    cfg,
    model_path,
    current_step,
    jit_logging_inference_fn,
    params,
    policy_params_fn_key,
    render_video=True,
    ppo_network=None,
):
    """Log evaluation rollout metrics and video to W&B (no latent logging).

    Simplified version of wandb_logging.rollout_logging_fn that skips
    latent space statistics since the expert MLP policy has no latent bottleneck.
    """
    _, reset_rng, act_rng = jax.random.split(policy_params_fn_key, 3)
    state = jit_reset(reset_rng)

    rollout = [state]

    # Calculate episode length from config
    physics_steps_per_ctrl = cfg.env_config.ctrl_dt / cfg.env_config.sim_dt
    steps_per_mocap_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_steps_per_ctrl
    )
    episode_length = int(cfg.env_config.clip_length * steps_per_mocap_frame)

    for _ in range(episode_length):
        _, act_rng = jax.random.split(act_rng)
        ctrl, extras = jit_logging_inference_fn(params, state.obs, act_rng)
        ctrl = jnp.squeeze(ctrl, axis=0) if ctrl.shape[0] == 1 else ctrl
        state = jit_step(state, ctrl)
        rollout.append(state)

    if render_video:
        wandb_logging._log_rollout_video(env, cfg, model_path, current_step, rollout)


@hydra.main(version_base=None, config_path="config", config_name="expert")
def main(cfg: DictConfig) -> None:
    """Main training entry point for expert (MLP-policy) PPO training."""
    _setup_environment()

    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except RuntimeError:
        n_devices = 1
        logging.info("Not using GPUs")

    cfg, cfg_dict, env_cfg_ml = utils.prepare_config(cfg)

    env_name = cfg.env_config.env_name

    run_id, checkpoint_path, existing_run_state = checkpointing.load_from_run_state(cfg)

    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="PPONetwork",
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    logging.info(f"Loading data: {cfg.env_config.reference_data_path}")
    reference_clips = registry.load_reference_clips(
        env_name,
        data_path=cfg.env_config.reference_data_path,
        n_frames_per_clip=cfg.env_config.clip_length,
        keep_clips_idx=cfg.env_config.keep_clips_idx,
    )
    key_split, _ = jax.random.split(
        jax.random.PRNGKey(cfg.train_setup.train_config.seed)
    )
    train_clips, test_clips = reference_clips.split(
        train_ratio=cfg.train_setup.train_subset_ratio,
        seed=key_split,
    )
    env = registry.load(
        env_name, config=env_cfg_ml, clips=train_clips, flatten_obs=False
    )
    test_env = registry.load(
        env_name, config=env_cfg_ml, clips=test_clips, flatten_obs=False
    )

    logging.info(f"Environment config: {cfg.env_config}")

    steps_per_frame = (1 / cfg.env_config.mocap_hz) / (cfg.env_config.ctrl_dt)
    episode_length = (
        cfg.env_config.clip_length
        - cfg.env_config.start_frame_range[-1]
        - cfg.env_config.reference_length
    ) * steps_per_frame
    logging.info(f"episode_length {episode_length}")

    logging.info("Using PPO Pipeline with MLP network")

    network_factory = functools.partial(
        expert_networks.make_mlp_ppo_networks,
        mlp_hidden_layer_sizes=tuple(cfg.network_config.mlp_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
    )

    wandb_logging.initialize_wandb_logging(
        logging_cfg=cfg.logging_config,
        cfg=cfg,
        run_id=run_id,
        existing_run_state=existing_run_state,
    )

    checkpointing.save_run_state(
        cfg=cfg,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        wandb_run_id=wandb.run.id,
    )

    checkpoint_callback = checkpointing.create_checkpoint_callback(
        cfg=cfg,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        wandb_run_id=wandb.run.id,
    )

    train_kwargs = dict(
        **cfg.train_setup.train_config,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.reset_every,
        episode_length=episode_length,
        network_factory=network_factory,
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
        config_dict=cfg_dict,
        eval_env_test_set=test_env,
        checkpoint_callback=checkpoint_callback,
        wrap_for_training=functools.partial(
            playground_wrappers.wrap_for_brax_training, full_reset=False
        ),
        randomization_fn=(
            domain_randomization_maker(
                floor_friction=cfg.env_config.domain_randomization.floor_friction,
                static_friction_scale=cfg.env_config.domain_randomization.static_friction_scale,
                armature_scale=cfg.env_config.domain_randomization.armature_scale,
                com_jitter=cfg.env_config.domain_randomization.com_jitter,
                link_mass_scale=cfg.env_config.domain_randomization.link_mass_scale,
                torso_mass_jitter=cfg.env_config.domain_randomization.torso_mass_jitter,
                qpos0_jitter=cfg.env_config.domain_randomization.qpos0_jitter,
            )
            if cfg.env_config.domain_randomization.use_domain_randomization
            else None
        ),
    )

    train_fn = functools.partial(expert_ppo.train, **train_kwargs)

    # Set the render env start frame to always be 0
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = registry.load(
        env_name, config=rollout_cfg, clips=None, flatten_obs=False
    )

    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)
    policy_params_fn = functools.partial(
        mlp_rollout_logging_fn,
        rollout_env,
        jit_reset,
        jit_step,
        cfg,
        checkpoint_path,
    )

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=wandb_logging.wandb_progress,
        policy_params_fn=policy_params_fn,
    )

    # Clean up run state after successful completion
    try:
        checkpointing.cleanup_run_state(cfg)
        logging.info("Training completed successfully, cleaned up run state")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")


if __name__ == "__main__":
    main()
