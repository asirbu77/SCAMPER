"""From-scratch three-network training entry point.

Trains the encoder + decoder + motion prior of ``train_distill_expert`` with
PPO on the imitation reward instead of DAgger against an expert teacher. The
latent KL still targets the motion prior, and the AR(1) term is unchanged, both
with the distillation weights and schedule.

Saves under ``step_prefix="DistillNetwork"`` so the checkpoint is consumable by
``train_behavior_prior`` and task_transfer's ``res_control`` mode.
"""

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
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig
from scamper.environment import registry

from scamper.utils import config_utils as utils
from scamper.utils import checkpointing, wandb_logging
from scamper.agent.scratch import (
    ppo as scratch_ppo,
    scratch_networks,
)
from scamper.agent.scratch.rollout_logging import scratch_rollout_logging_fn
from scamper.utils.domain_randomization import domain_randomization_maker


def _setup_environment() -> None:
    """Configure environment variables for JAX."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@hydra.main(version_base=None, config_path="config", config_name="scratch")
def main(cfg: DictConfig) -> None:
    """Run from-scratch three-network PPO training from a Hydra config."""
    _setup_environment()

    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except RuntimeError:
        n_devices = 1
        logging.info("Not using GPUs")

    # prepare_config must run before load_from_run_state so the run-state
    # config hash matches between discovery and saving.
    cfg, cfg_dict, env_cfg_ml = utils.prepare_config(cfg)

    env_name = cfg.env_config.env_name

    run_id, checkpoint_path, existing_run_state = (
        checkpointing.load_from_run_state_scratch(cfg)
    )

    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="DistillNetwork",
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
    # naconmax = nconmax * num_envs for the warp backend.
    if hasattr(env_cfg_ml, "nconmax"):
        env_cfg_ml.naconmax = env_cfg_ml.nconmax * cfg.train_setup.train_config.num_envs

    env = registry.load(
        env_name, config=env_cfg_ml, clips=train_clips, flatten_obs=False
    )
    test_env = registry.load(
        env_name, config=env_cfg_ml, clips=test_clips, flatten_obs=False
    )

    logging.info(f"Environment config: {cfg.env_config}")

    # episode_length = (clip_length - init_range - traj_length) * steps_per_frame
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / (cfg.env_config.ctrl_dt)
    episode_length = (
        cfg.env_config.clip_length
        - cfg.env_config.start_frame_range[-1]
        - cfg.env_config.reference_length
    ) * steps_per_frame
    logging.info(f"episode_length {episode_length}")

    latent_cfg = cfg.latent_config

    network_factory = functools.partial(
        scratch_networks.make_scratch_ppo_networks,
        latent_size=cfg.network_config.latent_size,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        # YAML attribute name kept as `prior_layer_sizes` for compat; sizes the
        # motion prior in this pipeline.
        motion_prior_hidden_layer_sizes=tuple(cfg.network_config.prior_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        proprioception_noise_std=cfg.network_config.get(
            "proprioception_noise_std", 0.0
        ),
        clamp_logvar=latent_cfg.get("clamp_logvar", True),
        logvar_min=latent_cfg.get("logvar_min", -5.0),
        logvar_max=latent_cfg.get("logvar_max", 0.0),
    )

    wandb_logging.initialize_wandb_logging(
        logging_cfg=cfg.logging_config,
        cfg=cfg,
        run_id=run_id,
        existing_run_state=existing_run_state,
    )

    # Always write run state after wandb init, so the current job has its own
    # run-state file even when restoring from a different job's state.
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
        latent_size=cfg.network_config.latent_size,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        motion_prior_hidden_layer_sizes=tuple(cfg.network_config.prior_layer_sizes),
        kl_weight=latent_cfg.kl_weight,
        ar1_weight=latent_cfg.ar1_weight,
        use_kl_schedule=latent_cfg.get("use_kl_schedule", False),
        kl_schedule_params=(
            dict(latent_cfg.kl_schedule_params)
            if latent_cfg.get("use_kl_schedule", False)
            else None
        ),
        use_ar1_schedule=latent_cfg.get("use_ar1_schedule", False),
        ar1_schedule_params=(
            dict(latent_cfg.ar1_schedule_params)
            if latent_cfg.get("use_ar1_schedule", False)
            else None
        ),
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
        # YAML block name kept as `prior_rollout_config` for compat with
        # distill-expert.yaml; configures motion-prior rollout eval.
        motion_prior_rollout_config=(
            dict(cfg.prior_rollout_config)
            if hasattr(cfg, "prior_rollout_config")
            and cfg.prior_rollout_config is not None
            else None
        ),
    )

    train_fn = functools.partial(scratch_ppo.train, **train_kwargs)

    # Render env always starts at frame 0 of each clip.
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = registry.load(
        env_name, config=rollout_cfg, clips=None, flatten_obs=False
    )

    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)
    policy_params_fn = functools.partial(
        scratch_rollout_logging_fn,
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

    try:
        checkpointing.cleanup_run_state(cfg)
        logging.info("Training completed, run state cleaned up")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")


if __name__ == "__main__":
    main()
