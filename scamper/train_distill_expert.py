"""Entry point for DAgger distillation of the expert teacher.

Distills a trained expert teacher into a VAE student (encoder +
decoder + motion prior) using online distillation (DAgger).
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
from omegaconf import DictConfig, OmegaConf
from scamper.environment.reference_clips import ReferenceClips
from scamper.environment.rodent import imitation

from scamper.utils import checkpointing, wandb_logging
from scamper.agent.distill_expert import distill_train
from scamper.agent.distill_expert.rollout_logging import distill_rollout_logging_fn
from scamper.agent.distill_expert.teacher import load_frozen_teacher
from scamper.utils.domain_randomization import domain_randomization_maker
from scamper.utils import config_utils as utils


def _setup_environment() -> None:
    """Configure environment variables for JAX."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@hydra.main(version_base=None, config_path="config", config_name="distill-expert")
def main(cfg: DictConfig):
    """Main function for DAgger distillation of the expert teacher."""

    _setup_environment()

    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except RuntimeError:
        n_devices = 1
        logging.info("Not using GPUs")

    # Validate teacher config
    if cfg.teacher_config.checkpoint_path is None:
        raise ValueError(
            "teacher_config.checkpoint_path must be specified for distillation"
        )

    # Prepare config BEFORE load_from_run_state so the config hash is consistent
    cfg, cfg_dict, env_cfg_ml = utils.prepare_config(cfg)

    # Determine how to load from checkpoint
    run_id, checkpoint_path, existing_run_state = (
        checkpointing.load_from_run_state_distill(cfg)
    )

    # Initialize checkpoint manager with DistillNetwork prefix
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="DistillNetwork",
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    # Create the reference clips
    logging.info(f"Loading data: {cfg.env_config.reference_data_path}")
    reference_clips = ReferenceClips(
        data_path=cfg.env_config.reference_data_path,
        n_frames_per_clip=cfg.env_config.clip_length,
        keep_clips_idx=cfg.env_config.keep_clips_idx,
    )
    # Create train/test split
    key_split, _ = jax.random.split(
        jax.random.PRNGKey(cfg.train_setup.train_config.seed)
    )
    train_clips, test_clips = reference_clips.split(
        train_ratio=cfg.train_setup.train_subset_ratio,
        seed=key_split,
    )
    # Compute naconmax for warp backend
    if hasattr(env_cfg_ml, "nconmax"):
        env_cfg_ml.naconmax = env_cfg_ml.nconmax * cfg.train_setup.train_config.num_envs

    # Create environments
    env = imitation.Imitation(config=env_cfg_ml, clips=train_clips)
    test_env = imitation.Imitation(config=env_cfg_ml, clips=test_clips)

    logging.info(f"Environment config: {cfg.env_config}")

    # Episode length
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / cfg.env_config.ctrl_dt
    episode_length = (
        cfg.env_config.clip_length
        - cfg.env_config.start_frame_range[-1]
        - cfg.env_config.reference_length
    ) * steps_per_frame
    logging.info(f"episode_length {episode_length}")

    logging.info("Using DAgger Distillation Pipeline")

    # Load teacher to extract obs/action sizes for config
    _, teacher_normalizer, teacher_cfg, _ = load_frozen_teacher(
        cfg.teacher_config.checkpoint_path,
        step=cfg.teacher_config.checkpoint_step,
    )
    teacher_net_cfg = teacher_cfg["network_config"]

    # Update network config with teacher dimensions
    OmegaConf.set_struct(cfg.network_config, False)
    OmegaConf.update(
        cfg.network_config,
        "action_size",
        teacher_net_cfg["action_size"],
        merge=False,
    )
    OmegaConf.update(
        cfg.network_config,
        "obs_sizes",
        teacher_net_cfg["obs_sizes"],
        merge=False,
    )
    OmegaConf.update(
        cfg.network_config,
        "intention_size",
        cfg.network_config.latent_size,
        merge=False,
    )
    OmegaConf.set_struct(cfg.network_config, True)

    # Initialize wandb logging
    wandb_logging.initialize_wandb_logging(
        logging_cfg=cfg.logging_config,
        cfg=cfg,
        run_id=run_id,
        existing_run_state=existing_run_state,
    )

    # Save run state after wandb initialization
    checkpointing.save_run_state(
        cfg=cfg,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        wandb_run_id=wandb.run.id,
    )

    # Create the checkpoint callback
    checkpoint_callback = checkpointing.create_checkpoint_callback(
        cfg=cfg,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        wandb_run_id=wandb.run.id,
    )

    # Get distill config
    distill_cfg = cfg.distill_config

    # Setup training function
    train_fn = functools.partial(
        distill_train.train,
        **cfg.train_setup.train_config,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.reset_every,
        episode_length=episode_length,
        teacher_checkpoint_path=cfg.teacher_config.checkpoint_path,
        teacher_checkpoint_step=cfg.teacher_config.checkpoint_step,
        kl_weight=distill_cfg.kl_weight,
        ar1_weight=distill_cfg.ar1_weight,
        grad_clip_norm=distill_cfg.grad_clip_norm,
        use_kl_schedule=distill_cfg.get("use_kl_schedule", False),
        kl_schedule_params=(
            dict(distill_cfg.kl_schedule_params)
            if distill_cfg.get("use_kl_schedule", False)
            else None
        ),
        use_ar1_schedule=distill_cfg.get("use_ar1_schedule", False),
        ar1_schedule_params=(
            dict(distill_cfg.ar1_schedule_params)
            if distill_cfg.get("use_ar1_schedule", False)
            else None
        ),
        latent_size=cfg.network_config.latent_size,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        # YAML attribute name kept as `prior_layer_sizes` for compat; sizes the
        # motion prior in this pipeline.
        motion_prior_hidden_layer_sizes=tuple(cfg.network_config.prior_layer_sizes),
        clamp_logvar=distill_cfg.get("clamp_logvar", True),
        logvar_min=distill_cfg.get("logvar_min", -5.0),
        logvar_max=distill_cfg.get("logvar_max", 0.0),
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
        hard_negative_mining=distill_cfg.get("hard_negative_mining", True),
        # YAML block name kept as `prior_rollout_config` for compat with saved
        # configs; in this pipeline it configures motion-prior rollout eval.
        motion_prior_rollout_config=(
            dict(cfg.prior_rollout_config)
            if hasattr(cfg, "prior_rollout_config")
            and cfg.prior_rollout_config is not None
            else None
        ),
    )

    # Create rollout logging environment and function
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = imitation.Imitation(config=rollout_cfg, clips=train_clips)

    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)
    policy_params_fn = functools.partial(
        distill_rollout_logging_fn,
        rollout_env,
        jit_reset,
        jit_step,
        cfg,
        checkpoint_path,
    )

    # Run training
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=wandb_logging.wandb_progress,
        policy_params_fn=policy_params_fn,
    )

    # Clean up run state after successful completion
    try:
        checkpointing.cleanup_run_state(cfg)
        logging.info("Distillation completed successfully, cleaned up run state")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")


if __name__ == "__main__":
    main()
