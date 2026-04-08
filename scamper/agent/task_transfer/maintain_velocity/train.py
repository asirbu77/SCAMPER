"""Maintain velocity task transfer training with Brax PPO.

Five training modes:
- decoder_only: Freeze decoder, train new encoder
- prior_decoder: Freeze prior + decoder, train residual encoder (from mlp_prior checkpoint)
- distill: Freeze prior + decoder, train residual encoder (from distill checkpoint)
- distill_prior: Freeze prior + rgm + decoder, train residual encoder (from distill_prior checkpoint)
- scratch: Train both policy and decoder from random initialization

Supports Orbax checkpointing and SLURM preemption recovery.

Usage:
    python -m scamper.agent.task_transfer.maintain_velocity.train mode=decoder_only
    python -m scamper.agent.task_transfer.maintain_velocity.train mode=prior_decoder
    python -m scamper.agent.task_transfer.maintain_velocity.train mode=distill
    python -m scamper.agent.task_transfer.maintain_velocity.train mode=distill_prior
    python -m scamper.agent.task_transfer.maintain_velocity.train mode=scratch
"""

import os

# IMPORTANT: MUJOCO_GL is read at mujoco import time, so this MUST be set
# before any module that imports mujoco (including mujoco_playground).
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Set XLA flags for JAX
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import functools
import json
import logging
from datetime import datetime
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import wandb
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as brax_ppo
from brax.training.acme import running_statistics, specs
from mujoco_playground import wrapper
from omegaconf import DictConfig, OmegaConf

from scamper.environment.rodent import consts as rodent_consts
from scamper.environment.rodent import maintain_velocity

from scamper.agent.task_transfer.maintain_velocity.checkpoint_utils import (
    load_prior_checkpoint,
    load_distill_prior_checkpoint,
    make_decoder_inference_fn,
    make_prior_inference_fn,
    make_rgm_inference_fn,
    # Checkpointing / run state
    discover_existing_run_state,
    save_run_state,
    cleanup_run_state,
    create_checkpoint_callback,
    save_task_transfer_checkpoint,
    load_task_transfer_metadata,
    load_task_transfer_brax_params,
)
from scamper.agent.task_transfer.maintain_velocity.logging import (
    rollout_logging_fn,
    make_wandb_progress_with_offset,
)
from scamper.agent.task_transfer.maintain_velocity.wrappers import (
    DecoderHighLevelWrapper,
    FlatObsWrapper,
    PriorDecoderHighLevelWrapper,
    PriorRGMDecoderHighLevelWrapper,
)
from scamper.agent.task_transfer.maintain_velocity.scratch_networks import (
    make_scratch_ppo_networks,
)

# Enable persistent compilation cache
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# Orbax step prefix for task transfer checkpoints
_STEP_PREFIX = "TaskTransfer"


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra config."""
    # Log device info
    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except Exception:
        n_devices = 1
        logging.info("Not using GPUs")

    # ----------------------------------------------------------------
    # Phase 1: Check for preemption recovery
    # ----------------------------------------------------------------
    model_path_base = hydra.utils.to_absolute_path(
        cfg.checkpointing.get("model_path", "./model_checkpoints")
    )
    Path(model_path_base).mkdir(parents=True, exist_ok=True)

    existing_run_state = None
    step_offset = 0
    initial_eval_counter = 0
    restore_params_for_brax = None
    remaining_timesteps = cfg.train.num_timesteps

    if cfg.checkpointing.get("enabled", True):
        existing_run_state = discover_existing_run_state(cfg, model_path_base)

    if existing_run_state is not None:
        # --- Preemption recovery path ---
        run_id = existing_run_state["run_id"]
        model_path = existing_run_state["checkpoint_path"]
        wandb_run_id_to_resume = existing_run_state["wandb_run_id"]

        ckpt_info = load_task_transfer_metadata(
            model_path, step=existing_run_state["latest_checkpoint_step"]
        )
        metadata = ckpt_info["training_metadata"]

        step_offset = metadata["completed_timesteps"]
        initial_eval_counter = metadata["eval_step"]
        remaining_timesteps = cfg.train.num_timesteps - step_offset

        if remaining_timesteps <= 0:
            logging.info("Training already complete, nothing to resume.")
            return

        logging.info(
            f"Resuming run {run_id} from step {step_offset}, "
            f"{remaining_timesteps} timesteps remaining"
        )
    else:
        # --- Fresh start path ---
        run_id = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        model_path = hydra.utils.to_absolute_path(f"{model_path_base}/{run_id}")
        Path(model_path).mkdir(parents=True, exist_ok=True)
        wandb_run_id_to_resume = None

    logging.info(f"Model checkpoint path: {model_path}")

    # ----------------------------------------------------------------
    # Phase 2: Load frozen networks from checkpoint
    # ----------------------------------------------------------------
    prior_checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint.path)

    if cfg.mode == "distill_prior":
        logging.info(f"Loading distill_prior checkpoint from: {prior_checkpoint_path}")
        (
            encoder_params,
            prior_params,
            decoder_params,
            rgm_params,
            normalizer_params,
            ckpt_cfg,
        ) = load_distill_prior_checkpoint(prior_checkpoint_path, cfg.checkpoint.step)
    elif cfg.mode == "distill":
        step_prefix = "DistillNetwork"
        logging.info(f"Loading distill checkpoint from: {prior_checkpoint_path}")
        (
            encoder_params,
            prior_params,
            decoder_params,
            normalizer_params,
            ckpt_cfg,
        ) = load_prior_checkpoint(
            prior_checkpoint_path, cfg.checkpoint.step, step_prefix=step_prefix
        )
    else:
        step_prefix = "PriorNetwork"
        logging.info(f"Loading mlp_prior checkpoint from: {prior_checkpoint_path}")
        (
            encoder_params,
            prior_params,
            decoder_params,
            normalizer_params,
            ckpt_cfg,
        ) = load_prior_checkpoint(
            prior_checkpoint_path, cfg.checkpoint.step, step_prefix=step_prefix
        )

    latent_size = ckpt_cfg["network_config"]["intention_size"]
    proprio_size = ckpt_cfg["network_config"]["obs_sizes"]["proprioception"]
    logging.info(f"Latent size: {latent_size}, Proprio size: {proprio_size}")

    # Build frozen_networks dict for checkpoint saving
    frozen_networks = {
        "encoder_params": encoder_params,
        "prior_params": prior_params,
        "decoder_params": decoder_params,
        "imitation_normalizer": normalizer_params,
    }
    if cfg.mode == "distill_prior":
        frozen_networks["rgm_params"] = rgm_params

    # Create inference functions
    decoder_fn = make_decoder_inference_fn(decoder_params, normalizer_params, ckpt_cfg)

    # ----------------------------------------------------------------
    # Phase 3: Create environments (unchanged)
    # ----------------------------------------------------------------
    env_cfg = maintain_velocity.default_config()
    xml_variant = cfg.env.get("xml_variant", "full-collisions")
    if xml_variant == "limited-collisions":
        env_cfg.walker_xml_path = rodent_consts.RODENT_XML_PATH
    elif xml_variant == "full-collisions":
        env_cfg.walker_xml_path = rodent_consts.RODENT_FULL_COLLISION_XML
    elif xml_variant == "no-tail-collisions":
        env_cfg.walker_xml_path = rodent_consts.RODENT_NO_TAIL_COLLISION_XML
    else:
        raise ValueError(f"Unknown xml_variant: {xml_variant}")
    logging.info(f"Using xml_variant: {xml_variant}")
    env_cfg.mujoco_impl = cfg.env.get("mujoco_impl", "jax")
    logging.info(f"Using mujoco_impl: {env_cfg.mujoco_impl}")
    env_cfg.ctrl_dt = ckpt_cfg["env_config"]["ctrl_dt"]
    env_cfg.init_z = cfg.env.init_z
    env_cfg.target_speed = cfg.env.target_speed
    env_cfg.reward_terms = {
        "forward_velocity": {"weight": cfg.env.reward_weights.forward_velocity},
        "lateral_velocity": {"weight": cfg.env.reward_weights.lateral_velocity},
        "angular_velocity_z": {"weight": cfg.env.reward_weights.angular_velocity_z},
        "control_diff_cost": {"weight": cfg.env.reward_weights.control_diff_cost},
    }
    logging.info(
        f"Environment config: ctrl_dt={env_cfg.ctrl_dt}, "
        f"target_speed={env_cfg.target_speed}"
    )
    logging.info(
        f"Reward weights: forward_velocity={cfg.env.reward_weights.forward_velocity}, "
        f"lateral_velocity={cfg.env.reward_weights.lateral_velocity}, "
        f"angular_velocity_z={cfg.env.reward_weights.angular_velocity_z}"
    )

    # Set actuator type: scratch mode can optionally use PD actuators;
    # decoder_only, prior_decoder, and distill always use torque (frozen
    # decoder was trained with torque actuators).
    if cfg.mode == "scratch":
        env_cfg.torque_actuators = cfg.scratch.get("torque_actuators", True)
        if not env_cfg.torque_actuators:
            logging.info("Scratch mode: using PD actuators")
        else:
            logging.info("Scratch mode: using torque actuators")
    else:
        env_cfg.torque_actuators = True

    def make_wrapped_env(is_eval: bool = False):
        base_env = maintain_velocity.MaintainVelocity(config=env_cfg)

        if cfg.mode == "decoder_only":
            return DecoderHighLevelWrapper(
                base_env, decoder_fn, latent_size, proprio_size
            )
        elif cfg.mode == "scratch":
            return FlatObsWrapper(base_env)
        elif cfg.mode == "distill_prior":
            prior_fn = make_prior_inference_fn(
                prior_params, normalizer_params, ckpt_cfg
            )
            rgm_fn = make_rgm_inference_fn(rgm_params, normalizer_params, ckpt_cfg)
            mode_cfg = cfg.distill_prior
            return PriorRGMDecoderHighLevelWrapper(
                base_env,
                prior_fn,
                rgm_fn,
                decoder_fn,
                latent_size,
                proprio_size,
                gate_bias=mode_cfg.gate_bias,
                rgm_noise=mode_cfg.rgm_noise,
            )
        else:  # prior_decoder or distill
            prior_fn = make_prior_inference_fn(
                prior_params, normalizer_params, ckpt_cfg
            )
            mode_cfg = cfg.distill if cfg.mode == "distill" else cfg.prior_decoder
            return PriorDecoderHighLevelWrapper(
                base_env,
                prior_fn,
                decoder_fn,
                latent_size,
                proprio_size,
                deterministic_prior=mode_cfg.deterministic_prior,
                noise_logvar=mode_cfg.noise_logvar,
            )

    env = make_wrapped_env(is_eval=False)
    eval_env = make_wrapped_env(is_eval=True)

    logging.info(f"Training mode: {cfg.mode}")
    logging.info(f"Environment action size (latent): {env.action_size}")

    # ----------------------------------------------------------------
    # Phase 4: Build config dict and save to model path
    # ----------------------------------------------------------------
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_dict["checkpoint_config"] = ckpt_cfg
    with open(Path(model_path) / "config.json", "w") as fp:
        json.dump(config_dict, fp, indent=4, default=str)

    # ----------------------------------------------------------------
    # Phase 5: Setup normalization and networks
    # ----------------------------------------------------------------
    normalize = lambda x, y: x
    if cfg.train.normalize_observations:
        normalize = running_statistics.normalize

    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    rng = jax.random.PRNGKey(cfg.train.seed)
    start_state = jit_reset(rng)

    action_size = env.action_size

    if cfg.mode == "scratch":
        obs_size = start_state.obs.shape[-1]
        task_obs_size = obs_size - proprio_size

        decoder_hidden_layer_sizes = tuple(
            ckpt_cfg["network_config"]["decoder_layer_sizes"]
        )

        def network_factory(*args, **kwargs):
            return make_scratch_ppo_networks(
                task_obs_size=task_obs_size,
                proprio_size=proprio_size,
                action_size=action_size,
                latent_size=latent_size,
                policy_hidden_layer_sizes=tuple(cfg.network.policy_layers),
                decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
                value_hidden_layer_sizes=tuple(cfg.network.value_layers),
            )

        scratch_network = network_factory()

        def make_scratch_logging_policy(scratch_networks):
            """Create logging policy for scratch mode."""
            policy_network = scratch_networks.policy_network
            parametric_action_distribution = (
                scratch_networks.parametric_action_distribution
            )

            def logging_policy(params, observations, key_sample):
                del key_sample  # Deterministic
                param_subset = (params[0], params[1])
                logits = policy_network.apply(*param_subset, observations)
                action = parametric_action_distribution.mode(logits)
                return action, {}

            return logging_policy

        jit_logging_inference_fn = jax.jit(make_scratch_logging_policy(scratch_network))
    else:
        obs_size = start_state.obs.shape[-1]

        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=tuple(cfg.network.policy_layers),
            value_hidden_layer_sizes=tuple(cfg.network.value_layers),
        )

        ppo_network = network_factory(
            obs_size,
            action_size,
            preprocess_observations_fn=normalize,
        )

        def make_logging_inference_fn(ppo_networks):
            """Create inference function for logging rollouts."""

            def make_logging_policy(deterministic=True):
                policy_network = ppo_networks.policy_network
                parametric_action_distribution = (
                    ppo_networks.parametric_action_distribution
                )

                def logging_policy(params, observations, key_sample):
                    param_subset = (params[0], params[1])
                    logits = policy_network.apply(*param_subset, observations)
                    if deterministic:
                        return parametric_action_distribution.mode(logits), {}
                    raw_actions = (
                        parametric_action_distribution.sample_no_postprocessing(
                            logits, key_sample
                        )
                    )
                    postprocessed_actions = parametric_action_distribution.postprocess(
                        raw_actions
                    )
                    return postprocessed_actions, {}

                return logging_policy

            return make_logging_policy

        make_logging_policy = make_logging_inference_fn(ppo_network)
        jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    # ----------------------------------------------------------------
    # Phase 6: If resuming, load Brax params from task transfer checkpoint
    # ----------------------------------------------------------------
    if existing_run_state is not None:
        # Build an abstract brax_params tuple for Orbax structure matching.
        # Brax params are (normalizer_params, policy_params, value_params).
        # We initialize a dummy network to get the correct pytree shapes.
        if cfg.mode == "scratch":
            dummy_net = network_factory()
            key_p, key_v = jax.random.split(jax.random.PRNGKey(0))
            dummy_normalizer = running_statistics.init_state(
                specs.Array(start_state.obs.shape[-1:], jnp.dtype("float32"))
            )
            dummy_policy = dummy_net.policy_network.init(key_p)
            dummy_value = dummy_net.value_network.init(key_v)
        else:
            dummy_net = network_factory(
                obs_size, action_size, preprocess_observations_fn=normalize
            )
            key_p, key_v = jax.random.split(jax.random.PRNGKey(0))
            dummy_normalizer = running_statistics.init_state(
                specs.Array(start_state.obs.shape[-1:], jnp.dtype("float32"))
            )
            dummy_policy = dummy_net.policy_network.init(key_p)
            dummy_value = dummy_net.value_network.init(key_v)

        abstract_brax_params = (dummy_normalizer, dummy_policy, dummy_value)

        restore_params_for_brax = load_task_transfer_brax_params(
            checkpoint_path=model_path,
            abstract_brax_params=abstract_brax_params,
            step=ckpt_info["step"],
        )
        logging.info("Loaded Brax params for resume")

    # ----------------------------------------------------------------
    # Phase 7: Initialize wandb (with resume support)
    # ----------------------------------------------------------------
    if wandb_run_id_to_resume is not None:
        wandb_run_id = wandb_run_id_to_resume
        wandb.init(
            project=cfg.logging.project,
            group=cfg.logging.group_name,
            config=config_dict,
            id=wandb_run_id,
            resume="allow",
            notes=f"checkpoint: {cfg.checkpoint.path}, mode: {cfg.mode}",
        )
    else:
        wandb_run_id = f"{cfg.logging.exp_name}_{run_id}"
        wandb.init(
            project=cfg.logging.project,
            group=cfg.logging.group_name,
            config=config_dict,
            id=wandb_run_id,
            notes=f"checkpoint: {cfg.checkpoint.path}, mode: {cfg.mode}",
        )

    # ----------------------------------------------------------------
    # Phase 8: Save run state and create checkpoint infrastructure
    # ----------------------------------------------------------------
    checkpointing_enabled = cfg.checkpointing.get("enabled", True)

    if checkpointing_enabled:
        save_run_state(
            cfg=cfg,
            model_path_base=model_path_base,
            run_id=run_id,
            checkpoint_path=model_path,
            wandb_run_id=wandb.run.id,
        )

        checkpoint_callback = create_checkpoint_callback(
            cfg=cfg,
            model_path_base=model_path_base,
            run_id=run_id,
            checkpoint_path=model_path,
            wandb_run_id=wandb.run.id,
        )

        max_to_keep = cfg.checkpointing.get("max_to_keep", 3)
        ckpt_mgr = ocp.CheckpointManager(
            model_path,
            options=ocp.CheckpointManagerOptions(
                create=True,
                step_prefix=_STEP_PREFIX,
                max_to_keep=max_to_keep,
            ),
        )
    else:
        checkpoint_callback = None
        ckpt_mgr = None

    # ----------------------------------------------------------------
    # Phase 9: Define policy_params_fn with checkpoint saving
    # ----------------------------------------------------------------
    eval_counter = [initial_eval_counter]

    def policy_params_fn(current_step, make_policy, params, jit_logging_inference_fn):
        del make_policy  # Unused, use our custom logging inference

        adjusted_step = current_step + step_offset

        # Skip the initial eval on resume — it duplicates the last pre-preemption eval
        if current_step == 0 and step_offset > 0:
            eval_counter[0] += 1
            return

        # Rollout logging
        rollout_logging_fn(
            env=eval_env,
            jit_reset=jit_reset,
            jit_step=jit_step,
            jit_inference_fn=lambda p, obs, rng: jit_logging_inference_fn(p, obs, rng),
            params=params,
            current_step=adjusted_step,
            model_path=model_path,
            episode_length=cfg.train.episode_length,
            render_camera=cfg.logging.render_camera,
            render_fps=cfg.logging.render_fps,
            render_height=cfg.logging.render_height,
            render_width=cfg.logging.render_width,
            render_every=cfg.logging.render_every,
            eval_count=eval_counter[0],
        )

        # Save checkpoint (skip on very first eval call which is at step 0)
        if checkpointing_enabled and ckpt_mgr is not None and current_step > 0:
            training_metadata = {
                "completed_timesteps": adjusted_step,
                "eval_step": eval_counter[0],
                "wandb_run_id": wandb.run.id,
                "run_id": run_id,
                "mode": cfg.mode,
            }
            save_task_transfer_checkpoint(
                ckpt_mgr=ckpt_mgr,
                step=eval_counter[0],
                brax_params=params,
                frozen_networks=frozen_networks,
                config=config_dict,
                training_metadata=training_metadata,
                checkpoint_callback=checkpoint_callback,
            )

        eval_counter[0] += 1

    # ----------------------------------------------------------------
    # Phase 10: Build training config and run
    # ----------------------------------------------------------------
    training_params = {
        "num_timesteps": remaining_timesteps,
        "episode_length": cfg.train.episode_length,
        "num_envs": cfg.train.num_envs,
        "batch_size": cfg.train.batch_size,
        "num_minibatches": cfg.train.num_minibatches,
        "num_updates_per_batch": cfg.train.num_updates_per_batch,
        "learning_rate": cfg.train.learning_rate,
        "entropy_cost": cfg.train.entropy_cost,
        "discounting": cfg.train.discounting,
        "clipping_epsilon": cfg.train.clipping_epsilon,
        "gae_lambda": cfg.train.gae_lambda,
        "max_grad_norm": cfg.train.max_grad_norm,
        "vf_loss_coefficient": cfg.train.vf_loss_coefficient,
        "reward_scaling": cfg.train.reward_scaling,
        "normalize_observations": cfg.train.normalize_observations,
        "unroll_length": cfg.train.unroll_length,
        "action_repeat": cfg.train.get("action_repeat", 1),
        "seed": cfg.train.seed,
    }

    num_evals = max(int(remaining_timesteps / cfg.train.eval_every), 1)
    progress_fn = make_wandb_progress_with_offset(
        step_offset, episode_length=cfg.train.episode_length
    )

    train_fn = functools.partial(
        brax_ppo.train,
        **training_params,
        num_evals=num_evals,
        network_factory=network_factory,
        restore_checkpoint_path=None,
        restore_params=restore_params_for_brax,
        progress_fn=progress_fn,
        wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training),
    )

    logging.info("Starting training...")
    make_inference_fn, params, _ = train_fn(
        environment=env,
        eval_env=eval_env,
        policy_params_fn=functools.partial(
            policy_params_fn, jit_logging_inference_fn=jit_logging_inference_fn
        ),
    )

    # ----------------------------------------------------------------
    # Phase 11: Save final checkpoint and cleanup
    # ----------------------------------------------------------------
    if checkpointing_enabled and ckpt_mgr is not None:
        training_metadata = {
            "completed_timesteps": cfg.train.num_timesteps,
            "eval_step": eval_counter[0],
            "wandb_run_id": wandb.run.id,
            "run_id": run_id,
            "mode": cfg.mode,
            "training_complete": True,
        }
        save_task_transfer_checkpoint(
            ckpt_mgr=ckpt_mgr,
            step=eval_counter[0],
            brax_params=params,
            frozen_networks=frozen_networks,
            config=config_dict,
            training_metadata=training_metadata,
            checkpoint_callback=checkpoint_callback,
        )

    if checkpointing_enabled:
        try:
            cleanup_run_state(cfg, model_path_base)
            logging.info("Training completed successfully, cleaned up run state")
        except Exception as e:
            logging.warning(f"Failed to cleanup run state: {e}")

    logging.info("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
