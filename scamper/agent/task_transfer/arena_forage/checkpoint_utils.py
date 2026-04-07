"""Checkpoint utilities for task transfer training.

This module provides functions to:
- Load prior, decoder, and encoder parameters from mlp_prior checkpoints
- Create inference functions for the frozen prior and decoder networks
- Save and load task transfer training checkpoints (frozen networks + PPO state)
- Manage run state files for SLURM preemption recovery

Handles both legacy flat normalizer format and new dict normalizer format.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from brax.training import distribution
from brax.training.acme import running_statistics, specs
from omegaconf import DictConfig, OmegaConf

from scamper.agent import checkpointing
from scamper.agent.imitation import intention_network
from scamper.agent.mlp_prior.prior_networks import Prior
from scamper.agent.observation_utils import (
    DictRunningStatisticsState,
    convert_flat_to_dict_normalizer,
)

# Orbax step prefix for task transfer checkpoints
_STEP_PREFIX = "TaskTransfer"


def load_prior_checkpoint(
    checkpoint_path: str,
    step: int | None = None,
    step_prefix: str = "PriorNetwork",
) -> Tuple[Dict, Dict, Dict, DictRunningStatisticsState, Dict]:
    """Load encoder, prior, and decoder parameters from an mlp_prior or distill checkpoint.

    Both checkpoint types share the same structure:
    (normalizer_params, {"params": {"encoder": ..., "decoder": ..., "prior": ...}})

    Handles both legacy flat normalizer and new dict normalizer formats.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        step: Specific step to load. If None, loads the latest.
        step_prefix: Orbax step prefix. "PriorNetwork" for mlp_prior,
            "DistillNetwork" for distill checkpoints.

    Returns:
        Tuple of (encoder_params, prior_params, decoder_params,
                  normalizer_params, config_dict).
        normalizer_params is always returned as DictRunningStatisticsState.
    """
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=step_prefix)

    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()

        # Load config
        cfg = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(config=ocp.args.JsonRestore()),
        )["config"]

    cfg = OmegaConf.create(cfg)

    # Check if using dict format or legacy flat format
    obs_sizes = cfg.network_config.get("obs_sizes", None)

    if obs_sizes is not None:
        # New dict format
        abstract_policy = _create_abstract_prior_policy_dict(cfg)

        with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
            policy_params = ckpt_mgr.restore(
                step,
                args=ocp.args.Composite(
                    policy=ocp.args.StandardRestore(abstract_policy)
                ),
            )["policy"]

        normalizer_params, network_params = policy_params
    else:
        # Legacy flat format - need to convert
        reference_obs_size = cfg.network_config.reference_obs_size
        observation_size = cfg.network_config.observation_size

        abstract_policy = _create_abstract_prior_policy_flat(cfg)

        with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
            policy_params = ckpt_mgr.restore(
                step,
                args=ocp.args.Composite(
                    policy=ocp.args.StandardRestore(abstract_policy)
                ),
            )["policy"]

        flat_normalizer_params, network_params = policy_params

        # Convert flat normalizer to dict format
        normalizer_params = convert_flat_to_dict_normalizer(
            flat_normalizer_params, reference_obs_size
        )

        # Add obs_sizes to config for downstream use
        cfg.network_config.obs_sizes = {
            "imitation_target": reference_obs_size,
            "proprioception": observation_size - reference_obs_size,
        }

    # Extract encoder, prior, and decoder params
    encoder_params = network_params["params"]["encoder"]
    prior_params = network_params["params"]["prior"]
    decoder_params = network_params["params"]["decoder"]

    return (
        encoder_params,
        prior_params,
        decoder_params,
        normalizer_params,
        OmegaConf.to_container(cfg),
    )


def _create_abstract_prior_policy_dict(cfg: OmegaConf) -> Tuple[Any, Any]:
    """Create abstract policy structure for dict-format checkpoint restoration.

    Args:
        cfg: Configuration from checkpoint (must have obs_sizes).

    Returns:
        Tuple of (normalizer_state, combined_policy_params) with correct structure.
    """
    latent_size = cfg.network_config.intention_size
    action_size = cfg.network_config.action_size
    obs_sizes = dict(cfg.network_config.obs_sizes)

    reference_obs_size = obs_sizes["imitation_target"]
    proprioceptive_obs_size = obs_sizes["proprioception"]

    encoder_hidden_layer_sizes = tuple(cfg.network_config.encoder_layer_sizes)
    decoder_hidden_layer_sizes = tuple(cfg.network_config.decoder_layer_sizes)
    prior_hidden_layer_sizes = tuple(
        cfg.network_config.get("prior_layer_sizes", [1024, 1024])
    )

    # Create network modules and initialize
    combined_params = _init_network_params(
        latent_size,
        action_size,
        reference_obs_size,
        proprioceptive_obs_size,
        encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes,
        prior_hidden_layer_sizes,
    )

    # Create dict normalizer state
    normalizer_state = DictRunningStatisticsState(
        imitation_target=running_statistics.init_state(
            specs.Array(reference_obs_size, jnp.dtype("float32"))
        ),
        proprioception=running_statistics.init_state(
            specs.Array(proprioceptive_obs_size, jnp.dtype("float32"))
        ),
    )

    return (normalizer_state, combined_params)


def _create_abstract_prior_policy_flat(cfg: OmegaConf) -> Tuple[Any, Any]:
    """Create abstract policy structure for flat-format (legacy) checkpoint restoration.

    Args:
        cfg: Configuration from checkpoint (has observation_size/reference_obs_size).

    Returns:
        Tuple of (normalizer_state, combined_policy_params) with correct structure.
    """
    latent_size = cfg.network_config.intention_size
    action_size = cfg.network_config.action_size
    observation_size = cfg.network_config.observation_size
    reference_obs_size = cfg.network_config.reference_obs_size
    proprioceptive_obs_size = observation_size - reference_obs_size

    encoder_hidden_layer_sizes = tuple(cfg.network_config.encoder_layer_sizes)
    decoder_hidden_layer_sizes = tuple(cfg.network_config.decoder_layer_sizes)
    prior_hidden_layer_sizes = tuple(
        cfg.network_config.get("prior_layer_sizes", [1024, 1024])
    )

    # Create network modules and initialize
    combined_params = _init_network_params(
        latent_size,
        action_size,
        reference_obs_size,
        proprioceptive_obs_size,
        encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes,
        prior_hidden_layer_sizes,
    )

    # Create flat normalizer state (legacy format)
    normalizer_state = running_statistics.init_state(
        specs.Array(observation_size, jnp.dtype("float32"))
    )

    return (normalizer_state, combined_params)


def _init_network_params(
    latent_size: int,
    action_size: int,
    reference_obs_size: int,
    proprioceptive_obs_size: int,
    encoder_hidden_layer_sizes: tuple,
    decoder_hidden_layer_sizes: tuple,
    prior_hidden_layer_sizes: tuple,
) -> Dict:
    """Initialize network parameters for checkpoint structure matching."""
    # Create encoder module
    encoder_module = intention_network.Encoder(
        layer_sizes=list(encoder_hidden_layer_sizes),
        latents=latent_size,
    )

    # Create decoder module
    action_param_size = action_size * 2  # mean and std
    decoder_module = intention_network.Decoder(
        layer_sizes=list(decoder_hidden_layer_sizes) + [action_param_size],
    )

    # Create prior module
    prior_module = Prior(
        layer_sizes=list(prior_hidden_layer_sizes),
        latents=latent_size,
    )

    # Initialize with dummy inputs
    key = jax.random.PRNGKey(0)
    key_enc, key_dec, key_prior = jax.random.split(key, 3)

    dummy_traj_obs = jnp.zeros((1, reference_obs_size))
    dummy_proprio_obs = jnp.zeros((1, proprioceptive_obs_size))
    dummy_decoder_input = jnp.zeros((1, latent_size + proprioceptive_obs_size))

    encoder_init = encoder_module.init(key_enc, dummy_traj_obs)
    decoder_init = decoder_module.init(key_dec, dummy_decoder_input)
    prior_init = prior_module.init(key_prior, dummy_proprio_obs)

    # Combine into expected structure
    return {
        "params": {
            "encoder": encoder_init["params"],
            "decoder": decoder_init["params"],
            "prior": prior_init["params"],
        }
    }


def make_decoder_inference_fn(
    decoder_params: Dict,
    normalizer_params: DictRunningStatisticsState,
    config: Dict,
) -> Callable:
    """Create decoder inference function for DecoderHighLevelWrapper.

    The decoder takes [latent, proprio] concatenated input and outputs action.

    Args:
        decoder_params: Frozen decoder parameters.
        normalizer_params: Dict normalizer with proprioception stats.
        config: Checkpoint config dict.

    Returns:
        Function (latent_proprio) -> (action, extras)
        where latent_proprio is [latent, proprio] concatenated.
    """
    action_size = config["network_config"]["action_size"]
    latent_size = config["network_config"]["intention_size"]
    decoder_hidden_layer_sizes = tuple(config["network_config"]["decoder_layer_sizes"])

    # Create action distribution
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    # Create decoder module
    decoder_module = intention_network.Decoder(
        layer_sizes=list(decoder_hidden_layer_sizes)
        + [parametric_action_distribution.param_size],
    )

    # Get proprioception normalizer stats
    proprio_normalizer = normalizer_params.proprioception

    def decoder_fn(latent_proprio: jnp.ndarray) -> Tuple[jnp.ndarray, Dict]:
        """Apply decoder to latent + proprio input.

        Args:
            latent_proprio: Concatenated [latent, proprio] array.

        Returns:
            Tuple of (action, extras_dict).
        """
        # Split input
        latent = latent_proprio[..., :latent_size]
        proprio = latent_proprio[..., latent_size:]

        # Normalize proprioception
        normalized_proprio = running_statistics.normalize(proprio, proprio_normalizer)

        # Concatenate normalized input for decoder
        decoder_input = jnp.concatenate([latent, normalized_proprio], axis=-1)

        # Apply decoder
        logits, _ = decoder_module.apply({"params": decoder_params}, decoder_input)

        # Get deterministic action (mode of distribution)
        action = parametric_action_distribution.mode(logits)

        return action, {"logits": logits}

    return decoder_fn


def make_prior_inference_fn(
    prior_params: Dict,
    normalizer_params: DictRunningStatisticsState,
    config: Dict,
) -> Callable:
    """Create prior inference function for PriorDecoderHighLevelWrapper.

    The prior takes proprio input and outputs latent distribution parameters.

    Args:
        prior_params: Frozen prior parameters.
        normalizer_params: Dict normalizer with proprioception stats.
        config: Checkpoint config dict.

    Returns:
        Function (proprio) -> (mean, logvar)
    """
    latent_size = config["network_config"]["intention_size"]
    prior_hidden_layer_sizes = tuple(
        config["network_config"].get("prior_layer_sizes", [1024, 1024])
    )

    # Create prior module
    prior_module = Prior(
        layer_sizes=list(prior_hidden_layer_sizes),
        latents=latent_size,
    )

    # Get proprioception normalizer stats
    proprio_normalizer = normalizer_params.proprioception

    def prior_fn(proprio: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply prior to proprioceptive observations.

        Args:
            proprio: Proprioceptive observations.

        Returns:
            Tuple of (mean, logvar) for latent distribution.
        """
        # Normalize proprioception
        normalized_proprio = running_statistics.normalize(proprio, proprio_normalizer)

        # Apply prior
        mean, logvar = prior_module.apply({"params": prior_params}, normalized_proprio)

        return mean, logvar

    return prior_fn


# ---------------------------------------------------------------------------
# Task transfer checkpointing and preemption recovery
# ---------------------------------------------------------------------------


def hash_task_transfer_config(cfg: DictConfig) -> str:
    """Create a short hash of the task-transfer config for run state matching.

    Hashes only the parts of the config that define the training run
    identity (mode, checkpoint source, network, train, env settings).
    Excludes model_path and run_id since those vary across preemptions.

    Args:
        cfg: Task transfer Hydra config.

    Returns:
        12-character MD5 hash string.
    """
    hashable = {
        "mode": cfg.mode,
        "checkpoint": {
            "path": cfg.checkpoint.path,
            "step": cfg.checkpoint.step,
        },
        "network": OmegaConf.to_container(cfg.network, resolve=True),
        "train": OmegaConf.to_container(cfg.train, resolve=True),
        "env": OmegaConf.to_container(cfg.env, resolve=True),
    }
    if cfg.mode == "prior_decoder":
        hashable["prior_decoder"] = OmegaConf.to_container(
            cfg.prior_decoder, resolve=True
        )
    elif cfg.mode == "distill":
        hashable["distill"] = OmegaConf.to_container(cfg.distill, resolve=True)
    cfg_str = json.dumps(hashable, sort_keys=True, default=str)
    return hashlib.md5(cfg_str.encode()).hexdigest()[:12]


def get_run_state_file_path(cfg: DictConfig, model_path_base: str) -> Path:
    """Get the path to the run state file for this job/config combination.

    Args:
        cfg: Task transfer config.
        model_path_base: Base directory for model checkpoints.

    Returns:
        Path to the run state JSON file.
    """
    base_path = Path(model_path_base).resolve()
    job_id = checkpointing.get_job_identifier()
    config_hash = hash_task_transfer_config(cfg)
    return base_path / f"run_state_{job_id}_{config_hash}.json"


def _find_run_state_file(cfg: DictConfig, model_path_base: str) -> Path | None:
    """Find a run state file for this config, tolerating SLURM job ID changes.

    First tries an exact match on job ID + config hash (fast path for
    preemption requeues where the job ID is preserved). Falls back to a
    glob search by config hash so that manual resubmissions with a new
    SLURM job ID can still discover the previous run state.

    Args:
        cfg: Task transfer config.
        model_path_base: Base directory for model checkpoints.

    Returns:
        Path to the run state file, or None if not found.
    """
    exact_path = get_run_state_file_path(cfg, model_path_base)
    if exact_path.exists():
        return exact_path

    # Fallback: search by config hash alone (handles new SLURM job IDs)
    config_hash = hash_task_transfer_config(cfg)
    base_path = Path(model_path_base).resolve()
    candidates = sorted(
        base_path.glob(f"run_state_*_{config_hash}.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        logging.info(
            f"No exact run state match; found {len(candidates)} candidate(s) "
            f"by config hash, using most recent: {candidates[0].name}"
        )
        return candidates[0]

    return None


def discover_existing_run_state(
    cfg: DictConfig, model_path_base: str
) -> dict[str, Any] | None:
    """Discover existing run state for preemption recovery.

    Searches for a run state file matching the current config (tolerating
    SLURM job ID changes across resubmissions), validates it, and checks
    that the checkpoint directory is accessible with valid steps.

    Args:
        cfg: Current task transfer config.
        model_path_base: Base directory where checkpoint subdirs are created.

    Returns:
        Run state dict with ``latest_checkpoint_step`` added if valid,
        None if no valid run state is found.
    """
    state_file_path = _find_run_state_file(cfg, model_path_base)
    if state_file_path is None:
        logging.info("No existing run state found")
        return None
    logging.info(f"Found task transfer run state at: {state_file_path}")

    run_state = checkpointing.read_json_with_lock(state_file_path)
    if not run_state:
        logging.info("Run state file exists but could not be read")
        return None

    required_keys = ["run_id", "checkpoint_path", "wandb_run_id", "config_hash"]
    if not all(key in run_state for key in required_keys):
        logging.warning("Run state file missing required keys, ignoring")
        return None

    current_hash = hash_task_transfer_config(cfg)
    if run_state["config_hash"] != current_hash:
        logging.warning(
            f"Config hash mismatch (saved: {run_state['config_hash']}, "
            f"current: {current_hash}), ignoring run state"
        )
        return None

    checkpoint_path = Path(run_state["checkpoint_path"])
    if not checkpoint_path.exists():
        logging.warning(f"Checkpoint directory {checkpoint_path} not found, ignoring")
        return None

    try:
        with ocp.CheckpointManager(
            str(checkpoint_path),
            options=ocp.CheckpointManagerOptions(
                create=False, step_prefix=_STEP_PREFIX
            ),
        ) as mgr:
            latest = mgr.latest_step()
        if latest is None:
            logging.warning("No valid checkpoints found in directory, ignoring")
            return None
        run_state["latest_checkpoint_step"] = latest
        logging.info(f"Found valid run state with checkpoint at step {latest}")
        return run_state
    except Exception as e:
        logging.warning(f"Failed to access checkpoint manager: {e}")
        return None


def save_run_state(
    cfg: DictConfig,
    model_path_base: str,
    run_id: str,
    checkpoint_path: str,
    wandb_run_id: str,
    latest_step: int | None = None,
) -> None:
    """Save run state for preemption recovery.

    Args:
        cfg: Task transfer config.
        model_path_base: Base directory for model checkpoints.
        run_id: Unique run identifier.
        checkpoint_path: Path to checkpoint directory.
        wandb_run_id: Wandb run ID for resuming logging.
        latest_step: Most recent checkpoint step, if known.
    """
    state_file_path = get_run_state_file_path(cfg, model_path_base)

    run_state = {
        "run_id": run_id,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "wandb_run_id": wandb_run_id,
        "config_hash": hash_task_transfer_config(cfg),
        "timestamp": time.time(),
    }
    if latest_step is not None:
        run_state["latest_checkpoint_step"] = latest_step

    try:
        checkpointing.atomic_write_json(state_file_path, run_state)
        logging.info(f"Saved run state to {state_file_path}")
    except Exception as e:
        logging.error(f"Failed to save run state: {e}")


def cleanup_run_state(cfg: DictConfig, model_path_base: str) -> None:
    """Remove run state file after successful training completion.

    Args:
        cfg: Task transfer config.
        model_path_base: Base directory for model checkpoints.
    """
    state_file_path = get_run_state_file_path(cfg, model_path_base)
    try:
        if state_file_path.exists():
            state_file_path.unlink()
            logging.info(f"Cleaned up run state: {state_file_path}")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")


def create_checkpoint_callback(
    cfg: DictConfig,
    model_path_base: str,
    run_id: str,
    checkpoint_path: str,
    wandb_run_id: str,
) -> Callable[[int], None]:
    """Create a callback that updates run state after each checkpoint save.

    Args:
        cfg: Task transfer config.
        model_path_base: Base directory for model checkpoints.
        run_id: Unique run identifier.
        checkpoint_path: Path to checkpoint directory.
        wandb_run_id: Wandb run ID.

    Returns:
        Callback function that takes a step number and updates run state.
    """

    def callback(step: int) -> None:
        try:
            save_run_state(
                cfg=cfg,
                model_path_base=model_path_base,
                run_id=run_id,
                checkpoint_path=checkpoint_path,
                wandb_run_id=wandb_run_id,
                latest_step=step,
            )
            logging.debug(f"Updated run state after checkpoint save at step {step}")
        except Exception as e:
            logging.warning(f"Failed to update run state after checkpoint save: {e}")

    return callback


def save_task_transfer_checkpoint(
    ckpt_mgr: ocp.CheckpointManager,
    step: int,
    brax_params: tuple,
    frozen_networks: dict,
    config: dict,
    training_metadata: dict,
    checkpoint_callback: Callable[[int], None] | None = None,
) -> None:
    """Save a full task transfer checkpoint.

    Saves frozen networks (encoder, prior, decoder, normalizer) alongside the
    PPO training state so the checkpoint is self-contained.

    Args:
        ckpt_mgr: Orbax checkpoint manager (reused across saves).
        step: Checkpoint step number.
        brax_params: Tuple of (normalizer_params, policy_params, value_params).
        frozen_networks: Dict with encoder_params, prior_params,
            decoder_params, imitation_normalizer.
        config: Full config dict.
        training_metadata: Dict with completed_timesteps, eval_step,
            wandb_run_id, run_id, mode.
        checkpoint_callback: Optional callback called with step after save.
    """
    ckpt_mgr.save(
        step=step,
        args=ocp.args.Composite(
            brax_params=ocp.args.StandardSave(brax_params),
            frozen_networks=ocp.args.StandardSave(frozen_networks),
            config=ocp.args.JsonSave(config),
            training_metadata=ocp.args.JsonSave(training_metadata),
        ),
    )
    logging.info(f"Saved task transfer checkpoint at step {step}")

    if checkpoint_callback is not None:
        try:
            checkpoint_callback(step)
        except Exception as e:
            logging.warning(f"Checkpoint callback failed: {e}")


def load_task_transfer_metadata(
    checkpoint_path: str,
    step: int | None = None,
) -> dict[str, Any]:
    """Load config and training metadata from a task transfer checkpoint.

    This is a lightweight load that only reads the JSON items (no array data),
    used to determine resume offsets before constructing abstract pytrees.

    Args:
        checkpoint_path: Path to checkpoint directory.
        step: Specific step to load. None = latest.

    Returns:
        Dict with keys ``config``, ``training_metadata``, and ``step``.
    """
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=_STEP_PREFIX)
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as mgr:
        if step is None:
            step = mgr.latest_step()

        json_data = mgr.restore(
            step,
            args=ocp.args.Composite(
                config=ocp.args.JsonRestore(),
                training_metadata=ocp.args.JsonRestore(),
            ),
        )

    return {
        "config": json_data["config"],
        "training_metadata": json_data["training_metadata"],
        "step": step,
    }


def load_task_transfer_brax_params(
    checkpoint_path: str,
    abstract_brax_params: Any,
    step: int | None = None,
) -> tuple:
    """Load Brax PPO params from a task transfer checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory.
        abstract_brax_params: Abstract pytree defining the expected structure
            of (normalizer_params, policy_params, value_params).
        step: Specific step to load. None = latest.

    Returns:
        Tuple of (normalizer_params, policy_params, value_params).
    """
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=_STEP_PREFIX)
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as mgr:
        if step is None:
            step = mgr.latest_step()

        data = mgr.restore(
            step,
            args=ocp.args.Composite(
                brax_params=ocp.args.StandardRestore(abstract_brax_params),
            ),
        )

    logging.info(f"Loaded Brax params from task transfer checkpoint at step {step}")
    return data["brax_params"]
