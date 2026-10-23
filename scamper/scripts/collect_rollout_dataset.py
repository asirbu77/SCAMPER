#!/usr/bin/env python3
"""Collect rollout dataset from distill or distill_prior checkpoint for discriminator training.

This script generates qpos trajectories and sampled latent vectors from:
1. Original reference data
2. Teacher (MLPPolicy) simulation rollouts (distill only)
3. Student encoder-decoder simulation rollouts (DistillEncoder + DistillDecoder)
4. Prior or RGM simulation rollouts with varying logvar values

For distill checkpoints (--checkpoint-type distill):
- teacher_qpos / teacher_latents: Teacher rollouts (latents are zeros — no bottleneck)
- student_encoder_decoder_qpos / student_encoder_decoder_latents: Student enc-dec rollouts
- prior_deterministic_qpos / prior_deterministic_latents: Prior rollouts using mean
- prior_predicted_logvar_qpos / prior_predicted_logvar_latents: Prior rollouts using predicted logvar
- prior_logvar_X_qpos / prior_logvar_X_latents: Prior rollouts with specified fixed logvar

For distill_prior checkpoints (--checkpoint-type distill_prior):
- student_encoder_decoder_qpos / student_encoder_decoder_latents: Student enc-dec rollouts
- rgm_deterministic_qpos / rgm_deterministic_latents: RGM rollouts using mean
- rgm_predicted_logvar_qpos / rgm_predicted_logvar_latents: RGM rollouts using predicted logvar
- rgm_logvar_X_qpos / rgm_logvar_X_latents: RGM rollouts with specified fixed logvar
"""

import argparse
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from brax.training import distribution
from brax.training.acme import running_statistics, specs
from jax import random
from ml_collections import ConfigDict
from mujoco import mjx
from omegaconf import OmegaConf
from tqdm import tqdm

from scamper.agent.distill_expert.distill_networks import (
    DistillNetworkParams,
    create_abstract_distill_policy,
    init_distill_params,
    make_distill_networks,
    make_student_inference_fn,
)
from scamper.agent.distill_expert.distill_train import DistillTrainingState
from scamper.agent.behavior_prior.behavior_prior_networks import (
    create_abstract_behavior_prior_policy,
    make_distill_encoder_decoder_inference_fn,
)
from scamper.agent.expert.mlp_network import MLPPolicy
from scamper.agent.imitation.prior_rollout_eval import (
    create_prior_policy,
    extract_prior_decoder_params,
)
from scamper.utils.observation_utils import (
    DictRunningStatisticsState,
    concat_flat_dict_obs,
    convert_flat_to_dict_normalizer,
    normalize_dict_obs,
)
from scamper.environment import wrappers
from scamper.environment.reference_clips import ReferenceClips
from scamper.environment.rodent import imitation


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect rollout data from distill checkpoint for discriminator training"
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to distill checkpoint directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path for output H5 file (e.g., rollout_dataset.h5)",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Checkpoint step to load (default: latest)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of clips to process in parallel (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--data-path-override",
        type=str,
        default=None,
        help="Override reference data path from checkpoint config",
    )
    parser.add_argument(
        "--logvars",
        type=float,
        nargs="*",
        default=[-4.0, -3.0, -2.0, -1.0, -0.5, 0.0],
        help="Logvar values for non-deterministic prior rollouts (default: -4 -3 -2 -1 -0.5 0)",
    )
    parser.add_argument(
        "--teacher-checkpoint-path",
        type=str,
        default=None,
        help="Override teacher checkpoint path (for loading teacher config)",
    )
    parser.add_argument(
        "--checkpoint-type",
        type=str,
        choices=["distill", "distill_prior"],
        default="distill_prior",
        help="Type of checkpoint to load (default: distill_prior)",
    )
    parser.add_argument(
        "--mujoco-impl",
        type=str,
        choices=["jax", "warp"],
        default=None,
        help="Override mujoco_impl backend (default: use checkpoint config)",
    )
    parser.add_argument(
        "--xml-variant",
        type=str,
        choices=["limited-collisions", "full-collisions", "no-tail-collisions"],
        default=None,
        help="Override walker XML variant (default: use checkpoint config)",
    )
    parser.add_argument(
        "--naconmax",
        type=int,
        default=None,
        help="Override naconmax (default: use checkpoint config)",
    )
    parser.add_argument(
        "--njmax",
        type=int,
        default=None,
        help="Override njmax (default: use checkpoint config)",
    )

    return parser.parse_args()


def get_observation_sizes(cfg: Any) -> Tuple[int, int, int]:
    """Extract observation sizes from config, supporting both legacy and dict formats.

    Args:
        cfg: OmegaConf configuration.

    Returns:
        Tuple of (reference_obs_size, proprioceptive_obs_size, total_obs_size).
    """
    network_config = cfg.network_config

    # Check for new dict-based format
    if hasattr(network_config, "obs_sizes") or "obs_sizes" in network_config:
        obs_sizes = network_config.obs_sizes
        reference_obs_size = int(obs_sizes["imitation_target"])
        proprioceptive_obs_size = int(obs_sizes["proprioception"])
        total_obs_size = reference_obs_size + proprioceptive_obs_size
    else:
        # Legacy flat format
        total_obs_size = int(network_config.observation_size)
        reference_obs_size = int(network_config.reference_obs_size)
        proprioceptive_obs_size = total_obs_size - reference_obs_size

    return reference_obs_size, proprioceptive_obs_size, total_obs_size


def fix_config_paths(
    cfg: Any, data_path_override: str | None = None, batch_size: int = 1
) -> Any:
    """Fix paths in config that may reference /tmp or other locations.

    Args:
        cfg: OmegaConf configuration.
        data_path_override: Optional override for reference_data_path.
        batch_size: Batch size for computing naconmax (nconmax * batch_size).

    Returns:
        Updated configuration.
    """
    PROJECT_ROOT = str(Path(__file__).parent.parent.parent)

    def fix_path(path_str):
        if not path_str:
            return path_str

        # Paths containing scamper/environment/ resolve via the current installation
        ENV_MARKER = "scamper/environment/"
        if ENV_MARKER in path_str:
            import scamper.environment

            env_root = str(Path(scamper.environment.__file__).parent)
            rel_path = path_str[path_str.index(ENV_MARKER) + len(ENV_MARKER) :]
            return os.path.join(env_root, rel_path)

        LEGACY_TASKS_MARKER = "vnl_playground/tasks/"
        if LEGACY_TASKS_MARKER in path_str:
            import scamper.environment

            env_root = str(Path(scamper.environment.__file__).parent)
            rel_path = path_str[
                path_str.index(LEGACY_TASKS_MARKER) + len(LEGACY_TASKS_MARKER) :
            ]
            return os.path.join(env_root, rel_path)

        # Any /tmp/<project>/... path: strip the prefix and resolve relative
        # to the current project root (handles /tmp/scamper, /tmp/track-mjx, etc.)
        if path_str.startswith("/tmp/"):
            parts = path_str.split("/")
            if len(parts) > 3:
                rest = "/".join(parts[3:])
                return os.path.join(PROJECT_ROOT, rest)

        return path_str

    OmegaConf.set_struct(cfg.env_config, False)

    if hasattr(cfg.env_config, "walker_xml_path"):
        OmegaConf.update(
            cfg.env_config, "walker_xml_path", fix_path(cfg.env_config.walker_xml_path)
        )
    if hasattr(cfg.env_config, "arena_xml_path"):
        OmegaConf.update(
            cfg.env_config, "arena_xml_path", fix_path(cfg.env_config.arena_xml_path)
        )

    if data_path_override:
        OmegaConf.update(cfg.env_config, "reference_data_path", data_path_override)
    elif hasattr(cfg.env_config, "reference_data_path"):
        OmegaConf.update(
            cfg.env_config,
            "reference_data_path",
            fix_path(cfg.env_config.reference_data_path),
        )

    # Compute naconmax for warp backend (naconmax = nconmax * batch_size)
    # Only compute if naconmax is not already set (legacy support)
    if hasattr(cfg.env_config, "nconmax") and not hasattr(cfg.env_config, "naconmax"):
        OmegaConf.update(
            cfg.env_config, "naconmax", cfg.env_config.nconmax * batch_size
        )

    OmegaConf.set_struct(cfg.env_config, True)
    return cfg


def create_environment(cfg: Any) -> Tuple[Any, ReferenceClips]:
    """Create the imitation environment from config.

    Args:
        cfg: Configuration from checkpoint.

    Returns:
        Tuple of (environment, reference_clips).
    """
    env_cfg = ConfigDict(OmegaConf.to_container(cfg.env_config, resolve=True))

    reference_clips = ReferenceClips(
        data_path=cfg.env_config.reference_data_path,
        n_frames_per_clip=cfg.env_config.clip_length,
        keep_clips_idx=cfg.env_config.get("keep_clips_idx", None),
    )

    env = wrappers.FlattenObsWrapper(
        imitation.Imitation(config=env_cfg, clips=reference_clips)
    )

    return env, reference_clips


def create_prior_policy_fn(
    policy_params: Tuple,
    cfg: Any,
    fixed_logvar: Optional[float],
    deterministic: bool,
) -> Callable:
    """Create prior policy using prior and decoder networks.

    Args:
        policy_params: Tuple of (normalizer_params, network_params).
        cfg: Configuration from checkpoint.
        fixed_logvar: Fixed log-variance for sampling. If None, uses predicted logvar.
        deterministic: If True, use mean instead of sampling.

    Returns:
        Policy function (obs, rng) -> (action, extras).
    """
    prior_params, decoder_params, normalizer_params = extract_prior_decoder_params(
        policy_params
    )

    _, proprioceptive_obs_size, _ = get_observation_sizes(cfg)

    # Get proprioceptive-only normalizer
    if isinstance(normalizer_params, DictRunningStatisticsState):
        # Dict normalizer - proprioception is already separate
        proprio_normalizer_params = normalizer_params.proprioception
    else:
        # Legacy flat normalizer - slice to get proprioceptive portion
        proprio_normalizer_params = running_statistics.RunningStatisticsState(
            count=normalizer_params.count,
            mean=normalizer_params.mean[-proprioceptive_obs_size:],
            summed_variance=normalizer_params.summed_variance[
                -proprioceptive_obs_size:
            ],
            std=normalizer_params.std[-proprioceptive_obs_size:],
        )

    prior_hidden_layer_sizes = tuple(
        cfg.network_config.get(
            "prior_layer_sizes", cfg.network_config.encoder_layer_sizes
        )
    )

    return create_prior_policy(
        prior_network_params=prior_params,
        decoder_network_params=decoder_params,
        normalizer_params=proprio_normalizer_params,
        intention_latent_size=cfg.network_config.intention_size,
        action_size=cfg.network_config.action_size,
        proprioceptive_obs_size=proprioceptive_obs_size,
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        prior_hidden_layer_sizes=prior_hidden_layer_sizes,
        preprocess_observations_fn=running_statistics.normalize,
        fixed_logvar=fixed_logvar,
        deterministic=deterministic,
    )


def create_batched_rollout_fn(
    env: Any,
    policy_fn: Callable,
    num_steps: int,
    proprioceptive_obs_size: int,
    reference_obs_size: int,
    is_prior_policy: bool,
    intention_size: int,
) -> Callable:
    """Create a batched rollout function for multiple clips.

    Args:
        env: The environment.
        policy_fn: Policy function (obs, rng) -> (action, extras).
        num_steps: Number of physics steps per rollout.
        proprioceptive_obs_size: Size of proprioceptive observations.
        reference_obs_size: Size of reference/imitation_target observations.
        is_prior_policy: If True, pass only proprioceptive obs to policy.
        intention_size: Dimension of the latent intention vector.

    Returns:
        A function that takes (initial_qpos_batch, clip_indices, rng_keys) and
        returns (qpos_trajectories, latents_trajectories).
    """
    jit_step = jax.jit(env.step)
    jit_reset = jax.jit(env.reset)
    mjx_model = env.mjx_model

    def single_rollout(
        initial_qpos: jax.Array, clip_idx: int, rng_key: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Run single rollout from initial_qpos, return qpos and latents trajectories."""
        reset_key, rollout_key = random.split(rng_key)

        # Reset environment to get proper state structure at specific clip
        state = jit_reset(reset_key, clip_idx=clip_idx, start_frame=0)

        # Replace qpos with initial_qpos and set qvel to zeros
        data = state.data.replace(qpos=initial_qpos)
        data = data.replace(qvel=jnp.zeros(mjx_model.nv))
        data = mjx.forward(mjx_model, data)
        state = state.replace(data=data)

        def step_fn(carry, _):
            state, key = carry
            key, action_key = random.split(key)

            # Get observations for policy
            if is_prior_policy:
                # Prior policy uses only proprioceptive observations
                obs = state.obs[..., -proprioceptive_obs_size:]
            else:
                # Encoder-decoder expects dict observations
                obs = {
                    "imitation_target": state.obs[..., :reference_obs_size],
                    "proprioception": state.obs[..., reference_obs_size:],
                }

            # Get action and extras (including sampled latent) from policy
            action, extras = policy_fn(obs, action_key)
            z = extras["intention"]
            # Capture the prior mean (mu) the sampled latent was drawn around,
            # so the injected-noise variance can be measured directly as
            # Var(z) - Var(mu) rather than estimated via lag-1 autocovariance.
            # `extras` is a static Python dict at trace time, so `.get`
            # selects the right traced array. Prior rollouts expose
            # "prior_mean", behavior-prior (RGM) rollouts "behavior_prior_mean";
            # policies without a distinct mean (teacher, encoder-decoder) fall
            # back to z, so their `_means` equal their `_latents`.
            mu = extras.get("prior_mean", extras.get("behavior_prior_mean", z))

            # Step environment
            next_state = jit_step(state, action)

            return (next_state, key), (next_state.data.qpos, z, mu)

        # Run rollout for num_steps - 1 (we already have initial qpos)
        initial_carry = (state, rollout_key)
        _, (qpos_trajectory, latents_trajectory, means_trajectory) = jax.lax.scan(
            step_fn, initial_carry, None, length=num_steps - 1
        )

        # Prepend initial qpos and zeros for latent/mean (no policy call at t=0)
        qpos_trajectory = jnp.concatenate(
            [initial_qpos[None, :], qpos_trajectory], axis=0
        )
        latents_trajectory = jnp.concatenate(
            [jnp.zeros((1, intention_size)), latents_trajectory], axis=0
        )
        means_trajectory = jnp.concatenate(
            [jnp.zeros((1, intention_size)), means_trajectory], axis=0
        )

        return (
            qpos_trajectory,
            latents_trajectory,
            means_trajectory,
        )

    # Vmap over batch of clips
    batched_rollout = jax.jit(jax.vmap(single_rollout))

    return batched_rollout


def collect_rollouts(
    env: Any,
    reference_clips: ReferenceClips,
    policy_fn: Callable,
    num_clips: int,
    num_steps: int,
    batch_size: int,
    proprioceptive_obs_size: int,
    reference_obs_size: int,
    intention_size: int,
    is_prior_policy: bool,
    rng_key: jax.Array,
    desc: str = "Collecting rollouts",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect rollouts for all clips.

    Args:
        env: Environment.
        reference_clips: ReferenceClips object.
        policy_fn: Policy function.
        num_clips: Total number of clips.
        num_steps: Steps per rollout.
        batch_size: Clips per batch.
        proprioceptive_obs_size: Size of proprioceptive observations.
        reference_obs_size: Size of reference/imitation_target observations.
        intention_size: Dimension of the latent intention vector.
        is_prior_policy: If True, pass only proprioceptive obs to policy.
        rng_key: Random key.
        desc: Progress bar description.

    Returns:
        Tuple of (qpos, latents, means) arrays with shapes
        (num_clips, num_steps, qpos_dim) and, for the latter two,
        (num_clips, num_steps, intention_size). `means` holds the prior mean
        the latent was sampled around (equal to `latents` for policies without
        a distinct mean, e.g. teacher and encoder-decoder).
    """
    batched_rollout = create_batched_rollout_fn(
        env,
        policy_fn,
        num_steps,
        proprioceptive_obs_size,
        reference_obs_size,
        is_prior_policy,
        intention_size,
    )

    all_qpos = []
    all_latents = []
    all_means = []
    num_batches = (num_clips + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=desc):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_clips)
        actual_batch_size = end_idx - start_idx

        # Get initial qpos for this batch (first frame of each clip)
        initial_qpos_batch = jnp.stack(
            [
                reference_clips.at(clip=i, frame=0).qpos
                for i in range(start_idx, end_idx)
            ]
        )

        # Clip indices for this batch
        clip_indices = jnp.arange(start_idx, end_idx)

        # Generate random keys for this batch
        rng_key, batch_key = random.split(rng_key)
        batch_keys = random.split(batch_key, actual_batch_size)

        # Run batched rollouts
        qpos_batch, latents_batch, means_batch = batched_rollout(
            initial_qpos_batch, clip_indices, batch_keys
        )

        # Transfer to CPU and append
        all_qpos.append(np.array(qpos_batch))
        all_latents.append(np.array(latents_batch))
        all_means.append(np.array(means_batch))

    return (
        np.concatenate(all_qpos, axis=0),
        np.concatenate(all_latents, axis=0),
        np.concatenate(all_means, axis=0),
    )


def get_original_qpos(reference_clips: ReferenceClips) -> np.ndarray:
    """Extract original qpos from reference clips.

    Args:
        reference_clips: ReferenceClips object.

    Returns:
        qpos array of shape (num_clips, num_steps, qpos_dim).
    """
    return np.array(reference_clips.qpos)


def save_results(
    output_path: str,
    original_qpos: np.ndarray,
    results: Dict[str, np.ndarray],
    latents_results: Dict[str, np.ndarray],
    means_results: Dict[str, np.ndarray],
    checkpoint_path: str,
    num_clips: int,
    num_steps: int,
    logvars: Sequence[float],
    intention_size: int,
) -> None:
    """Save all rollout data to H5 file.

    Args:
        output_path: Path to output H5 file.
        original_qpos: Original reference qpos.
        results: Dictionary mapping rollout name to qpos arrays.
        latents_results: Dictionary mapping rollout name to latents arrays.
        means_results: Dictionary mapping rollout name to prior-mean arrays
            (the mu the latent was sampled around; equals latents for policies
            without a distinct mean).
        checkpoint_path: Path to source checkpoint.
        num_clips: Number of clips.
        num_steps: Steps per clip.
        logvars: List of logvar values used for non-deterministic rollouts.
        intention_size: Dimension of the latent intention vector.
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("original_qpos", data=original_qpos, dtype="float32")

        for name, data in results.items():
            f.create_dataset(f"{name}_qpos", data=data, dtype="float32")
        for name, data in latents_results.items():
            f.create_dataset(f"{name}_latents", data=data, dtype="float32")
        for name, data in means_results.items():
            f.create_dataset(f"{name}_means", data=data, dtype="float32")

        f.attrs["num_clips"] = num_clips
        f.attrs["num_steps"] = num_steps
        f.attrs["qpos_dim"] = original_qpos.shape[-1]
        f.attrs["intention_size"] = intention_size
        f.attrs["checkpoint_path"] = checkpoint_path
        f.attrs["logvars"] = list(logvars)

    print(f"Saved results to {output_path}")


def load_teacher_config(teacher_checkpoint_path: str) -> Any:
    """Load teacher config from its original checkpoint.

    Args:
        teacher_checkpoint_path: Path to the expert teacher checkpoint.

    Returns:
        OmegaConf config from the teacher checkpoint.
    """
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix="PPONetwork")
    with ocp.CheckpointManager(
        teacher_checkpoint_path, options=mgr_options
    ) as ckpt_mgr:
        step = ckpt_mgr.latest_step()
        cfg = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(config=ocp.args.JsonRestore()),
        )["config"]
    return OmegaConf.create(cfg)


def _fix_teacher_path(path_str: str) -> str:
    """Apply path fixing logic to teacher checkpoint path.

    Handles /tmp/<project>/... paths by resolving relative to project root.
    """
    if not path_str:
        return path_str

    PROJECT_ROOT = str(Path(__file__).parent.parent.parent)

    if path_str.startswith("/tmp/"):
        parts = path_str.split("/")
        if len(parts) > 3:
            rest = "/".join(parts[3:])
            return os.path.join(PROJECT_ROOT, rest)

    return path_str


def create_abstract_teacher(
    obs_sizes: Mapping[str, int],
    action_size: int,
    mlp_layer_sizes: Sequence[int],
) -> Tuple:
    """Create abstract tree for teacher checkpoint restoration.

    Args:
        obs_sizes: Dict mapping observation keys to sizes.
        action_size: Size of the action space.
        mlp_layer_sizes: Hidden layer sizes for the teacher MLPPolicy.

    Returns:
        Tuple of (normalizer_state, mlp_params) with correct structure.
    """
    action_param_size = action_size * 2
    mlp_module = MLPPolicy(layer_sizes=list(mlp_layer_sizes) + [action_param_size])

    total_obs_size = sum(obs_sizes.values())
    dummy_obs = jnp.zeros((1, total_obs_size))
    init_params = mlp_module.init(jax.random.PRNGKey(0), dummy_obs)

    normalizer_state = DictRunningStatisticsState(
        imitation_target=running_statistics.init_state(
            specs.Array(obs_sizes["imitation_target"], jnp.dtype("float32"))
        ),
        proprioception=running_statistics.init_state(
            specs.Array(obs_sizes["proprioception"], jnp.dtype("float32"))
        ),
    )

    return (normalizer_state, init_params)


def create_abstract_distill_training_state(
    obs_sizes: Mapping[str, int],
    action_size: int,
    latent_size: int,
    encoder_hidden_layer_sizes: Sequence[int],
    decoder_hidden_layer_sizes: Sequence[int],
    motion_prior_hidden_layer_sizes: Sequence[int],
) -> DistillTrainingState:
    """Create abstract DistillTrainingState for checkpoint restoration.

    Args:
        obs_sizes: Dict mapping observation keys to sizes.
        action_size: Size of the action space.
        latent_size: Dimension of the latent space.
        encoder_hidden_layer_sizes: Hidden layer sizes for encoder.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder.
        motion_prior_hidden_layer_sizes: Hidden layer sizes for motion prior.

    Returns:
        DistillTrainingState with correct structure for Orbax restoration.
    """
    encoder, decoder, motion_prior, _ = make_distill_networks(
        obs_sizes=obs_sizes,
        action_size=action_size,
        latent_size=latent_size,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        motion_prior_hidden_layer_sizes=motion_prior_hidden_layer_sizes,
    )

    rng = jax.random.PRNGKey(0)
    init_params = init_distill_params(
        encoder, decoder, motion_prior, obs_sizes, latent_size, rng
    )

    # Must match the optimizer structure from distill_train.py
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(learning_rate=1e-4, weight_decay=0.0, eps=1e-5),
    )

    normalizer_params = DictRunningStatisticsState(
        imitation_target=running_statistics.init_state(
            specs.Array(obs_sizes["imitation_target"], jnp.dtype("float32"))
        ),
        proprioception=running_statistics.init_state(
            specs.Array(obs_sizes["proprioception"], jnp.dtype("float32"))
        ),
    )

    return DistillTrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        normalizer_params=normalizer_params,
        env_steps=jnp.int32(0),
    )


def load_distill_checkpoint(
    checkpoint_path: str,
    step: int | None = None,
    teacher_checkpoint_path_override: str | None = None,
) -> Tuple[Any, Tuple, DistillTrainingState, Tuple, Tuple[int, ...]]:
    """Load a distill checkpoint with all components.

    Args:
        checkpoint_path: Path to distill checkpoint directory.
        step: Specific step to load. If None, loads latest.
        teacher_checkpoint_path_override: Override for teacher checkpoint path.

    Returns:
        Tuple of (config, policy_params, train_state, teacher_params,
                  mlp_layer_sizes).
    """
    step_prefix = "DistillNetwork"
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=step_prefix)

    # Pass 1: Load config
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()
        print(f"Loading distill checkpoint from {checkpoint_path} at step {step}")

        cfg = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(config=ocp.args.JsonRestore()),
        )["config"]
        cfg = OmegaConf.create(cfg)

    # Load teacher config to get mlp_layer_sizes
    teacher_ckpt_path = teacher_checkpoint_path_override
    if teacher_ckpt_path is None:
        teacher_ckpt_path = _fix_teacher_path(cfg.teacher_config.checkpoint_path)

    print(f"Loading teacher config from {teacher_ckpt_path}")
    teacher_cfg = load_teacher_config(teacher_ckpt_path)
    mlp_layer_sizes = tuple(teacher_cfg.network_config.mlp_layer_sizes)

    # Extract architecture params
    obs_sizes = dict(cfg.network_config.obs_sizes)
    action_size = int(cfg.network_config.action_size)
    latent_size = int(cfg.network_config.intention_size)
    encoder_layer_sizes = tuple(cfg.network_config.encoder_layer_sizes)
    decoder_layer_sizes = tuple(cfg.network_config.decoder_layer_sizes)
    # YAML attribute name kept as `prior_layer_sizes` for compat with saved
    # configs; in this pipeline it sizes the motion prior.
    motion_prior_layer_sizes = tuple(cfg.network_config.prior_layer_sizes)

    # Construct abstract trees
    abstract_policy = create_abstract_distill_policy(
        obs_sizes=obs_sizes,
        latent_size=latent_size,
        action_size=action_size,
        encoder_hidden_layer_sizes=encoder_layer_sizes,
        decoder_hidden_layer_sizes=decoder_layer_sizes,
        motion_prior_hidden_layer_sizes=motion_prior_layer_sizes,
    )

    abstract_train_state = create_abstract_distill_training_state(
        obs_sizes=obs_sizes,
        action_size=action_size,
        latent_size=latent_size,
        encoder_hidden_layer_sizes=encoder_layer_sizes,
        decoder_hidden_layer_sizes=decoder_layer_sizes,
        motion_prior_hidden_layer_sizes=motion_prior_layer_sizes,
    )

    abstract_teacher = create_abstract_teacher(
        obs_sizes=obs_sizes,
        action_size=action_size,
        mlp_layer_sizes=mlp_layer_sizes,
    )

    # Pass 2: Load all items
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        restored = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(
                policy=ocp.args.StandardRestore(abstract_policy),
                train_state=ocp.args.StandardRestore(abstract_train_state),
                teacher=ocp.args.StandardRestore(abstract_teacher),
            ),
        )

    return (
        cfg,
        restored["policy"],
        restored["train_state"],
        restored["teacher"],
        mlp_layer_sizes,
    )


def create_teacher_policy(
    teacher_params: Tuple,
    mlp_layer_sizes: Sequence[int],
    obs_sizes: Mapping[str, int],
    action_size: int,
    intention_size: int,
) -> Callable:
    """Create teacher policy from frozen MLPPolicy.

    The teacher maps observations directly to actions with no latent bottleneck.
    Returns zeros for the intention vector.

    Args:
        teacher_params: Tuple of (normalizer_params, raw_params).
        mlp_layer_sizes: Hidden layer sizes for the MLPPolicy.
        obs_sizes: Dict mapping observation keys to sizes.
        action_size: Size of the action space.
        intention_size: Dimension of the latent space (for zero-padding).

    Returns:
        Policy function (obs_dict, rng) -> (action, extras).
    """
    teacher_normalizer_params, teacher_raw_params = teacher_params

    action_param_size = action_size * 2
    mlp_module = MLPPolicy(layer_sizes=list(mlp_layer_sizes) + [action_param_size])
    action_dist = distribution.NormalTanhDistribution(event_size=action_size)

    def policy_fn(obs, rng):
        norm_obs = normalize_dict_obs(obs, teacher_normalizer_params)
        flat_obs = concat_flat_dict_obs(norm_obs)
        logits = mlp_module.apply(teacher_raw_params, flat_obs)
        action = action_dist.mode(logits)
        return action, {"intention": jnp.zeros(intention_size)}

    return policy_fn


def create_student_encoder_decoder_policy(
    train_state: DistillTrainingState,
    cfg: Any,
) -> Callable:
    """Create student encoder-decoder policy from DistillEncoder + DistillDecoder.

    Uses the actual trained DistillEncoder (takes both imitation_target and
    proprioception) and DistillDecoder from the train_state.

    Args:
        train_state: DistillTrainingState with actual student params.
        cfg: Configuration from checkpoint.

    Returns:
        Policy function (obs_dict, rng) -> (action, extras).
    """
    obs_sizes = dict(cfg.network_config.obs_sizes)
    action_size = int(cfg.network_config.action_size)
    latent_size = int(cfg.network_config.intention_size)

    # cfg.network_config.prior_layer_sizes attribute kept for compat; sizes
    # the motion prior in this pipeline.
    encoder, decoder, _, action_dist = make_distill_networks(
        obs_sizes=obs_sizes,
        action_size=action_size,
        latent_size=latent_size,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        motion_prior_hidden_layer_sizes=tuple(cfg.network_config.prior_layer_sizes),
    )

    make_policy = make_student_inference_fn(
        encoder, decoder, action_dist, obs_sizes, latent_size
    )
    return make_policy(
        (train_state.normalizer_params, train_state.params), deterministic=True
    )


def load_distill_prior_checkpoint_for_rollouts(
    checkpoint_path: str,
    step: int | None = None,
) -> Tuple[Any, Tuple]:
    """Load a distill_prior checkpoint for rollout collection.

    Args:
        checkpoint_path: Path to distill_prior checkpoint directory.
        step: Specific step to load. If None, loads latest.

    Returns:
        Tuple of (config, policy_params) where policy_params is
        (normalizer_params, network_params_dict).
    """
    step_prefix = "DistillPriorNetwork"
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=step_prefix)

    # Load config
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()
        print(f"Loading distill_prior checkpoint from {checkpoint_path} at step {step}")

        cfg = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(config=ocp.args.JsonRestore()),
        )["config"]
        cfg = OmegaConf.create(cfg)

    # Create abstract policy for restoration
    rgm_hidden_layer_sizes = tuple(cfg.network_config.rgm_layer_sizes)
    abstract_policy = create_abstract_behavior_prior_policy(
        OmegaConf.to_container(cfg), rgm_hidden_layer_sizes
    )

    # Restore policy params
    mgr_options2 = ocp.CheckpointManagerOptions(create=False, step_prefix=step_prefix)
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options2) as ckpt_mgr:
        restored = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(
                policy=ocp.args.StandardRestore(abstract_policy),
            ),
        )

    return cfg, restored["policy"]


def create_distill_prior_encoder_decoder_policy(
    policy_params: Tuple,
    cfg: Any,
) -> Callable:
    """Create encoder-decoder policy from distill_prior checkpoint.

    Uses the frozen DistillEncoder and DistillDecoder stored in the checkpoint.

    Args:
        policy_params: Tuple of (normalizer_params, network_params).
        cfg: Configuration from checkpoint.

    Returns:
        Policy function (obs_dict, rng) -> (action, extras).
    """
    normalizer_params, network_params = policy_params
    encoder_params = {"params": network_params["params"]["encoder"]}
    decoder_params = {"params": network_params["params"]["decoder"]}

    obs_sizes = dict(cfg.network_config.obs_sizes)

    return make_distill_encoder_decoder_inference_fn(
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        normalizer_params=normalizer_params,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        latent_size=int(cfg.network_config.intention_size),
        action_size=int(cfg.network_config.action_size),
        obs_sizes=obs_sizes,
        deterministic=True,
    )


def create_behavior_prior_policy_fn(
    policy_params: Tuple,
    cfg: Any,
    fixed_logvar: float | None,
    deterministic: bool,
) -> Callable:
    """Create RGM policy using RGM and decoder networks from distill_prior checkpoint.

    The RGM has the same architecture as Prior, so we reuse create_prior_policy
    with the RGM parameters.

    Args:
        policy_params: Tuple of (normalizer_params, network_params).
        cfg: Configuration from checkpoint.
        fixed_logvar: Fixed log-variance for sampling. If None, uses predicted logvar.
        deterministic: If True, use mean instead of sampling.

    Returns:
        Policy function (obs, rng) -> (action, extras).
    """
    normalizer_params, network_params = policy_params
    rgm_params = network_params["params"]["rgm"]
    decoder_params = network_params["params"]["decoder"]

    _, proprioceptive_obs_size, _ = get_observation_sizes(cfg)

    if isinstance(normalizer_params, DictRunningStatisticsState):
        proprio_normalizer_params = normalizer_params.proprioception
    else:
        proprio_normalizer_params = running_statistics.RunningStatisticsState(
            count=normalizer_params.count,
            mean=normalizer_params.mean[-proprioceptive_obs_size:],
            summed_variance=normalizer_params.summed_variance[
                -proprioceptive_obs_size:
            ],
            std=normalizer_params.std[-proprioceptive_obs_size:],
        )

    rgm_hidden_layer_sizes = tuple(cfg.network_config.rgm_layer_sizes)

    return create_prior_policy(
        prior_network_params=rgm_params,
        decoder_network_params=decoder_params,
        normalizer_params=proprio_normalizer_params,
        intention_latent_size=cfg.network_config.intention_size,
        action_size=cfg.network_config.action_size,
        proprioceptive_obs_size=proprioceptive_obs_size,
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        prior_hidden_layer_sizes=rgm_hidden_layer_sizes,
        preprocess_observations_fn=running_statistics.normalize,
        fixed_logvar=fixed_logvar,
        deterministic=deterministic,
    )


def _setup_environment(args, cfg):
    """Set up environment and compute rollout parameters.

    Args:
        args: Parsed CLI arguments.
        cfg: Configuration from checkpoint.

    Returns:
        Tuple of (cfg, env, reference_clips, num_clips, num_steps,
                  reference_obs_size, proprioceptive_obs_size, intention_size).
    """
    # Apply CLI overrides to checkpoint config before environment creation
    OmegaConf.set_struct(cfg.env_config, False)
    if args.mujoco_impl is not None:
        print(
            f"  Overriding mujoco_impl: {cfg.env_config.mujoco_impl} -> {args.mujoco_impl}"
        )
        OmegaConf.update(cfg.env_config, "mujoco_impl", args.mujoco_impl)
    if args.xml_variant is not None:
        from scamper.environment.rodent import consts as rodent_consts

        xml_map = {
            "limited-collisions": str(rodent_consts.RODENT_XML_PATH),
            "full-collisions": str(rodent_consts.RODENT_FULL_COLLISION_XML),
            "no-tail-collisions": str(rodent_consts.RODENT_NO_TAIL_COLLISION_XML),
        }
        new_xml_path = xml_map[args.xml_variant]
        print(
            f"  Overriding walker_xml_path: {cfg.env_config.get('walker_xml_path', 'N/A')} -> {new_xml_path}"
        )
        OmegaConf.update(cfg.env_config, "walker_xml_path", new_xml_path)
    if args.naconmax is not None:
        print(
            f"  Overriding naconmax: {cfg.env_config.get('naconmax', 'N/A')} -> {args.naconmax}"
        )
        OmegaConf.update(cfg.env_config, "naconmax", args.naconmax)
    if args.njmax is not None:
        print(
            f"  Overriding njmax: {cfg.env_config.get('njmax', 'N/A')} -> {args.njmax}"
        )
        OmegaConf.update(cfg.env_config, "njmax", args.njmax)
    OmegaConf.set_struct(cfg.env_config, True)

    cfg = fix_config_paths(cfg, args.data_path_override, args.batch_size)

    reference_obs_size, proprioceptive_obs_size, total_obs_size = get_observation_sizes(
        cfg
    )

    print(f"  Observation size: {total_obs_size}")
    print(f"  Action size: {cfg.network_config.action_size}")
    print(f"  Intention size: {cfg.network_config.intention_size}")
    print(f"  Reference obs size: {reference_obs_size}")
    print(f"  Proprioceptive obs size: {proprioceptive_obs_size}")

    print("\nCreating environment...")
    env, reference_clips = create_environment(cfg)

    num_clips = reference_clips.qpos.shape[0]
    clip_length = reference_clips.qpos.shape[1]

    steps_per_frame = (1 / cfg.env_config.mocap_hz) / cfg.env_config.ctrl_dt
    num_steps = int((clip_length - cfg.env_config.reference_length) * steps_per_frame)

    print(f"  Number of clips: {num_clips}")
    print(f"  Clip length (mocap frames): {clip_length}")
    print(f"  Steps per frame: {steps_per_frame}")
    print(f"  Simulation steps per clip: {num_steps}")

    intention_size = int(cfg.network_config.intention_size)

    return (
        cfg,
        env,
        reference_clips,
        num_clips,
        num_steps,
        reference_obs_size,
        proprioceptive_obs_size,
        intention_size,
    )


def _collect_and_log(
    name,
    env,
    reference_clips,
    policy_fn,
    num_clips,
    num_steps,
    batch_size,
    proprioceptive_obs_size,
    reference_obs_size,
    intention_size,
    is_prior_policy,
    rng_key,
    desc,
    results,
    latents_results,
    means_results,
):
    """Collect rollouts for a single policy and store in results dicts."""
    t_start = time.time()
    (
        results[name],
        latents_results[name],
        means_results[name],
    ) = collect_rollouts(
        env,
        reference_clips,
        policy_fn,
        num_clips,
        num_steps,
        batch_size,
        proprioceptive_obs_size,
        reference_obs_size,
        intention_size,
        is_prior_policy=is_prior_policy,
        rng_key=rng_key,
        desc=desc,
    )
    print(f"  Completed in {time.time() - t_start:.1f}s")
    print(f"  qpos shape: {results[name].shape}")
    print(f"  latents shape: {latents_results[name].shape}")


def main_distill(args):
    """Main flow for distill checkpoints."""
    print(f"\nLoading distill checkpoint from {args.checkpoint_path}")
    cfg, policy_params, train_state, teacher_params, mlp_layer_sizes = (
        load_distill_checkpoint(
            args.checkpoint_path,
            args.checkpoint_step,
            args.teacher_checkpoint_path,
        )
    )

    (
        cfg,
        env,
        reference_clips,
        num_clips,
        num_steps,
        reference_obs_size,
        proprioceptive_obs_size,
        intention_size,
    ) = _setup_environment(args, cfg)

    obs_sizes = dict(cfg.network_config.obs_sizes)
    action_size = int(cfg.network_config.action_size)
    rng = jax.random.PRNGKey(args.seed)
    results = {}
    latents_results = {}
    means_results = {}

    # 1. Get original qpos
    print("\n" + "=" * 60)
    print("Extracting original reference qpos...")
    print("=" * 60)
    original_qpos = get_original_qpos(reference_clips)
    print(f"  Shape: {original_qpos.shape}")

    # 2. Teacher rollouts
    print("\n" + "=" * 60)
    print("Collecting teacher (MLPPolicy) rollouts...")
    print("=" * 60)
    print(f"  Teacher MLP layer sizes: {mlp_layer_sizes}")
    teacher_policy = create_teacher_policy(
        teacher_params, mlp_layer_sizes, obs_sizes, action_size, intention_size
    )
    rng, key = jax.random.split(rng)
    _collect_and_log(
        "teacher",
        env,
        reference_clips,
        teacher_policy,
        num_clips,
        num_steps,
        args.batch_size,
        proprioceptive_obs_size,
        reference_obs_size,
        intention_size,
        is_prior_policy=False,
        rng_key=key,
        desc="Teacher",
        results=results,
        latents_results=latents_results,
        means_results=means_results,
    )

    # 3. Student encoder-decoder rollouts
    print("\n" + "=" * 60)
    print("Collecting student encoder-decoder rollouts...")
    print("=" * 60)
    student_enc_dec_policy = create_student_encoder_decoder_policy(train_state, cfg)
    rng, key = jax.random.split(rng)
    _collect_and_log(
        "student_encoder_decoder",
        env,
        reference_clips,
        student_enc_dec_policy,
        num_clips,
        num_steps,
        args.batch_size,
        proprioceptive_obs_size,
        reference_obs_size,
        intention_size,
        is_prior_policy=False,
        rng_key=key,
        desc="Student encoder-decoder",
        results=results,
        latents_results=latents_results,
        means_results=means_results,
    )

    # 4. Prior rollouts deterministic
    print("\n" + "=" * 60)
    print("Collecting prior rollouts (deterministic, mean only)...")
    print("=" * 60)
    prior_policy_det = create_prior_policy_fn(
        policy_params, cfg, fixed_logvar=0.0, deterministic=True
    )
    rng, key = jax.random.split(rng)
    _collect_and_log(
        "prior_deterministic",
        env,
        reference_clips,
        prior_policy_det,
        num_clips,
        num_steps,
        args.batch_size,
        proprioceptive_obs_size,
        reference_obs_size,
        intention_size,
        is_prior_policy=True,
        rng_key=key,
        desc="Prior (deterministic)",
        results=results,
        latents_results=latents_results,
        means_results=means_results,
    )

    # 5. Prior rollouts with predicted logvar
    print("\n" + "=" * 60)
    print("Collecting prior rollouts (predicted logvar, per-dimension)...")
    print("=" * 60)
    prior_policy_predicted = create_prior_policy_fn(
        policy_params, cfg, fixed_logvar=None, deterministic=False
    )
    rng, key = jax.random.split(rng)
    _collect_and_log(
        "prior_predicted_logvar",
        env,
        reference_clips,
        prior_policy_predicted,
        num_clips,
        num_steps,
        args.batch_size,
        proprioceptive_obs_size,
        reference_obs_size,
        intention_size,
        is_prior_policy=True,
        rng_key=key,
        desc="Prior (predicted logvar)",
        results=results,
        latents_results=latents_results,
        means_results=means_results,
    )

    # 6. Prior rollouts with specified fixed logvar values
    for logvar in args.logvars:
        std = np.exp(logvar / 2)
        print("\n" + "=" * 60)
        print(f"Collecting prior rollouts (logvar={logvar}, std~{std:.2f})...")
        print("=" * 60)
        prior_policy = create_prior_policy_fn(
            policy_params, cfg, fixed_logvar=logvar, deterministic=False
        )
        rng, key = jax.random.split(rng)
        key_name = (
            f"prior_logvar_{int(logvar)}"
            if logvar == int(logvar)
            else f"prior_logvar_{logvar}"
        )
        _collect_and_log(
            key_name,
            env,
            reference_clips,
            prior_policy,
            num_clips,
            num_steps,
            args.batch_size,
            proprioceptive_obs_size,
            reference_obs_size,
            intention_size,
            is_prior_policy=True,
            rng_key=key,
            desc=f"Prior (logvar={logvar})",
            results=results,
            latents_results=latents_results,
            means_results=means_results,
        )

    return original_qpos, results, latents_results, means_results


def main_distill_prior(args):
    """Main flow for distill_prior checkpoints.

    Uses the RGM network for random generation rollouts instead of the prior.
    """
    print(f"\nLoading distill_prior checkpoint from {args.checkpoint_path}")
    cfg, policy_params = load_distill_prior_checkpoint_for_rollouts(
        args.checkpoint_path,
        args.checkpoint_step,
    )

    (
        cfg,
        env,
        reference_clips,
        num_clips,
        num_steps,
        reference_obs_size,
        proprioceptive_obs_size,
        intention_size,
    ) = _setup_environment(args, cfg)

    rng = jax.random.PRNGKey(args.seed)
    results = {}
    latents_results = {}
    means_results = {}

    # 1. Get original qpos
    print("\n" + "=" * 60)
    print("Extracting original reference qpos...")
    print("=" * 60)
    original_qpos = get_original_qpos(reference_clips)
    print(f"  Shape: {original_qpos.shape}")

    # 2. Encoder-decoder rollouts
    print("\n" + "=" * 60)
    print("Collecting encoder-decoder rollouts...")
    print("=" * 60)
    enc_dec_policy = create_distill_prior_encoder_decoder_policy(policy_params, cfg)
    rng, key = jax.random.split(rng)
    _collect_and_log(
        "student_encoder_decoder",
        env,
        reference_clips,
        enc_dec_policy,
        num_clips,
        num_steps,
        args.batch_size,
        proprioceptive_obs_size,
        reference_obs_size,
        intention_size,
        is_prior_policy=False,
        rng_key=key,
        desc="Encoder-decoder",
        results=results,
        latents_results=latents_results,
        means_results=means_results,
    )

    # 3. RGM rollouts deterministic
    print("\n" + "=" * 60)
    print("Collecting RGM rollouts (deterministic, mean only)...")
    print("=" * 60)
    behavior_prior_policy_det = create_behavior_prior_policy_fn(
        policy_params, cfg, fixed_logvar=0.0, deterministic=True
    )
    rng, key = jax.random.split(rng)
    _collect_and_log(
        "rgm_deterministic",
        env,
        reference_clips,
        behavior_prior_policy_det,
        num_clips,
        num_steps,
        args.batch_size,
        proprioceptive_obs_size,
        reference_obs_size,
        intention_size,
        is_prior_policy=True,
        rng_key=key,
        desc="RGM (deterministic)",
        results=results,
        latents_results=latents_results,
        means_results=means_results,
    )

    # 4. RGM rollouts with predicted logvar
    print("\n" + "=" * 60)
    print("Collecting RGM rollouts (predicted logvar, per-dimension)...")
    print("=" * 60)
    behavior_prior_policy_predicted = create_behavior_prior_policy_fn(
        policy_params, cfg, fixed_logvar=None, deterministic=False
    )
    rng, key = jax.random.split(rng)
    _collect_and_log(
        "rgm_predicted_logvar",
        env,
        reference_clips,
        behavior_prior_policy_predicted,
        num_clips,
        num_steps,
        args.batch_size,
        proprioceptive_obs_size,
        reference_obs_size,
        intention_size,
        is_prior_policy=True,
        rng_key=key,
        desc="RGM (predicted logvar)",
        results=results,
        latents_results=latents_results,
        means_results=means_results,
    )

    # 5. RGM rollouts with specified fixed logvar values
    for logvar in args.logvars:
        std = np.exp(logvar / 2)
        print("\n" + "=" * 60)
        print(f"Collecting RGM rollouts (logvar={logvar}, std~{std:.2f})...")
        print("=" * 60)
        behavior_prior_policy = create_behavior_prior_policy_fn(
            policy_params, cfg, fixed_logvar=logvar, deterministic=False
        )
        rng, key = jax.random.split(rng)
        key_name = (
            f"rgm_logvar_{int(logvar)}"
            if logvar == int(logvar)
            else f"rgm_logvar_{logvar}"
        )
        _collect_and_log(
            key_name,
            env,
            reference_clips,
            behavior_prior_policy,
            num_clips,
            num_steps,
            args.batch_size,
            proprioceptive_obs_size,
            reference_obs_size,
            intention_size,
            is_prior_policy=True,
            rng_key=key,
            desc=f"RGM (logvar={logvar})",
            results=results,
            latents_results=latents_results,
            means_results=means_results,
        )

    return original_qpos, results, latents_results, means_results


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print(f"Rollout Dataset Collection (checkpoint type: {args.checkpoint_type})")
    print("=" * 60)

    if args.checkpoint_type == "distill_prior":
        original_qpos, results, latents_results, means_results = main_distill_prior(
            args
        )
    else:
        original_qpos, results, latents_results, means_results = main_distill(args)

    intention_size = original_qpos.shape[0]  # not used, just for save_results
    # Re-derive intention_size from results
    first_key = next(iter(latents_results))
    intention_size = latents_results[first_key].shape[-1]

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    save_results(
        args.output_path,
        original_qpos,
        results,
        latents_results,
        means_results,
        args.checkpoint_path,
        original_qpos.shape[0],
        next(iter(results.values())).shape[1],
        args.logvars,
        intention_size,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output file: {args.output_path}")
    print(f"Datasets:")
    print(f"  - original_qpos: {original_qpos.shape}")
    for name, data in results.items():
        print(
            f"  - {name}_qpos: {data.shape}  |  "
            f"{name}_latents: {latents_results[name].shape}"
        )
    print("\nDone!")


if __name__ == "__main__":
    main()
