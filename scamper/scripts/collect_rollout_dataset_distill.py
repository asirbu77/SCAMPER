#!/usr/bin/env python3
"""Collect rollout dataset from distill checkpoint for discriminator training.

This script generates qpos trajectories and sampled latent vectors from:
1. Original reference data
2. Teacher (MLPPolicy) simulation rollouts
3. Student encoder-decoder simulation rollouts (DistillEncoder + DistillDecoder)
4. Prior simulation rollouts with varying logvar values

Output H5 always contains:
- original_qpos: Reference data (num_clips, num_steps, 74)
- teacher_qpos / teacher_latents: Teacher rollouts (latents are zeros — no bottleneck)
- student_encoder_decoder_qpos / student_encoder_decoder_latents: Student enc-dec rollouts
- prior_deterministic_qpos / prior_deterministic_latents: Prior rollouts using mean (no sampling)
- prior_predicted_logvar_qpos / prior_predicted_logvar_latents: Prior rollouts using predicted logvar

Additionally, for each logvar value specified via --logvars:
- prior_logvar_X_qpos / prior_logvar_X_latents: Prior rollouts with the specified fixed logvar
"""

import argparse
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Tuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from brax.training import distribution
from brax.training.acme import running_statistics, specs
from omegaconf import OmegaConf

from scamper.agent.distill.distill_networks import (
    DistillNetworkParams,
    create_abstract_distill_policy,
    init_distill_params,
    make_distill_networks,
    make_student_inference_fn,
)
from scamper.agent.distill.distill_train import DistillTrainingState
from scamper.agent.imitation_mlp.mlp_network import MLPPolicy
from scamper.agent.observation_utils import (
    DictRunningStatisticsState,
    concat_flat_dict_obs,
    normalize_dict_obs,
)
from scamper.scripts.collect_rollout_dataset import (
    collect_rollouts,
    create_batched_rollout_fn,
    create_environment,
    create_prior_policy_fn,
    fix_config_paths,
    get_observation_sizes,
    get_original_qpos,
    save_results,
)


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
        default=[-4.0, -2.0, 0.0],
        help="Logvar values for non-deterministic prior rollouts (default: -4 -2 0)",
    )
    parser.add_argument(
        "--teacher-checkpoint-path",
        type=str,
        default=None,
        help="Override teacher checkpoint path (for loading teacher config)",
    )

    return parser.parse_args()


def load_teacher_config(teacher_checkpoint_path: str) -> Any:
    """Load teacher config from its original checkpoint.

    Args:
        teacher_checkpoint_path: Path to the imitation_mlp teacher checkpoint.

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
    prior_hidden_layer_sizes: Sequence[int],
) -> DistillTrainingState:
    """Create abstract DistillTrainingState for checkpoint restoration.

    Args:
        obs_sizes: Dict mapping observation keys to sizes.
        action_size: Size of the action space.
        latent_size: Dimension of the latent space.
        encoder_hidden_layer_sizes: Hidden layer sizes for encoder.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder.
        prior_hidden_layer_sizes: Hidden layer sizes for prior.

    Returns:
        DistillTrainingState with correct structure for Orbax restoration.
    """
    encoder, decoder, prior, _ = make_distill_networks(
        obs_sizes=obs_sizes,
        action_size=action_size,
        latent_size=latent_size,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        prior_hidden_layer_sizes=prior_hidden_layer_sizes,
    )

    rng = jax.random.PRNGKey(0)
    init_params = init_distill_params(
        encoder, decoder, prior, obs_sizes, latent_size, rng
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
    prior_layer_sizes = tuple(cfg.network_config.prior_layer_sizes)

    # Construct abstract trees
    abstract_policy = create_abstract_distill_policy(
        obs_sizes=obs_sizes,
        latent_size=latent_size,
        action_size=action_size,
        encoder_hidden_layer_sizes=encoder_layer_sizes,
        decoder_hidden_layer_sizes=decoder_layer_sizes,
        prior_hidden_layer_sizes=prior_layer_sizes,
    )

    abstract_train_state = create_abstract_distill_training_state(
        obs_sizes=obs_sizes,
        action_size=action_size,
        latent_size=latent_size,
        encoder_hidden_layer_sizes=encoder_layer_sizes,
        decoder_hidden_layer_sizes=decoder_layer_sizes,
        prior_hidden_layer_sizes=prior_layer_sizes,
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

    encoder, decoder, _, action_dist = make_distill_networks(
        obs_sizes=obs_sizes,
        action_size=action_size,
        latent_size=latent_size,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        prior_hidden_layer_sizes=tuple(cfg.network_config.prior_layer_sizes),
    )

    make_policy = make_student_inference_fn(
        encoder, decoder, action_dist, obs_sizes, latent_size
    )
    return make_policy(
        (train_state.normalizer_params, train_state.params), deterministic=True
    )


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Distill Rollout Dataset Collection")
    print("=" * 60)

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint_path}")
    cfg, policy_params, train_state, teacher_params, mlp_layer_sizes = (
        load_distill_checkpoint(
            args.checkpoint_path,
            args.checkpoint_step,
            args.teacher_checkpoint_path,
        )
    )

    # Fix paths in config
    cfg = fix_config_paths(cfg, args.data_path_override, args.batch_size)

    # Extract observation sizes
    reference_obs_size, proprioceptive_obs_size, total_obs_size = get_observation_sizes(
        cfg
    )
    obs_sizes = dict(cfg.network_config.obs_sizes)

    print(f"  Observation size: {total_obs_size}")
    print(f"  Action size: {cfg.network_config.action_size}")
    print(f"  Intention size: {cfg.network_config.intention_size}")
    print(f"  Reference obs size: {reference_obs_size}")
    print(f"  Proprioceptive obs size: {proprioceptive_obs_size}")
    print(f"  Teacher MLP layer sizes: {mlp_layer_sizes}")

    # Create environment
    print("\nCreating environment...")
    env, reference_clips = create_environment(cfg)

    num_clips = reference_clips.qpos.shape[0]
    clip_length = reference_clips.qpos.shape[1]

    # Compute actual simulation episode length
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / cfg.env_config.ctrl_dt
    num_steps = int((clip_length - cfg.env_config.reference_length) * steps_per_frame)

    print(f"  Number of clips: {num_clips}")
    print(f"  Clip length (mocap frames): {clip_length}")
    print(f"  Steps per frame: {steps_per_frame}")
    print(f"  Simulation steps per clip: {num_steps}")

    # Initialize random key
    rng = jax.random.PRNGKey(args.seed)
    intention_size = int(cfg.network_config.intention_size)
    action_size = int(cfg.network_config.action_size)

    results = {}
    latents_results = {}

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
    teacher_policy = create_teacher_policy(
        teacher_params, mlp_layer_sizes, obs_sizes, action_size, intention_size
    )
    rng, key = jax.random.split(rng)
    t_start = time.time()
    results["teacher"], latents_results["teacher"] = collect_rollouts(
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
    )
    print(f"  Completed in {time.time() - t_start:.1f}s")
    print(f"  qpos shape: {results['teacher'].shape}")
    print(f"  latents shape: {latents_results['teacher'].shape}")

    # 3. Student encoder-decoder rollouts
    print("\n" + "=" * 60)
    print("Collecting student encoder-decoder rollouts...")
    print("=" * 60)
    student_enc_dec_policy = create_student_encoder_decoder_policy(train_state, cfg)
    rng, key = jax.random.split(rng)
    t_start = time.time()
    results["student_encoder_decoder"], latents_results["student_encoder_decoder"] = (
        collect_rollouts(
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
        )
    )
    print(f"  Completed in {time.time() - t_start:.1f}s")
    print(f"  qpos shape: {results['student_encoder_decoder'].shape}")
    print(f"  latents shape: {latents_results['student_encoder_decoder'].shape}")

    # 4. Prior rollouts deterministic
    print("\n" + "=" * 60)
    print("Collecting prior rollouts (deterministic, mean only)...")
    print("=" * 60)
    prior_policy_det = create_prior_policy_fn(
        policy_params, cfg, fixed_logvar=0.0, deterministic=True
    )
    rng, key = jax.random.split(rng)
    t_start = time.time()
    results["prior_deterministic"], latents_results["prior_deterministic"] = (
        collect_rollouts(
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
        )
    )
    print(f"  Completed in {time.time() - t_start:.1f}s")
    print(f"  qpos shape: {results['prior_deterministic'].shape}")
    print(f"  latents shape: {latents_results['prior_deterministic'].shape}")

    # 5. Prior rollouts with predicted logvar
    print("\n" + "=" * 60)
    print("Collecting prior rollouts (predicted logvar, per-dimension)...")
    print("=" * 60)
    prior_policy_predicted = create_prior_policy_fn(
        policy_params, cfg, fixed_logvar=None, deterministic=False
    )
    rng, key = jax.random.split(rng)
    t_start = time.time()
    results["prior_predicted_logvar"], latents_results["prior_predicted_logvar"] = (
        collect_rollouts(
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
        )
    )
    print(f"  Completed in {time.time() - t_start:.1f}s")
    print(f"  qpos shape: {results['prior_predicted_logvar'].shape}")
    print(f"  latents shape: {latents_results['prior_predicted_logvar'].shape}")

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
        t_start = time.time()
        key_name = (
            f"prior_logvar_{int(logvar)}"
            if logvar == int(logvar)
            else f"prior_logvar_{logvar}"
        )
        results[key_name], latents_results[key_name] = collect_rollouts(
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
        )
        print(f"  Completed in {time.time() - t_start:.1f}s")
        print(f"  qpos shape: {results[key_name].shape}")
        print(f"  latents shape: {latents_results[key_name].shape}")

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    save_results(
        args.output_path,
        original_qpos,
        results,
        latents_results,
        args.checkpoint_path,
        num_clips,
        num_steps,
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
