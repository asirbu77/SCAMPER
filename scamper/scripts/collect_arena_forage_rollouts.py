#!/usr/bin/env python3
"""Collect arena forage rollouts for inception distance analysis.

Generates qpos trajectories from three sources and saves them in an H5 file
compatible with inception_distance.py:

1. student_encoder_decoder_qpos: Encoder-decoder rollouts on the imitation task
   (the "real" distribution — what the trained motion model produces on reference clips)
2. distill_prior_qpos: Arena forage rollouts with distill_prior mode and a random
   (untrained) PPO policy. The gate is mostly 0 at init, so the RGM drives behavior.
3. distill_prior_no_rgm_qpos: Arena forage rollouts with distill_prior_no_rgm mode
   and a random PPO policy. The prior + random residual drives behavior.

This lets us measure whether the RGM adds realism to rodent motion when the agent
hasn't received any task-specific reward signal yet (initial eval / explore mode).

Usage:
    python -m scamper.scripts.collect_arena_forage_rollouts \
        --checkpoint-path scratch/distill_prior_checkpoints/260406_130524_381481 \
        --checkpoint-step 54 \
        --output-path data/arena_forage_inception_rollouts.h5

    # Then run inception distance:
    python -m scamper.scripts.inception_distance \
        --h5-path data/arena_forage_inception_rollouts.h5 \
        --real-dataset student_encoder_decoder_qpos \
        --fake-datasets distill_prior_qpos distill_prior_no_rgm_qpos
"""

import argparse
import os
import time
from pathlib import Path
from typing import Any, Callable, Tuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics, specs
from mujoco import mjx
from tqdm import tqdm

from scamper.agent.task_transfer.arena_forage.checkpoint_utils import (
    load_distill_prior_checkpoint,
    make_decoder_inference_fn,
    make_prior_inference_fn,
    make_rgm_inference_fn,
)
from scamper.agent.task_transfer.arena_forage.wrappers import (
    PriorDecoderHighLevelWrapper,
    PriorRGMDecoderHighLevelWrapper,
)
from scamper.environment.rodent import arena_forage
from scamper.environment.rodent import consts as rodent_consts
from scamper.scripts.collect_rollout_dataset import (
    collect_rollouts,
    get_observation_sizes,
)
from scamper.scripts.collect_rollout_dataset_distill import (
    _collect_and_log,
    _setup_environment,
    create_distill_prior_encoder_decoder_policy,
    load_distill_prior_checkpoint_for_rollouts,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect arena forage rollouts for inception distance analysis"
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to distill_prior checkpoint directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path for output H5 file",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Checkpoint step to load (default: latest)",
    )

    # Rollout parameters
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=842,
        help="Number of arena forage rollouts (default: 842)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=490,
        help="Time steps per arena forage rollout (default: 490)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for parallel rollouts (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    # Environment overrides (shared with imitation env setup)
    parser.add_argument(
        "--data-path-override",
        type=str,
        default=None,
        help="Override reference data path from checkpoint config",
    )
    parser.add_argument(
        "--mujoco-impl",
        type=str,
        choices=["jax", "warp"],
        default="warp",
        help="MuJoCo backend (default: warp)",
    )
    parser.add_argument(
        "--xml-variant",
        type=str,
        choices=["limited-collisions", "full-collisions", "no-tail-collisions"],
        default="no-tail-collisions",
        help="Walker XML variant (default: no-tail-collisions)",
    )
    parser.add_argument(
        "--naconmax",
        type=int,
        default=None,
        help="Override naconmax",
    )
    parser.add_argument(
        "--njmax",
        type=int,
        default=None,
        help="Override njmax",
    )

    # Network architecture
    parser.add_argument(
        "--policy-layers",
        type=int,
        nargs="+",
        default=[2048, 1024, 512],
        help="Policy hidden layer sizes (default: 2048 1024 512)",
    )
    parser.add_argument(
        "--value-layers",
        type=int,
        nargs="+",
        default=[2048, 1024, 1024, 512],
        help="Value hidden layer sizes (default: 2048 1024 1024 512)",
    )

    # distill_prior gate parameters
    parser.add_argument(
        "--gate-bias",
        type=float,
        default=0.5,
        help="Gate bias threshold (default: 0.5)",
    )
    parser.add_argument(
        "--gate-margin",
        type=float,
        default=0.2,
        help="Gate hysteresis margin (default: 0.2)",
    )
    parser.add_argument(
        "--rgm-noise",
        action="store_true",
        default=True,
        help="Sample RGM with noise (default: True)",
    )
    parser.add_argument(
        "--no-rgm-noise",
        action="store_true",
        help="Use RGM mean only (deterministic)",
    )
    parser.add_argument(
        "--rgm-logvar",
        type=float,
        default=None,
        help="Fixed log-variance for RGM sampling (default: predicted)",
    )

    # distill_prior_no_rgm parameters
    parser.add_argument(
        "--deterministic-prior",
        action="store_true",
        default=True,
        help="Use prior mean only, no noise (default: True)",
    )
    parser.add_argument(
        "--noise-logvar",
        type=float,
        default=-2.0,
        help="Fixed noise log-variance for no-RGM mode (default: -2.0)",
    )

    # Arena forage environment parameters
    parser.add_argument("--num-treats", type=int, default=10)
    parser.add_argument("--treat-sphere-radius", type=float, default=0.06)
    parser.add_argument("--treat-height", type=float, default=0.03)
    parser.add_argument("--spawn-radius", type=float, default=2.0)
    parser.add_argument("--min-spawn-radius", type=float, default=0.75)
    parser.add_argument("--scent-sigma", type=float, default=0.2)

    args = parser.parse_args()

    # Handle --no-rgm-noise flag
    if args.no_rgm_noise:
        args.rgm_noise = False

    return args


def create_arena_forage_env(
    args: argparse.Namespace,
    ckpt_cfg: dict,
) -> arena_forage.ArenaForage:
    """Create the arena forage base environment.

    Args:
        args: Parsed CLI arguments.
        ckpt_cfg: Checkpoint config dict.

    Returns:
        Configured ArenaForage environment.
    """
    env_cfg = arena_forage.default_config()

    # XML variant
    xml_map = {
        "limited-collisions": rodent_consts.RODENT_XML_PATH,
        "full-collisions": rodent_consts.RODENT_FULL_COLLISION_XML,
        "no-tail-collisions": rodent_consts.RODENT_NO_TAIL_COLLISION_XML,
    }
    env_cfg.walker_xml_path = xml_map[args.xml_variant]
    env_cfg.mujoco_impl = args.mujoco_impl
    env_cfg.ctrl_dt = ckpt_cfg["env_config"]["ctrl_dt"]
    env_cfg.torque_actuators = True

    # Arena forage task parameters
    env_cfg.num_treats = args.num_treats
    env_cfg.treat_sphere_radius = args.treat_sphere_radius
    env_cfg.treat_height = args.treat_height
    env_cfg.min_spawn_radius = args.min_spawn_radius
    env_cfg.spawn_radius = args.spawn_radius
    env_cfg.scent_sigma = args.scent_sigma
    env_cfg.reward_terms = {
        "treat_collection": {"weight": 500.0},
        "scent_proximity": {"weight": 0.0},
    }

    # Override contact limits if specified
    if args.naconmax is not None:
        env_cfg.naconmax = args.naconmax
    if args.njmax is not None:
        env_cfg.njmax = args.njmax

    base_env = arena_forage.ArenaForage(config=env_cfg)
    base_env._mjx_model = mjx.put_model(base_env._mj_model, impl=env_cfg.mujoco_impl)

    return base_env


def create_random_ppo_policy(
    obs_size: int,
    action_size: int,
    policy_layers: list[int],
    value_layers: list[int],
    seed: int = 0,
) -> Tuple[Callable, Tuple]:
    """Create a randomly-initialized PPO policy with deterministic inference.

    Args:
        obs_size: Observation size.
        action_size: Action size.
        policy_layers: Policy hidden layer sizes.
        value_layers: Value hidden layer sizes.
        seed: Random seed for weight initialization.

    Returns:
        Tuple of (policy_fn, params) where policy_fn(params, obs, key) -> (action, {}).
    """
    ppo_network = ppo_networks.make_ppo_networks(
        obs_size,
        action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(policy_layers),
        value_hidden_layer_sizes=tuple(value_layers),
    )

    # count=0 means normalize returns input unchanged (identity)
    normalizer_params = running_statistics.init_state(
        specs.Array(obs_size, jnp.dtype("float32"))
    )
    policy_params = ppo_network.policy_network.init(jax.random.PRNGKey(seed))
    params = (normalizer_params, policy_params)

    parametric_action_distribution = ppo_network.parametric_action_distribution
    policy_network = ppo_network.policy_network

    def policy_fn(params, observations, key_sample):
        del key_sample  # Deterministic
        param_subset = (params[0], params[1])
        logits = policy_network.apply(*param_subset, observations)
        action = parametric_action_distribution.mode(logits)
        return action, {}

    return policy_fn, params


def collect_arena_forage_rollouts(
    env: Any,
    policy_fn: Callable,
    params: Tuple,
    num_rollouts: int,
    num_steps: int,
    batch_size: int,
    rng_key: jax.Array,
    desc: str = "Collecting arena forage rollouts",
) -> np.ndarray:
    """Collect rollouts from the arena forage environment.

    Resets the environment with random seeds and runs the policy forward,
    recording qpos at each step.

    Args:
        env: Wrapped arena forage environment.
        policy_fn: Policy function (params, obs, key) -> (action, extras).
        params: Policy parameters (normalizer_params, policy_params).
        num_rollouts: Total number of rollouts to collect.
        num_steps: Steps per rollout.
        batch_size: Number of rollouts per batch (vmapped).
        rng_key: Random key.
        desc: Progress bar description.

    Returns:
        qpos array of shape (num_rollouts, num_steps, qpos_dim).
    """
    jit_step = jax.jit(env.step)
    jit_reset = jax.jit(env.reset)

    def single_rollout(rng_key: jax.Array) -> jax.Array:
        reset_key, rollout_key = jax.random.split(rng_key)
        state = jit_reset(reset_key)
        initial_qpos = state.data.qpos

        def step_fn(carry, _):
            state, key = carry
            key, act_key = jax.random.split(key)
            action, _ = policy_fn(params, state.obs, act_key)
            next_state = jit_step(state, action)
            return (next_state, key), next_state.data.qpos

        _, qpos_trajectory = jax.lax.scan(
            step_fn, (state, rollout_key), None, length=num_steps - 1
        )

        # Prepend initial qpos
        qpos_trajectory = jnp.concatenate(
            [initial_qpos[None, :], qpos_trajectory], axis=0
        )
        return qpos_trajectory  # (num_steps, qpos_dim)

    batched_rollout = jax.jit(jax.vmap(single_rollout))

    all_qpos = []
    num_batches = (num_rollouts + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=desc):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_rollouts)
        actual_batch_size = end_idx - start_idx

        rng_key, batch_key = jax.random.split(rng_key)
        batch_keys = jax.random.split(batch_key, actual_batch_size)

        qpos_batch = batched_rollout(batch_keys)
        all_qpos.append(np.array(qpos_batch))

    return np.concatenate(all_qpos, axis=0)


def save_results(
    output_path: str,
    results: dict[str, np.ndarray],
    checkpoint_path: str,
) -> None:
    """Save rollout datasets to H5 file.

    Args:
        output_path: Path to output H5 file.
        results: Dictionary mapping dataset name to qpos array.
        checkpoint_path: Source checkpoint path (stored as metadata).
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        for name, data in results.items():
            f.create_dataset(name, data=data, dtype="float32")

        # Metadata from first dataset
        first = next(iter(results.values()))
        f.attrs["num_clips"] = first.shape[0]
        f.attrs["num_steps"] = first.shape[1]
        f.attrs["qpos_dim"] = first.shape[2]
        f.attrs["checkpoint_path"] = checkpoint_path

    print(f"Saved results to {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Arena Forage Initial-Eval Rollout Collection")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Checkpoint step: {args.checkpoint_step}")
    print(f"  Num rollouts: {args.num_rollouts}")
    print(f"  Num steps: {args.num_steps}")
    print(f"  Backend: {args.mujoco_impl}")
    print(f"  XML variant: {args.xml_variant}")
    print(f"  Policy layers: {args.policy_layers}")
    print(f"  Gate bias: {args.gate_bias}, margin: {args.gate_margin}")
    print(f"  RGM noise: {args.rgm_noise}, logvar: {args.rgm_logvar}")
    print(
        f"  Deterministic prior: {args.deterministic_prior}, noise logvar: {args.noise_logvar}"
    )

    rng = jax.random.PRNGKey(args.seed)
    results = {}

    # ------------------------------------------------------------------
    # Step 1: Load distill_prior checkpoint (two interfaces needed)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Loading distill_prior checkpoint...")
    print("=" * 60)

    # Interface 1: For imitation encoder-decoder rollouts
    cfg, policy_params = load_distill_prior_checkpoint_for_rollouts(
        args.checkpoint_path, args.checkpoint_step
    )

    # Interface 2: For arena forage frozen networks (separate param dicts)
    (
        encoder_params,
        prior_params,
        decoder_params,
        rgm_params,
        normalizer_params,
        ckpt_cfg,
    ) = load_distill_prior_checkpoint(args.checkpoint_path, args.checkpoint_step)

    latent_size = ckpt_cfg["network_config"]["intention_size"]
    proprio_size = ckpt_cfg["network_config"]["obs_sizes"]["proprioception"]
    print(f"  Latent size: {latent_size}, Proprio size: {proprio_size}")

    # ------------------------------------------------------------------
    # Step 2: Collect student encoder-decoder rollouts on imitation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Setting up imitation environment for encoder-decoder rollouts...")
    print("=" * 60)

    (
        cfg,
        imitation_env,
        reference_clips,
        num_clips,
        num_steps_imitation,
        reference_obs_size,
        proprioceptive_obs_size,
        intention_size,
    ) = _setup_environment(args, cfg)

    print(f"  Reference clips: {num_clips} clips, {num_steps_imitation} sim steps")

    enc_dec_policy = create_distill_prior_encoder_decoder_policy(policy_params, cfg)

    imitation_results = {}
    imitation_latents = {}
    rng, key = jax.random.split(rng)
    _collect_and_log(
        "student_encoder_decoder",
        imitation_env,
        reference_clips,
        enc_dec_policy,
        num_clips,
        num_steps_imitation,
        args.batch_size,
        proprioceptive_obs_size,
        reference_obs_size,
        intention_size,
        is_prior_policy=False,
        rng_key=key,
        desc="Encoder-decoder (imitation)",
        results=imitation_results,
        latents_results=imitation_latents,
    )

    results["student_encoder_decoder_qpos"] = imitation_results[
        "student_encoder_decoder"
    ]
    print(f"  Shape: {results['student_encoder_decoder_qpos'].shape}")

    # ------------------------------------------------------------------
    # Step 3: Arena forage rollouts — distill_prior mode
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Setting up arena forage environment (distill_prior mode)...")
    print("=" * 60)

    base_env_dp = create_arena_forage_env(args, ckpt_cfg)
    decoder_fn = make_decoder_inference_fn(decoder_params, normalizer_params, ckpt_cfg)
    prior_fn = make_prior_inference_fn(prior_params, normalizer_params, ckpt_cfg)
    rgm_fn = make_rgm_inference_fn(rgm_params, normalizer_params, ckpt_cfg)

    dp_env = PriorRGMDecoderHighLevelWrapper(
        base_env_dp,
        prior_fn,
        rgm_fn,
        decoder_fn,
        latent_size,
        proprio_size,
        gate_bias=args.gate_bias,
        gate_margin=args.gate_margin,
        rgm_noise=args.rgm_noise,
        rgm_logvar=args.rgm_logvar,
    )

    # Get obs size from a test reset
    rng, reset_key = jax.random.split(rng)
    test_state = jax.jit(dp_env.reset)(reset_key)
    dp_obs_size = test_state.obs.shape[-1]
    dp_action_size = dp_env.action_size  # latent_size + 1
    print(f"  Obs size: {dp_obs_size}, Action size: {dp_action_size}")
    print(
        f"  Gate bias: {args.gate_bias}, margin: {args.gate_margin}, "
        f"RGM noise: {args.rgm_noise}"
    )

    dp_policy_fn, dp_params = create_random_ppo_policy(
        dp_obs_size,
        dp_action_size,
        args.policy_layers,
        args.value_layers,
        seed=args.seed,
    )

    print(
        f"\nCollecting {args.num_rollouts} distill_prior rollouts "
        f"({args.num_steps} steps each)..."
    )
    t_start = time.time()
    rng, key = jax.random.split(rng)
    results["distill_prior_qpos"] = collect_arena_forage_rollouts(
        dp_env,
        dp_policy_fn,
        dp_params,
        args.num_rollouts,
        args.num_steps,
        args.batch_size,
        key,
        desc="distill_prior (arena forage)",
    )
    print(f"  Completed in {time.time() - t_start:.1f}s")
    print(f"  Shape: {results['distill_prior_qpos'].shape}")

    # ------------------------------------------------------------------
    # Step 4: Arena forage rollouts — distill_prior_no_rgm mode
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Setting up arena forage environment (distill_prior_no_rgm mode)...")
    print("=" * 60)

    base_env_no_rgm = create_arena_forage_env(args, ckpt_cfg)
    decoder_fn_2 = make_decoder_inference_fn(
        decoder_params, normalizer_params, ckpt_cfg
    )
    prior_fn_2 = make_prior_inference_fn(prior_params, normalizer_params, ckpt_cfg)

    no_rgm_env = PriorDecoderHighLevelWrapper(
        base_env_no_rgm,
        prior_fn_2,
        decoder_fn_2,
        latent_size,
        proprio_size,
        deterministic_prior=args.deterministic_prior,
        noise_logvar=args.noise_logvar,
    )

    # Get obs size from a test reset
    rng, reset_key = jax.random.split(rng)
    test_state_2 = jax.jit(no_rgm_env.reset)(reset_key)
    no_rgm_obs_size = test_state_2.obs.shape[-1]
    no_rgm_action_size = no_rgm_env.action_size  # latent_size
    print(f"  Obs size: {no_rgm_obs_size}, Action size: {no_rgm_action_size}")
    print(
        f"  Deterministic prior: {args.deterministic_prior}, "
        f"noise logvar: {args.noise_logvar}"
    )

    no_rgm_policy_fn, no_rgm_params = create_random_ppo_policy(
        no_rgm_obs_size,
        no_rgm_action_size,
        args.policy_layers,
        args.value_layers,
        seed=args.seed,
    )

    print(
        f"\nCollecting {args.num_rollouts} distill_prior_no_rgm rollouts "
        f"({args.num_steps} steps each)..."
    )
    t_start = time.time()
    rng, key = jax.random.split(rng)
    results["distill_prior_no_rgm_qpos"] = collect_arena_forage_rollouts(
        no_rgm_env,
        no_rgm_policy_fn,
        no_rgm_params,
        args.num_rollouts,
        args.num_steps,
        args.batch_size,
        key,
        desc="distill_prior_no_rgm (arena forage)",
    )
    print(f"  Completed in {time.time() - t_start:.1f}s")
    print(f"  Shape: {results['distill_prior_no_rgm_qpos'].shape}")

    # ------------------------------------------------------------------
    # Step 5: Save results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    save_results(args.output_path, results, args.checkpoint_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output file: {args.output_path}")
    for name, data in results.items():
        print(f"  {name}: {data.shape}")
    print(
        f"\nTo run inception distance:\n"
        f"  python -m scamper.scripts.inception_distance \\\n"
        f"      --h5-path {args.output_path} \\\n"
        f"      --real-dataset student_encoder_decoder_qpos \\\n"
        f"      --fake-datasets distill_prior_qpos distill_prior_no_rgm_qpos"
    )


if __name__ == "__main__":
    main()
