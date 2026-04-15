#!/usr/bin/env python3
"""Behavior Prior (RGM) rollout: run rollouts, render videos, and save qpos to H5.

For each noise level and each clip/frame start position, runs NUM_ROLLOUTS rollouts
using the RGM, renders each as an .mp4 video, and saves all qpos trajectories to an
H5 file for later re-rendering.
"""

import argparse
import os
import time
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import h5py
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from mujoco import mjx

from scamper.agent.observation_utils import flatten_obs_dict
from scamper.scripts.collect_rollout_dataset import (
    create_environment,
    fix_config_paths,
    get_observation_sizes,
)
from scamper.scripts.collect_rollout_dataset_distill import (
    create_rgm_policy_fn,
    load_distill_prior_checkpoint_for_rollouts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavior prior (RGM) rollout visualization"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to distill_prior checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for videos and H5 file",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Checkpoint step to load (default: latest)",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=5,
        help="Number of rollouts per (noise_level, start_position) combination",
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        nargs="+",
        default=["deterministic", "-4.0", "-3.0", "-2.0", "-1.0", "0.0", "predicted"],
        help="Noise levels: 'deterministic', 'predicted', or float logvar values",
    )
    parser.add_argument(
        "--clip-configs",
        type=str,
        nargs="+",
        default=["0,0", "1,0", "2,0", "3,0", "4,0", "5,0"],
        help="Starting positions as clip_idx,start_frame pairs",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-fps", type=int, default=50)
    parser.add_argument(
        "--camera-name", type=str, default="close_profile-rodent"
    )
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--render-width", type=int, default=1280)
    return parser.parse_args()


def parse_noise_levels(noise_strs: list[str]) -> list:
    """Convert CLI noise level strings to internal representation."""
    levels = []
    for s in noise_strs:
        if s == "deterministic":
            levels.append("deterministic")
        elif s == "predicted":
            levels.append(None)
        else:
            levels.append(float(s))
    return levels


def parse_clip_configs(clip_strs: list[str]) -> list[tuple[int, int]]:
    """Convert CLI clip config strings to (clip_idx, start_frame) tuples."""
    configs = []
    for s in clip_strs:
        parts = s.split(",")
        configs.append((int(parts[0]), int(parts[1])))
    return configs


def noise_level_name(noise_level) -> str:
    if noise_level == "deterministic":
        return "deterministic"
    elif noise_level is None:
        return "predicted_logvar"
    else:
        return f"logvar_{noise_level}"


def create_clip_state(env, clip_idx, start_frame, rng_key):
    """Create environment state initialized from a specific clip and frame."""
    state = env.reset(rng_key)
    unwrapped_env = env
    while hasattr(unwrapped_env, "_env"):
        unwrapped_env = unwrapped_env._env

    reference = unwrapped_env.reference_clips.at(clip=clip_idx, frame=start_frame)
    mjx_model = env.mjx_model
    data = state.data.replace(qpos=reference.qpos)
    data = data.replace(qvel=jnp.zeros(mjx_model.nv))
    data = mjx.forward(mjx_model, data)

    info = dict(state.info)
    info["start_frame"] = start_frame
    info["reference_clip"] = clip_idx
    state = state.replace(data=data, info=info)
    return state


def run_rollouts(env, policy_fn, starting_state, max_steps, rng_keys):
    """Run vmapped rollouts from a starting state and return all states."""
    jit_step = jax.jit(env.step)

    def single_rollout_fn(rng_key):
        state = starting_state

        def step_fn(carry, _):
            state, key = carry
            key, key_action = random.split(key)

            if hasattr(state.obs, "get") or isinstance(state.obs, dict):
                flat_obs = flatten_obs_dict(state.obs)
                proprio = flat_obs["proprioception"]
            else:
                proprio = state.obs

            action, _ = policy_fn(proprio, key_action)
            next_state = jit_step(state, action)
            return (next_state, key), next_state

        initial_carry = (state, rng_key)
        _, all_states = jax.lax.scan(step_fn, initial_carry, None, length=max_steps)
        return all_states

    vmapped_rollout = jax.jit(jax.vmap(single_rollout_fn))
    all_states = vmapped_rollout(rng_keys)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), all_states)
    return all_states


def render_rollout(env, states, num_steps, camera, height=480, width=640):
    """Render a single rollout to a list of frames."""
    states_list = []
    for i in range(num_steps):
        state_i = jax.tree_util.tree_map(lambda x: x[i], states)
        states_list.append(state_i)
    return env.render(
        states_list, camera=camera, height=height, width=width, render_ghost=False
    )


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_levels = parse_noise_levels(args.noise_levels)
    clip_frame_starts = parse_clip_configs(args.clip_configs)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    cfg, policy_params = load_distill_prior_checkpoint_for_rollouts(
        args.checkpoint_path, args.checkpoint_step
    )

    cfg = fix_config_paths(cfg)
    env, reference_clips = create_environment(cfg)

    reference_obs_size, proprioceptive_obs_size, total_obs_size = (
        get_observation_sizes(cfg)
    )
    print(f"Action size: {cfg.network_config.action_size}")
    print(f"Intention size: {cfg.network_config.intention_size}")
    print(f"Proprioceptive obs size: {proprioceptive_obs_size}")
    print(f"Reference obs size: {reference_obs_size}")
    print(f"Number of reference clips: {reference_clips.qpos.shape[0]}")

    # Run rollouts
    rng = random.PRNGKey(args.seed)
    video_paths = []
    h5_path = output_dir / "behavior_prior_rollouts.h5"

    with h5py.File(str(h5_path), "w") as h5f:
        h5f.attrs["checkpoint_path"] = args.checkpoint_path
        h5f.attrs["checkpoint_step"] = (
            args.checkpoint_step if args.checkpoint_step is not None else -1
        )
        h5f.attrs["max_steps"] = args.max_steps
        h5f.attrs["seed"] = args.seed
        h5f.attrs["num_rollouts"] = args.num_rollouts

        for noise_level in noise_levels:
            nl_name = noise_level_name(noise_level)

            if noise_level == "deterministic":
                rgm_policy = create_rgm_policy_fn(
                    policy_params, cfg, fixed_logvar=0.0, deterministic=True
                )
            else:
                rgm_policy = create_rgm_policy_fn(
                    policy_params, cfg, fixed_logvar=noise_level, deterministic=False
                )

            for clip_idx, start_frame in clip_frame_starts:
                pose_name = f"clip{clip_idx}_frame{start_frame}"
                group_name = f"{nl_name}/{pose_name}"
                print(f"\n{'='*60}")
                print(
                    f"Noise: {nl_name} | Start: {pose_name} | "
                    f"Rollouts: {args.num_rollouts}"
                )
                print(f"{'='*60}")

                rng, rng_init, rng_rollout = random.split(rng, 3)
                starting_state = create_clip_state(
                    env, clip_idx, start_frame, rng_init
                )

                # 1 for deterministic since they're identical
                n = 1 if noise_level == "deterministic" else args.num_rollouts
                rollout_keys = random.split(rng_rollout, n)

                t0 = time.time()
                all_states = run_rollouts(
                    env, rgm_policy, starting_state, args.max_steps, rollout_keys
                )
                elapsed = time.time() - t0
                print(f"  Time: {elapsed:.1f}s")

                # Extract qpos and save to H5
                # all_states.data.qpos shape: (n, max_steps, qpos_dim)
                qpos_np = np.array(all_states.data.qpos)
                grp = h5f.create_group(group_name)
                grp.create_dataset("qpos", data=qpos_np, compression="gzip")
                grp.attrs["clip_idx"] = clip_idx
                grp.attrs["start_frame"] = start_frame
                grp.attrs["noise_level"] = (
                    nl_name if noise_level is None else str(noise_level)
                )
                grp.attrs["num_rollouts"] = n
                print(f"  Saved qpos to H5: {group_name} {qpos_np.shape}")

                # Render each rollout
                for ri in range(n):
                    rollout_states = jax.tree_util.tree_map(
                        lambda x: x[ri], all_states
                    )

                    frames = render_rollout(
                        env,
                        rollout_states,
                        args.max_steps,
                        camera=args.camera_name,
                        height=args.render_height,
                        width=args.render_width,
                    )

                    video_filename = f"{nl_name}_{pose_name}_rollout{ri:02d}.mp4"
                    video_path = output_dir / video_filename
                    with imageio.get_writer(
                        str(video_path), fps=args.render_fps
                    ) as writer:
                        for frame in frames:
                            writer.append_data(frame)

                    video_paths.append(video_path)
                    print(f"  Saved: {video_filename}")

                del all_states

    print(f"\nDone! {len(video_paths)} videos saved to {output_dir}")
    print(f"H5 file saved to {h5_path}")


if __name__ == "__main__":
    main()
