"""Evaluate a single bowl escape checkpoint and save distances to a numpy file.

Usage:
    python -m scamper.scripts.eval_bowl_escape_single <checkpoint_path> <output_path> [--num_seeds 20] [--episode_length 2000] [--step STEP]

Designed to be called from a notebook via subprocess for GPU memory isolation.
Loads the model once and runs multiple rollout seeds to amortize compilation cost.
"""

import argparse
import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_GL"] = "egl"

import functools
import json

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_networks

from vnl_playground.tasks.rodent import consts as rodent_consts
from vnl_playground.tasks.rodent import bowl_escape

from scamper.agent.task_transfer.bowl_escape.checkpoint_utils import (
    load_prior_checkpoint,
    load_task_transfer_metadata,
    load_task_transfer_brax_params,
    make_decoder_inference_fn,
    make_prior_inference_fn,
)
from scamper.agent.task_transfer.bowl_escape.wrappers import (
    DecoderHighLevelWrapper,
    FlatObsWrapper,
    PriorDecoderHighLevelWrapper,
)
from scamper.agent.task_transfer.bowl_escape.scratch_networks import (
    make_scratch_ppo_networks,
)


def load_model(checkpoint_path, checkpoint_step=None):
    ckpt_info = load_task_transfer_metadata(checkpoint_path, step=checkpoint_step)
    config = ckpt_info["config"]
    ckpt_step = ckpt_info["step"]
    mode = config["mode"]
    ckpt_cfg = config["checkpoint_config"]

    print(f"  Mode: {mode}, Step: {ckpt_step}", flush=True)

    prior_checkpoint_path = config["checkpoint"]["path"]
    prior_checkpoint_step = config["checkpoint"]["step"]

    (
        encoder_params,
        prior_params,
        decoder_params,
        normalizer_params,
        _,
    ) = load_prior_checkpoint(prior_checkpoint_path, prior_checkpoint_step)

    latent_size = ckpt_cfg["network_config"]["intention_size"]
    proprio_size = ckpt_cfg["network_config"]["obs_sizes"]["proprioception"]

    env_cfg = bowl_escape.default_config()

    xml_variant = config["env"].get("xml_variant", "full-collisions")
    if xml_variant == "limited-collisions":
        env_cfg.walker_xml_path = rodent_consts.RODENT_XML_PATH
    elif xml_variant == "full-collisions":
        env_cfg.walker_xml_path = rodent_consts.RODENT_FULL_COLLISION_XML
    elif xml_variant == "no-tail-collisions":
        env_cfg.walker_xml_path = rodent_consts.RODENT_NO_TAIL_COLLISION_XML
    else:
        raise ValueError(f"Unknown xml_variant: {xml_variant}")

    env_cfg.mujoco_impl = config["env"].get("mujoco_impl", "jax")
    env_cfg.ctrl_dt = ckpt_cfg["env_config"]["ctrl_dt"]
    env_cfg.torque_actuators = True

    if mode == "scratch":
        env_cfg.torque_actuators = config["scratch"].get("torque_actuators", True)

    decoder_fn = make_decoder_inference_fn(decoder_params, normalizer_params, ckpt_cfg)

    import mujoco
    from mujoco import mjx

    base_env = bowl_escape.BowlEscape(config=env_cfg)
    base_env._mj_model.opt.ccd_iterations = 50

    hfield_friction = config["env"].get("hfield_friction")
    if hfield_friction is not None:
        for i in range(base_env._mj_model.ngeom):
            if base_env._mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_HFIELD:
                base_env._mj_model.geom_friction[i, 0] = hfield_friction
                break

    base_env._mjx_model = mjx.put_model(base_env._mj_model, impl=env_cfg.mujoco_impl)

    if mode == "decoder_only":
        env = DecoderHighLevelWrapper(base_env, decoder_fn, latent_size, proprio_size)
    elif mode == "scratch":
        env = FlatObsWrapper(base_env)
    elif mode == "prior_decoder":
        prior_fn = make_prior_inference_fn(prior_params, normalizer_params, ckpt_cfg)
        env = PriorDecoderHighLevelWrapper(
            base_env,
            prior_fn,
            decoder_fn,
            latent_size,
            proprio_size,
            deterministic_prior=config["prior_decoder"]["deterministic_prior"],
            noise_logvar=config["prior_decoder"]["noise_logvar"],
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    start_state = jit_reset(rng)
    obs_size = start_state.obs.shape[-1]
    normalize = running_statistics.normalize
    policy_action_size = env.action_size

    if mode == "scratch":

        def network_factory(*args, **kwargs):
            return make_scratch_ppo_networks(
                obs_size=obs_size,
                action_size=policy_action_size,
                policy_hidden_layer_sizes=tuple(config["network"]["policy_layers"]),
                value_hidden_layer_sizes=tuple(config["network"]["value_layers"]),
            )

        dummy_net = network_factory()
    else:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=tuple(config["network"]["policy_layers"]),
            value_hidden_layer_sizes=tuple(config["network"]["value_layers"]),
        )
        dummy_net = network_factory(
            obs_size, policy_action_size, preprocess_observations_fn=normalize
        )

    key_p, key_v = jax.random.split(jax.random.PRNGKey(0))
    dummy_normalizer = running_statistics.init_state(
        specs.Array(start_state.obs.shape[-1:], jnp.dtype("float32"))
    )
    dummy_policy = dummy_net.policy_network.init(key_p)
    dummy_value = dummy_net.value_network.init(key_v)
    abstract_brax_params = (dummy_normalizer, dummy_policy, dummy_value)

    brax_params = load_task_transfer_brax_params(
        checkpoint_path, abstract_brax_params, step=ckpt_step
    )

    if mode == "scratch":
        ppo_network = network_factory()
    else:
        ppo_network = network_factory(
            obs_size, policy_action_size, preprocess_observations_fn=normalize
        )

    policy_network = ppo_network.policy_network
    parametric_action_distribution = ppo_network.parametric_action_distribution

    def inference_fn(params, observations, key_sample):
        del key_sample
        param_subset = (params[0], params[1])
        logits = policy_network.apply(*param_subset, observations)
        action = parametric_action_distribution.mode(logits)
        return action, {}

    jit_inference_fn = jax.jit(inference_fn)

    print(f"  Model loaded successfully.", flush=True)
    return jit_reset, jit_step, jit_inference_fn, brax_params, config


def run_rollout(
    jit_reset, jit_step, jit_inference_fn, brax_params, seed, episode_length
):
    rng = jax.random.PRNGKey(seed)
    reset_rng, act_rng = jax.random.split(rng)

    state = jit_reset(reset_rng)
    distances = []

    for _ in range(episode_length):
        _, act_rng = jax.random.split(act_rng)
        action, _ = jit_inference_fn(brax_params, state.obs, act_rng)
        state = jit_step(state, action)
        x, y = float(state.data.qpos[0]), float(state.data.qpos[1])
        distances.append(np.sqrt(x**2 + y**2))

    return np.array(distances)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--episode_length", type=int, default=2000)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()

    jit_reset, jit_step, jit_inference_fn, brax_params, config = load_model(
        args.checkpoint_path, checkpoint_step=args.step
    )
    mode = config["mode"]

    # Run multiple seeds, reusing the same compiled model
    all_distances = []
    for seed in range(args.num_seeds):
        distances = run_rollout(
            jit_reset,
            jit_step,
            jit_inference_fn,
            brax_params,
            seed,
            args.episode_length,
        )
        all_distances.append(distances)
        print(f"  Seed {seed}: final distance = {distances[-1]:.3f} m", flush=True)

    # Save as (num_seeds, episode_length) array
    np.savez(args.output_path, distances=np.stack(all_distances), mode=np.array(mode))


if __name__ == "__main__":
    main()
