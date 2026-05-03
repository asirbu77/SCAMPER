"""Load and query a frozen expert teacher for DAgger distillation.

The teacher is a trained MLPPolicy checkpoint (from train_expert) that maps
dictionary observations (imitation_target + proprioception) to action
distribution parameters.
"""

import logging
from typing import Any, Callable

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from brax.training import distribution
from brax.training.acme import running_statistics, specs
from omegaconf import OmegaConf

from scamper.agent.expert.mlp_network import MLPPolicy
from scamper.utils.observation_utils import (
    DictRunningStatisticsState,
    concat_flat_dict_obs,
    normalize_dict_obs,
)


def load_frozen_teacher(
    checkpoint_path: str,
    step: int | None = None,
) -> tuple[Callable, DictRunningStatisticsState, dict, Any]:
    """Load a trained expert teacher from checkpoint.

    Args:
        checkpoint_path: Path to expert checkpoint directory (from train_expert).
        step: Specific step to load. None = latest.

    Returns:
        Tuple of (teacher_query_fn, teacher_normalizer_params, teacher_config,
                  teacher_policy_params).
        teacher_query_fn: (normalized_obs_dict) -> action_params [B, 76].
        teacher_policy_params: Raw Flax params for use with MLPPolicy.apply().
    """
    # Load config
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix="PPONetwork")
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()

        cfg = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(config=ocp.args.JsonRestore()),
        )["config"]

    cfg = OmegaConf.create(cfg)
    network_config = cfg.network_config

    # Validate this is an MLP checkpoint
    arch_name = network_config.get("arch_name", "")
    if arch_name not in ("mlp", ""):
        logging.warning(
            f"Teacher checkpoint arch_name is '{arch_name}', expected 'mlp'. "
            "Attempting to load anyway."
        )

    obs_sizes = dict(network_config.obs_sizes)
    action_size = network_config.action_size
    mlp_hidden_layer_sizes = tuple(network_config.mlp_layer_sizes)
    action_param_size = action_size * 2  # mean + logstd

    # Create MLP module matching checkpoint architecture
    mlp_module = MLPPolicy(
        layer_sizes=list(mlp_hidden_layer_sizes) + [action_param_size],
    )

    # Create abstract tree for Orbax restoration
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
    abstract_policy = (normalizer_state, init_params)

    # Load actual parameters
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        policy_data = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(policy=ocp.args.StandardRestore(abstract_policy)),
        )["policy"]

    teacher_normalizer, teacher_params = policy_data

    logging.info(f"Loaded expert teacher from {checkpoint_path} at step {step}")

    # Create query function
    teacher_fn = make_teacher_query_fn(mlp_module, teacher_params)

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    return teacher_fn, teacher_normalizer, config_dict, teacher_params


def make_teacher_query_fn(
    mlp_module: MLPPolicy,
    teacher_params: Any,
) -> Callable:
    """Create a JAX-traceable function that queries the teacher.

    The returned function takes normalized dict observations and returns
    raw action logits. It does NOT normalize — the caller must normalize first.

    Args:
        mlp_module: MLPPolicy module matching the teacher architecture.
        teacher_params: Loaded teacher network parameters.

    Returns:
        teacher_fn(normalized_obs_dict) -> action_params [B, action_param_size].
        normalized_obs_dict should already be normalized.
    """

    def teacher_fn(normalized_obs_dict: dict[str, jnp.ndarray]) -> jnp.ndarray:
        flat_obs = concat_flat_dict_obs(normalized_obs_dict)
        return mlp_module.apply(teacher_params, flat_obs)

    return teacher_fn
