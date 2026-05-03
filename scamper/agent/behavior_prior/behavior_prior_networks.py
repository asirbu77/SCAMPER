"""Network definitions for behavior-prior training.

This module provides:
- Loading frozen encoder/decoder/motion-prior from a distill_expert checkpoint
- Creating a trainable behavior-prior network (same architecture as
  mlp_prior.Prior, which is reused as the underlying class)
- Combining networks for checkpointing

The key difference from mlp_prior is that the DistillEncoder takes
concat([imitation_target, proprioception]) rather than just imitation_target.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from brax.training import distribution, networks, types
from brax.training.acme import running_statistics, specs
from flax import linen as nn
from omegaconf import OmegaConf

from scamper.agent import checkpointing
from scamper.agent.distill_expert.distill_networks import (
    DistillDecoder,
    DistillEncoder,
    DistillNetworkParams,
    DistillMotionPrior,
    init_distill_params,
    make_distill_networks,
    reparameterize,
)
from scamper.agent.distill_expert.distill_train import DistillTrainingState
from scamper.agent.imitation.prior import Prior
from scamper.agent.observation_utils import (
    DictRunningStatisticsState,
    flatten_obs_dict,
    normalize_dict_obs,
)

# Alias for clarity: the behavior prior reuses mlp_prior.Prior's architecture.
BehaviorPrior = Prior


def load_frozen_distill_networks(
    checkpoint_path: str,
    step: int | None = None,
) -> Tuple[Dict, Dict, Dict, DictRunningStatisticsState, Dict]:
    """Load encoder, decoder, and prior parameters from a distill checkpoint.

    Loads from the ``train_state`` key (not ``policy``) because the ``policy``
    key stores a dummy standard-shape encoder for task-transfer compatibility,
    whereas ``train_state`` contains the actual DistillEncoder parameters.

    Args:
        checkpoint_path: Path to the distill checkpoint directory.
        step: Specific step to load. If None, loads the latest.

    Returns:
        Tuple of (encoder_params, decoder_params, motion_prior_params,
        normalizer_params, config_dict).  The ``*_params`` dicts are in
        Flax init format (``{"params": {...}}``).
    """
    mgr_options = ocp.CheckpointManagerOptions(
        create=False, step_prefix="DistillNetwork"
    )

    # Load config
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()
        cfg = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(config=ocp.args.JsonRestore()),
        )["config"]

    cfg = OmegaConf.create(cfg)

    # Extract network config
    obs_sizes = dict(cfg.network_config.obs_sizes)
    action_size = cfg.network_config.action_size
    latent_size = cfg.network_config.intention_size
    encoder_hidden = tuple(cfg.network_config.encoder_layer_sizes)
    decoder_hidden = tuple(cfg.network_config.decoder_layer_sizes)
    # cfg attribute `prior_layer_sizes` kept for compat; sizes the motion prior.
    motion_prior_hidden = tuple(cfg.network_config.prior_layer_sizes)

    # Create abstract structures to match the checkpoint pytree.
    encoder, decoder, motion_prior, _ = make_distill_networks(
        obs_sizes=obs_sizes,
        action_size=action_size,
        latent_size=latent_size,
        encoder_hidden_layer_sizes=encoder_hidden,
        decoder_hidden_layer_sizes=decoder_hidden,
        motion_prior_hidden_layer_sizes=motion_prior_hidden,
    )

    init_params = init_distill_params(
        encoder, decoder, motion_prior, obs_sizes, latent_size, jax.random.PRNGKey(0)
    )

    # Recreate the optimizer used during distill training (structure must match)
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(learning_rate=1e-4, weight_decay=0.0, eps=1e-5),
    )

    normalizer_state = DictRunningStatisticsState(
        imitation_target=running_statistics.init_state(
            specs.Array(obs_sizes["imitation_target"], jnp.dtype("float32"))
        ),
        proprioception=running_statistics.init_state(
            specs.Array(obs_sizes["proprioception"], jnp.dtype("float32"))
        ),
    )

    abstract_state = DistillTrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        normalizer_params=normalizer_state,
        env_steps=jnp.int32(0),
    )

    # Restore actual training state
    train_state = checkpointing.load_training_state(
        checkpoint_path,
        abstract_state,
        step_prefix="DistillNetwork",
        step=step,
    )

    return (
        train_state.params.encoder,
        train_state.params.decoder,
        train_state.params.prior,
        train_state.normalizer_params,
        OmegaConf.to_container(cfg),
    )


def make_behavior_prior_networks(
    latent_size: int,
    proprioceptive_obs_size: int,
    behavior_prior_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> Tuple[BehaviorPrior, networks.FeedForwardNetwork]:
    """Create the trainable behavior-prior network.

    The behavior prior reuses ``mlp_prior.Prior``'s architecture: an MLP that
    maps proprioceptive observations to ``(mean, logvar)`` of a latent
    distribution.

    Args:
        latent_size: Dimension of the latent intention space.
        proprioceptive_obs_size: Size of proprioceptive observations.
        behavior_prior_hidden_layer_sizes: Hidden layer sizes for the
            behavior-prior MLP.

    Returns:
        Tuple of (behavior_prior_module, behavior_prior_feedforward_network).
    """
    behavior_prior_module = BehaviorPrior(
        layer_sizes=list(behavior_prior_hidden_layer_sizes),
        latents=latent_size,
    )

    def init_fn(key):
        dummy_obs = jnp.zeros((1, proprioceptive_obs_size))
        return behavior_prior_module.init(key, dummy_obs)

    def apply_fn(params, obs):
        return behavior_prior_module.apply(params, obs)

    return behavior_prior_module, networks.FeedForwardNetwork(
        init=init_fn, apply=apply_fn
    )


def make_distill_encoder_apply_fn(
    encoder_hidden_layer_sizes: Sequence[int],
    latent_size: int,
    obs_sizes: Mapping[str, int],
    clamp_logvar: bool = True,
    logvar_min: float = -5.0,
    logvar_max: float = 0.0,
) -> Callable:
    """Create a distill encoder apply function.

    Unlike ``mlp_prior.make_encoder_apply_fn`` which takes only
    ``imitation_target``, this function concatenates both observation keys
    before encoding.

    Args:
        encoder_hidden_layer_sizes: Hidden layer sizes for the encoder.
        latent_size: Dimension of the latent space.
        obs_sizes: Dict with observation sizes.
        clamp_logvar: Whether to clamp logvar values.
        logvar_min: Minimum logvar value.
        logvar_max: Maximum logvar value.

    Returns:
        Function ``(params, imitation_target, proprioception, key) ->
        (mean, logvar)``.
    """
    encoder_module = DistillEncoder(
        layer_sizes=list(encoder_hidden_layer_sizes),
        latents=latent_size,
        clamp_logvar=clamp_logvar,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )

    def apply_fn(
        params: Dict,
        imitation_target: jnp.ndarray,
        proprioception: jnp.ndarray,
        key: jax.Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        encoder_input = jnp.concatenate([imitation_target, proprioception], axis=-1)
        return encoder_module.apply(params, encoder_input)

    return apply_fn


def make_distill_encoder_decoder_inference_fn(
    encoder_params: Dict,
    decoder_params: Dict,
    normalizer_params: DictRunningStatisticsState,
    encoder_hidden_layer_sizes: Sequence[int],
    decoder_hidden_layer_sizes: Sequence[int],
    latent_size: int,
    action_size: int,
    obs_sizes: Mapping[str, int],
    deterministic: bool = True,
    clamp_logvar: bool = True,
    logvar_min: float = -5.0,
    logvar_max: float = 0.0,
) -> Callable:
    """Create a frozen distill encoder+decoder policy for data collection.

    Args:
        encoder_params: Frozen DistillEncoder parameters.
        decoder_params: Frozen DistillDecoder parameters.
        normalizer_params: Dict observation normalizer parameters.
        encoder_hidden_layer_sizes: Hidden layer sizes for encoder.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder.
        latent_size: Dimension of the latent space.
        action_size: Size of the action space.
        obs_sizes: Dict with observation sizes.
        deterministic: If True, use mean of distributions.
        clamp_logvar: Whether to clamp encoder logvar.
        logvar_min: Minimum logvar value.
        logvar_max: Maximum logvar value.

    Returns:
        Policy function ``(obs_dict, key) -> (action, extras)``.
    """
    action_param_size = action_size * 2
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    encoder_module = DistillEncoder(
        layer_sizes=list(encoder_hidden_layer_sizes),
        latents=latent_size,
        clamp_logvar=clamp_logvar,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )

    decoder_module = DistillDecoder(
        layer_sizes=list(decoder_hidden_layer_sizes) + [action_param_size],
    )

    def policy_fn(
        obs: Mapping[str, jnp.ndarray], key: jax.Array
    ) -> Tuple[jnp.ndarray, Dict]:
        key_encoder, key_action = jax.random.split(key)

        flat_obs = flatten_obs_dict(obs)
        normalized_obs = normalize_dict_obs(flat_obs, normalizer_params)

        traj_obs = normalized_obs["imitation_target"]
        proprio_obs = normalized_obs["proprioception"]

        # DistillEncoder takes concat([imitation_target, proprioception])
        encoder_input = jnp.concatenate([traj_obs, proprio_obs], axis=-1)
        latent_mean, latent_logvar = encoder_module.apply(encoder_params, encoder_input)

        if deterministic:
            z = latent_mean
        else:
            z = reparameterize(key_encoder, latent_mean, latent_logvar)

        decoder_input = jnp.concatenate([z, proprio_obs], axis=-1)
        logits, _ = decoder_module.apply(decoder_params, decoder_input)

        if deterministic:
            action = parametric_action_distribution.mode(logits)
        else:
            raw_action = parametric_action_distribution.sample_no_postprocessing(
                logits, key_action
            )
            action = parametric_action_distribution.postprocess(raw_action)

        extras = {
            "latent_mean": latent_mean,
            "latent_logvar": latent_logvar,
            "intention": z,
            "logits": logits,
        }

        return action, extras

    return policy_fn


def create_combined_checkpoint_params(
    encoder_params: Dict,
    decoder_params: Dict,
    motion_prior_params: Dict,
    behavior_prior_params: Dict,
    normalizer_params: DictRunningStatisticsState,
) -> Tuple[DictRunningStatisticsState, Dict]:
    """Combine parameters into checkpoint-compatible format.

    Creates a checkpoint with encoder, decoder, motion prior (from
    distill_expert), and behavior prior (newly trained).

    Args:
        encoder_params: DistillEncoder parameters (Flax init format).
        decoder_params: DistillDecoder parameters (Flax init format).
        motion_prior_params: DistillMotionPrior parameters (Flax init format).
        behavior_prior_params: Behavior-prior parameters (Flax init format or
            inner params).
        normalizer_params: Dict observation normalizer parameters.

    Returns:
        Tuple of ``(normalizer_params, combined_network_params)``.
    """

    def _extract_inner(params):
        if "params" in params:
            return params["params"]
        return params

    # Dict keys 'prior' and 'rgm' kept for compat with task_transfer loaders;
    # 'prior' stores the motion prior, 'rgm' stores the behavior prior.
    combined_network_params = {
        "params": {
            "encoder": _extract_inner(encoder_params),
            "decoder": _extract_inner(decoder_params),
            "prior": _extract_inner(motion_prior_params),
            "rgm": _extract_inner(behavior_prior_params),
        }
    }

    return (normalizer_params, combined_network_params)


def create_abstract_behavior_prior_policy(
    cfg: Dict,
    behavior_prior_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> Tuple[Any, Any]:
    """Create abstract policy structure for checkpoint restoration.

    Args:
        cfg: Configuration dictionary from checkpoint.
        behavior_prior_hidden_layer_sizes: Hidden layer sizes for the
            behavior-prior network.

    Returns:
        Tuple of ``(normalizer_state, combined_policy_params)`` with correct
        structure.
    """
    latent_size = cfg["network_config"]["intention_size"]
    action_size = cfg["network_config"]["action_size"]
    obs_sizes = cfg["network_config"]["obs_sizes"]
    reference_obs_size = obs_sizes["imitation_target"]
    proprioceptive_obs_size = obs_sizes["proprioception"]

    encoder_hidden = tuple(cfg["network_config"]["encoder_layer_sizes"])
    decoder_hidden = tuple(cfg["network_config"]["decoder_layer_sizes"])
    # cfg attribute `prior_layer_sizes` kept for compat; sizes the motion prior.
    motion_prior_hidden = tuple(cfg["network_config"]["prior_layer_sizes"])

    action_param_size = action_size * 2

    # Create all modules
    enc = DistillEncoder(layer_sizes=list(encoder_hidden), latents=latent_size)
    dec = DistillDecoder(layer_sizes=list(decoder_hidden) + [action_param_size])
    motion_pri = DistillMotionPrior(
        layer_sizes=list(motion_prior_hidden), latents=latent_size
    )
    behavior_pri = BehaviorPrior(
        layer_sizes=list(behavior_prior_hidden_layer_sizes), latents=latent_size
    )

    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    total_obs_size = reference_obs_size + proprioceptive_obs_size
    enc_init = enc.init(k1, jnp.zeros((1, total_obs_size)))
    dec_init = dec.init(k2, jnp.zeros((1, latent_size + proprioceptive_obs_size)))
    motion_pri_init = motion_pri.init(k3, jnp.zeros((1, proprioceptive_obs_size)))
    behavior_pri_init = behavior_pri.init(k4, jnp.zeros((1, proprioceptive_obs_size)))

    # Dict keys 'prior' and 'rgm' kept for compat with task_transfer loaders.
    combined_params = {
        "params": {
            "encoder": enc_init["params"],
            "decoder": dec_init["params"],
            "prior": motion_pri_init["params"],
            "rgm": behavior_pri_init["params"],
        }
    }

    normalizer_state = DictRunningStatisticsState(
        imitation_target=running_statistics.init_state(
            specs.Array(reference_obs_size, jnp.dtype("float32"))
        ),
        proprioception=running_statistics.init_state(
            specs.Array(proprioceptive_obs_size, jnp.dtype("float32"))
        ),
    )

    return (normalizer_state, combined_params)
