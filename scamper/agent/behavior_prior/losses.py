"""Loss functions for behavior-prior training.

The behavior-prior training loss is KL divergence between the frozen distill
encoder and the trainable behavior-prior distributions. The key difference
from mlp_prior is that the distill encoder takes
concat([imitation_target, proprioception]) rather than just imitation_target.
"""

from typing import Any, Callable, Dict, Tuple

from brax.training import types

import jax
import jax.numpy as jnp

from scamper.agent.imitation.prior import (
    compute_encoder_prior_kl_loss,
    create_ramp_schedule,
)
from scamper.agent.observation_utils import (
    DictRunningStatisticsState,
    normalize_dict_obs,
)


def compute_behavior_prior_training_loss(
    behavior_prior_params: Dict,
    frozen_encoder_params: Dict,
    normalizer_params: DictRunningStatisticsState,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    encoder_apply_fn: Callable,
    behavior_prior_apply_fn: Callable,
    kl_weight: float = 1.0,
    kl_schedule: Callable | None = None,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Compute the behavior-prior training loss.

    KL divergence between the frozen distill encoder and the trainable
    behavior prior.

    Args:
        behavior_prior_params: Trainable behavior-prior network parameters
            (BehaviorPriorTrainingParams; the inner field is named ``rgm``
            for checkpoint-key compat).
        frozen_encoder_params: Frozen DistillEncoder parameters.
        normalizer_params: Dict observation normalizer parameters.
        data: Transition data with shape [B, T], observation is a dict.
        rng: Random key.
        step: Current training step (for scheduling).
        encoder_apply_fn: Function ``(params, imitation_target,
            proprioception, key) -> (mean, logvar)``.
        behavior_prior_apply_fn: Function ``(params, proprioception) ->
            (mean, logvar)``.
        kl_weight: Weight for KL divergence loss.
        kl_schedule: Optional schedule function for KL weight.

    Returns:
        Tuple of ``(total_loss, metrics_dict)``.
    """
    _, encoder_key = jax.random.split(rng, 2)

    # [B, T] -> [T, B] -> [T*B]
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    data = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), data)

    # Normalize dict observations
    normalized_obs = normalize_dict_obs(data.observation, normalizer_params)
    traj_obs = normalized_obs["imitation_target"]
    proprio_obs = normalized_obs["proprioception"]

    # Distill encoder takes both obs keys (frozen — apply stop_gradient)
    encoder_mean, encoder_logvar = encoder_apply_fn(
        frozen_encoder_params, traj_obs, proprio_obs, encoder_key
    )
    encoder_mean = jax.lax.stop_gradient(encoder_mean)
    encoder_logvar = jax.lax.stop_gradient(encoder_logvar)

    # Behavior prior takes only proprioception (trainable). Inner field name
    # 'rgm' is kept for checkpoint-key compat (see BehaviorPriorTrainingParams).
    behavior_prior_mean, behavior_prior_logvar = behavior_prior_apply_fn(
        behavior_prior_params.rgm, proprio_obs
    )

    # KL(encoder || behavior_prior). Reusing the mlp_prior helper, whose name
    # is unchanged because that's a separate package.
    kl_loss = compute_encoder_prior_kl_loss(
        encoder_mean, encoder_logvar, behavior_prior_mean, behavior_prior_logvar
    )

    # Apply schedule if provided
    current_kl_weight = kl_weight
    if kl_schedule is not None:
        current_kl_weight = kl_schedule(step)

    total_loss = current_kl_weight * kl_loss

    metrics = {
        "kl_loss": kl_loss,
        "kl_weight": current_kl_weight,
        "mean_diff_l2": jnp.mean(
            jnp.linalg.norm(encoder_mean - behavior_prior_mean, axis=-1)
        ),
        "logvar_diff_l2": jnp.mean(
            jnp.linalg.norm(encoder_logvar - behavior_prior_logvar, axis=-1)
        ),
    }

    return total_loss, metrics
