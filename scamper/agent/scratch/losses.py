# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright 2026 SCAMPER authors. Adapted from the imitation PPO
# losses, with the KL to a standard Gaussian replaced by the KL to a jointly
# trained motion prior (as in DAgger distillation).

"""PPO losses for from-scratch three-network training.

Standard PPO (clipped surrogate + value + entropy) plus the two latent
regularizers used by ``train_distill_expert``:

- ``KL(encoder || motion prior)`` -- the only term that trains the motion
  prior, which is not in the action path. No stop-gradient on either side, so
  encoder and prior are pulled towards each other exactly as in distillation.
- AR(1) temporal smoothness on consecutive encoder means.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from brax.training import types

from scamper.agent.distill_expert.losses import compute_encoder_motion_prior_kl_loss
from scamper.agent.imitation.losses import (
    PPONetworkParams,
    compute_ar1_temporal_loss,
    compute_gae,
)
from scamper.agent.scratch import scratch_networks

__all__ = [
    "PPONetworkParams",
    "compute_ar1_temporal_loss",
    "compute_encoder_motion_prior_kl_loss",
    "compute_gae",
    "compute_ppo_loss",
]


def compute_ppo_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    ppo_network: scratch_networks.PPOScratchNetworks,
    entropy_cost: float = 1e-4,
    kl_weight: float = 1e-3,
    ar1_weight: float = 5e-3,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vf_coefficient: float = 0.5,
    kl_schedule_fn: Callable[[int], jnp.ndarray] | None = None,
    ar1_schedule_fn: Callable[[int], jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, types.Metrics]:
    """Compute the PPO loss with motion-prior KL and AR(1) regularization.

    The first five arguments are positional because
    ``utils.gradients.gradient_update_fn`` passes them that way; everything
    else is bound via ``functools.partial``.

    Args:
        params: PPO network parameters (policy and value).
        normalizer_params: Running statistics for observation normalization.
        data: Transition batch with shape [B, T]. Required extra fields:
            - ``data.extras["state_extras"]["truncation"]``
            - ``data.extras["policy_extras"]["raw_action"]``
            - ``data.extras["policy_extras"]["log_prob"]``
        rng: JAX random key.
        step: Current eval iteration (drives the weight schedules).
        ppo_network: Policy/value/action-distribution container.
        entropy_cost: Entropy bonus coefficient.
        kl_weight: Base weight for KL(encoder || motion prior). Ignored when
            ``kl_schedule_fn`` is provided.
        ar1_weight: Base weight for the AR(1) loss. Ignored when
            ``ar1_schedule_fn`` is provided.
        discounting: Discount factor (gamma) for GAE.
        reward_scaling: Multiplier applied to rewards.
        gae_lambda: GAE lambda parameter.
        clipping_epsilon: PPO clipping range for the policy ratio.
        normalize_advantage: Whether to normalize advantages.
        vf_coefficient: Coefficient for the value function loss.
        kl_schedule_fn: Optional schedule ``step -> kl_weight``.
        ar1_schedule_fn: Optional schedule ``step -> ar1_weight``.

    Returns:
        Tuple of (total_loss, metrics dict).
    """
    _, policy_key, entropy_key = jax.random.split(rng, 3)
    parametric_action_distribution = ppo_network.parametric_action_distribution
    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply

    # Put the time dimension first.
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Stored per-sample RNGs let us replay the exact latent sample drawn at
    # rollout time, which the PPO ratio depends on.
    policy_rng = data.extras["policy_extras"].get("policy_rng")

    if policy_rng is None:
        # Standard path: use fresh RNG
        (
            policy_logits,
            latent_mean,
            latent_logvar,
            motion_prior_mean,
            motion_prior_logvar,
        ) = policy_apply(normalizer_params, params.policy, data.observation, policy_key)
    else:
        # Deterministic replay: use stored per-sample RNGs.
        # Flatten [T, B] into a single batch dim to avoid vmap (better for
        # buffer donation).
        obs_leaf = jax.tree_util.tree_leaves(data.observation)[0]
        T, B = obs_leaf.shape[:2]

        flat_obs = jax.tree_util.tree_map(
            lambda x: x.reshape((T * B,) + x.shape[2:]),
            data.observation,
        )
        flat_rng = policy_rng.reshape((T * B, 2))

        (
            flat_logits,
            flat_mean,
            flat_logvar,
            flat_prior_mean,
            flat_prior_logvar,
        ) = policy_apply(normalizer_params, params.policy, flat_obs, flat_rng)

        def _unflatten(x):
            return x.reshape((T, B) + x.shape[1:])

        policy_logits = _unflatten(flat_logits)
        latent_mean = _unflatten(flat_mean)
        latent_logvar = _unflatten(flat_logvar)
        motion_prior_mean = _unflatten(flat_prior_mean)
        motion_prior_logvar = _unflatten(flat_prior_logvar)

    baseline = value_apply(normalizer_params, params.value, data.observation)

    # Get the last timestep from dict observation (tree_map handles dict structure)
    last_next_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(normalizer_params, params.value, last_next_obs)

    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras["policy_extras"]["raw_action"]
    )
    behaviour_action_log_probs = data.extras["policy_extras"]["log_prob"]

    vs, advantages = compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting,
    )
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = (
        jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    )

    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss
    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * vf_coefficient

    # Entropy reward
    entropy = jnp.mean(
        parametric_action_distribution.entropy(policy_logits, entropy_key)
    )
    entropy_loss = entropy_cost * -entropy

    # Latent regularization weights (schedules are indexed by eval iteration)
    current_kl_weight = (
        kl_schedule_fn(step) if kl_schedule_fn is not None else kl_weight
    )
    current_ar1_weight = (
        ar1_schedule_fn(step) if ar1_schedule_fn is not None else ar1_weight
    )

    # KL(encoder || motion prior): sum over latent dims, mean over samples.
    kl_loss = compute_encoder_motion_prior_kl_loss(
        latent_mean, latent_logvar, motion_prior_mean, motion_prior_logvar
    )

    # L2 loss to previous latent mean (AR(1) temporal smoothness)
    ar1_loss = compute_ar1_temporal_loss(latent_mean, data.discount, truncation)

    kl_loss_weighted = current_kl_weight * kl_loss
    ar1_loss_weighted = current_ar1_weight * ar1_loss
    latent_loss = kl_loss_weighted + ar1_loss_weighted

    total_loss = policy_loss + v_loss + entropy_loss + latent_loss

    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        "total_latent_loss": latent_loss,
        "kl_loss": kl_loss,
        "ar1_loss": ar1_loss,
        "latent_kl_loss": kl_loss_weighted,
        "latent_ar1_loss": ar1_loss_weighted,
        "kl_weight": current_kl_weight,
        "ar1_weight": current_ar1_weight,
        "mean_diff_l2": jnp.mean(
            jnp.linalg.norm(latent_mean - motion_prior_mean, axis=-1)
        ),
        "logvar_diff_l2": jnp.mean(
            jnp.linalg.norm(latent_logvar - motion_prior_logvar, axis=-1)
        ),
    }
