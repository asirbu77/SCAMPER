"""Prior network module + KL utilities (formerly under scamper.agent.mlp_prior).

This module hosts the pieces of the old ``mlp_prior`` package that other code
still depends on after the prior-training pipeline was removed:

- ``Prior``: the proprioception-conditioned latent prior network. Its Flax
  submodule names (``hidden_*``, ``fc2_mean``, ``fc2_logvar``) are part of the
  saved-checkpoint structure for any pipeline that persists a prior under the
  ``"prior"`` dict key — load-bearing, do not rename.
- ``compute_encoder_prior_kl_loss``: KL divergence between two diagonal
  Gaussians (encoder vs. prior). Reused by behavior_prior training losses.
- ``create_ramp_schedule``: a (start_value, end_value)-style ramp distinct
  from the ``imitation.losses.create_ramp_schedule`` (which uses min/max
  semantics). Reused by behavior_prior training schedules.
"""

from collections.abc import Sequence
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from brax.training import networks
from flax import linen as nn


class Prior(nn.Module):
    """Prior network mapping proprioception -> (mean, logvar) of latents.

    Attributes:
        layer_sizes: Hidden layer dimensions for the MLP.
        latents: Dimension of the latent intention space.
        activation: Activation function (default: SiLU).
        kernel_init: Weight initializer (default: LeCun uniform).
        bias: Whether to use bias terms in Dense layers.
    """

    layer_sizes: Sequence[int]
    latents: int
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, get_activation: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray] | Tuple[Tuple[jnp.ndarray, jnp.ndarray], dict]:
        activations = {}
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            x = self.activation(x)
            x = nn.LayerNorm()(x)
            if get_activation:
                activations[f"layer_{i}"] = x

        mean_x = nn.Dense(self.latents, name="fc2_mean")(x)
        logvar_x = nn.Dense(self.latents, name="fc2_logvar")(x)

        if get_activation:
            activations["mean"] = mean_x
            activations["logvar"] = logvar_x
            return (mean_x, logvar_x), activations
        return mean_x, logvar_x


def compute_encoder_prior_kl_loss(
    encoder_mean: jnp.ndarray,
    encoder_logvar: jnp.ndarray,
    prior_mean: jnp.ndarray,
    prior_logvar: jnp.ndarray,
) -> jnp.ndarray:
    """KL(encoder || prior) for diagonal Gaussians.

    Sum over latent dimensions, mean over samples.

    Args:
        encoder_mean: Mean of encoder distribution [..., latent_dim].
        encoder_logvar: Log-variance of encoder distribution [..., latent_dim].
        prior_mean: Mean of prior distribution [..., latent_dim].
        prior_logvar: Log-variance of prior distribution [..., latent_dim].

    Returns:
        Scalar KL divergence loss.
    """
    log_var_diff = prior_logvar - encoder_logvar
    var_ratio = jnp.exp(encoder_logvar - prior_logvar)
    mean_diff_sq = jnp.square(encoder_mean - prior_mean) / jnp.exp(prior_logvar)

    element_wise_kl = 0.5 * (log_var_diff + var_ratio + mean_diff_sq - 1)
    kl_per_sample = jnp.sum(element_wise_kl, axis=-1)
    return jnp.mean(kl_per_sample)


def create_ramp_schedule(
    start_value: float = 0.0001,
    end_value: float = 0.1,
    total_steps: int = 100,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    schedule: str = "linear",
) -> Callable[[int], jnp.ndarray]:
    """Create a (start_value, end_value)-style ramp schedule.

    Distinct from ``imitation.losses.create_ramp_schedule`` (which uses
    (min_value, max_value) and supports cyclic schedules); this version is
    used by the behavior-prior training schedules.

    Args:
        start_value: Schedule value at and before the start fraction.
        end_value: Schedule value at and after the end fraction.
        total_steps: Total number of evaluation steps spanned by the schedule.
        start_frac: Fraction of total_steps at which ramping begins.
        end_frac: Fraction of total_steps at which ramping completes.
        schedule: ``"linear"`` or ``"cosine"``.

    Returns:
        A function mapping step -> schedule value.
    """
    delay_steps = int(start_frac * total_steps)
    ramp_steps = int((end_frac - start_frac) * total_steps)
    ramp_steps = max(ramp_steps, 1)

    def schedule_fn(step: int) -> jnp.ndarray:
        step = jnp.asarray(step, dtype=jnp.float32)

        if schedule == "linear":
            effective_step = step - delay_steps
            progress = jnp.clip(effective_step / ramp_steps, 0.0, 1.0)
            is_delayed = step < delay_steps
            ramped_value = start_value + progress * (end_value - start_value)
            return jnp.where(is_delayed, start_value, ramped_value)
        elif schedule == "cosine":
            effective_step = jnp.maximum(step - delay_steps, 0.0)
            is_delayed = step < delay_steps
            progress = jnp.clip(effective_step / ramp_steps, 0.0, 1.0)
            cosine_value = start_value + 0.5 * (end_value - start_value) * (
                1 - jnp.cos(jnp.pi * progress)
            )
            return jnp.where(is_delayed, start_value, cosine_value)
        else:
            raise ValueError(f"schedule must be 'linear' or 'cosine', not {schedule}")

    return schedule_fn
