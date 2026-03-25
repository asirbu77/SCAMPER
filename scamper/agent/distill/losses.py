"""Loss functions for DAgger distillation.

Three loss terms:
- L_action: MSE between student and teacher action parameters
- L_KL: KL(encoder || learned_prior) for diagonal Gaussians
- L_AR1: Temporal smoothness of consecutive encoder means
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp

from scamper.agent.distill.distill_networks import (
    DistillDecoder,
    DistillEncoder,
    DistillNetworkParams,
    DistillPrior,
    reparameterize,
)
from scamper.agent.observation_utils import (
    DictRunningStatisticsState,
    concat_flat_dict_obs,
    normalize_dict_obs,
)


def compute_action_matching_loss(
    student_action_params: jnp.ndarray,
    teacher_action_params: jnp.ndarray,
) -> jnp.ndarray:
    """MSE between student and teacher action distribution parameters.

    Args:
        student_action_params: Student decoder output [N, action_param_size].
        teacher_action_params: Teacher MLP output [N, action_param_size].

    Returns:
        Scalar MSE loss.
    """
    return jnp.mean(jnp.square(student_action_params - teacher_action_params))


def compute_encoder_prior_kl_loss(
    encoder_mean: jnp.ndarray,
    encoder_logvar: jnp.ndarray,
    prior_mean: jnp.ndarray,
    prior_logvar: jnp.ndarray,
) -> jnp.ndarray:
    """KL(encoder || prior) for diagonal Gaussians.

    KL_j = 0.5 * (log(sig_p^2/sig_q^2) + sig_q^2/sig_p^2
                   + (mu_q - mu_p)^2/sig_p^2 - 1)
    Sum over latent dims, mean over samples.

    Args:
        encoder_mean: [N, latent_dim]
        encoder_logvar: [N, latent_dim]
        prior_mean: [N, latent_dim]
        prior_logvar: [N, latent_dim]

    Returns:
        Scalar KL divergence loss.
    """
    log_var_diff = prior_logvar - encoder_logvar
    var_ratio = jnp.exp(encoder_logvar - prior_logvar)
    mean_diff_sq = jnp.square(encoder_mean - prior_mean) / jnp.exp(prior_logvar)

    element_wise_kl = 0.5 * (log_var_diff + var_ratio + mean_diff_sq - 1)
    kl_per_sample = jnp.sum(element_wise_kl, axis=-1)
    return jnp.mean(kl_per_sample)


def compute_ar1_temporal_loss(
    encoder_means: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """AR(1) temporal smoothness: ||mu_e_t - mu_e_{t-1}||^2.

    Args:
        encoder_means: [T, B, latent_dim] encoder means over time.
        valid_mask: [T, B] binary mask (1 where consecutive steps are same episode).

    Returns:
        Scalar L2 loss between consecutive encoder means.
    """
    z_prev = encoder_means[:-1]  # [T-1, B, D]
    z_curr = encoder_means[1:]  # [T-1, B, D]
    l2_diff = jnp.mean(jnp.square(z_curr - z_prev), axis=-1)  # [T-1, B]
    mask = valid_mask[:-1]
    return jnp.sum(l2_diff * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def create_ramp_schedule(
    start_value: float = 0.0001,
    end_value: float = 0.1,
    total_steps: int = 100,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    schedule: str = "linear",
) -> Callable[[int], jnp.ndarray]:
    """Create a schedule for loss weights (linear or cosine ramp).

    Args:
        start_value: Starting value.
        end_value: Ending value.
        total_steps: Total number of evaluation steps.
        start_frac: Fraction of total_steps at which ramping begins.
        end_frac: Fraction of total_steps by which ramping completes.
        schedule: "linear" or "cosine".

    Returns:
        Function step -> scheduled_value.
    """
    delay_steps = int(start_frac * total_steps)
    ramp_steps = max(int((end_frac - start_frac) * total_steps), 1)

    def schedule_fn(step: int) -> jnp.ndarray:
        step = jnp.asarray(step, dtype=jnp.float32)
        effective_step = step - delay_steps
        progress = jnp.clip(effective_step / ramp_steps, 0.0, 1.0)
        is_delayed = step < delay_steps

        if schedule == "linear":
            ramped = start_value + progress * (end_value - start_value)
        elif schedule == "cosine":
            ramped = start_value + 0.5 * (end_value - start_value) * (
                1 - jnp.cos(jnp.pi * progress)
            )
        else:
            raise ValueError(f"schedule must be 'linear' or 'cosine', not {schedule}")

        return jnp.where(is_delayed, start_value, ramped)

    return schedule_fn


def compute_distill_loss(
    params: DistillNetworkParams,
    normalizer_params: DictRunningStatisticsState,
    data: Any,
    rng: jnp.ndarray,
    step: int,
    *,
    teacher_fn: Callable,
    teacher_normalizer_params: DictRunningStatisticsState,
    encoder: DistillEncoder,
    decoder: DistillDecoder,
    prior: DistillPrior,
    kl_weight: float,
    ar1_weight: float,
    kl_schedule_fn: Callable | None,
    ar1_schedule_fn: Callable | None,
) -> tuple[jnp.ndarray, dict]:
    """Compute the full distillation loss.

    The first five positional args match what gradient_update_fn passes from
    minibatch_step: (params, normalizer_params, data, rng, step).
    All remaining args are keyword-only and supplied via functools.partial.

    Data arrives as [B, T, ...]. We swap to [T, B, ...] for temporal loss,
    then merge to [T*B, ...] for the main losses.

    Args:
        params: Trainable DistillNetworkParams (encoder, decoder, prior).
        normalizer_params: Student's observation normalizer.
        data: Transition data with shape [B, T].
        rng: Random key.
        step: Current training step.
        teacher_fn: Frozen teacher query function (normalized_obs -> action_params).
        teacher_normalizer_params: Teacher's observation normalizer.
        encoder: DistillEncoder module.
        decoder: DistillDecoder module.
        prior: DistillPrior module.
        kl_weight: Base weight for KL loss.
        ar1_weight: Base weight for AR(1) loss.
        kl_schedule_fn: Optional schedule for KL weight.
        ar1_schedule_fn: Optional schedule for AR(1) weight.

    Returns:
        Tuple of (total_loss, metrics_dict).
    """
    _, enc_key = jax.random.split(rng, 2)

    # [B, T, ...] -> [T, B, ...]
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    T, B = data.observation["proprioception"].shape[:2]

    # Normalize observations for student
    obs_flat = jax.tree_util.tree_map(
        lambda x: x.reshape(-1, *x.shape[2:]), data.observation
    )
    normalized_obs = normalize_dict_obs(obs_flat, normalizer_params)
    traj_obs = normalized_obs["imitation_target"]
    proprio_obs = normalized_obs["proprioception"]

    # --- Student forward pass ---
    encoder_input = jnp.concatenate([traj_obs, proprio_obs], axis=-1)
    latent_mean, latent_logvar = encoder.apply(params.encoder, encoder_input)

    z = reparameterize(enc_key, latent_mean, latent_logvar)

    decoder_input = jnp.concatenate([z, proprio_obs], axis=-1)
    student_action_params, _ = decoder.apply(params.decoder, decoder_input)

    # --- Prior forward pass ---
    prior_mean, prior_logvar = prior.apply(params.prior, proprio_obs)

    # --- Teacher forward pass (frozen) ---
    teacher_normalized_obs = normalize_dict_obs(obs_flat, teacher_normalizer_params)
    teacher_action_params = jax.lax.stop_gradient(teacher_fn(teacher_normalized_obs))

    # --- Losses ---
    action_loss = compute_action_matching_loss(
        student_action_params, teacher_action_params
    )
    kl_loss = compute_encoder_prior_kl_loss(
        latent_mean, latent_logvar, prior_mean, prior_logvar
    )

    # AR(1) loss needs [T, B, D] shape
    encoder_means_tb = latent_mean.reshape(T, B, -1)

    # Valid mask: 1 where consecutive steps belong to same episode
    discount = data.discount.reshape(T, B)
    truncation = data.extras["state_extras"]["truncation"].reshape(T, B)
    valid_mask = discount * (1.0 - truncation)

    ar1_loss = compute_ar1_temporal_loss(encoder_means_tb, valid_mask)

    # Apply schedules
    current_kl_weight = (
        kl_schedule_fn(step) if kl_schedule_fn is not None else kl_weight
    )
    current_ar1_weight = (
        ar1_schedule_fn(step) if ar1_schedule_fn is not None else ar1_weight
    )

    total_loss = (
        action_loss + current_kl_weight * kl_loss + current_ar1_weight * ar1_loss
    )

    metrics = {
        "total_loss": total_loss,
        "action_loss": action_loss,
        "kl_loss": kl_loss,
        "ar1_loss": ar1_loss,
        "kl_weight": current_kl_weight,
        "ar1_weight": current_ar1_weight,
        "mean_diff_l2": jnp.mean(jnp.linalg.norm(latent_mean - prior_mean, axis=-1)),
        "logvar_diff_l2": jnp.mean(
            jnp.linalg.norm(latent_logvar - prior_logvar, axis=-1)
        ),
    }

    return total_loss, metrics
