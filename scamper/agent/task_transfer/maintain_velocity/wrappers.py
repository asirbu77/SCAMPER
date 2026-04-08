"""Environment wrappers for maintain_velocity task transfer training.

Four wrapper classes for different transfer learning modes:
- DecoderHighLevelWrapper: Policy outputs latent -> frozen decoder -> ctrl
- PriorDecoderHighLevelWrapper: Policy outputs residual -> add to prior mean -> decoder -> ctrl
- PriorRGMDecoderHighLevelWrapper: Policy outputs residual -> add to prior mean + rgm mean -> decoder -> ctrl
- FlatObsWrapper: Converts dict observations to flat arrays (for scratch mode)

Only supports dict observations format from base environment.
"""

from typing import Any, Callable, Mapping

import jax
import jax.numpy as jp
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env

from scamper.agent.task_transfer.maintain_velocity.observation_utils import (
    concat_flat_dict_obs,
    flatten_obs_dict,
)


def _get_proprio(obs: jax.Array | Mapping[str, Any], proprio_size: int) -> jax.Array:
    """Extract proprioception from observation.

    Args:
        obs: Either a dict with 'proprioception' key, or a flat array where
            proprioception is the last proprio_size elements.
        proprio_size: Size of proprioceptive observations.

    Returns:
        Flattened proprioceptive observation array.
    """
    if isinstance(obs, Mapping):
        flat_obs = flatten_obs_dict(obs)
        return flat_obs["proprioception"]
    else:
        # Flat format - proprio is last proprio_size elements
        return obs[..., -proprio_size:]


def _flatten_obs_for_policy(obs: Mapping[str, Any]) -> jax.Array:
    """Flatten observations for the high-level policy network.

    Args:
        obs: Dict with observation keys.

    Returns:
        Flat observation array suitable for the PPO policy network.
    """
    flat_obs = flatten_obs_dict(obs)
    return concat_flat_dict_obs(flat_obs)


class FlatObsWrapper(wrapper.Wrapper):
    """Wrapper that converts dict observations to flat arrays.

    This minimal wrapper enables scratch mode training with Brax's standard
    normalizer by ensuring observations are flat arrays rather than dicts.
    The flattening order is: sorted non-proprio keys, then proprioception.
    """

    def __init__(self, env: mjx_env.MjxEnv):
        """Initialize the flat observation wrapper.

        Args:
            env: Base environment with dict observations.
        """
        super().__init__(env)

    def reset(
        self,
        rng: jax.Array,
        **kwargs: Any,
    ) -> mjx_env.State:
        """Reset the environment and flatten observations.

        Args:
            rng: Random key for environment reset.
            **kwargs: Additional arguments passed to base env reset.

        Returns:
            Initial environment state with flat observation array.
        """
        state = self.env.reset(rng, **kwargs)
        return state.replace(obs=_flatten_obs_for_policy(state.obs))

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Take a step and flatten the resulting observation.

        Args:
            state: Current environment state.
            action: Action to apply.

        Returns:
            Next environment state with flat observation array.
        """
        next_state = super().step(state, action)
        return next_state.replace(obs=_flatten_obs_for_policy(next_state.obs))


class DecoderHighLevelWrapper(wrapper.Wrapper):
    """Mode 1: Policy outputs latent intentions -> frozen decoder -> ctrl.

    This wrapper translates high-level latent intentions into low-level
    control signals using a frozen decoder from a pretrained checkpoint.

    The policy learns to output latent vectors that the decoder converts
    to motor commands.
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        decoder_inference_fn: Callable,
        latent_size: int,
        proprio_size: int,
    ):
        """Initialize the decoder wrapper.

        Args:
            env: Base environment with dict observations.
            decoder_inference_fn: Function (latent_proprio) -> (action, extras)
                where latent_proprio is [latent, proprio] concatenated.
            latent_size: Dimension of the latent intention space.
            proprio_size: Size of proprioceptive observations.
        """
        self._decoder_fn = decoder_inference_fn
        self._latent_size = latent_size
        self._proprio_size = proprio_size
        super().__init__(env)

    def reset(
        self,
        rng: jax.Array,
        **kwargs: Any,
    ) -> mjx_env.State:
        """Reset the environment.

        Args:
            rng: Random key for environment reset.
            **kwargs: Additional arguments passed to base env reset.

        Returns:
            Initial environment state with decoder extras initialized.
        """
        state = self.env.reset(rng, **kwargs)

        proprio = _get_proprio(state.obs, self._proprio_size)
        dummy_latent = jp.zeros(proprio.shape[:-1] + (self._latent_size,))
        latent_proprio = jp.concatenate([dummy_latent, proprio], axis=-1)
        _, extras = self._decoder_fn(latent_proprio)
        state.info["decoder_extras"] = extras

        state = state.replace(obs=_flatten_obs_for_policy(state.obs))

        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Take a step using latent intention as action.

        Args:
            state: Current environment state.
            action: Latent intention vector from the policy.

        Returns:
            Next environment state after applying decoded control.
        """
        proprio = _get_proprio(state.obs, self._proprio_size)

        latent_proprio = jp.concatenate([action, proprio], axis=-1)

        ctrl, extras = self._decoder_fn(latent_proprio)

        state.info["decoder_extras"] = extras

        next_state = super().step(state, ctrl)

        next_state = next_state.replace(obs=_flatten_obs_for_policy(next_state.obs))

        return next_state

    @property
    def action_size(self) -> int:
        """Return the latent size as the action size for the policy."""
        return self._latent_size


class PriorDecoderHighLevelWrapper(wrapper.Wrapper):
    """Mode 2: Policy outputs residual -> add to prior mean -> decoder -> ctrl.

    This wrapper combines a learned residual policy with a frozen prior network.
    The final latent is: residual + prior_mean + optional_noise

    This allows the policy to learn task-specific corrections while leveraging
    the pretrained prior's knowledge of natural movements.
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        prior_inference_fn: Callable,
        decoder_inference_fn: Callable,
        latent_size: int,
        proprio_size: int,
        deterministic_prior: bool = True,
        noise_logvar: float = -2.0,
    ):
        """Initialize the prior+decoder wrapper.

        Args:
            env: Base environment with dict observations.
            prior_inference_fn: Function (proprio) -> (mean, logvar)
            decoder_inference_fn: Function (latent_proprio) -> (action, extras)
            latent_size: Dimension of the latent intention space.
            proprio_size: Size of proprioceptive observations.
            deterministic_prior: If True, use prior mean only (no noise).
            noise_logvar: Fixed log-variance for noise sampling (used when
                deterministic_prior=False).
        """
        self._prior_fn = prior_inference_fn
        self._decoder_fn = decoder_inference_fn
        self._latent_size = latent_size
        self._proprio_size = proprio_size
        self._deterministic = deterministic_prior
        self._noise_logvar = noise_logvar
        super().__init__(env)

    def reset(
        self,
        rng: jax.Array,
        **kwargs: Any,
    ) -> mjx_env.State:
        """Reset the environment.

        Args:
            rng: Random key for environment reset.
            **kwargs: Additional arguments passed to base env reset.

        Returns:
            Initial environment state with prior/decoder extras initialized.
        """
        state = self.env.reset(rng, **kwargs)

        proprio = _get_proprio(state.obs, self._proprio_size)

        prior_mean, prior_logvar = self._prior_fn(proprio)
        state.info["prior_mean"] = prior_mean
        state.info["prior_logvar"] = prior_logvar

        dummy_latent = jp.zeros_like(prior_mean)
        latent_proprio = jp.concatenate([dummy_latent, proprio], axis=-1)
        _, decoder_extras = self._decoder_fn(latent_proprio)
        state.info["decoder_extras"] = decoder_extras

        state.info["final_latent"] = jp.zeros_like(prior_mean)
        state.info["rng"] = rng

        state = state.replace(obs=_flatten_obs_for_policy(state.obs))

        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Take a step using residual + prior mean as latent.

        Args:
            state: Current environment state.
            action: Residual vector from the policy.

        Returns:
            Next environment state after applying decoded control.
        """
        proprio = _get_proprio(state.obs, self._proprio_size)

        prior_mean, prior_logvar = self._prior_fn(proprio)

        if self._deterministic:
            latent = action + prior_mean
        else:
            rng = state.info.get("rng", jax.random.PRNGKey(0))
            rng, noise_rng = jax.random.split(rng)
            std = jp.exp(0.5 * self._noise_logvar)
            noise = jax.random.normal(noise_rng, shape=prior_mean.shape) * std
            latent = action + prior_mean + noise
            state.info["rng"] = rng

        latent_proprio = jp.concatenate([latent, proprio], axis=-1)

        ctrl, decoder_extras = self._decoder_fn(latent_proprio)

        state.info["decoder_extras"] = decoder_extras
        state.info["prior_mean"] = prior_mean
        state.info["prior_logvar"] = prior_logvar
        state.info["final_latent"] = latent

        next_state = super().step(state, ctrl)

        next_state = next_state.replace(obs=_flatten_obs_for_policy(next_state.obs))

        return next_state

    @property
    def action_size(self) -> int:
        """Return the latent size as the action size for the policy."""
        return self._latent_size


class PriorRGMDecoderHighLevelWrapper(wrapper.Wrapper):
    """Mode 3: Learned gate blends policy+prior with RGM -> decoder -> ctrl.

    The policy outputs latent_size + 1: a residual vector and a single scalar
    gate logit. The scalar gate controls the blend between
    (residual + prior_mean) and rgm_z for all latent dimensions.
    The gate is initialised near zero so the RGM dominates at the start.

    latent = gate * (residual + prior_mean) + (1 - gate) * rgm_z

    A prior consistency penalty (Mahalanobis distance between the final
    latent and the prior distribution) encourages the gate to open by
    penalizing deviation from the prior's predictions.

    At init:  gate ~ 0.05 -> latent ~ rgm_z  (naturalistic movement)
    Trained:  gate ~ 1.0  -> latent ~ residual + prior_mean  (task transfer)
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        prior_inference_fn: Callable,
        rgm_inference_fn: Callable,
        decoder_inference_fn: Callable,
        latent_size: int,
        proprio_size: int,
        gate_scale: float = 5.0,
        gate_bias: float = 3.0,
        rgm_noise: bool = False,
        prior_consistency_weight: float = 0.0,
    ):
        """Initialize the gated prior+rgm+decoder wrapper.

        Args:
            env: Base environment with dict observations.
            prior_inference_fn: Function (proprio) -> (mean, logvar)
            rgm_inference_fn: Function (proprio) -> (mean, logvar)
            decoder_inference_fn: Function (latent_proprio) -> (action, extras)
            latent_size: Dimension of the latent intention space.
            proprio_size: Size of proprioceptive observations.
            gate_scale: Multiplier on gate logit before sigmoid. Larger values
                make the gate more binary (on/off).
            gate_bias: Negative bias so the gate starts near 0 at init.
                sigmoid(-gate_bias) is the initial gate value (~0.05 for 3.0).
            rgm_noise: If True, sample RGM with its predicted logvar. If False,
                use rgm_mean only (deterministic). False is recommended since
                the policy+prior side already adds variation.
            prior_consistency_weight: Weight for the prior consistency penalty
                (Mahalanobis distance between final latent and prior). Higher
                values encourage the gate to open by penalizing reliance on
                the RGM. 0.0 disables the penalty.
        """
        self._prior_fn = prior_inference_fn
        self._rgm_fn = rgm_inference_fn
        self._decoder_fn = decoder_inference_fn
        self._latent_size = latent_size
        self._proprio_size = proprio_size
        self._gate_scale = gate_scale
        self._gate_bias = gate_bias
        self._rgm_noise = rgm_noise
        self._prior_consistency_weight = prior_consistency_weight
        super().__init__(env)

    def reset(
        self,
        rng: jax.Array,
        **kwargs: Any,
    ) -> mjx_env.State:
        """Reset the environment.

        Args:
            rng: Random key for environment reset.
            **kwargs: Additional arguments passed to base env reset.

        Returns:
            Initial environment state with prior/rgm/decoder extras initialized.
        """
        state = self.env.reset(rng, **kwargs)

        proprio = _get_proprio(state.obs, self._proprio_size)

        prior_mean, prior_logvar = self._prior_fn(proprio)
        state.info["prior_mean"] = prior_mean
        state.info["prior_logvar"] = prior_logvar

        rgm_mean, rgm_logvar = self._rgm_fn(proprio)
        state.info["rgm_mean"] = rgm_mean
        state.info["rgm_logvar"] = rgm_logvar

        dummy_latent = jp.zeros_like(prior_mean)
        latent_proprio = jp.concatenate([dummy_latent, proprio], axis=-1)
        _, decoder_extras = self._decoder_fn(latent_proprio)
        state.info["decoder_extras"] = decoder_extras

        state.info["final_latent"] = jp.zeros_like(prior_mean)
        state.info["rng"] = rng

        # Gate metrics in state.metrics so Brax's EpisodeWrapper logs them
        state = state.replace(
            metrics={
                **state.metrics,
                "gate": jp.float32(0.0),
                "prior_consistency": jp.float32(0.0),
            },
            obs=_flatten_obs_for_policy(state.obs),
        )

        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Take a step using gated blend of policy+prior and RGM.

        The action has size latent_size + 1: the residual vector followed by
        a single scalar gate logit that controls all dimensions.

        Args:
            state: Current environment state.
            action: Concatenated [residual, gate_logit] from the policy.

        Returns:
            Next environment state after applying decoded control.
        """
        proprio = _get_proprio(state.obs, self._proprio_size)

        # Split policy output into residual and scalar gate
        residual = action[..., : self._latent_size]
        gate_logit = action[..., self._latent_size]

        # Scalar gate: biased toward 0 at init so RGM dominates
        gate = jax.nn.sigmoid(gate_logit * self._gate_scale - self._gate_bias)

        # Frozen network outputs
        prior_mean, prior_logvar = self._prior_fn(proprio)
        rgm_mean, rgm_logvar = self._rgm_fn(proprio)

        # RGM latent: optionally sample with predicted logvar
        rng = state.info.get("rng", jax.random.PRNGKey(0))
        rng, noise_rng = jax.random.split(rng)
        if self._rgm_noise:
            rgm_std = jp.exp(0.5 * rgm_logvar)
            rgm_z = rgm_mean + jax.random.normal(
                noise_rng, shape=rgm_mean.shape
            ) * rgm_std
        else:
            rgm_z = rgm_mean

        # Scalar blend: gate=0 -> RGM, gate=1 -> policy+prior
        latent = gate * (residual + prior_mean) + (1.0 - gate) * rgm_z

        # Prior consistency penalty: Mahalanobis distance between final
        # latent and the prior distribution. Penalizes the latent for
        # deviating from the prior's predictions, encouraging gate opening.
        prior_var = jp.exp(prior_logvar)
        prior_consistency = jp.sum(
            (latent - prior_mean) ** 2 / prior_var, axis=-1
        )

        state.info["rng"] = rng

        latent_proprio = jp.concatenate([latent, proprio], axis=-1)

        ctrl, decoder_extras = self._decoder_fn(latent_proprio)

        state.info["decoder_extras"] = decoder_extras
        state.info["prior_mean"] = prior_mean
        state.info["prior_logvar"] = prior_logvar
        state.info["rgm_mean"] = rgm_mean
        state.info["rgm_logvar"] = rgm_logvar
        state.info["final_latent"] = latent

        # Gate metrics in state.metrics so EpisodeWrapper aggregates them
        state = state.replace(
            metrics={
                **state.metrics,
                "gate": gate,
                "prior_consistency": prior_consistency,
            }
        )

        next_state = super().step(state, ctrl)

        # Apply prior consistency penalty to reward
        if self._prior_consistency_weight > 0.0:
            next_state = next_state.replace(
                reward=next_state.reward
                - self._prior_consistency_weight * prior_consistency,
            )

        next_state = next_state.replace(obs=_flatten_obs_for_policy(next_state.obs))

        return next_state

    @property
    def action_size(self) -> int:
        """Return latent_size + 1 (residual + scalar gate) as the action size."""
        return self._latent_size + 1
