"""Environment wrappers for arena forage task transfer training.

Four wrapper classes for different transfer learning modes:
- DecoderHighLevelWrapper: Policy outputs latent -> frozen decoder -> ctrl
- PriorDecoderHighLevelWrapper: Policy outputs residual -> add to prior mean -> decoder -> ctrl
- PriorRGMDecoderHighLevelWrapper: Policy outputs residual -> add to prior mean + rgm mean -> decoder -> ctrl
- FlatObsWrapper: Converts dict observations to flat arrays (for scratch mode)

Supports both legacy flat observations and new dict observations format.
"""

from typing import Any, Callable, Mapping

import jax
import jax.numpy as jp
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env

from scamper.agent.task_transfer.arena_forage.observation_utils import (
    concat_flat_dict_obs,
    flatten_obs_dict,
)


def _get_proprio(obs: jax.Array | Mapping[str, Any], proprio_size: int) -> jax.Array:
    """Extract proprioception from observation, handling both dict and flat formats.

    Args:
        obs: Either a flat array (legacy) or dict with 'proprioception' key (new).
        proprio_size: Size of proprioceptive observations (used for flat format).

    Returns:
        Flattened proprioceptive observation array.
    """
    if isinstance(obs, Mapping):
        # New dict format - use flatten_obs_dict to handle nested structures
        flat_obs = flatten_obs_dict(obs)
        return flat_obs["proprioception"]
    else:
        # Legacy flat format - proprio is last proprio_size elements
        return obs[..., -proprio_size:]


def _flatten_obs_for_policy(obs: jax.Array | Mapping[str, Any]) -> jax.Array:
    """Flatten observations for the high-level policy network.

    Args:
        obs: Either a flat array (legacy) or dict with observation keys (new).

    Returns:
        Flat observation array suitable for the PPO policy network.
    """
    if isinstance(obs, Mapping):
        # New dict format - flatten and concatenate
        flat_obs = flatten_obs_dict(obs)
        return concat_flat_dict_obs(flat_obs)
    else:
        # Legacy flat format - already flat
        return obs


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

    Supports both legacy flat observations and new dict observations format.
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
            env: Base environment (with dict or flat observations).
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

        # Initialize decoder_extras with correct pytree structure by running
        # decoder once with dummy latent. This ensures lax.scan carry structure
        # matches between reset and step.
        proprio = _get_proprio(state.obs, self._proprio_size)
        dummy_latent = jp.zeros(proprio.shape[:-1] + (self._latent_size,))
        latent_proprio = jp.concatenate([dummy_latent, proprio], axis=-1)
        _, extras = self._decoder_fn(latent_proprio)
        state.info["decoder_extras"] = extras

        # Flatten observations for the high-level policy network
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
        # Extract proprioception from observation (handles both dict and flat)
        proprio = _get_proprio(state.obs, self._proprio_size)

        # Concatenate latent and proprio for decoder
        latent_proprio = jp.concatenate([action, proprio], axis=-1)

        # Apply frozen decoder to get control signal
        ctrl, extras = self._decoder_fn(latent_proprio)

        # Store decoder extras in state info
        state.info["decoder_extras"] = extras

        # Step the base environment with decoded control
        next_state = super().step(state, ctrl)

        # Flatten observations for the high-level policy network
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

    Supports both legacy flat observations and new dict observations format.
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
            env: Base environment (with dict or flat observations).
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

        # Extract proprioception (handles both dict and flat formats)
        proprio = _get_proprio(state.obs, self._proprio_size)

        # Initialize prior_mean/prior_logvar with correct shape by running prior
        prior_mean, prior_logvar = self._prior_fn(proprio)
        state.info["prior_mean"] = prior_mean
        state.info["prior_logvar"] = prior_logvar

        # Initialize decoder_extras with correct pytree structure
        dummy_latent = jp.zeros_like(prior_mean)
        latent_proprio = jp.concatenate([dummy_latent, proprio], axis=-1)
        _, decoder_extras = self._decoder_fn(latent_proprio)
        state.info["decoder_extras"] = decoder_extras

        # Initialize final_latent with correct shape
        state.info["final_latent"] = jp.zeros_like(prior_mean)
        state.info["rng"] = rng

        # Flatten observations for the high-level policy network
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
        # Extract proprioception from observation (handles both dict and flat)
        proprio = _get_proprio(state.obs, self._proprio_size)

        # Get prior distribution from proprioception
        prior_mean, prior_logvar = self._prior_fn(proprio)

        # Compute final latent
        if self._deterministic:
            latent = action + prior_mean
        else:
            # Sample noise with fixed logvar
            rng = state.info.get("rng", jax.random.PRNGKey(0))
            rng, noise_rng = jax.random.split(rng)
            std = jp.exp(0.5 * self._noise_logvar)
            noise = jax.random.normal(noise_rng, shape=prior_mean.shape) * std
            latent = action + prior_mean + noise
            state.info["rng"] = rng

        # Concatenate latent and proprio for decoder
        latent_proprio = jp.concatenate([latent, proprio], axis=-1)

        # Apply frozen decoder
        ctrl, decoder_extras = self._decoder_fn(latent_proprio)

        # Store extras in state info
        state.info["decoder_extras"] = decoder_extras
        state.info["prior_mean"] = prior_mean
        state.info["prior_logvar"] = prior_logvar
        state.info["final_latent"] = latent

        # Step the base environment
        next_state = super().step(state, ctrl)

        # Flatten observations for the high-level policy network
        next_state = next_state.replace(obs=_flatten_obs_for_policy(next_state.obs))

        return next_state

    @property
    def action_size(self) -> int:
        """Return the latent size as the action size for the policy."""
        return self._latent_size


class PriorRGMDecoderHighLevelWrapper(wrapper.Wrapper):
    """Mode 3: Stochastic binary gate selects policy+prior or RGM.

    The policy outputs latent_size + 1: a residual vector and a single scalar
    gate logit. The gate is binarized via a hard threshold — PPO's Gaussian
    sampling provides stochastic exploration of the binary choice, and PPO's
    policy gradient learns which choice is better.

    gate = 1 if gate_logit > gate_bias else 0

    When gate=1, the residual is applied via a Latent Action Barrier (LAB):
    latent = prior_mean + lab_lambda * prior_std * tanh(residual)

    This bounds the residual within lab_lambda standard deviations of the
    prior, scaled per-dimension by the prior's own uncertainty. At random
    init the residual is near zero (tanh(~0) ≈ 0), so latent ≈ prior_mean.

    When gate=0, the RGM mean (or sample) is used directly:
    latent = rgm_z

    Each step uses either pure RGM or pure policy+prior, never a blend.
    The policy learns state-dependent gating: e.g., RGM for exploration,
    prior+residual when task reward is available.

    At init:  gate = 1 (prior+residual via LAB, near-zero residual)
    Trained:  gate -> 0 in states where RGM is better
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        prior_inference_fn: Callable,
        rgm_inference_fn: Callable,
        decoder_inference_fn: Callable,
        latent_size: int,
        proprio_size: int,
        gate_bias: float = 0.0,
        gate_margin: float = 0.0,
        rgm_noise: bool = False,
        lab_lambda: float = 1.0,
    ):
        """Initialize the binary-gated prior+rgm+decoder wrapper.

        Args:
            env: Base environment with dict observations.
            prior_inference_fn: Function (proprio) -> (mean, logvar)
            rgm_inference_fn: Function (proprio) -> (mean, logvar)
            decoder_inference_fn: Function (latent_proprio) -> (action, extras)
            latent_size: Dimension of the latent intention space.
            proprio_size: Size of proprioceptive observations.
            gate_bias: Threshold for the binary gate. Higher values bias
                toward RGM initially. gate = 1 if gate_logit > gate_bias.
            gate_margin: Hysteresis margin for bistable gate switching.
                Switching on requires gate_logit > gate_bias + gate_margin,
                switching off requires gate_logit < gate_bias - gate_margin.
                0.0 disables hysteresis.
            rgm_noise: If True, sample RGM with its predicted logvar. If False,
                use rgm_mean only (deterministic). False is recommended since
                the policy+prior side already adds variation.
            lab_lambda: Latent Action Barrier scale factor. Controls how many
                prior standard deviations the residual can deviate. Higher
                values allow larger corrections. At random init the residual
                is near zero regardless of this value.
        """
        self._prior_fn = prior_inference_fn
        self._rgm_fn = rgm_inference_fn
        self._decoder_fn = decoder_inference_fn
        self._latent_size = latent_size
        self._proprio_size = proprio_size
        self._gate_bias = gate_bias
        self._gate_margin = gate_margin
        self._rgm_noise = rgm_noise
        self._lab_lambda = lab_lambda
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
        state.info["prev_gate"] = jp.float32(1.0)

        state = state.replace(
            metrics={
                **state.metrics,
                "gate": jp.float32(0.0),
                "gate_logit": jp.float32(0.0),
            },
            obs=_flatten_obs_for_policy(state.obs),
        )

        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Take a step using stochastic binary gate to select motor program.

        The action has size latent_size + 1: the residual vector followed by
        a single scalar gate logit. The gate is binarized via hard threshold
        — PPO's Gaussian sampling provides exploration of the binary choice.

        Args:
            state: Current environment state.
            action: Concatenated [residual, gate_logit] from the policy.

        Returns:
            Next environment state after applying decoded control.
        """
        proprio = _get_proprio(state.obs, self._proprio_size)

        residual = action[..., : self._latent_size]
        gate_logit = action[..., self._latent_size]

        prev_gate = state.info["prev_gate"]
        upper = self._gate_bias + self._gate_margin
        lower = self._gate_bias - self._gate_margin
        gate = jp.where(
            prev_gate > 0.5,
            jp.where(gate_logit < lower, 0.0, 1.0),
            jp.where(gate_logit > upper, 1.0, 0.0),
        )

        prior_mean, prior_logvar = self._prior_fn(proprio)
        rgm_mean, rgm_logvar = self._rgm_fn(proprio)

        rng = state.info.get("rng", jax.random.PRNGKey(0))
        rng, noise_rng = jax.random.split(rng)
        if self._rgm_noise:
            rgm_std = jp.exp(0.5 * rgm_logvar)
            rgm_z = rgm_mean + jax.random.normal(
                noise_rng, shape=rgm_mean.shape
            ) * rgm_std
        else:
            rgm_z = rgm_mean

        prior_std = jp.exp(0.5 * prior_logvar)
        lab_residual = prior_mean + self._lab_lambda * prior_std * jp.tanh(residual)

        latent = gate * lab_residual + (1.0 - gate) * rgm_z

        state.info["rng"] = rng
        state.info["prev_gate"] = gate

        latent_proprio = jp.concatenate([latent, proprio], axis=-1)

        ctrl, decoder_extras = self._decoder_fn(latent_proprio)

        state.info["decoder_extras"] = decoder_extras
        state.info["prior_mean"] = prior_mean
        state.info["prior_logvar"] = prior_logvar
        state.info["rgm_mean"] = rgm_mean
        state.info["rgm_logvar"] = rgm_logvar
        state.info["final_latent"] = latent

        state = state.replace(
            metrics={
                **state.metrics,
                "gate": gate,
                "gate_logit": gate_logit,
            }
        )

        next_state = super().step(state, ctrl)

        next_state = next_state.replace(obs=_flatten_obs_for_policy(next_state.obs))

        return next_state

    @property
    def action_size(self) -> int:
        """Return latent_size + 1 (residual + scalar gate) as the action size."""
        return self._latent_size + 1
