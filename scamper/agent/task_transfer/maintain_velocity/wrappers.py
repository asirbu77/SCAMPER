"""Environment wrappers for maintain_velocity task transfer training.

Five wrapper classes for different transfer learning modes:
- DecoderHighLevelWrapper: Policy outputs latent -> frozen decoder -> ctrl
- PriorDecoderHighLevelWrapper: Policy outputs residual -> add to prior mean -> decoder -> ctrl
- RGMDecoderHighLevelWrapper: Policy outputs residual -> add to RGM mean -> decoder -> ctrl
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


class RGMDecoderHighLevelWrapper(wrapper.Wrapper):
    """Mode: Policy outputs residual -> add to RGM mean -> decoder -> ctrl.

    Like PriorDecoderHighLevelWrapper but uses the frozen RGM instead of
    the prior. The final latent is: rgm_mean + residual + optional_noise.

    No gate, no prior involvement.
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        rgm_inference_fn: Callable,
        decoder_inference_fn: Callable,
        latent_size: int,
        proprio_size: int,
        rgm_noise: bool = False,
        rgm_logvar: float | None = None,
    ):
        """Initialize the rgm+decoder wrapper.

        Args:
            env: Base environment with dict observations.
            rgm_inference_fn: Function (proprio) -> (mean, logvar)
            decoder_inference_fn: Function (latent_proprio) -> (action, extras)
            latent_size: Dimension of the latent intention space.
            proprio_size: Size of proprioceptive observations.
            rgm_noise: If True, sample RGM latent with noise. If False,
                use rgm_mean only (deterministic).
            rgm_logvar: Fixed log-variance for RGM sampling. Only used when
                rgm_noise=True. If None, uses the RGM's predicted logvar.
                If a float, overrides the predicted logvar with this value.
        """
        self._rgm_fn = rgm_inference_fn
        self._decoder_fn = decoder_inference_fn
        self._latent_size = latent_size
        self._proprio_size = proprio_size
        self._rgm_noise = rgm_noise
        self._rgm_logvar = rgm_logvar
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
            Initial environment state with rgm/decoder extras initialized.
        """
        state = self.env.reset(rng, **kwargs)

        proprio = _get_proprio(state.obs, self._proprio_size)

        rgm_mean, rgm_logvar = self._rgm_fn(proprio)
        state.info["rgm_mean"] = rgm_mean
        state.info["rgm_logvar"] = rgm_logvar

        dummy_latent = jp.zeros_like(rgm_mean)
        latent_proprio = jp.concatenate([dummy_latent, proprio], axis=-1)
        _, decoder_extras = self._decoder_fn(latent_proprio)
        state.info["decoder_extras"] = decoder_extras

        state.info["final_latent"] = jp.zeros_like(rgm_mean)
        state.info["rng"] = rng

        state = state.replace(obs=_flatten_obs_for_policy(state.obs))

        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Take a step using residual + RGM mean as latent.

        Args:
            state: Current environment state.
            action: Residual vector from the policy.

        Returns:
            Next environment state after applying decoded control.
        """
        proprio = _get_proprio(state.obs, self._proprio_size)

        rgm_mean, rgm_logvar = self._rgm_fn(proprio)

        rng = state.info.get("rng", jax.random.PRNGKey(0))
        rng, noise_rng = jax.random.split(rng)
        if self._rgm_noise:
            if self._rgm_logvar is not None:
                rgm_std = jp.exp(0.5 * self._rgm_logvar)
            else:
                rgm_std = jp.exp(0.5 * rgm_logvar)
            noise = jax.random.normal(noise_rng, shape=rgm_mean.shape) * rgm_std
            latent = action + rgm_mean + noise
        else:
            latent = action + rgm_mean

        state.info["rng"] = rng

        latent_proprio = jp.concatenate([latent, proprio], axis=-1)

        ctrl, decoder_extras = self._decoder_fn(latent_proprio)

        state.info["decoder_extras"] = decoder_extras
        state.info["rgm_mean"] = rgm_mean
        state.info["rgm_logvar"] = rgm_logvar
        state.info["final_latent"] = latent

        next_state = super().step(state, ctrl)

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

    When gate=1, the residual is added to the prior mean:
    latent = prior_mean + residual

    When gate=0, the RGM mean (or sample) is used directly:
    latent = rgm_z

    Each step uses either pure RGM or pure policy+prior, never a blend.
    The policy learns state-dependent gating: e.g., RGM for exploration,
    prior+residual when task reward is available.

    At init:  gate = 1 (prior+residual, near-zero residual)
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
        rgm_logvar: float | None = None,
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
            rgm_noise: If True, sample RGM latent with noise. If False,
                use rgm_mean only (deterministic).
            rgm_logvar: Fixed log-variance for RGM sampling. Only used when
                rgm_noise=True. If None, uses the RGM's predicted logvar.
                If a float, overrides the predicted logvar with this value.
        """
        self._prior_fn = prior_inference_fn
        self._rgm_fn = rgm_inference_fn
        self._decoder_fn = decoder_inference_fn
        self._latent_size = latent_size
        self._proprio_size = proprio_size
        self._gate_bias = gate_bias
        self._gate_margin = gate_margin
        self._rgm_noise = rgm_noise
        self._rgm_logvar = rgm_logvar
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

        # Gate metrics in state.metrics so Brax's EpisodeWrapper logs them
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

        # Split policy output into residual and scalar gate
        residual = action[..., : self._latent_size]
        gate_logit = action[..., self._latent_size]

        # Binary gate with hysteresis: once committed to a state, the gate
        # stays until the logit crosses the far threshold (bistable dynamics)
        prev_gate = state.info["prev_gate"]
        upper = self._gate_bias + self._gate_margin
        lower = self._gate_bias - self._gate_margin
        gate = jp.where(
            prev_gate > 0.5,
            # Currently 1 (prior+residual): stay unless logit drops below lower
            jp.where(gate_logit < lower, 0.0, 1.0),
            # Currently 0 (RGM): stay unless logit rises above upper
            jp.where(gate_logit > upper, 1.0, 0.0),
        )

        # Frozen network outputs
        prior_mean, prior_logvar = self._prior_fn(proprio)
        rgm_mean, rgm_logvar = self._rgm_fn(proprio)

        # RGM latent: optionally sample with predicted or fixed logvar
        rng = state.info.get("rng", jax.random.PRNGKey(0))
        rng, noise_rng = jax.random.split(rng)
        if self._rgm_noise:
            if self._rgm_logvar is not None:
                rgm_std = jp.exp(0.5 * self._rgm_logvar)
            else:
                rgm_std = jp.exp(0.5 * rgm_logvar)
            rgm_z = rgm_mean + jax.random.normal(
                noise_rng, shape=rgm_mean.shape
            ) * rgm_std
        else:
            rgm_z = rgm_mean

        # Prior + residual
        prior_residual = prior_mean + residual

        # Binary selection: gate=0 -> pure RGM, gate=1 -> prior + residual
        latent = gate * prior_residual + (1.0 - gate) * rgm_z

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

        # Gate metrics in state.metrics so EpisodeWrapper aggregates them
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
