"""Environment wrappers for mouse_reach task transfer training.

Two wrapper classes for different transfer learning modes:
- DecoderHighLevelWrapper: Policy outputs latent -> frozen decoder -> ctrl
- PriorDecoderHighLevelWrapper: Policy outputs residual -> add to prior mean -> decoder -> ctrl

The mouse reach env has proprioception = [qpos, qvel, wrist_pos] (11 dim)
but the frozen decoder/prior expect only [qpos, qvel] (8 dim). The wrappers
handle this by extracting the decoder-compatible slice for the frozen networks
while passing the full observation to the policy.
"""

from typing import Any, Callable, Mapping

import jax
import jax.numpy as jp
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env

from scamper.agent.task_transfer.mouse_reach.observation_utils import (
    concat_flat_dict_obs,
    flatten_obs_dict,
)


def _get_decoder_proprio(
    obs: jax.Array | Mapping[str, Any], decoder_proprio_size: int
) -> jax.Array:
    """Extract decoder-compatible proprioception from observation.

    The mouse reach env returns proprioception as [qpos, qvel, wrist_pos]
    but the frozen decoder/prior only expect [qpos, qvel]. This function
    extracts the first decoder_proprio_size elements.

    Args:
        obs: Either a dict with 'proprioception' key, or a flat array.
        decoder_proprio_size: Size of decoder-compatible proprioception
            (qpos + qvel from the prior training).

    Returns:
        Decoder-compatible proprioceptive observation array.
    """
    if isinstance(obs, Mapping):
        flat_obs = flatten_obs_dict(obs)
        proprio = flat_obs["proprioception"]
    else:
        # Flat format - full proprio is at end, but we need to slice carefully
        proprio = obs[..., -decoder_proprio_size:]
        return proprio

    # Take only the decoder-compatible slice (qpos + qvel, no wrist_pos)
    return proprio[..., :decoder_proprio_size]


def _flatten_obs_for_policy(obs: Mapping[str, Any]) -> jax.Array:
    """Flatten observations for the high-level policy network.

    The policy sees the FULL observation including wrist_pos in proprioception.

    Args:
        obs: Dict with observation keys.

    Returns:
        Flat observation array suitable for the PPO policy network.
    """
    flat_obs = flatten_obs_dict(obs)
    return concat_flat_dict_obs(flat_obs)


class DecoderHighLevelWrapper(wrapper.Wrapper):
    """Mode 1: Policy outputs latent intentions -> frozen decoder -> ctrl.

    The policy learns to output latent vectors that the decoder converts
    to motor commands. The decoder receives only the decoder-compatible
    proprioception (qpos + qvel), not wrist_pos.
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        decoder_inference_fn: Callable,
        latent_size: int,
        decoder_proprio_size: int,
        full_proprio_size: int,
    ):
        """Initialize the decoder wrapper.

        Args:
            env: Base environment with dict observations.
            decoder_inference_fn: Function (latent_proprio) -> (action, extras)
                where latent_proprio is [latent, proprio] concatenated.
            latent_size: Dimension of the latent intention space.
            decoder_proprio_size: Size of decoder-compatible proprioception
                (from prior checkpoint, e.g. 8 for qpos+qvel).
            full_proprio_size: Size of full proprioception in the flat obs
                (e.g. 11 for qpos+qvel+wrist_pos). Used to extract proprio
                from the flat observation in step().
        """
        self._decoder_fn = decoder_inference_fn
        self._latent_size = latent_size
        self._decoder_proprio_size = decoder_proprio_size
        self._full_proprio_size = full_proprio_size
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

        proprio = _get_decoder_proprio(state.obs, self._decoder_proprio_size)
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
        # state.obs is flat from _flatten_obs_for_policy in previous step/reset
        # Flat format: [task_obs | full_proprio]. Extract decoder-compatible
        # proprio (qpos + qvel) from the full proprioception at the end.
        full_proprio = state.obs[..., -self._full_proprio_size:]
        decoder_proprio = full_proprio[..., : self._decoder_proprio_size]

        latent_proprio = jp.concatenate([action, decoder_proprio], axis=-1)

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

    The final latent is: residual + prior_mean + optional_noise.
    Both the prior and decoder receive only decoder-compatible proprioception
    (qpos + qvel), not wrist_pos.
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        prior_inference_fn: Callable,
        decoder_inference_fn: Callable,
        latent_size: int,
        decoder_proprio_size: int,
        full_proprio_size: int,
        deterministic_prior: bool = True,
        noise_logvar: float = -2.0,
    ):
        """Initialize the prior+decoder wrapper.

        Args:
            env: Base environment with dict observations.
            prior_inference_fn: Function (proprio) -> (mean, logvar)
            decoder_inference_fn: Function (latent_proprio) -> (action, extras)
            latent_size: Dimension of the latent intention space.
            decoder_proprio_size: Size of decoder-compatible proprioception.
            full_proprio_size: Size of full proprioception in the flat obs
                (e.g. 11 for qpos+qvel+wrist_pos). Used to extract proprio
                from the flat observation in step().
            deterministic_prior: If True, use prior mean only (no noise).
            noise_logvar: Fixed log-variance for noise sampling.
        """
        self._prior_fn = prior_inference_fn
        self._decoder_fn = decoder_inference_fn
        self._latent_size = latent_size
        self._decoder_proprio_size = decoder_proprio_size
        self._full_proprio_size = full_proprio_size
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

        proprio = _get_decoder_proprio(state.obs, self._decoder_proprio_size)

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
        # Extract decoder-compatible proprio from flat obs
        full_proprio = state.obs[..., -self._full_proprio_size:]
        decoder_proprio = full_proprio[..., : self._decoder_proprio_size]

        prior_mean, prior_logvar = self._prior_fn(decoder_proprio)

        if self._deterministic:
            latent = action + prior_mean
        else:
            rng = state.info.get("rng", jax.random.PRNGKey(0))
            rng, noise_rng = jax.random.split(rng)
            std = jp.exp(0.5 * self._noise_logvar)
            noise = jax.random.normal(noise_rng, shape=prior_mean.shape) * std
            latent = action + prior_mean + noise
            state.info["rng"] = rng

        latent_proprio = jp.concatenate([latent, decoder_proprio], axis=-1)

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
