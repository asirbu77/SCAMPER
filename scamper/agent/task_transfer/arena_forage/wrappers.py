"""Environment wrappers for arena forage task transfer training.

Wrapper classes for different transfer learning modes:
- DecoderHighLevelWrapper: Policy outputs latent -> frozen decoder -> ctrl (mimic_mjx mode)
- PriorDecoderHighLevelWrapper: Policy outputs residual -> add to motion prior mean -> decoder -> ctrl (res_control mode)
- RGMDecoderHighLevelWrapper: Policy outputs residual -> add to behavior prior mean -> decoder -> ctrl (bp_res_control mode)
- PriorRGMDecoderHighLevelWrapper: Policy outputs residual + gate -> select motion prior or behavior prior -> decoder -> ctrl (prior_gating mode)
- PriorRGMInterpDecoderHighLevelWrapper: Fixed linear blend of motion prior and behavior prior -> add policy residual on top -> decoder -> ctrl (prior_interp mode)
- PriorScheduleDecoderHighLevelWrapper: Policy outputs residual -> scheduled stochastic gate selects behavior prior or motion prior -> decoder -> ctrl (prior_schedule mode)
- FlatObsWrapper: Converts dict observations to flat arrays (scratch mode)

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

# Key under which `BraxAutoResetWrapper` (with full_reset=True) passes info
# through an episode boundary instead of clobbering it with the reset value.
# This is the documented hook for curriculum state -- see
# `mujoco_playground._src.wrapper.BraxAutoResetWrapper`. Anything that must
# survive autoreset (e.g. training-progress counters) has to live in here.
_PRESERVE_INFO_KEY = "AutoResetWrapper_preserve_info"
_PROGRESS_KEY = "sched_progress"
_PREV_GATE_KEY = "prev_gate"


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
        state.info["prev_gate"] = jp.float32(0.0)

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
            if self._rgm_logvar is not None:
                rgm_std = jp.exp(0.5 * self._rgm_logvar)
            else:
                rgm_std = jp.exp(0.5 * rgm_logvar)
            rgm_z = (
                rgm_mean + jax.random.normal(noise_rng, shape=rgm_mean.shape) * rgm_std
            )
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


class PriorScheduleDecoderHighLevelWrapper(wrapper.Wrapper):
    """Mode: Scheduled stochastic gate between behavior prior and motion prior.

    Same latent construction as PriorRGMDecoderHighLevelWrapper -- each step is
    either pure behavior prior or pure motion prior + residual, never a blend:

    latent = gate * (prior_mean + residual) + (1 - gate) * rgm_z

    but the gate is no longer an output of the policy. It is drawn from a fixed
    exploration schedule that anneals from "sample the behavior prior" to
    "residually control the motion prior" over the course of training. The
    policy therefore outputs only the residual (action size = latent_size).

    Schedule
    --------
    `progress` runs 0 -> 1 over the full training run. The target probability of
    running in motion-prior+residual mode is

        frac = clip((progress - warmup_frac) / ramp_frac, 0, 1)
        p    = p_start + (p_end - p_start) * shape(frac)

    where `shape` is identity (linear) or a raised cosine.

    Rather than draw `gate ~ Bernoulli(p)` independently each step -- which
    would chatter at control rate and produce motor noise rather than
    behaviour -- the gate is a two-state Markov chain:

        P(0 -> 1) = p / bout_tau      P(1 -> 0) = (1 - p) / bout_tau

    whose stationary probability of gate=1 is exactly `p`, with mean bout
    lengths of bout_tau / (1 - p) steps in mode 1 and bout_tau / p steps in
    mode 0. This is the scheduled analogue of `prior_gating`'s `gate_margin`
    hysteresis. At p=0 the chain is pinned to the behavior prior; at p=1 it is
    absorbed into motion-prior+residual control.

    Tracking training progress
    --------------------------
    The env is stepped inside the jitted PPO loop, so progress cannot be pushed
    in from the host. Instead it is carried in `state.info` under
    `_PRESERVE_INFO_KEY`, which `BraxAutoResetWrapper(full_reset=True)` passes
    through episode boundaries specifically for curriculum state. Progress
    advances by `progress_per_step` on every step; the caller sets that to
    `num_envs / num_timesteps` for the training env (each of the `num_envs`
    parallel copies advances by one env-step's worth of global progress) and to
    0.0 for eval envs, which are pinned at `initial_progress`.

    Note on gradients: on gate=0 steps the residual has no effect on the
    environment, but PPO still includes its log-prob in the ratio. The advantage
    is independent of the residual sample there, so the expected gradient is
    zero (the score function has zero mean) -- it is variance, not bias. A
    nonzero `p_start` gives the residual head real signal throughout and keeps
    the entropy bonus from inflating its scale unopposed during a long warmup.
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        prior_inference_fn: Callable,
        rgm_inference_fn: Callable,
        decoder_inference_fn: Callable,
        latent_size: int,
        proprio_size: int,
        progress_per_step: float = 0.0,
        initial_progress: float = 0.0,
        p_start: float = 0.0,
        p_end: float = 1.0,
        warmup_frac: float = 0.1,
        ramp_frac: float = 0.6,
        ramp_shape: str = "cosine",
        bout_tau: float = 25.0,
        rgm_noise: bool = True,
        rgm_logvar: float | None = None,
    ):
        """Initialize the scheduled-gate prior+rgm+decoder wrapper.

        Args:
            env: Base environment with dict observations.
            prior_inference_fn: Function (proprio) -> (mean, logvar)
            rgm_inference_fn: Function (proprio) -> (mean, logvar)
            decoder_inference_fn: Function (latent_proprio) -> (action, extras)
            latent_size: Dimension of the latent intention space.
            proprio_size: Size of proprioceptive observations.
            progress_per_step: Amount schedule progress advances per env step.
                Use `num_envs / num_timesteps` for the training env and 0.0 to
                freeze the schedule (eval / rendering).
            initial_progress: Schedule progress at reset, in [0, 1]. Set to
                `completed_timesteps / num_timesteps` when resuming from a
                preemption checkpoint, otherwise the curriculum restarts.
            p_start: Gate probability at the start of the ramp.
            p_end: Gate probability at the end of the ramp.
            warmup_frac: Fraction of training held at `p_start` before ramping.
            ramp_frac: Fraction of training spent ramping `p_start` -> `p_end`.
                Progress beyond `warmup_frac + ramp_frac` holds at `p_end`.
            ramp_shape: 'linear' or 'cosine' interpolation across the ramp.
            bout_tau: Switching timescale in env steps; larger values give
                longer, less chattery bouts in each mode.
            rgm_noise: If True, sample the behavior prior latent with noise.
                If False, use rgm_mean only (deterministic).
            rgm_logvar: Fixed log-variance for RGM sampling. Only used when
                rgm_noise=True. If None, uses the RGM's predicted logvar.

        Raises:
            ValueError: If the schedule parameters are out of range.
        """
        if ramp_frac <= 0.0:
            raise ValueError(f"prior_schedule ramp_frac must be > 0, got {ramp_frac}")
        if ramp_shape not in ("linear", "cosine"):
            raise ValueError(
                f"prior_schedule ramp_shape must be 'linear' or 'cosine', "
                f"got {ramp_shape!r}"
            )
        if bout_tau < 1.0:
            raise ValueError(
                f"prior_schedule bout_tau must be >= 1 (it is a switching "
                f"timescale in env steps), got {bout_tau}"
            )
        if not 0.0 <= p_start <= 1.0 or not 0.0 <= p_end <= 1.0:
            raise ValueError(
                f"prior_schedule p_start/p_end must be in [0, 1], got "
                f"{p_start}/{p_end}"
            )
        self._prior_fn = prior_inference_fn
        self._rgm_fn = rgm_inference_fn
        self._decoder_fn = decoder_inference_fn
        self._latent_size = latent_size
        self._proprio_size = proprio_size
        self._progress_per_step = progress_per_step
        self._initial_progress = initial_progress
        self._p_start = p_start
        self._p_end = p_end
        self._warmup_frac = warmup_frac
        self._ramp_frac = ramp_frac
        self._ramp_shape = ramp_shape
        self._bout_tau = bout_tau
        self._rgm_noise = rgm_noise
        self._rgm_logvar = rgm_logvar
        super().__init__(env)

    def gate_prob(self, progress: jax.Array | float) -> jax.Array:
        """Target probability of motion-prior+residual mode at a given progress.

        Args:
            progress: Training progress in [0, 1]. Accepts a traced array or a
                plain Python float.

        Returns:
            Scalar gate probability in [p_start, p_end].
        """
        frac = jp.clip((progress - self._warmup_frac) / self._ramp_frac, 0.0, 1.0)
        if self._ramp_shape == "cosine":
            frac = 0.5 - 0.5 * jp.cos(jp.pi * frac)
        return jp.float32(self._p_start + (self._p_end - self._p_start) * frac)

    def _step_gate(
        self, prev_gate: jax.Array, gate_prob: jax.Array, rng: jax.Array
    ) -> jax.Array:
        """Advance the two-state gate chain by one step.

        Args:
            prev_gate: Previous gate value (0.0 or 1.0).
            gate_prob: Stationary probability of gate=1 at the current progress.
            rng: Random key for the switch draw.

        Returns:
            New gate value (0.0 or 1.0).
        """
        p_off_on = jp.clip(gate_prob / self._bout_tau, 0.0, 1.0)
        p_on_off = jp.clip((1.0 - gate_prob) / self._bout_tau, 0.0, 1.0)
        switch_prob = jp.where(prev_gate > 0.5, p_on_off, p_off_on)
        switched = jax.random.uniform(rng, shape=jp.shape(prev_gate)) < switch_prob
        return jp.where(switched, 1.0 - prev_gate, prev_gate)

    def override_schedule_progress(
        self, state: mjx_env.State, progress: float, rng: jax.Array
    ) -> mjx_env.State:
        """Pin a reset state to a given schedule progress.

        The eval env is a separate instance that gets reset on every
        evaluation, so its own counter would always read "start of training".
        This lets the caller stamp the real training progress onto a freshly
        reset state so logging rollouts mirror what training is actually doing.
        The returned state has an identical pytree structure, so a previously
        traced `jit(env.step)` still hits its compilation cache.

        Args:
            state: State returned by `reset`.
            progress: Training progress in [0, 1] to pin the schedule to.
            rng: Random key used to draw the initial gate from its stationary
                distribution at `progress`.

        Returns:
            State with schedule progress and initial gate replaced.
        """
        progress_arr = jp.float32(progress)
        gate_prob = self.gate_prob(progress_arr)
        gate = (jax.random.uniform(rng) < gate_prob).astype(jp.float32)
        info = dict(state.info)
        info[_PRESERVE_INFO_KEY] = {
            _PROGRESS_KEY: progress_arr,
            _PREV_GATE_KEY: gate,
        }
        return state.replace(
            info=info,
            metrics={
                **state.metrics,
                "gate": gate,
                "gate_prob": gate_prob,
                "gate_progress": progress_arr,
            },
        )

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
            Initial environment state with prior/rgm/decoder extras and the
            schedule state initialized.
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

        rng, gate_rng = jax.random.split(rng)
        state.info["rng"] = rng

        # Draw the starting gate from its stationary distribution so an env
        # pinned to a late progress (eval / rendering) starts in the right
        # regime instead of taking ~bout_tau steps to switch into it.
        progress = jp.float32(self._initial_progress)
        gate_prob = self.gate_prob(progress)
        gate = (jax.random.uniform(gate_rng) < gate_prob).astype(jp.float32)

        # Schedule state lives under the preserve key so it survives autoreset.
        state.info[_PRESERVE_INFO_KEY] = {
            _PROGRESS_KEY: progress,
            _PREV_GATE_KEY: gate,
        }

        state = state.replace(
            metrics={
                **state.metrics,
                "gate": gate,
                "gate_prob": gate_prob,
                "gate_progress": progress,
            },
            obs=_flatten_obs_for_policy(state.obs),
        )

        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Take a step using the scheduled gate to select the motor program.

        Args:
            state: Current environment state.
            action: Residual vector from the policy.

        Returns:
            Next environment state after applying decoded control.
        """
        proprio = _get_proprio(state.obs, self._proprio_size)

        residual = action

        sched = state.info[_PRESERVE_INFO_KEY]
        progress = sched[_PROGRESS_KEY]
        gate_prob = self.gate_prob(progress)

        rng = state.info.get("rng", jax.random.PRNGKey(0))
        rng, gate_rng, noise_rng = jax.random.split(rng, 3)

        gate = self._step_gate(sched[_PREV_GATE_KEY], gate_prob, gate_rng)

        prior_mean, prior_logvar = self._prior_fn(proprio)
        rgm_mean, rgm_logvar = self._rgm_fn(proprio)

        if self._rgm_noise:
            if self._rgm_logvar is not None:
                rgm_std = jp.exp(0.5 * self._rgm_logvar)
            else:
                rgm_std = jp.exp(0.5 * rgm_logvar)
            rgm_z = (
                rgm_mean + jax.random.normal(noise_rng, shape=rgm_mean.shape) * rgm_std
            )
        else:
            rgm_z = rgm_mean

        # Prior + residual
        prior_residual = prior_mean + residual

        # Binary selection: gate=0 -> pure RGM, gate=1 -> prior + residual
        latent = gate * prior_residual + (1.0 - gate) * rgm_z

        state.info["rng"] = rng
        state.info[_PRESERVE_INFO_KEY] = {
            _PROGRESS_KEY: jp.clip(progress + self._progress_per_step, 0.0, 1.0),
            _PREV_GATE_KEY: gate,
        }

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
                "gate_prob": gate_prob,
                "gate_progress": progress,
            }
        )

        next_state = super().step(state, ctrl)

        next_state = next_state.replace(obs=_flatten_obs_for_policy(next_state.obs))

        return next_state

    @property
    def action_size(self) -> int:
        """Return the latent size as the action size (residual only)."""
        return self._latent_size


class PriorRGMInterpDecoderHighLevelWrapper(wrapper.Wrapper):
    """Mode: Fixed linear interpolation between motion prior and behavior prior.

    Like PriorRGMDecoderHighLevelWrapper but the blend weight is a constant
    hyperparameter instead of a learned binary gate. Every step mixes both
    priors, then applies the residual on top:

    latent = alpha * prior_mean + (1 - alpha) * rgm_z + residual

    alpha controls only the prior mixture; the residual keeps full authority at
    every alpha.

    The policy outputs only the residual, so the action size is latent_size
    (no gate head, no hysteresis, no carried gate state).

    Both endpoints reproduce an existing mode exactly: alpha=1 gives
    prior_mean + residual (res_control), and alpha=0 gives rgm_z + residual
    (bp_res_control).
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        prior_inference_fn: Callable,
        rgm_inference_fn: Callable,
        decoder_inference_fn: Callable,
        latent_size: int,
        proprio_size: int,
        alpha: float = 0.5,
        rgm_noise: bool = False,
        rgm_logvar: float | None = None,
    ):
        """Initialize the interpolated prior+rgm+decoder wrapper.

        Args:
            env: Base environment with dict observations.
            prior_inference_fn: Function (proprio) -> (mean, logvar)
            rgm_inference_fn: Function (proprio) -> (mean, logvar)
            decoder_inference_fn: Function (latent_proprio) -> (action, extras)
            latent_size: Dimension of the latent intention space.
            proprio_size: Size of proprioceptive observations.
            alpha: Fixed blend weight in [0, 1] applied to the two priors only.
                1.0 uses the motion prior (== res_control), 0.0 uses the
                behavior prior (== bp_res_control), and intermediate values
                mix them. The residual is added afterwards at full strength
                regardless of alpha.
            rgm_noise: If True, sample RGM latent with noise. If False,
                use rgm_mean only (deterministic).
            rgm_logvar: Fixed log-variance for RGM sampling. Only used when
                rgm_noise=True. If None, uses the RGM's predicted logvar.
                If a float, overrides the predicted logvar with this value.

        Raises:
            ValueError: If alpha is outside [0, 1].
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"prior_interp alpha must be in [0, 1], got {alpha}.")
        self._prior_fn = prior_inference_fn
        self._rgm_fn = rgm_inference_fn
        self._decoder_fn = decoder_inference_fn
        self._latent_size = latent_size
        self._proprio_size = proprio_size
        self._alpha = alpha
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

        state = state.replace(obs=_flatten_obs_for_policy(state.obs))

        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Take a step using a fixed blend of both priors as the latent.

        Args:
            state: Current environment state.
            action: Residual vector from the policy.

        Returns:
            Next environment state after applying decoded control.
        """
        proprio = _get_proprio(state.obs, self._proprio_size)

        residual = action

        prior_mean, prior_logvar = self._prior_fn(proprio)
        rgm_mean, rgm_logvar = self._rgm_fn(proprio)

        rng = state.info.get("rng", jax.random.PRNGKey(0))
        rng, noise_rng = jax.random.split(rng)
        if self._rgm_noise:
            if self._rgm_logvar is not None:
                rgm_std = jp.exp(0.5 * self._rgm_logvar)
            else:
                rgm_std = jp.exp(0.5 * rgm_logvar)
            rgm_z = (
                rgm_mean + jax.random.normal(noise_rng, shape=rgm_mean.shape) * rgm_std
            )
        else:
            rgm_z = rgm_mean

        # Fixed linear interpolation between the two priors, then the residual
        # is applied on top at full strength (independent of alpha).
        blend = self._alpha * prior_mean + (1.0 - self._alpha) * rgm_z
        latent = blend + residual

        state.info["rng"] = rng

        latent_proprio = jp.concatenate([latent, proprio], axis=-1)

        ctrl, decoder_extras = self._decoder_fn(latent_proprio)

        state.info["decoder_extras"] = decoder_extras
        state.info["prior_mean"] = prior_mean
        state.info["prior_logvar"] = prior_logvar
        state.info["rgm_mean"] = rgm_mean
        state.info["rgm_logvar"] = rgm_logvar
        state.info["final_latent"] = latent

        next_state = super().step(state, ctrl)

        next_state = next_state.replace(obs=_flatten_obs_for_policy(next_state.obs))

        return next_state

    @property
    def action_size(self) -> int:
        """Return the latent size as the action size (residual only)."""
        return self._latent_size
