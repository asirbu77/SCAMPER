"""Multi-mode behavior-prior rollout evaluation.

Evaluates the behavior-prior network's ability to generate plausible actions
by running rollouts using BehaviorPrior + DistillDecoder in 6 different modes:
1. Deterministic: z = behavior_prior_mean (no sampling)
2. logvar=0: z sampled with std=1.0
3. logvar=-2: z sampled with std≈0.368
4. logvar=-3: z sampled with std≈0.187
5. logvar=-4: z sampled with std≈0.092
6. predicted_logvar: z sampled using network-predicted logvar per dimension

All six modes are evaluated and rendered separately.
"""

from typing import Callable, Optional, Sequence, Tuple, Any, Dict

import jax
import jax.numpy as jnp
from jax import random
from brax.training import distribution, types
from brax.training.acme import running_statistics

from scamper.agent.distill_expert.distill_networks import DistillDecoder
from scamper.agent.mlp_prior.prior_networks import Prior as BehaviorPrior
from scamper.agent.mlp_prior.prior_rollout_eval import (
    check_termination_nan,
    compute_world_zaxis_termination,
    reparameterize,
)
from scamper.agent.observation_utils import DictRunningStatisticsState


def create_behavior_prior_policy(
    behavior_prior_network_params: Dict,
    decoder_network_params: Dict,
    normalizer_params: running_statistics.RunningStatisticsState,
    intention_latent_size: int,
    action_size: int,
    proprioceptive_obs_size: int,
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024, 512),
    behavior_prior_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    fixed_logvar: Optional[float] = -2.0,
    deterministic: bool = False,
) -> Callable:
    """Create a policy that uses only the behavior prior + DistillDecoder.

    Args:
        behavior_prior_network_params: Parameters for the behavior-prior network.
        decoder_network_params: Parameters for the DistillDecoder network.
        normalizer_params: Running statistics for observation normalization.
        intention_latent_size: Size of the intention latent space.
        action_size: Size of the action space.
        proprioceptive_obs_size: Size of proprioceptive observations.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder.
        behavior_prior_hidden_layer_sizes: Hidden layer sizes for the
            behavior prior.
        preprocess_observations_fn: Function to normalize observations.
        fixed_logvar: Fixed log-variance to use for latent sampling. If None,
            uses the network-predicted logvar per dimension.
        deterministic: If True, use mean of the behavior-prior distribution
            instead of sampling.

    Returns:
        Policy function ``(proprioceptive_obs, rng_key) -> (actions, extras)``.
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    action_param_size = action_size * 2

    behavior_prior_module = BehaviorPrior(
        layer_sizes=list(behavior_prior_hidden_layer_sizes),
        latents=intention_latent_size,
    )

    decoder_module = DistillDecoder(
        layer_sizes=list(decoder_hidden_layer_sizes) + [action_param_size],
    )

    def policy_fn(
        obs: jax.Array,
        rng_key: jax.Array,
    ) -> Tuple[jax.Array, Dict[str, Any]]:
        _, key_sample = random.split(rng_key)

        # Get proprioceptive observations (last part of obs)
        proprioceptive_obs = obs[..., -proprioceptive_obs_size:]

        # Normalize observations
        normalized_obs = preprocess_observations_fn(
            proprioceptive_obs, normalizer_params
        )

        # Get behavior-prior distribution
        behavior_prior_mean, behavior_prior_logvar = behavior_prior_module.apply(
            {"params": behavior_prior_network_params}, normalized_obs
        )

        # Determine which logvar to use for sampling
        if fixed_logvar is not None:
            logvar_for_sampling = jnp.full_like(behavior_prior_mean, fixed_logvar)
        else:
            logvar_for_sampling = behavior_prior_logvar

        # Sample from behavior prior
        if deterministic:
            z = behavior_prior_mean
        else:
            z = reparameterize(key_sample, behavior_prior_mean, logvar_for_sampling)

        # Decode to action distribution parameters
        decoder_input = jnp.concatenate([z, normalized_obs], axis=-1)
        logits, _ = decoder_module.apply(
            {"params": decoder_network_params}, decoder_input
        )

        # Always use mode of action distribution for deterministic actions
        action = parametric_action_distribution.mode(logits)

        extras = {
            "behavior_prior_mean": behavior_prior_mean,
            "behavior_prior_logvar": behavior_prior_logvar,
            "intention": z,
            "logits": logits,
        }

        return action, extras

    return policy_fn


def extract_behavior_prior_decoder_params(
    policy_params: Tuple,
) -> Tuple[Dict, Dict, DictRunningStatisticsState]:
    """Extract behavior-prior and decoder parameters from full policy params.

    Args:
        policy_params: Tuple of (normalizer_params, network_params).

    Returns:
        Tuple of (behavior_prior_params, decoder_params, normalizer_params).
        Behavior-prior params are read from the saved ``"rgm"`` dict key
        (kept for checkpoint-compat with task_transfer loaders).
    """
    normalizer_params, network_params = policy_params
    # Dict key 'rgm' kept for compat; holds the behavior-prior params.
    behavior_prior_params = network_params["params"]["rgm"]
    decoder_params = network_params["params"]["decoder"]
    return behavior_prior_params, decoder_params, normalizer_params


class MultiModeBehaviorPriorRolloutEvaluator:
    """Evaluator for running behavior-prior-only rollouts in multiple modes.

    Runs evaluation in 6 modes with the same logvar values as
    ``mlp_prior.prior_rollout_eval.MultiModePriorRolloutEvaluator``.
    Uses BehaviorPrior + DistillDecoder instead of Prior + Decoder.
    """

    def __init__(
        self,
        env,
        intention_latent_size: int,
        action_size: int,
        proprioceptive_obs_size: int,
        decoder_hidden_layer_sizes: Sequence[int],
        behavior_prior_hidden_layer_sizes: Sequence[int],
        preprocess_observations_fn: types.PreprocessObservationFn,
        max_steps: int = 200,
        eval_interval: int = 1,
        render_fps: int = 50,
        render_camera_name: str = "close_profile",
        model_path: str = "",
    ):
        self.env = env
        self.intention_latent_size = intention_latent_size
        self.action_size = action_size
        self.proprioceptive_obs_size = proprioceptive_obs_size
        self.decoder_hidden_layer_sizes = decoder_hidden_layer_sizes
        self.behavior_prior_hidden_layer_sizes = behavior_prior_hidden_layer_sizes
        self.preprocess_observations_fn = preprocess_observations_fn
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.render_fps = render_fps
        self.render_camera_name = f"{render_camera_name}-rodent"
        self.model_path = model_path

        self._key = random.PRNGKey(0)

        # Same 6 evaluation modes as mlp_prior
        self.evaluation_modes = [
            {"name": "deterministic", "deterministic": True, "fixed_logvar": -2.0},
            {"name": "logvar_0", "deterministic": False, "fixed_logvar": 0.0},
            {"name": "logvar_-2", "deterministic": False, "fixed_logvar": -2.0},
            {"name": "logvar_-3", "deterministic": False, "fixed_logvar": -3.0},
            {"name": "logvar_-4", "deterministic": False, "fixed_logvar": -4.0},
            {"name": "predicted_logvar", "deterministic": False, "fixed_logvar": None},
        ]

    def _build_single_rollout_fn(self, policy_params: Tuple, mode: Dict):
        """Build the jitted single rollout function for a specific mode."""
        (
            behavior_prior_params,
            decoder_params,
            normalizer_params,
        ) = extract_behavior_prior_decoder_params(policy_params)

        # Extract proprioceptive-only normalizer from dict normalizer
        proprio_normalizer_params = normalizer_params.proprioception

        policy_fn = create_behavior_prior_policy(
            behavior_prior_network_params=behavior_prior_params,
            decoder_network_params=decoder_params,
            normalizer_params=proprio_normalizer_params,
            intention_latent_size=self.intention_latent_size,
            action_size=self.action_size,
            proprioceptive_obs_size=self.proprioceptive_obs_size,
            decoder_hidden_layer_sizes=self.decoder_hidden_layer_sizes,
            behavior_prior_hidden_layer_sizes=self.behavior_prior_hidden_layer_sizes,
            preprocess_observations_fn=self.preprocess_observations_fn,
            fixed_logvar=mode["fixed_logvar"],
            deterministic=mode["deterministic"],
        )

        jit_step = jax.jit(self.env.step)

        @jax.jit
        def single_rollout_fn(
            initial_state: Any, rng_key: jax.Array
        ) -> Tuple[jax.Array, jax.Array, Any]:
            state = initial_state

            def step_fn(carry, _):
                state, key, nan_terminated = carry
                key, key_action = random.split(key)

                if hasattr(state.obs, "get") or isinstance(state.obs, dict):
                    proprio = state.obs.get("proprioception", state.obs)
                    if isinstance(proprio, dict):
                        from jax import flatten_util

                        proprio, _ = flatten_util.ravel_pytree(proprio)
                else:
                    proprio = state.obs

                action, _ = policy_fn(proprio, key_action)

                next_state = jit_step(state, action)

                step_nan = check_termination_nan(next_state.data)
                new_nan_terminated = jnp.logical_or(nan_terminated, step_nan)

                return (next_state, key, new_nan_terminated), next_state

            initial_carry = (state, rng_key, jnp.array(False))
            (_, _, nan_terminated), all_states = jax.lax.scan(
                step_fn, initial_carry, None, length=self.max_steps
            )

            upside_down_flags = compute_world_zaxis_termination(
                self.env, all_states.data
            )

            any_upside_down = jnp.any(upside_down_flags)
            first_upside_down_step = jnp.argmax(upside_down_flags)

            terminated = jnp.logical_or(nan_terminated, any_upside_down)
            step_count = jnp.where(
                any_upside_down, first_upside_down_step + 1, self.max_steps
            )

            return step_count, terminated, all_states

        return single_rollout_fn

    def _render_rollout(
        self, states: Any, step_count: int, current_step: int, mode_name: str
    ) -> None:
        """Render a rollout and log to wandb."""
        import wandb
        import imageio

        states_list = []
        for i in range(step_count):
            state_i = jax.tree_util.tree_map(lambda x: x[i], states)
            states_list.append(state_i)

        frames = self.env.render(
            states_list, camera=self.render_camera_name, height=480, width=640
        )

        if len(frames) > 0:
            video_path = (
                f"{self.model_path}/behavior_prior_rollout_"
                f"{mode_name}_{current_step}.mp4"
            )
            with imageio.get_writer(video_path, fps=self.render_fps) as writer:
                for frame in frames:
                    writer.append_data(frame)

            wandb.log(
                {
                    f"videos/behavior_prior_rollout_{mode_name}": wandb.Video(
                        video_path, format="mp4"
                    )
                },
                commit=False,
            )

    def run_evaluation(
        self,
        policy_params: Tuple,
        eval_step: int,
        reset_key: jax.Array,
    ) -> Optional[Dict[str, float]]:
        """Run behavior-prior rollout evaluation in all 6 modes from a shared initial state.

        Args:
            policy_params: Current policy parameters.
            eval_step: Current evaluation step number.
            reset_key: Random key for resetting environment.

        Returns:
            Dictionary of metrics if evaluation was run, None otherwise.
        """
        if eval_step % self.eval_interval != 0:
            return None

        jit_reset = jax.jit(self.env.reset)
        initial_state = jit_reset(reset_key)

        self._key, rollout_key = random.split(self._key)

        for mode in self.evaluation_modes:
            mode_name = mode["name"]

            single_rollout_fn = self._build_single_rollout_fn(policy_params, mode)

            rollout_key, mode_key = random.split(rollout_key)

            step_count, _, states = single_rollout_fn(initial_state, mode_key)
            step_count = int(step_count)

            try:
                self._render_rollout(states, self.max_steps, eval_step, mode_name)
            except Exception as e:
                import logging

                logging.warning(f"Failed to render {mode_name} rollout: {e}")

        return {}
