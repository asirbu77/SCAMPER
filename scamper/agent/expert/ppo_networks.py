"""PPO network definitions for the expert (MLP-policy) trainer.

Provides network containers, inference functions, and factory functions
for PPO training with a plain MLP policy (no VAE/latent bottleneck).

Observations are expected as dictionaries with keys:
- "imitation_target": Reference trajectory observations (flat array)
- "proprioception": Proprioceptive state observations (flat array)
"""

from collections.abc import Callable, Mapping, Sequence

import flax
import jax
from brax.training import distribution, networks, types
from brax.training.types import PRNGKey
from jax import numpy as jnp

from scamper.agent.imitation.ppo_networks import make_dict_value_network
from scamper.agent.expert import mlp_network


@flax.struct.dataclass
class PPOMLPNetworks:
    """Container for MLP PPO network components.

    Attributes:
        policy_network: MLP policy network (dict obs -> action params).
        value_network: Feedforward value function network.
        parametric_action_distribution: Action distribution (NormalTanh).
    """

    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(
    ppo_networks: PPOMLPNetworks,
) -> Callable[..., types.Policy]:
    """Create a policy factory function for inference.

    Args:
        ppo_networks: PPO network components.

    Returns:
        A make_policy function: (params, deterministic) -> policy_fn.
    """

    def make_policy(
        params: types.PolicyParams,
        deterministic: bool = False,
    ) -> types.Policy:
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def policy(
            observations: types.Observation,
            key_sample: PRNGKey,
        ) -> tuple[types.Action, types.Extra]:
            logits = policy_network.apply(*params, observations)

            if deterministic:
                return jnp.array(parametric_action_distribution.mode(logits)), {}

            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )

            return jnp.array(postprocessed_actions), {
                "log_prob": log_prob,
                "raw_action": raw_actions,
                "logits": logits,
            }

        return policy

    return make_policy


def make_logging_inference_fn(
    ppo_networks: PPOMLPNetworks,
) -> Callable[[bool], Callable]:
    """Create a policy factory for logging/evaluation with explicit params.

    Args:
        ppo_networks: PPO network components.

    Returns:
        A make_logging_policy function: (deterministic) -> logging_policy_fn.
        Where logging_policy_fn: (params, obs, key) -> (action, extras).
    """

    def make_logging_policy(deterministic: bool = False) -> Callable:
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def logging_policy(
            params: types.PolicyParams,
            observations: types.Observation,
            key_sample: PRNGKey,
        ) -> tuple[types.Action, types.Extra]:
            logits = policy_network.apply(*params, observations)

            if deterministic:
                return jnp.array(parametric_action_distribution.mode(logits)), {}

            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )

            return jnp.array(postprocessed_actions), {
                "log_prob": log_prob,
                "raw_action": raw_actions,
                "logits": logits,
            }

        return logging_policy

    return make_logging_policy


def make_mlp_ppo_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
    mlp_hidden_layer_sizes: Sequence[int] = (2048, 1024, 1024, 512),
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
) -> PPOMLPNetworks:
    """Create MLP-based PPO networks for imitation learning.

    Args:
        obs_sizes: Dict mapping observation keys to their sizes.
        action_size: Action dimension.
        mlp_hidden_layer_sizes: Hidden layer sizes for the MLP policy.
        value_hidden_layer_sizes: Hidden layer sizes for the value network.

    Returns:
        PPOMLPNetworks containing policy, value, and action distribution.
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    policy_network = mlp_network.make_mlp_policy(
        parametric_action_distribution.param_size,
        obs_sizes=obs_sizes,
        mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
    )

    value_network = make_dict_value_network(
        obs_sizes=obs_sizes,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )

    return PPOMLPNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
