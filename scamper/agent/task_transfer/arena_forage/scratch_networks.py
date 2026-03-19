"""Network architectures for scratch mode training.

Scratch mode trains a single-MLP policy from random initialization.
The policy takes flat observations and directly outputs action distribution
parameters. Uses SiLU activation with LayerNorm on hidden layers.

Expects flat observations (via FlatObsWrapper).
"""

from collections.abc import Sequence

import flax
import jax.numpy as jnp
from brax.training import distribution, networks
from brax.training.acme import running_statistics
from flax import linen as nn


def make_scratch_policy(
    action_param_size: int,
    obs_size: int,
    hidden_layer_sizes: Sequence[int] = (1024, 512, 256),
) -> networks.FeedForwardNetwork:
    """Create a scratch policy network (single MLP, obs -> action params).

    Args:
        action_param_size: Output dimension (typically 2x action_size for
            Gaussian mean and variance).
        obs_size: Total flat observation size.
        hidden_layer_sizes: Hidden layer sizes for policy MLP.

    Returns:
        FeedForwardNetwork with init and apply methods.
    """
    policy_module = networks.MLP(
        layer_sizes=list(hidden_layer_sizes) + [action_param_size],
        activation=nn.silu,
        layer_norm=True,
    )

    def apply(
        processor_params: running_statistics.RunningStatisticsState,
        policy_params,
        obs: jnp.ndarray,
    ) -> jnp.ndarray:
        normalized_obs = running_statistics.normalize(obs, processor_params)
        return policy_module.apply(policy_params, normalized_obs)

    dummy_obs = jnp.zeros((1, obs_size))

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs),
        apply=apply,
    )


def make_scratch_value_network(
    obs_size: int,
    hidden_layer_sizes: Sequence[int] = (1024, 512, 256),
) -> networks.FeedForwardNetwork:
    """Create a value network for scratch mode.

    Args:
        obs_size: Total flat observation size.
        hidden_layer_sizes: MLP layer sizes for value network.

    Returns:
        FeedForwardNetwork that accepts flat observations.
    """
    base_value_network = networks.make_value_network(
        obs_size,
        hidden_layer_sizes=hidden_layer_sizes,
    )

    def apply(
        processor_params: running_statistics.RunningStatisticsState,
        value_params,
        obs: jnp.ndarray,
    ):
        normalized_obs = running_statistics.normalize(obs, processor_params)
        return base_value_network.apply((), value_params, normalized_obs)

    return networks.FeedForwardNetwork(
        init=lambda key: base_value_network.init(key),
        apply=apply,
    )


@flax.struct.dataclass
class ScratchPPONetworks:
    """Container for scratch mode PPO network components.

    Attributes:
        policy_network: Single-MLP policy network.
        value_network: Feedforward value function network.
        parametric_action_distribution: Action distribution (NormalTanh).
    """

    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_scratch_ppo_networks(
    obs_size: int,
    action_size: int,
    policy_hidden_layer_sizes: Sequence[int] = (1024, 512, 256),
    value_hidden_layer_sizes: Sequence[int] = (1024, 512, 256),
) -> ScratchPPONetworks:
    """Create PPO networks for scratch mode training.

    Creates a single-MLP policy that directly maps observations to action
    distribution parameters, with no latent bottleneck or decoder.

    Args:
        obs_size: Total flat observation size.
        action_size: Action dimension.
        policy_hidden_layer_sizes: Hidden layer sizes for policy MLP.
        value_hidden_layer_sizes: Hidden layer sizes for value network.

    Returns:
        ScratchPPONetworks containing policy, value, and action distribution.
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    policy_network = make_scratch_policy(
        action_param_size=parametric_action_distribution.param_size,
        obs_size=obs_size,
        hidden_layer_sizes=policy_hidden_layer_sizes,
    )

    value_network = make_scratch_value_network(
        obs_size=obs_size,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )

    return ScratchPPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
