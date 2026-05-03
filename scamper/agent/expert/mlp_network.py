"""MLP policy network for the expert trainer (no latent bottleneck).

Provides a simple MLP that takes concatenated observations (imitation_target +
proprioception) and directly outputs action distribution parameters. No encoder,
decoder, or VAE components.
"""

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
from brax.training import networks
from flax import linen as nn

from scamper.utils.observation_utils import (
    DictRunningStatisticsState,
    concat_flat_dict_obs,
    normalize_dict_obs,
)


class MLPPolicy(nn.Module):
    """MLP policy that maps flat observations to action distribution parameters.

    Attributes:
        layer_sizes: All layer dimensions including final output size.
            Hidden layers get SiLU + LayerNorm; the final layer has no activation.
        activation: Activation function (default: SiLU).
        kernel_init: Weight initializer (default: LeCun uniform).
        bias: Whether to use bias terms in Dense layers.
    """

    layer_sizes: Sequence[int]
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            if i != len(self.layer_sizes) - 1:
                x = self.activation(x)
                x = nn.LayerNorm()(x)
        return x


def make_mlp_policy(
    action_param_size: int,
    obs_sizes: Mapping[str, int],
    mlp_hidden_layer_sizes: Sequence[int] = (2048, 1024, 1024, 512),
) -> networks.FeedForwardNetwork:
    """Create an MLP policy network that accepts dictionary observations.

    Args:
        action_param_size: Output dimension (typically 2x action_size for
            Gaussian mean and log-variance).
        obs_sizes: Dict mapping observation keys to their sizes.
        mlp_hidden_layer_sizes: Hidden layer sizes for the MLP.

    Returns:
        FeedForwardNetwork with init and apply methods. The apply function
        takes (processor_params, policy_params, obs) and returns action logits.
    """
    policy_module = MLPPolicy(
        layer_sizes=list(mlp_hidden_layer_sizes) + [action_param_size],
    )

    total_obs_size = sum(obs_sizes.values())
    dummy_obs = jnp.zeros((1, total_obs_size))

    def apply(
        processor_params: DictRunningStatisticsState,
        policy_params,
        obs: Mapping[str, jnp.ndarray],
    ):
        normalized_obs = normalize_dict_obs(obs, processor_params)
        flat_obs = concat_flat_dict_obs(normalized_obs)
        return policy_module.apply(policy_params, flat_obs)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs),
        apply=apply,
    )
