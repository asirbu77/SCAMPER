"""Networks for from-scratch PPO training of the three-network system.

Composes the distillation networks -- ``DistillEncoder``, ``DistillDecoder``
and ``DistillMotionPrior`` -- into a single Flax module so that the policy
parameter tree is natively ``{"params": {"encoder", "decoder", "prior"}}``,
which is the layout ``task_transfer`` checkpoint loading expects.

Key constraint (inherited from ``distill_expert.distill_networks``): the
decoder and motion-prior parameter trees must exactly match
``imitation.intention_network.Decoder`` and ``imitation.prior.Prior``, so the
saved checkpoint stays loadable by the task-transfer wrappers.
"""

from collections.abc import Callable, Mapping, Sequence

import flax
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.types import PRNGKey
from flax import linen as nn

from scamper.agent.distill_expert.distill_networks import (
    DistillDecoder,
    DistillEncoder,
    DistillMotionPrior,
)
from scamper.agent.imitation import intention_network
from scamper.agent.imitation.ppo_networks import make_dict_value_network
from scamper.utils.observation_utils import (
    DictRunningStatisticsState,
    normalize_dict_obs,
)


class ScratchIntentionNetwork(nn.Module):
    """Encoder + decoder + motion prior as one policy module.

    Follows ``intention_network.IntentionNetwork`` in how it handles keys and
    proprioception noise, but the encoder consumes
    ``concat(imitation_target, proprioception)`` (as in distillation) and a
    proprioception-conditioned motion prior is evaluated alongside it.

    Submodule names (``encoder``, ``decoder``, ``prior``) are load-bearing:
    they become the keys of the saved parameter tree.

    Attributes:
        encoder_layers: Hidden layer sizes for the encoder MLP.
        decoder_layers: Layer sizes for the decoder, including the final
            action-parameter output size.
        motion_prior_layers: Hidden layer sizes for the motion prior MLP.
        latents: Dimension of the latent intention space.
        proprioception_noise_std: Stddev of multiplicative Gaussian noise
            applied to proprioception on stochastic passes (0 = disabled).
        clamp_logvar: Whether to clamp encoder/prior logvars.
        logvar_min: Lower clamp bound (PULSE default -5).
        logvar_max: Upper clamp bound (PULSE default 0).
    """

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    motion_prior_layers: Sequence[int]
    latents: int = 16
    proprioception_noise_std: float = 0.0
    clamp_logvar: bool = True
    logvar_min: float = -5.0
    logvar_max: float = 0.0

    def setup(self):
        """Initialize encoder, decoder and motion-prior submodules."""
        self.encoder = DistillEncoder(
            layer_sizes=self.encoder_layers,
            latents=self.latents,
            clamp_logvar=self.clamp_logvar,
            logvar_min=self.logvar_min,
            logvar_max=self.logvar_max,
        )
        self.decoder = DistillDecoder(layer_sizes=self.decoder_layers)
        # Submodule name kept as 'prior' for checkpoint-key compatibility with
        # task_transfer loaders; the network is a motion prior.
        self.prior = DistillMotionPrior(
            layer_sizes=self.motion_prior_layers,
            latents=self.latents,
            clamp_logvar=self.clamp_logvar,
            logvar_min=self.logvar_min,
            logvar_max=self.logvar_max,
        )

    def __call__(
        self,
        obs: Mapping[str, jnp.ndarray],
        key: jax.Array,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        """Run encoder -> reparameterize -> decoder, plus the motion prior.

        Returns:
            ``(action_params, latent_mean, latent_logvar, motion_prior_mean,
            motion_prior_logvar)``, with a trailing activations dict when
            ``get_activation`` is set.
        """
        traj = obs["imitation_target"]
        egocentric_obs = obs["proprioception"]

        obs_is_batched = traj.ndim >= 2

        # Key handling must match between rollout and loss replay: the PPO loss
        # re-runs this module with the per-sample keys stored in policy_extras.
        if key.ndim == 1:
            noise_rng, encoder_rng = jax.random.split(key)
        elif not obs_is_batched:
            noise_rng, encoder_rng = jax.random.split(key[0])
        else:
            noise_rng, encoder_rng = jax.vmap(jax.random.split)(key).swapaxes(0, 1)

        # Multiplicative Gaussian noise on proprioception (training only)
        if not deterministic and self.proprioception_noise_std > 0.0:
            if noise_rng.ndim == 1:
                noise = jax.random.normal(noise_rng, egocentric_obs.shape)
            elif not obs_is_batched:
                noise = jax.random.normal(noise_rng[0], egocentric_obs.shape)
            else:
                noise = jax.vmap(
                    lambda rng_key, obs_i: jax.random.normal(rng_key, obs_i.shape)
                )(noise_rng, egocentric_obs)
            egocentric_obs = egocentric_obs * (
                1.0 + self.proprioception_noise_std * noise
            )

        encoder_input = jnp.concatenate([traj, egocentric_obs], axis=-1)
        latent_mean, latent_logvar = self.encoder(encoder_input)

        # Uses the mean in the case of deterministic evaluation
        if deterministic:
            z = latent_mean
        else:
            z = intention_network.reparameterize(
                encoder_rng, latent_mean, latent_logvar
            )

        action, _ = self.decoder(jnp.concatenate([z, egocentric_obs], axis=-1))

        motion_prior_mean, motion_prior_logvar = self.prior(egocentric_obs)

        if get_activation:
            return (
                action,
                latent_mean,
                latent_logvar,
                motion_prior_mean,
                motion_prior_logvar,
                {
                    "egocentric_obs": egocentric_obs,
                    "traj_obs": traj,
                    "intention": z,
                },
            )
        return action, latent_mean, latent_logvar, motion_prior_mean, motion_prior_logvar


@flax.struct.dataclass
class PPOScratchNetworks:
    """Container for the from-scratch PPO network components.

    Attributes:
        policy_network: Encoder + decoder + motion-prior policy network.
        value_network: Feedforward value function network.
        parametric_action_distribution: Action distribution (NormalTanh).
    """

    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_scratch_policy(
    action_param_size: int,
    latent_size: int,
    obs_sizes: Mapping[str, int],
    encoder_hidden_layer_sizes: Sequence[int] = (2048, 1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024, 512),
    motion_prior_hidden_layer_sizes: Sequence[int] = (1536, 1024, 512),
    proprioception_noise_std: float = 0.0,
    clamp_logvar: bool = True,
    logvar_min: float = -5.0,
    logvar_max: float = 0.0,
) -> networks.FeedForwardNetwork:
    """Create the encoder/decoder/motion-prior policy network.

    Args:
        action_param_size: Output dimension (2x action_size for NormalTanh).
        latent_size: Dimension of the latent intention space.
        obs_sizes: Dict mapping observation keys to their sizes.
        encoder_hidden_layer_sizes: Hidden layer sizes for the encoder MLP.
        decoder_hidden_layer_sizes: Hidden layer sizes for the decoder MLP.
        motion_prior_hidden_layer_sizes: Hidden layer sizes for the motion prior.
        proprioception_noise_std: Stddev for multiplicative proprioception noise.
        clamp_logvar: Whether to clamp encoder/prior logvars.
        logvar_min: Lower clamp bound.
        logvar_max: Upper clamp bound.

    Returns:
        FeedForwardNetwork whose apply returns ``(action_params, latent_mean,
        latent_logvar, motion_prior_mean, motion_prior_logvar)``.
    """
    policy_module = ScratchIntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes) + [action_param_size],
        motion_prior_layers=list(motion_prior_hidden_layer_sizes),
        latents=latent_size,
        proprioception_noise_std=proprioception_noise_std,
        clamp_logvar=clamp_logvar,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )

    def apply(
        processor_params: DictRunningStatisticsState,
        policy_params,
        obs: Mapping[str, jnp.ndarray],
        key,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        """Apply the policy with observation normalization."""
        obs = normalize_dict_obs(obs, processor_params)
        return policy_module.apply(
            policy_params,
            obs=obs,
            key=key,
            deterministic=deterministic,
            get_activation=get_activation,
        )

    dummy_obs = {
        "imitation_target": jnp.zeros((1, obs_sizes["imitation_target"])),
        "proprioception": jnp.zeros((1, obs_sizes["proprioception"])),
    }
    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs, dummy_key),
        apply=apply,
    )


def make_scratch_ppo_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
    latent_size: int = 16,
    encoder_hidden_layer_sizes: Sequence[int] = (2048, 1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024, 512),
    motion_prior_hidden_layer_sizes: Sequence[int] = (1536, 1024, 512),
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    proprioception_noise_std: float = 0.0,
    clamp_logvar: bool = True,
    logvar_min: float = -5.0,
    logvar_max: float = 0.0,
) -> PPOScratchNetworks:
    """Create the PPO networks for from-scratch three-network training.

    The trainer calls this positionally as ``factory(obs_sizes, action_size)``;
    every other argument is bound by ``functools.partial`` in the entry point.

    Args:
        obs_sizes: Dict mapping observation keys to their sizes, e.g.
            ``{"imitation_target": 3716, "proprioception": 226}``.
        action_size: Action dimension.
        latent_size: Dimension of the latent intention space.
        encoder_hidden_layer_sizes: Hidden layer sizes for the encoder MLP.
        decoder_hidden_layer_sizes: Hidden layer sizes for the decoder MLP.
        motion_prior_hidden_layer_sizes: Hidden layer sizes for the motion prior.
        value_hidden_layer_sizes: Hidden layer sizes for the value network.
        proprioception_noise_std: Stddev for multiplicative proprioception noise.
        clamp_logvar: Whether to clamp encoder/prior logvars.
        logvar_min: Lower clamp bound.
        logvar_max: Upper clamp bound.

    Returns:
        PPOScratchNetworks containing policy, value, and action distribution.
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    policy_network = make_scratch_policy(
        parametric_action_distribution.param_size,
        latent_size=latent_size,
        obs_sizes=obs_sizes,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        motion_prior_hidden_layer_sizes=motion_prior_hidden_layer_sizes,
        proprioception_noise_std=proprioception_noise_std,
        clamp_logvar=clamp_logvar,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )

    value_network = make_dict_value_network(
        obs_sizes=obs_sizes,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )

    return PPOScratchNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def _split_keys_for_obs(observations, key_network: PRNGKey):
    """Return one key per sample when observations are batched."""
    obs_leaf = jax.tree_util.tree_leaves(observations)[0]
    if obs_leaf.ndim >= 2:
        return jax.random.split(key_network, obs_leaf.shape[0])
    return key_network


def make_inference_fn(
    ppo_networks: PPOScratchNetworks,
) -> Callable[..., types.Policy]:
    """Create a policy factory for rollout during training.

    The per-sample keys are stored in the returned extras as ``policy_rng`` so
    the PPO loss can replay the exact latent sample.

    Args:
        ppo_networks: PPO network components.

    Returns:
        A ``make_policy(params, deterministic) -> policy_fn`` factory.
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
            key_sample, key_network = jax.random.split(key_sample)
            per_sample_keys = _split_keys_for_obs(observations, key_network)

            logits, latent_mean, latent_logvar, _, _ = policy_network.apply(
                *params, observations, per_sample_keys, deterministic=deterministic
            )

            if deterministic:
                return jnp.array(parametric_action_distribution.mode(logits)), {
                    "latent_mean": latent_mean,
                    "latent_logvar": latent_logvar,
                }

            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )

            return jnp.array(postprocessed_actions), {
                "latent_mean": latent_mean,
                "latent_logvar": latent_logvar,
                "log_prob": log_prob,
                "raw_action": raw_actions,
                "logits": logits,
                "policy_rng": per_sample_keys,
            }

        return policy

    return make_policy


def make_logging_inference_fn(
    ppo_networks: PPOScratchNetworks,
) -> Callable[[bool], Callable]:
    """Create a policy factory for logging/evaluation with explicit params.

    Unlike ``make_inference_fn`` the returned policy takes params as an
    argument, and it also reports the motion-prior distribution so rollout
    logging can track encoder-vs-prior drift.

    Args:
        ppo_networks: PPO network components.

    Returns:
        A ``make_logging_policy(deterministic) -> fn(params, obs, key)`` factory.
    """

    def make_logging_policy(deterministic: bool = False) -> Callable:
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def logging_policy(
            params: types.PolicyParams,
            observations: types.Observation,
            key_sample: PRNGKey,
        ) -> tuple[types.Action, types.Extra]:
            key_sample, key_network = jax.random.split(key_sample)
            per_sample_keys = _split_keys_for_obs(observations, key_network)

            (
                logits,
                latent_mean,
                latent_logvar,
                motion_prior_mean,
                motion_prior_logvar,
            ) = policy_network.apply(
                *params, observations, per_sample_keys, deterministic=deterministic
            )

            extras = {
                "latent_mean": latent_mean,
                "latent_logvar": latent_logvar,
                "motion_prior_mean": motion_prior_mean,
                "motion_prior_logvar": motion_prior_logvar,
                "logits": logits,
            }

            if deterministic:
                return jnp.array(parametric_action_distribution.mode(logits)), extras

            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            extras["log_prob"] = parametric_action_distribution.log_prob(
                logits, raw_actions
            )
            extras["raw_action"] = raw_actions
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )

            return jnp.array(postprocessed_actions), extras

        return logging_policy

    return make_logging_policy


def create_compatible_checkpoint_params(
    policy_params,
    normalizer_params: DictRunningStatisticsState,
    obs_sizes: Mapping[str, int],
    latent_size: int,
    encoder_hidden_layer_sizes: Sequence[int],
) -> tuple:
    """Create checkpoint params compatible with task_transfer loading.

    Mirrors ``distill_networks.create_compatible_checkpoint_params``: the saved
    encoder is a dummy standard-shape ``intention_network.Encoder`` (the
    task-transfer abstract tree expects an imitation_target-only encoder, and
    only the motion prior + decoder are consumed downstream). The trained
    ``ScratchIntentionNetwork`` encoder survives in the ``train_state`` item.

    Args:
        policy_params: Policy parameter tree, i.e.
            ``{"params": {"encoder", "decoder", "prior"}}``.
        normalizer_params: Observation normalizer state.
        obs_sizes: Dict mapping observation keys to their sizes.
        latent_size: Dimension of the latent intention space.
        encoder_hidden_layer_sizes: Encoder hidden layer sizes (for the dummy).

    Returns:
        ``(normalizer_params, combined_params)`` tuple.
    """
    dummy_encoder = intention_network.Encoder(
        layer_sizes=list(encoder_hidden_layer_sizes),
        latents=latent_size,
    )
    dummy_traj = jnp.zeros((1, obs_sizes["imitation_target"]))
    dummy_encoder_params = dummy_encoder.init(jax.random.PRNGKey(42), dummy_traj)[
        "params"
    ]

    combined_params = {
        "params": {
            "encoder": dummy_encoder_params,
            "decoder": policy_params["params"]["decoder"],
            # Dict key 'prior' kept for checkpoint compat with task_transfer.
            "prior": policy_params["params"]["prior"],
        }
    }

    return (normalizer_params, combined_params)
