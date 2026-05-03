"""Network architectures for DAgger distillation of the expert teacher.

Defines encoder, decoder, and motion-prior networks for distilling an expert
teacher into a VAE student with a jointly trained motion prior.

Key constraint: the Decoder and motion-prior parameter trees must exactly
match those of intention_network.Decoder and mlp_prior.Prior respectively,
so that checkpoints are compatible with task_transfer wrappers.
"""

from collections.abc import Mapping, Sequence

import flax
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.acme import running_statistics, specs
from flax import linen as nn

from scamper.agent.imitation import intention_network
from scamper.agent.mlp_prior.prior_networks import Prior as OriginalPrior
from scamper.agent.observation_utils import (
    DictRunningStatisticsState,
    normalize_dict_obs,
)

# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------


class DistillEncoder(nn.Module):
    """Encoder: maps [imitation_target, proprioception] -> (mu, logvar).

    Takes the concatenation of both observation keys as input (unlike the
    imitation encoder which takes only imitation_target).

    Attributes:
        layer_sizes: Hidden layer dimensions.
        latents: Latent dimension.
        clamp_logvar: Whether to clamp logvar to [logvar_min, logvar_max].
        logvar_min: Minimum logvar value (default -5, matching PULSE).
        logvar_max: Maximum logvar value (default 0, matching PULSE).
    """

    layer_sizes: Sequence[int]
    latents: int
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True
    clamp_logvar: bool = True
    logvar_min: float = -5.0
    logvar_max: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            x = self.activation(x)
            x = nn.LayerNorm()(x)

        mean = nn.Dense(self.latents, name="fc_mean")(x)
        logvar = nn.Dense(self.latents, name="fc_logvar")(x)
        if self.clamp_logvar:
            logvar = jnp.clip(logvar, self.logvar_min, self.logvar_max)
        return mean, logvar


class DistillDecoder(nn.Module):
    """Decoder: maps [z, proprioception] -> action_params.

    Layer naming matches intention_network.Decoder (hidden_0, hidden_1, ...)
    so the parameter tree is checkpoint-compatible with task_transfer.

    Attributes:
        layer_sizes: All layer dims including final output (action_param_size).
    """

    layer_sizes: Sequence[int]
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activation(x)
                x = nn.LayerNorm()(x)
        return x, {}


class DistillMotionPrior(nn.Module):
    """Motion prior: maps proprioception -> (mu, logvar).

    Layer and output head naming matches mlp_prior.Prior (hidden_0, hidden_1,
    ..., fc2_mean, fc2_logvar) for checkpoint compatibility with task_transfer.

    Attributes:
        layer_sizes: Hidden layer dimensions.
        latents: Latent dimension (must match encoder).
        clamp_logvar: Whether to clamp logvar to [logvar_min, logvar_max].
        logvar_min: Minimum logvar value (default -5, matching PULSE).
        logvar_max: Maximum logvar value (default 0, matching PULSE).
    """

    layer_sizes: Sequence[int]
    latents: int
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True
    clamp_logvar: bool = True
    logvar_min: float = -5.0
    logvar_max: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            x = self.activation(x)
            x = nn.LayerNorm()(x)

        mean = nn.Dense(self.latents, name="fc2_mean")(x)
        logvar = nn.Dense(self.latents, name="fc2_logvar")(x)
        if self.clamp_logvar:
            logvar = jnp.clip(logvar, self.logvar_min, self.logvar_max)
        return mean, logvar


# ---------------------------------------------------------------------------
# Params container
# ---------------------------------------------------------------------------


@flax.struct.dataclass
class DistillNetworkParams:
    """Trainable parameters for the distillation student."""

    encoder: types.Params
    decoder: types.Params
    # Field name kept as 'prior' for checkpoint-key compatibility (it becomes
    # the "prior" key in the saved params tree, which downstream
    # task_transfer loaders depend on); the underlying network is a
    # DistillMotionPrior.
    prior: types.Params


# ---------------------------------------------------------------------------
# Reparameterization
# ---------------------------------------------------------------------------


def reparameterize(
    rng: jax.Array, mean: jnp.ndarray, logvar: jnp.ndarray
) -> jnp.ndarray:
    """Sample z = mu + exp(0.5*logvar) * eps, eps ~ N(0, I)."""
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, logvar.shape)
    return mean + eps * std


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def make_distill_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
    latent_size: int = 16,
    encoder_hidden_layer_sizes: Sequence[int] = (2048, 1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024, 512),
    motion_prior_hidden_layer_sizes: Sequence[int] = (1536, 1024, 512),
    clamp_logvar: bool = True,
    logvar_min: float = -5.0,
    logvar_max: float = 0.0,
) -> tuple[
    DistillEncoder,
    DistillDecoder,
    DistillMotionPrior,
    distribution.ParametricDistribution,
]:
    """Create the three network modules and the action distribution.

    Returns:
        Tuple of (encoder, decoder, motion_prior, action_distribution).
    """
    action_param_size = action_size * 2  # mean + logstd for NormalTanh

    encoder = DistillEncoder(
        layer_sizes=list(encoder_hidden_layer_sizes),
        latents=latent_size,
        clamp_logvar=clamp_logvar,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )
    decoder = DistillDecoder(
        layer_sizes=list(decoder_hidden_layer_sizes) + [action_param_size],
    )
    motion_prior = DistillMotionPrior(
        layer_sizes=list(motion_prior_hidden_layer_sizes),
        latents=latent_size,
        clamp_logvar=clamp_logvar,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )
    action_dist = distribution.NormalTanhDistribution(event_size=action_size)

    return encoder, decoder, motion_prior, action_dist


def init_distill_params(
    encoder: DistillEncoder,
    decoder: DistillDecoder,
    motion_prior: DistillMotionPrior,
    obs_sizes: Mapping[str, int],
    latent_size: int,
    rng: jax.Array,
) -> DistillNetworkParams:
    """Initialize all three network parameters.

    Returns:
        DistillNetworkParams with encoder, decoder, motion-prior params.
    """
    rng_enc, rng_dec, rng_motion_prior = jax.random.split(rng, 3)

    total_obs_size = sum(obs_sizes.values())
    proprio_size = obs_sizes["proprioception"]

    dummy_encoder_input = jnp.zeros((1, total_obs_size))
    dummy_decoder_input = jnp.zeros((1, latent_size + proprio_size))
    dummy_motion_prior_input = jnp.zeros((1, proprio_size))

    encoder_params = encoder.init(rng_enc, dummy_encoder_input)
    decoder_params = decoder.init(rng_dec, dummy_decoder_input)
    motion_prior_params = motion_prior.init(rng_motion_prior, dummy_motion_prior_input)

    # Field name kept as 'prior' for checkpoint-key compat (see DistillNetworkParams).
    return DistillNetworkParams(
        encoder=encoder_params,
        decoder=decoder_params,
        prior=motion_prior_params,
    )


# ---------------------------------------------------------------------------
# Student inference function (for environment rollout)
# ---------------------------------------------------------------------------


def make_student_inference_fn(
    encoder: DistillEncoder,
    decoder: DistillDecoder,
    action_dist: distribution.ParametricDistribution,
    obs_sizes: Mapping[str, int],
    latent_size: int,
):
    """Create a policy factory for environment rollout during DAgger.

    Returns:
        make_policy(params, deterministic) -> policy_fn(obs, key) -> (action, extras)
        where params = (normalizer_params, DistillNetworkParams).
    """

    def make_policy(
        params: tuple[DictRunningStatisticsState, DistillNetworkParams],
        deterministic: bool = False,
    ):
        normalizer_params, network_params = params

        def policy(observations, key_sample):
            # Normalize observations
            norm_obs = normalize_dict_obs(observations, normalizer_params)
            traj_obs = norm_obs["imitation_target"]
            proprio_obs = norm_obs["proprioception"]

            # Encoder: concat(imitation_target, proprioception) -> (mu, logvar)
            encoder_input = jnp.concatenate([traj_obs, proprio_obs], axis=-1)
            latent_mean, latent_logvar = encoder.apply(
                network_params.encoder, encoder_input
            )

            # Sample or use mean
            if deterministic:
                z = latent_mean
            else:
                _, enc_rng = jax.random.split(key_sample)
                z = reparameterize(enc_rng, latent_mean, latent_logvar)

            # Decoder: concat(z, proprioception) -> action_params
            decoder_input = jnp.concatenate([z, proprio_obs], axis=-1)
            logits, _ = decoder.apply(network_params.decoder, decoder_input)

            if deterministic:
                action = action_dist.mode(logits)
            else:
                raw_action = action_dist.sample_no_postprocessing(logits, key_sample)
                action = action_dist.postprocess(raw_action)

            return jnp.array(action), {
                "latent_mean": latent_mean,
                "latent_logvar": latent_logvar,
                "intention": z,
                "logits": logits,
            }

        return policy

    return make_policy


def make_logging_inference_fn(
    encoder: DistillEncoder,
    decoder: DistillDecoder,
    action_dist: distribution.ParametricDistribution,
    obs_sizes: Mapping[str, int],
    latent_size: int,
    motion_prior: DistillMotionPrior | None = None,
):
    """Create a logging policy factory where params are passed explicitly.

    Returns:
        make_logging_policy(deterministic) -> fn(params, obs, key) -> (action, extras)
    """

    def make_logging_policy(deterministic: bool = False):
        def logging_policy(params, observations, key_sample):
            normalizer_params, network_params = params

            norm_obs = normalize_dict_obs(observations, normalizer_params)
            traj_obs = norm_obs["imitation_target"]
            proprio_obs = norm_obs["proprioception"]

            encoder_input = jnp.concatenate([traj_obs, proprio_obs], axis=-1)
            latent_mean, latent_logvar = encoder.apply(
                network_params.encoder, encoder_input
            )

            if deterministic:
                z = latent_mean
            else:
                _, enc_rng = jax.random.split(key_sample)
                z = reparameterize(enc_rng, latent_mean, latent_logvar)

            decoder_input = jnp.concatenate([z, proprio_obs], axis=-1)
            logits, _ = decoder.apply(network_params.decoder, decoder_input)

            if deterministic:
                action = action_dist.mode(logits)
            else:
                raw_action = action_dist.sample_no_postprocessing(logits, key_sample)
                action = action_dist.postprocess(raw_action)

            extras = {
                "latent_mean": latent_mean,
                "latent_logvar": latent_logvar,
                "intention": z,
                "logits": logits,
            }

            if motion_prior is not None:
                # network_params.prior holds the motion-prior params; field
                # name kept as 'prior' for checkpoint-key compat.
                motion_prior_mean, motion_prior_logvar = motion_prior.apply(
                    network_params.prior, proprio_obs
                )
                extras["motion_prior_mean"] = motion_prior_mean
                extras["motion_prior_logvar"] = motion_prior_logvar

            return jnp.array(action), extras

        return logging_policy

    return make_logging_policy


# ---------------------------------------------------------------------------
# Checkpoint compatibility
# ---------------------------------------------------------------------------


def create_compatible_checkpoint_params(
    network_params: DistillNetworkParams,
    normalizer_params: DictRunningStatisticsState,
    obs_sizes: Mapping[str, int],
    latent_size: int,
    encoder_hidden_layer_sizes: Sequence[int],
) -> tuple:
    """Create checkpoint params compatible with task_transfer loading.

    Task transfer's checkpoint_utils.load_prior_checkpoint expects:
    (normalizer_params, {"params": {"encoder": ..., "decoder": ..., "prior": ...}})
    The "prior" key here stores the motion-prior params; the key string is
    kept for checkpoint-compat with task_transfer.

    The encoder in the checkpoint is a dummy (standard-shape intention_network.Encoder)
    since task_transfer only uses motion-prior + decoder.

    Returns:
        (normalizer_params, combined_params) tuple.
    """
    # Create dummy encoder with standard shape (imitation_target only)
    # for checkpoint structural compatibility.
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
            "decoder": network_params.decoder["params"],
            "prior": network_params.prior["params"],
        }
    }

    return (normalizer_params, combined_params)


def create_abstract_distill_policy(
    obs_sizes: Mapping[str, int],
    latent_size: int,
    action_size: int,
    encoder_hidden_layer_sizes: Sequence[int],
    decoder_hidden_layer_sizes: Sequence[int],
    motion_prior_hidden_layer_sizes: Sequence[int],
) -> tuple:
    """Create abstract policy structure for checkpoint restoration.

    This matches the task_transfer compatible format for Orbax.
    """
    reference_obs_size = obs_sizes["imitation_target"]
    proprioceptive_obs_size = obs_sizes["proprioception"]
    action_param_size = action_size * 2

    # Use standard-shape encoder for compatibility
    enc = intention_network.Encoder(
        layer_sizes=list(encoder_hidden_layer_sizes), latents=latent_size
    )
    dec = intention_network.Decoder(
        layer_sizes=list(decoder_hidden_layer_sizes) + [action_param_size]
    )
    motion_pri = OriginalPrior(
        layer_sizes=list(motion_prior_hidden_layer_sizes), latents=latent_size
    )

    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)

    enc_init = enc.init(k1, jnp.zeros((1, reference_obs_size)))
    dec_init = dec.init(k2, jnp.zeros((1, latent_size + proprioceptive_obs_size)))
    motion_pri_init = motion_pri.init(k3, jnp.zeros((1, proprioceptive_obs_size)))

    # Dict key 'prior' kept for checkpoint compat with task_transfer loaders.
    combined_params = {
        "params": {
            "encoder": enc_init["params"],
            "decoder": dec_init["params"],
            "prior": motion_pri_init["params"],
        }
    }

    normalizer_state = DictRunningStatisticsState(
        imitation_target=running_statistics.init_state(
            specs.Array(reference_obs_size, jnp.dtype("float32"))
        ),
        proprioception=running_statistics.init_state(
            specs.Array(proprioceptive_obs_size, jnp.dtype("float32"))
        ),
    )

    return (normalizer_state, combined_params)
