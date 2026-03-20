"""Quick script to load prior checkpoints and compute KL loss to verify which is best."""

import sys
sys.path.insert(0, "/root/vast/eric/SCAMPER")

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from scamper.agent import checkpointing
from scamper.agent.mlp_prior import prior_networks, losses

CKPT_DIR = "/root/vast/eric/SCAMPER/model_checkpoints/260319_053201_537706"
STEPS_TO_CHECK = [2, 3, 4]


def main():
    # Load config from one checkpoint
    cfg = checkpointing.load_config_from_checkpoint(
        CKPT_DIR, step_prefix="PriorNetwork", step=3
    )

    # Extract network sizes from config
    net_cfg = cfg["network_config"]
    latent_size = net_cfg["intention_size"]
    obs_sizes = net_cfg["obs_sizes"]
    reference_obs_size = obs_sizes["imitation_target"]
    proprioceptive_obs_size = obs_sizes["proprioception"]
    encoder_layer_sizes = tuple(net_cfg["encoder_layer_sizes"])
    prior_layer_sizes = tuple(net_cfg["prior_layer_sizes"])

    print(f"Latent size: {latent_size}")
    print(f"Reference obs size: {reference_obs_size}")
    print(f"Proprioceptive obs size: {proprioceptive_obs_size}")
    print(f"Encoder layers: {encoder_layer_sizes}")
    print(f"Prior layers: {prior_layer_sizes}")
    print()

    # Create abstract policy for loading
    abstract_policy = prior_networks.create_abstract_prior_policy(
        cfg, prior_hidden_layer_sizes=prior_layer_sizes
    )

    # Create network apply functions
    encoder_apply_fn = prior_networks.make_encoder_apply_fn(
        encoder_hidden_layer_sizes=encoder_layer_sizes,
        latent_size=latent_size,
        reference_obs_size=reference_obs_size,
    )

    _, prior_network = prior_networks.make_prior_networks(
        latent_size=latent_size,
        proprioceptive_obs_size=proprioceptive_obs_size,
        prior_hidden_layer_sizes=prior_layer_sizes,
    )

    # Generate fixed random observations for consistent comparison
    key = jax.random.PRNGKey(42)
    num_samples = 1024
    key_traj, key_proprio, key_enc = jax.random.split(key, 3)
    dummy_traj_obs = jax.random.normal(key_traj, (num_samples, reference_obs_size))
    dummy_proprio_obs = jax.random.normal(key_proprio, (num_samples, proprioceptive_obs_size))

    # Load each checkpoint and compute KL
    mgr_options = ocp.CheckpointManagerOptions(
        create=False, step_prefix="PriorNetwork"
    )

    for step in STEPS_TO_CHECK:
        with ocp.CheckpointManager(CKPT_DIR, options=mgr_options) as ckpt_mgr:
            policy = ckpt_mgr.restore(
                step,
                args=ocp.args.Composite(
                    policy=ocp.args.StandardRestore(abstract_policy)
                ),
            )["policy"]

        normalizer_params, network_params = policy
        encoder_params = network_params["params"]["encoder"]
        prior_params = network_params["params"]["prior"]

        # Normalize observations using checkpoint's normalizer
        from scamper.agent.observation_utils import normalize_dict_obs
        obs_dict = {
            "imitation_target": dummy_traj_obs,
            "proprioception": dummy_proprio_obs,
        }
        normalized_obs = normalize_dict_obs(obs_dict, normalizer_params)

        traj_obs = normalized_obs["imitation_target"]
        proprio_obs = normalized_obs["proprioception"]

        # Get encoder outputs
        encoder_mean, encoder_logvar = encoder_apply_fn(
            encoder_params, traj_obs, key_enc
        )

        # Get prior outputs - prior_network.apply calls module.apply(params, obs)
        # so params needs {"params": ...} wrapping for flax
        prior_mean, prior_logvar = prior_network.apply({"params": prior_params}, proprio_obs)

        # Compute KL
        kl_loss = losses.compute_encoder_prior_kl_loss(
            encoder_mean, encoder_logvar, prior_mean, prior_logvar
        )

        mean_diff = jnp.mean(jnp.linalg.norm(encoder_mean - prior_mean, axis=-1))
        logvar_diff = jnp.mean(jnp.linalg.norm(encoder_logvar - prior_logvar, axis=-1))

        print(f"PriorNetwork_{step}:")
        print(f"  KL loss:        {float(kl_loss):.6f}")
        print(f"  Mean diff L2:   {float(mean_diff):.6f}")
        print(f"  Logvar diff L2: {float(logvar_diff):.6f}")
        print()


if __name__ == "__main__":
    main()
