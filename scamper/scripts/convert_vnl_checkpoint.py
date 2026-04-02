"""Convert a vnl-playground mouse intention checkpoint to SCAMPER format.

vnl-playground saves checkpoints via ocp.PyTreeCheckpointer as a flat tuple:
    (normalizer_params, policy_params, value_params)

where:
    - normalizer_params: RunningStatisticsState over flat obs [proprio | task_obs]
    - policy_params: IntentionPolicy params with encoder/decoder submodules
    - value_params: Value network params (discarded)

The encoder uses fc_mean/fc_logvar naming. SCAMPER expects fc2_mean/fc2_logvar.

The flat obs is ordered alphabetically by key: [proprioception | task_obs].
SCAMPER's dict normalizer needs: imitation_target (=task_obs), proprioception.

This script auto-detects all sizes from the checkpoint params — no manual
specification needed. Just point it at input and output paths.
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from brax.training.acme import running_statistics

# Add SCAMPER to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scamper.agent.observation_utils import DictRunningStatisticsState


def load_vnl_checkpoint(checkpoint_path: str):
    """Load a vnl-playground PyTreeCheckpointer checkpoint.

    Returns:
        Tuple of (normalizer_params, policy_params, value_params)
        normalizer_params is a plain dict, not a RunningStatisticsState.
    """
    checkpointer = ocp.PyTreeCheckpointer()
    raw = checkpointer.restore(checkpoint_path)
    normalizer_params, policy_params, value_params = raw
    return normalizer_params, policy_params, value_params


def infer_sizes(normalizer_params, policy_params, value_params):
    """Auto-detect all network sizes from checkpoint params.

    Returns dict with: proprio_size, task_obs_size, action_size, latent_size,
    encoder_layer_sizes, decoder_layer_sizes, value_layer_sizes.
    """
    encoder = policy_params["params"]["encoder"]
    decoder = policy_params["params"]["decoder"]

    # Latent size from encoder output heads (fc_mean or fc2_mean)
    mean_key = "fc_mean" if "fc_mean" in encoder else "fc2_mean"
    latent_size = encoder[mean_key]["kernel"].shape[1]

    # Encoder hidden layer sizes from hidden_* kernels
    encoder_layer_sizes = []
    i = 0
    while f"hidden_{i}" in encoder:
        encoder_layer_sizes.append(encoder[f"hidden_{i}"]["kernel"].shape[1])
        i += 1

    # Decoder hidden layer sizes from hidden_* kernels (last is action output)
    decoder_all_sizes = []
    i = 0
    while f"hidden_{i}" in decoder:
        decoder_all_sizes.append(decoder[f"hidden_{i}"]["kernel"].shape[1])
        i += 1
    # Last hidden_* is the action output layer, not a hidden layer
    decoder_layer_sizes = decoder_all_sizes[:-1]
    action_param_size = decoder_all_sizes[-1]
    action_size = action_param_size // 2  # NormalTanhDistribution: mean + std

    # Decoder input = latent + proprio, first decoder kernel shape[0] tells us
    decoder_input_size = decoder["hidden_0"]["kernel"].shape[0]
    proprio_size = decoder_input_size - latent_size

    # Normalizer total size = proprio + task_obs
    total_obs = normalizer_params["mean"].shape[0]
    task_obs_size = total_obs - proprio_size

    # Value network hidden layer sizes
    value_layer_sizes = []
    if isinstance(value_params, dict) and "params" in value_params:
        vp = value_params["params"]
        i = 0
        while f"hidden_{i}" in vp:
            value_layer_sizes.append(vp[f"hidden_{i}"]["kernel"].shape[1])
            i += 1
    if not value_layer_sizes:
        # Fallback: use encoder layer sizes
        value_layer_sizes = list(encoder_layer_sizes)

    return {
        "proprio_size": proprio_size,
        "task_obs_size": task_obs_size,
        "action_size": action_size,
        "latent_size": latent_size,
        "encoder_layer_sizes": encoder_layer_sizes,
        "decoder_layer_sizes": decoder_layer_sizes,
        "value_layer_sizes": value_layer_sizes,
    }


def remap_encoder_params(encoder_params: dict) -> dict:
    """Remap vnl-playground encoder param names to SCAMPER conventions.

    fc_mean → fc2_mean, fc_logvar → fc2_logvar.
    If already using fc2_* naming, passes through unchanged.
    """
    remapped = {}
    for key, value in encoder_params.items():
        if key == "fc_mean":
            remapped["fc2_mean"] = value
        elif key == "fc_logvar":
            remapped["fc2_logvar"] = value
        else:
            remapped[key] = value
    return remapped


def split_normalizer(
    flat_normalizer: dict,
    proprio_size: int,
    task_obs_size: int,
) -> DictRunningStatisticsState:
    """Split a flat normalizer dict into SCAMPER's dict normalizer.

    vnl-playground flat obs order (alphabetical): [proprioception | task_obs]
    SCAMPER dict normalizer keys: imitation_target, proprioception
    """
    count = flat_normalizer["count"]
    std_eps = flat_normalizer["std_eps"]
    mode = flat_normalizer.get("mode", 0)

    return DictRunningStatisticsState(
        imitation_target=running_statistics.RunningStatisticsState(
            mean=flat_normalizer["mean"][proprio_size:],
            std=flat_normalizer["std"][proprio_size:],
            count=count,
            summed_variance=flat_normalizer["summed_variance"][proprio_size:],
            std_eps=std_eps,
            mode=mode,
        ),
        proprioception=running_statistics.RunningStatisticsState(
            mean=flat_normalizer["mean"][:proprio_size],
            std=flat_normalizer["std"][:proprio_size],
            count=count,
            summed_variance=flat_normalizer["summed_variance"][:proprio_size],
            std_eps=std_eps,
            mode=mode,
        ),
    )


def convert_checkpoint(input_path: str, output_path: str, output_step: int = 0):
    """Convert a vnl-playground checkpoint to SCAMPER format.

    All sizes are auto-detected from the checkpoint params.
    """
    print(f"Loading vnl-playground checkpoint from {input_path}...")
    normalizer_params, policy_params, value_params = load_vnl_checkpoint(input_path)

    # Auto-detect all sizes
    sizes = infer_sizes(normalizer_params, policy_params, value_params)
    print(f"  Auto-detected sizes:")
    for k, v in sizes.items():
        print(f"    {k}: {v}")

    # Extract and remap encoder params
    encoder_params = remap_encoder_params(policy_params["params"]["encoder"])
    decoder_params = policy_params["params"]["decoder"]

    print(f"  Encoder keys: {sorted(encoder_params.keys())}")
    print(f"  Decoder keys: {sorted(decoder_params.keys())}")

    # Split normalizer
    dict_normalizer = split_normalizer(
        normalizer_params, sizes["proprio_size"], sizes["task_obs_size"]
    )
    print(f"  Normalizer imitation_target: {dict_normalizer.imitation_target.mean.shape}")
    print(f"  Normalizer proprioception: {dict_normalizer.proprioception.mean.shape}")

    # Build SCAMPER policy structure: (normalizer, {"params": {"encoder": ..., "decoder": ...}})
    policy = (
        dict_normalizer,
        {"params": {"encoder": encoder_params, "decoder": decoder_params}},
    )

    # Build config
    config = {
        "network_config": {
            "arch_name": "intention",
            "obs_sizes": {
                "imitation_target": sizes["task_obs_size"],
                "proprioception": sizes["proprio_size"],
            },
            "action_size": sizes["action_size"],
            "intention_size": sizes["latent_size"],
            "encoder_layer_sizes": sizes["encoder_layer_sizes"],
            "decoder_layer_sizes": sizes["decoder_layer_sizes"],
            "critic_layer_sizes": sizes["value_layer_sizes"],
        },
    }

    # Save in SCAMPER format
    print(f"Saving SCAMPER checkpoint to {output_path} (step={output_step})...")
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="PPONetwork",
    )
    with ocp.CheckpointManager(output_path, options=mgr_options) as ckpt_mgr:
        ckpt_mgr.save(
            step=output_step,
            args=ocp.args.Composite(
                policy=ocp.args.StandardSave(policy),
                config=ocp.args.JsonSave(config),
            ),
        )
        ckpt_mgr.wait_until_finished()

    print("Conversion complete!")
    print(f"\nTo use in prior training:")
    print(f"  teacher_config.checkpoint_path={output_path}")
    print(f"  teacher_config.checkpoint_step={output_step}")


def verify_converted_checkpoint(checkpoint_path: str, step: int = 0):
    """Verify a converted checkpoint can be loaded by SCAMPER."""
    from scamper.agent.mlp_prior.prior_networks import load_frozen_encoder_decoder

    print(f"\nVerifying converted checkpoint at {checkpoint_path}...")
    encoder_params, decoder_params, normalizer_params, cfg = (
        load_frozen_encoder_decoder(checkpoint_path, step=step)
    )

    obs_sizes = cfg["network_config"]["obs_sizes"]
    print(f"  obs_sizes: {obs_sizes}")
    print(f"  action_size: {cfg['network_config']['action_size']}")
    print(f"  intention_size: {cfg['network_config']['intention_size']}")
    print(f"  encoder_layer_sizes: {cfg['network_config']['encoder_layer_sizes']}")
    print(f"  decoder_layer_sizes: {cfg['network_config']['decoder_layer_sizes']}")
    print(f"  Encoder fc2_mean kernel: {encoder_params['fc2_mean']['kernel'].shape}")
    print(f"  Encoder fc2_logvar kernel: {encoder_params['fc2_logvar']['kernel'].shape}")
    print(f"  Normalizer imitation_target mean: {normalizer_params.imitation_target.mean.shape}")
    print(f"  Normalizer proprioception mean: {normalizer_params.proprioception.mean.shape}")
    print("Verification passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert vnl-playground intention checkpoint to SCAMPER format. "
        "All network sizes are auto-detected from the checkpoint."
    )
    parser.add_argument(
        "input_path",
        help="Path to vnl-playground checkpoint step directory",
    )
    parser.add_argument(
        "output_path",
        help="Output directory for SCAMPER checkpoint",
    )
    parser.add_argument(
        "--output-step", type=int, default=0,
        help="Step number for the output checkpoint (default: 0)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify the converted checkpoint can be loaded by SCAMPER",
    )

    args = parser.parse_args()

    convert_checkpoint(
        input_path=args.input_path,
        output_path=args.output_path,
        output_step=args.output_step,
    )

    if args.verify:
        verify_converted_checkpoint(args.output_path, step=args.output_step)


if __name__ == "__main__":
    main()
