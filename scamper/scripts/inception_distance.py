#!/usr/bin/env python3
"""Compute Frechet Inception Distance (FID) and Kernel Inception Distance (KID).

Trains a VAE on the "real" dataset, then encodes both real and fake datasets
through the encoder to obtain mu vectors. These learned features are used to
compute FID and KID between distributions.

Example usage:
    python -m scamper.scripts.inception_distance \
        --h5-path data/discriminator_dataset.h5 \
        --output-dir results/inception_distance \
        --seeds 0 1 2 3 4
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Set environment variables before importing JAX
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import flax
import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import scipy.linalg
from tqdm import tqdm

from scamper.analysis.discriminator.data_loading import (
    list_h5_datasets,
    load_h5_dataset,
    load_h5_metadata,
)
from scamper.analysis.vae_network import make_vae_network

# Indices of joints that are always zero in the rodent data
ZERO_JOINT_INDICES = [
    18,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    57,
    65,
    73,
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute FID and KID between motion clip datasets using VAE features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--h5-path",
        type=str,
        required=True,
        help="Path to H5 dataset file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory (results + VAE weights)",
    )

    # Dataset selection
    parser.add_argument(
        "--real-dataset",
        type=str,
        default="encoder_decoder_qpos",
        help="Name of the 'real' dataset in H5 file",
    )
    parser.add_argument(
        "--fake-datasets",
        type=str,
        nargs="+",
        default=None,
        help="Names of 'fake' datasets. Auto-discovers if not provided.",
    )

    # VAE architecture
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64,
        help="Dimensionality of VAE latent space",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[512, 256],
        help="Encoder hidden layer sizes (decoder uses reversed)",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.1,
        help="Dropout rate for training",
    )
    parser.add_argument(
        "--no-layer-norm",
        action="store_true",
        help="Disable layer normalization",
    )

    # VAE training
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of VAE training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for VAE training",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Target KL weight (beta-VAE)",
    )
    parser.add_argument(
        "--beta-warmup-epochs",
        type=int,
        default=None,
        help="Number of epochs for linear beta warmup (default: num_epochs // 2)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
    )

    # Seeds & data
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Random seeds for VAE training",
    )
    parser.add_argument(
        "--exclude-root",
        action="store_true",
        help="Exclude root position/orientation (first 7 qpos dims)",
    )
    parser.add_argument(
        "--exclude-zero-joints",
        action="store_true",
        help="Exclude joints that are always zero",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for split-FID/KID computation",
    )

    return parser.parse_args()


def filter_joints(
    data: np.ndarray,
    exclude_root: bool,
    exclude_zero_joints: bool,
) -> np.ndarray:
    """Filter qpos dimensions by excluding root and/or zero joints.

    Args:
        data: Array of shape (..., qpos_dim) with last dim = 74.
        exclude_root: If True, exclude first 7 dims (root pos + quat).
        exclude_zero_joints: If True, exclude always-zero joints.

    Returns:
        Filtered array with reduced last dimension.
    """
    if not exclude_root and not exclude_zero_joints:
        return data

    all_indices = set(range(74))
    indices_to_exclude = set()

    if exclude_root:
        indices_to_exclude.update(range(7))
    if exclude_zero_joints:
        indices_to_exclude.update(ZERO_JOINT_INDICES)

    indices_to_keep = sorted(all_indices - indices_to_exclude)
    return data[..., indices_to_keep]


@flax.struct.dataclass
class VAETrainingState:
    """Training state for VAE."""

    optimizer_state: optax.OptState
    params: Dict
    epoch: int
    step: int


def vae_loss(
    params: dict,
    apply_fn,
    batch: jnp.ndarray,
    rng: jax.Array,
    beta: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute VAE loss: MSE reconstruction + beta * KL divergence."""
    dropout_rng, sample_rng = jax.random.split(rng)
    reconstruction, mu, logvar = apply_fn(
        params, batch, rng=sample_rng, training=True, rngs={"dropout": dropout_rng}
    )

    # Flatten input for reconstruction loss
    flat_batch = batch.reshape(batch.shape[0], -1)
    recon_loss = jnp.mean(jnp.sum((reconstruction - flat_batch) ** 2, axis=-1))

    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * jnp.mean(jnp.sum(1.0 + logvar - mu**2 - jnp.exp(logvar), axis=-1))

    total_loss = recon_loss + beta * kl_loss
    metrics = {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "beta": jnp.array(beta),
    }
    return total_loss, metrics


def create_vae_train_step(apply_fn, optimizer: optax.GradientTransformation):
    """Create JIT-compiled VAE training step."""

    @jax.jit
    def train_step(
        state: VAETrainingState,
        batch: jnp.ndarray,
        rng: jax.Array,
        beta: float,
    ) -> Tuple[VAETrainingState, Dict[str, jnp.ndarray]]:
        (loss, metrics), grads = jax.value_and_grad(vae_loss, has_aux=True)(
            state.params, apply_fn, batch, rng, beta
        )

        updates, new_optimizer_state = optimizer.update(
            grads, state.optimizer_state, state.params
        )
        new_params = optax.apply_updates(state.params, updates)

        new_state = VAETrainingState(
            optimizer_state=new_optimizer_state,
            params=new_params,
            epoch=state.epoch,
            step=state.step + 1,
        )
        return new_state, metrics

    return train_step


def train_vae(
    data: np.ndarray,
    input_size: int,
    latent_dim: int,
    encoder_hidden_layer_sizes: Tuple[int, ...],
    decoder_hidden_layer_sizes: Tuple[int, ...] | None,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    dropout_rate: float,
    use_layer_norm: bool,
    target_beta: float,
    beta_warmup_epochs: int,
    seed: int,
) -> Tuple[dict, dict, Dict]:
    """Train VAE on the provided data.

    Returns:
        Tuple of (trained_params, encode_fn_info, final_metrics) where
        encode_fn_info is (vae_module, init_fn, apply_fn, encode_fn).
    """
    vae, init_fn, apply_fn, encode_fn = make_vae_network(
        input_size=input_size,
        latent_dim=latent_dim,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        dropout_rate=dropout_rate,
        use_layer_norm=use_layer_norm,
    )

    # Initialize
    key = jax.random.PRNGKey(seed)
    init_params = init_fn(key)

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )

    state = VAETrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        epoch=0,
        step=0,
    )

    train_step_fn = create_vae_train_step(apply_fn, optimizer)

    # Flatten data for training: (N, steps, dim) -> (N, steps*dim)
    flat_data = data.reshape(data.shape[0], -1).astype(np.float32)
    n_samples = len(flat_data)

    rng = np.random.default_rng(seed)
    final_metrics = {}

    for epoch in tqdm(range(num_epochs), desc=f"VAE training (seed={seed})"):
        # Linear beta warmup
        if beta_warmup_epochs > 0:
            beta = min(target_beta, target_beta * (epoch + 1) / beta_warmup_epochs)
        else:
            beta = target_beta

        # Shuffle indices
        indices = rng.permutation(n_samples)

        epoch_metrics = []
        key, epoch_key = jax.random.split(key)
        n_batches = n_samples // batch_size
        batch_keys = jax.random.split(epoch_key, max(1, n_batches))

        for i in range(n_batches):
            batch_idx = indices[i * batch_size : (i + 1) * batch_size]
            batch = jnp.array(flat_data[batch_idx])

            state, metrics = train_step_fn(state, batch, batch_keys[i], beta)
            epoch_metrics.append({k: float(v) for k, v in metrics.items()})

        state = state.replace(epoch=epoch + 1)

        if epoch_metrics:
            final_metrics = {
                k: float(np.mean([m[k] for m in epoch_metrics]))
                for k in epoch_metrics[0]
            }

    return state.params, (vae, init_fn, apply_fn, encode_fn), final_metrics


def extract_features(
    params: dict,
    encode_fn,
    data: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    """Extract mu vectors from data using the trained VAE encoder.

    Args:
        params: Trained VAE parameters.
        encode_fn: Function (params, x) -> mu.
        data: Array of shape (N, steps, dim) or (N, flat_dim).
        batch_size: Batch size for encoding.

    Returns:
        Array of mu vectors with shape (N, latent_dim).
    """
    flat_data = data.reshape(data.shape[0], -1).astype(np.float32)
    n_samples = len(flat_data)

    mu_batches = []
    for i in range(0, n_samples, batch_size):
        batch = jnp.array(flat_data[i : i + batch_size])
        mu = encode_fn(params, batch)
        mu_batches.append(np.array(mu))

    return np.concatenate(mu_batches, axis=0)


def compute_fid(mu_real: np.ndarray, mu_fake: np.ndarray) -> float:
    """Compute Frechet Inception Distance between two sets of features.

    FID = ||mu_1 - mu_2||^2 + Tr(Sigma_1 + Sigma_2 - 2*sqrtm(Sigma_1 @ Sigma_2))

    Args:
        mu_real: Feature vectors for real data, shape (N, d).
        mu_fake: Feature vectors for fake data, shape (M, d).

    Returns:
        FID score (lower is better, 0 means identical distributions).
    """
    mu1 = np.mean(mu_real, axis=0)
    mu2 = np.mean(mu_fake, axis=0)
    sigma1 = np.cov(mu_real, rowvar=False)
    sigma2 = np.cov(mu_fake, rowvar=False)

    diff = mu1 - mu2
    mean_diff_sq = np.dot(diff, diff)

    # Compute sqrt of product of covariances
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Handle numerical issues: remove imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = mean_diff_sq + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def compute_kid(
    mu_real: np.ndarray,
    mu_fake: np.ndarray,
    degree: int = 3,
    num_subsets: int = 100,
    subset_size: int | None = None,
) -> Tuple[float, float]:
    """Compute Kernel Inception Distance using polynomial kernel.

    Uses unbiased MMD^2 estimator over random subsets.

    Args:
        mu_real: Feature vectors for real data, shape (N, d).
        mu_fake: Feature vectors for fake data, shape (M, d).
        degree: Degree of polynomial kernel.
        num_subsets: Number of random subsets for estimating mean/std.
        subset_size: Size of each subset. Defaults to min(N, M, 1000).

    Returns:
        Tuple of (KID_mean, KID_std).
    """
    n_real = len(mu_real)
    n_fake = len(mu_fake)
    d = mu_real.shape[1]

    if subset_size is None:
        subset_size = min(n_real, n_fake, 1000)

    rng = np.random.default_rng(42)
    kid_values = []

    for _ in range(num_subsets):
        # Sample subsets
        real_idx = rng.choice(n_real, size=subset_size, replace=(subset_size > n_real))
        fake_idx = rng.choice(n_fake, size=subset_size, replace=(subset_size > n_fake))

        x = mu_real[real_idx]
        y = mu_fake[fake_idx]

        # Polynomial kernel: k(x,y) = (x.y / d + 1)^degree
        kxx = (x @ x.T / d + 1.0) ** degree
        kyy = (y @ y.T / d + 1.0) ** degree
        kxy = (x @ y.T / d + 1.0) ** degree

        # Unbiased MMD^2 estimator
        m = subset_size
        # Remove diagonal for unbiased estimate
        np.fill_diagonal(kxx, 0.0)
        np.fill_diagonal(kyy, 0.0)

        mmd2 = (
            np.sum(kxx) / (m * (m - 1))
            + np.sum(kyy) / (m * (m - 1))
            - 2.0 * np.sum(kxy) / (m * m)
        )
        kid_values.append(float(mmd2))

    return float(np.mean(kid_values)), float(np.std(kid_values))


def create_barplot(
    data: Dict[str, Dict[str, float]],
    metric_name: str,
    reference_value: float | None,
    reference_label: str,
    output_path: str,
) -> None:
    """Create horizontal bar chart for FID or KID results.

    Args:
        data: Dict mapping dataset name to {"mean": ..., "std": ...}.
        metric_name: "FID" or "KID" for labels.
        reference_value: Split baseline value to draw as reference line.
        reference_label: Label for the reference line.
        output_path: Path to save the plot.
    """
    # Sort by mean value
    sorted_items = sorted(data.items(), key=lambda x: x[1]["mean"])
    names = [item[0].replace("_qpos", "") for item in sorted_items]
    means = [item[1]["mean"] for item in sorted_items]
    stds = [item[1]["std"] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5 + 1)))
    y_pos = np.arange(len(names))

    ax.barh(y_pos, means, xerr=stds, align="center", alpha=0.8, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel(metric_name)
    ax.set_title(f"{metric_name} by Dataset")

    if reference_value is not None:
        ax.axvline(
            x=reference_value,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=reference_label,
        )
        ax.legend()

    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    if args.beta_warmup_epochs is None:
        args.beta_warmup_epochs = args.num_epochs // 2

    # Auto-discover fake datasets
    if args.fake_datasets is None:
        all_datasets = list_h5_datasets(args.h5_path)
        args.fake_datasets = [ds for ds in all_datasets if ds != args.real_dataset]
        if not args.fake_datasets:
            raise ValueError(
                f"No fake datasets found. H5 file only contains: {all_datasets}"
            )
        print(f"Auto-discovered {len(args.fake_datasets)} fake datasets")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 70)
    print("INCEPTION DISTANCE (FID + KID) COMPUTATION")
    print("=" * 70)
    print(f"H5 path: {args.h5_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Real dataset: {args.real_dataset}")
    print(f"Fake datasets: {args.fake_datasets}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Encoder hidden layers: {args.hidden_layers}")
    print(f"Decoder hidden layers: {list(reversed(args.hidden_layers))}")
    print(f"Num epochs: {args.num_epochs}")
    print(f"Beta: {args.beta} (warmup: {args.beta_warmup_epochs} epochs)")
    print(f"Seeds: {args.seeds}")
    print(f"Exclude root: {args.exclude_root}")
    print(f"Exclude zero joints: {args.exclude_zero_joints}")
    print("=" * 70)

    # Load real dataset
    print(f"\nLoading real dataset: {args.real_dataset}")
    real_data = load_h5_dataset(args.h5_path, args.real_dataset)
    real_data = filter_joints(real_data, args.exclude_root, args.exclude_zero_joints)
    print(f"  Shape: {real_data.shape}")

    # Load fake datasets
    fake_datasets = {}
    for fake_name in args.fake_datasets:
        print(f"Loading fake dataset: {fake_name}")
        fake_data = load_h5_dataset(args.h5_path, fake_name)
        fake_data = filter_joints(
            fake_data, args.exclude_root, args.exclude_zero_joints
        )
        fake_datasets[fake_name] = fake_data
        print(f"  Shape: {fake_data.shape}")

    # Determine input size
    input_size = int(np.prod(real_data.shape[1:]))
    print(f"\nFlattened input size: {input_size}")

    # Per-seed results
    all_seed_results = []

    for seed in args.seeds:
        print(f"\n{'=' * 60}")
        print(f"SEED {seed}")
        print("=" * 60)

        # Train VAE on real dataset
        start_time = time.time()
        trained_params, network_fns, train_metrics = train_vae(
            data=real_data,
            input_size=input_size,
            latent_dim=args.latent_dim,
            encoder_hidden_layer_sizes=tuple(args.hidden_layers),
            decoder_hidden_layer_sizes=None,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            dropout_rate=args.dropout_rate,
            use_layer_norm=not args.no_layer_norm,
            target_beta=args.beta,
            beta_warmup_epochs=args.beta_warmup_epochs,
            seed=seed,
        )
        train_time = time.time() - start_time
        print(f"VAE training complete ({train_time:.1f}s)")
        print(f"  Final metrics: {train_metrics}")

        _, _, _, encode_fn = network_fns

        # Save VAE weights
        weights_path = output_dir / f"vae_weights_seed_{seed}.msgpack"
        with open(weights_path, "wb") as f:
            f.write(flax.serialization.to_bytes(trained_params))
        print(f"  VAE weights saved to: {weights_path}")

        # Extract features for real dataset
        print("Extracting features...")
        mu_real = extract_features(trained_params, encode_fn, real_data)
        print(f"  Real mu shape: {mu_real.shape}")

        # Self-FID/KID (real vs real, should be ~0)
        self_fid = compute_fid(mu_real, mu_real)
        self_kid_mean, self_kid_std = compute_kid(mu_real, mu_real)
        print(f"  Self-FID: {self_fid:.6f}")
        print(f"  Self-KID: {self_kid_mean:.6f} +/- {self_kid_std:.6f}")

        # Split-FID/KID (random split of real data, noise floor baseline)
        split_rng = np.random.default_rng(args.split_seed)
        split_indices = split_rng.permutation(len(mu_real))
        mid = len(mu_real) // 2
        mu_real_a = mu_real[split_indices[:mid]]
        mu_real_b = mu_real[split_indices[mid : 2 * mid]]

        split_fid = compute_fid(mu_real_a, mu_real_b)
        split_kid_mean, split_kid_std = compute_kid(mu_real_a, mu_real_b)
        print(f"  Split-FID: {split_fid:.6f}")
        print(f"  Split-KID: {split_kid_mean:.6f} +/- {split_kid_std:.6f}")

        # Compute FID/KID for each fake dataset
        seed_results = {
            "seed": seed,
            "train_metrics": train_metrics,
            "train_time": train_time,
            "self_fid": self_fid,
            "self_kid_mean": self_kid_mean,
            "self_kid_std": self_kid_std,
            "split_fid": split_fid,
            "split_kid_mean": split_kid_mean,
            "split_kid_std": split_kid_std,
            "datasets": {},
        }

        for fake_name, fake_data in fake_datasets.items():
            mu_fake = extract_features(trained_params, encode_fn, fake_data)
            fid = compute_fid(mu_real, mu_fake)
            kid_mean, kid_std = compute_kid(mu_real, mu_fake)

            seed_results["datasets"][fake_name] = {
                "fid": fid,
                "kid_mean": kid_mean,
                "kid_std": kid_std,
            }
            print(f"  {fake_name}: FID={fid:.4f}, KID={kid_mean:.6f} +/- {kid_std:.6f}")

        all_seed_results.append(seed_results)

    # Aggregate across seeds
    print(f"\n{'=' * 60}")
    print("AGGREGATING ACROSS SEEDS")
    print("=" * 60)

    aggregated = {}

    # Self metrics
    self_fids = [r["self_fid"] for r in all_seed_results]
    self_kid_means = [r["self_kid_mean"] for r in all_seed_results]
    aggregated["self_comparison"] = {
        "fid_mean": float(np.mean(self_fids)),
        "fid_std": float(np.std(self_fids)),
        "kid_mean": float(np.mean(self_kid_means)),
        "kid_std": float(np.std(self_kid_means)),
    }

    # Split metrics
    split_fids = [r["split_fid"] for r in all_seed_results]
    split_kid_means = [r["split_kid_mean"] for r in all_seed_results]
    aggregated["split_baseline"] = {
        "fid_mean": float(np.mean(split_fids)),
        "fid_std": float(np.std(split_fids)),
        "kid_mean": float(np.mean(split_kid_means)),
        "kid_std": float(np.std(split_kid_means)),
    }

    # Per-dataset metrics
    for fake_name in args.fake_datasets:
        fids = [r["datasets"][fake_name]["fid"] for r in all_seed_results]
        kid_means = [r["datasets"][fake_name]["kid_mean"] for r in all_seed_results]
        aggregated[fake_name] = {
            "fid_mean": float(np.mean(fids)),
            "fid_std": float(np.std(fids)),
            "kid_mean": float(np.mean(kid_means)),
            "kid_std": float(np.std(kid_means)),
        }

    # Save results.json
    output_data = {
        "metadata": {
            "h5_path": args.h5_path,
            "real_dataset": args.real_dataset,
            "fake_datasets": args.fake_datasets,
            "latent_dim": args.latent_dim,
            "hidden_layers": args.hidden_layers,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "beta": args.beta,
            "beta_warmup_epochs": args.beta_warmup_epochs,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "dropout_rate": args.dropout_rate,
            "use_layer_norm": not args.no_layer_norm,
            "seeds": args.seeds,
            "exclude_root": args.exclude_root,
            "exclude_zero_joints": args.exclude_zero_joints,
            "split_seed": args.split_seed,
            "timestamp": datetime.now().isoformat(),
        },
        "per_seed_results": all_seed_results,
        "aggregated": aggregated,
    }

    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Save results.csv
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "fid_mean", "fid_std", "kid_mean", "kid_std"])
        for name, vals in aggregated.items():
            writer.writerow(
                [
                    name,
                    f"{vals['fid_mean']:.6f}",
                    f"{vals['fid_std']:.6f}",
                    f"{vals['kid_mean']:.6f}",
                    f"{vals['kid_std']:.6f}",
                ]
            )
    print(f"CSV saved to: {csv_path}")

    # Create bar plots (only for fake datasets, not self/split)
    fake_plot_data = {
        name: vals
        for name, vals in aggregated.items()
        if name not in ("self_comparison", "split_baseline")
    }

    if fake_plot_data:
        # FID barplot
        fid_data = {
            name: {"mean": vals["fid_mean"], "std": vals["fid_std"]}
            for name, vals in fake_plot_data.items()
        }
        split_fid_ref = aggregated["split_baseline"]["fid_mean"]
        create_barplot(
            fid_data,
            "FID",
            split_fid_ref,
            f"Split baseline ({split_fid_ref:.2f})",
            str(output_dir / "fid_barplot.png"),
        )

        # KID barplot
        kid_data = {
            name: {"mean": vals["kid_mean"], "std": vals["kid_std"]}
            for name, vals in fake_plot_data.items()
        }
        split_kid_ref = aggregated["split_baseline"]["kid_mean"]
        create_barplot(
            kid_data,
            "KID",
            split_kid_ref,
            f"Split baseline ({split_kid_ref:.6f})",
            str(output_dir / "kid_barplot.png"),
        )

    # Print summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<30} {'FID':>14} {'KID':>18}")
    print("-" * 80)
    for name, vals in aggregated.items():
        fid_str = f"{vals['fid_mean']:.4f} +/- {vals['fid_std']:.4f}"
        kid_str = f"{vals['kid_mean']:.6f} +/- {vals['kid_std']:.6f}"
        print(f"{name:<30} {fid_str:>14} {kid_str:>18}")
    print("=" * 80)


if __name__ == "__main__":
    main()
