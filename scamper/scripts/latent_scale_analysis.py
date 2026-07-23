#!/usr/bin/env python3
"""Latent-scale analysis of motion-prior vs behavior-prior random generation.

Addresses the question of whether the behavioral difference between the two
priors reflects a difference in the *scale* of the latents handed to the
decoder rather than a difference in the underlying distributions.

For every (prior, sampling-noise) condition we measure the latent code that
the decoder actually received, and decompose its variance into a slow
"signal" component carried by the prior mean and a white "injected noise"
component contributed by stochastic sampling. Because the injected noise is
i.i.d. across timesteps while the prior mean varies smoothly, the within-
rollout lag-1 autocovariance isolates the two:

    Var(z_t)                  = Var(mu_t) + sigma^2
    Cov(z_t, z_{t+1})         = Cov(mu_t, mu_{t+1})  ~=  Var(mu_t)
    => sigma^2                ~=  Var(z_t) - Cov(z_t, z_{t+1})

This lets us re-express the diversity results against *realized* latent scale
instead of the nominal sampling noise, i.e. compare the two priors at matched
perturbation magnitude.

Behavioral diversity is recomputed with the same pipeline used for the main
text (egocentric kinematic windows -> shared PCA(50) -> cosine Vendi score
with a clip-level bootstrap), so the numbers here are directly comparable to
the noise-sweep figure.

Outputs (under --output-dir):
  latent_statistics.csv      per-condition latent + behavioral metrics
  latent_scale_analysis.png  multi-panel summary figure
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

# Conditions are keyed by the fixed log-variance used for sampling; the
# deterministic condition (mean only) has no associated sigma.
CONDITIONS: List[Tuple[str, float]] = [
    ("deterministic", float("-inf")),
    ("logvar_-4", -4.0),
    ("logvar_-3", -3.0),
    ("logvar_-2", -2.0),
    ("logvar_-1", -1.0),
    ("logvar_-0.5", -0.5),
    ("logvar_0", 0.0),
]

# The two rollout files use different h5 key prefixes for their prior.
PRIORS: Dict[str, Dict[str, str]] = {
    "Motion Prior": {"file": "generation_rollout_motionprior.h5", "prefix": "prior"},
    "Behavior Prior": {"file": "generation_rollout_behaviorprior.h5", "prefix": "rgm"},
}

WINDOW_LENGTH = 30
WINDOW_STRIDE = 15
PCA_DIM = 50
PCA_FIT_SAMPLES = 50_000
BOOTSTRAP_B = 500
RANDOM_SEED = 0
CTRL_DT = 0.02


def dataset_key(prefix: str, condition: str, kind: str) -> str:
    """Build the h5 dataset name for a (prior, condition, qpos|latents) triple."""
    return f"{prefix}_{condition}_{kind}"


# --------------------------------------------------------------------------
# Latent statistics
# --------------------------------------------------------------------------


def latent_variance_decomposition(z: np.ndarray) -> Dict[str, float]:
    """Split latent variance into slow-signal and white-noise components.

    Args:
        z: Latents of shape (num_rollouts, num_steps, latent_dim).

    Returns:
        Dict of scalar statistics, each averaged over latent dimensions.
    """
    n_dim = z.shape[-1]

    # Pool across rollouts and time for the marginal (total) variance.
    flat = z.reshape(-1, n_dim)
    var_total = flat.var(axis=0)

    # Lag-1 autocovariance, computed within a rollout so the difference never
    # spans a trajectory boundary. Centering uses the per-dimension marginal
    # mean so the covariance is on the same footing as var_total.
    mu = flat.mean(axis=0)
    centered = z - mu
    autocov = (centered[:, :-1, :] * centered[:, 1:, :]).mean(axis=(0, 1))

    # The injected noise is white, so it contributes to var_total but not to
    # the lag-1 autocovariance. Clip at zero: the estimator is unbiased but
    # can go slightly negative in the deterministic condition.
    var_noise = np.maximum(var_total - autocov, 0.0)
    var_signal = np.maximum(autocov, 0.0)

    # One-step latent displacement: the quantity that shows up as visible
    # jitter in the rendered motion.
    step = np.linalg.norm(np.diff(z, axis=1), axis=-1)

    return {
        "latent_norm": float(np.linalg.norm(flat, axis=-1).mean()),
        "std_total": float(np.sqrt(var_total).mean()),
        "std_signal": float(np.sqrt(var_signal).mean()),
        "std_noise": float(np.sqrt(var_noise).mean()),
        # Kept per-dimension so the deterministic condition can be subtracted
        # off as a floor (see floor_correct_noise).
        "var_noise_perdim": var_noise,
        "lag1_autocorr": float((autocov / np.maximum(var_total, 1e-12)).mean()),
        "latent_step": float(step.mean()),
        "participation_ratio": participation_ratio(flat),
    }


def floor_correct_noise(
    var_noise: np.ndarray, var_noise_floor: np.ndarray
) -> float:
    """Subtract the deterministic-condition noise floor from a noise estimate.

    The lag-1 estimator attributes any high-frequency component of the prior
    mean to the "white" bin, so even the deterministic rollouts register a
    small non-zero noise variance. That floor is a property of the prior's own
    mean dynamics, not of injected noise, so we subtract it in quadrature.
    """
    return float(np.sqrt(np.maximum(var_noise - var_noise_floor, 0.0)).mean())


def matched_scale_gap(
    df, x_col: str, y_col: str, n_grid: int = 25
) -> "pd.DataFrame":
    """Interpolate each prior's diversity onto a shared realized-scale axis.

    Rather than rescaling latents off-manifold, we compare the priors at the
    sampling noise levels that produce the *same* realized latent scale, by
    linear interpolation within the range both priors actually cover.
    """
    import pandas as pd

    curves = {}
    for prior, sub in df.groupby("prior"):
        sub = sub.sort_values(x_col)
        curves[prior] = (sub[x_col].to_numpy(), sub[y_col].to_numpy())

    lo = max(x.min() for x, _ in curves.values())
    hi = min(x.max() for x, _ in curves.values())
    grid = np.linspace(lo, hi, n_grid)

    out = {x_col: grid}
    for prior, (x, y) in curves.items():
        out[prior] = np.interp(grid, x, y)
    frame = pd.DataFrame(out)
    frame["gap"] = frame["Behavior Prior"] - frame["Motion Prior"]
    return frame


def participation_ratio(x: np.ndarray) -> float:
    """Effective dimensionality of a point cloud: (sum l)^2 / sum l^2."""
    cov = np.cov(x, rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    eig = eig[eig > 0]
    return float(eig.sum() ** 2 / np.square(eig).sum())


# --------------------------------------------------------------------------
# Behavioral features (matches the main-text diversity pipeline)
# --------------------------------------------------------------------------


def quat_to_euler_wxyz(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a wxyz quaternion to roll, pitch, yaw."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def per_frame_features(qpos: np.ndarray, ctrl_dt: float = CTRL_DT) -> np.ndarray:
    """Egocentric kinematic features: heading- and position-invariant."""
    z = qpos[..., 2:3]
    quat = qpos[..., 3:7]
    joints = qpos[..., 7:]
    roll, pitch, yaw = quat_to_euler_wxyz(quat)
    roll = roll[..., None]
    pitch = pitch[..., None]

    xyz = qpos[..., :3]
    vel_world = np.zeros_like(xyz)
    vel_world[:, 1:-1] = (xyz[:, 2:] - xyz[:, :-2]) / (2.0 * ctrl_dt)
    vel_world[:, 0] = (xyz[:, 1] - xyz[:, 0]) / ctrl_dt
    vel_world[:, -1] = (xyz[:, -1] - xyz[:, -2]) / ctrl_dt
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    vx_body = cos_y * vel_world[..., 0] + sin_y * vel_world[..., 1]
    vy_body = -sin_y * vel_world[..., 0] + cos_y * vel_world[..., 1]
    vel_body = np.stack([vx_body, vy_body, vel_world[..., 2]], axis=-1)

    yaw_unwrapped = np.unwrap(yaw, axis=-1)
    yaw_rate = np.zeros_like(yaw_unwrapped)
    yaw_rate[:, 1:-1] = (yaw_unwrapped[:, 2:] - yaw_unwrapped[:, :-2]) / (2.0 * ctrl_dt)
    yaw_rate[:, 0] = (yaw_unwrapped[:, 1] - yaw_unwrapped[:, 0]) / ctrl_dt
    yaw_rate[:, -1] = (yaw_unwrapped[:, -1] - yaw_unwrapped[:, -2]) / ctrl_dt
    yaw_rate = yaw_rate[..., None]

    return np.concatenate(
        [z, roll, pitch, vel_body, yaw_rate, joints], axis=-1
    ).astype(np.float32)


def make_windows(feats: np.ndarray, length: int, stride: int) -> np.ndarray:
    """Stack frames into flattened sliding windows, one row per window."""
    n_roll, n_steps, n_feat = feats.shape
    starts = np.arange(0, n_steps - length + 1, stride)
    out = np.empty((n_roll, len(starts), length, n_feat), dtype=np.float32)
    for wi, s in enumerate(starts):
        out[:, wi] = feats[:, s : s + length, :]
    return out.reshape(n_roll * len(starts), length * n_feat)


def net_displacement(qpos: np.ndarray) -> np.ndarray:
    """Straight-line distance from first to last frame of each rollout."""
    return np.linalg.norm(qpos[:, -1, :2] - qpos[:, 0, :2], axis=-1)


def fast_vendi_cosine(x: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine Vendi score via the dual (PCA_DIM x PCA_DIM) eigendecomposition."""
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)
    gram = (xn.T @ xn) / x.shape[0]
    eigvals = np.linalg.eigvalsh(gram)
    eigvals = eigvals[eigvals > eps]
    return float(np.exp(-np.sum(eigvals * np.log(eigvals))))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/random_rollouts"),
        help="Directory holding the generation rollout h5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plotting_data/latent_scale"),
        help="Where to write the CSV and figure.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=BOOTSTRAP_B,
        help="Number of clip-level bootstrap iterations for the Vendi CIs.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    conditions = [
        (prior, cond, logvar)
        for prior in PRIORS
        for cond, logvar in CONDITIONS
    ]

    # ---- Pass 1: latent statistics, plus a subsample for the shared PCA ----
    print("Pass 1: latent statistics and PCA fit subsample")
    stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    fit_chunks: List[np.ndarray] = []
    n_per_condition = PCA_FIT_SAMPLES // len(conditions)

    for prior, cond, _ in conditions:
        spec = PRIORS[prior]
        with h5py.File(args.data_dir / spec["file"], "r") as f:
            z = f[dataset_key(spec["prefix"], cond, "latents")][:]
            qpos = f[dataset_key(spec["prefix"], cond, "qpos")][:]

        stats[(prior, cond)] = latent_variance_decomposition(z)
        stats[(prior, cond)]["net_displacement"] = float(
            net_displacement(qpos).mean()
        )

        windows = make_windows(
            per_frame_features(qpos), WINDOW_LENGTH, WINDOW_STRIDE
        )
        take = min(n_per_condition, windows.shape[0])
        fit_chunks.append(windows[rng.choice(windows.shape[0], take, replace=False)])
        print(f"  {prior:15s} {cond:15s} windows={windows.shape}")
        del windows

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x_fit = np.concatenate(fit_chunks, axis=0)
    del fit_chunks
    scaler = StandardScaler().fit(x_fit)
    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_SEED).fit(
        scaler.transform(x_fit)
    )
    print(
        f"  PCA cumulative variance @ {PCA_DIM}: "
        f"{pca.explained_variance_ratio_.sum():.3f}"
    )
    del x_fit

    # ---- Pass 2: project every condition and bootstrap the Vendi score ----
    print("Pass 2: behavioral diversity")
    embeddings: Dict[Tuple[str, str], np.ndarray] = {}
    n_clips = None
    for prior, cond, _ in conditions:
        spec = PRIORS[prior]
        with h5py.File(args.data_dir / spec["file"], "r") as f:
            qpos = f[dataset_key(spec["prefix"], cond, "qpos")][:]
        windows = make_windows(
            per_frame_features(qpos), WINDOW_LENGTH, WINDOW_STRIDE
        )
        emb = pca.transform(scaler.transform(windows)).astype(np.float32)
        if n_clips is None:
            n_clips = qpos.shape[0]
        embeddings[(prior, cond)] = emb.reshape(n_clips, -1, PCA_DIM)
        del windows, emb

    boot_idx = np.random.default_rng(RANDOM_SEED + 1).integers(
        0, n_clips, size=(args.bootstrap, n_clips)
    )
    for key, emb in embeddings.items():
        samples = np.empty(args.bootstrap)
        for b in range(args.bootstrap):
            samples[b] = fast_vendi_cosine(emb[boot_idx[b]].reshape(-1, PCA_DIM))
        stats[key]["vendi"] = float(samples.mean())
        stats[key]["vendi_lo"] = float(np.percentile(samples, 2.5))
        stats[key]["vendi_hi"] = float(np.percentile(samples, 97.5))
        print(
            f"  {key[0]:15s} {key[1]:15s} vendi={samples.mean():6.3f} "
            f"[{np.percentile(samples, 2.5):.3f}, {np.percentile(samples, 97.5):.3f}]"
        )

    # ---- Write results ----
    import pandas as pd

    rows = []
    for prior, cond, logvar in conditions:
        entry = stats[(prior, cond)]
        floor = stats[(prior, "deterministic")]["var_noise_perdim"]
        row = {
            "prior": prior,
            "condition": cond,
            "logvar": logvar,
            "sigma_nominal": 0.0 if np.isinf(logvar) else float(np.exp(0.5 * logvar)),
            "sigma_realized": floor_correct_noise(entry["var_noise_perdim"], floor),
        }
        row.update({k: v for k, v in entry.items() if k != "var_noise_perdim"})
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "latent_statistics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Matched-scale comparisons: the controlled version of the noise sweep.
    stochastic = df[np.isfinite(df["logvar"])]
    for x_col, label in [
        ("std_total", "per-dim latent s.d."),
        ("latent_step", "one-step latent displacement"),
        ("sigma_realized", "realized injected noise"),
    ]:
        matched = matched_scale_gap(stochastic, x_col, "vendi")
        out_path = args.output_dir / f"matched_by_{x_col}.csv"
        matched.to_csv(out_path, index=False)
        print(
            f"\nVendi gap at matched {label} "
            f"[{matched[x_col].min():.3f}, {matched[x_col].max():.3f}]: "
            f"min={matched['gap'].min():.2f} "
            f"mean={matched['gap'].mean():.2f} "
            f"max={matched['gap'].max():.2f}"
        )

    make_figure(df, args.output_dir / "latent_scale_analysis.png")
    print(f"\nWrote {args.output_dir / 'latent_scale_analysis.png'}")


def make_figure(df, path: Path) -> None:
    """Four-panel summary: scale matching, then diversity vs realized scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "Behavior Prior": (0 / 255, 170 / 255, 188 / 255),
        "Motion Prior": (148 / 255, 103 / 255, 189 / 255),
    }
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))

    stochastic = df[np.isfinite(df["logvar"])]

    # (a) Realized injected noise tracks the nominal sigma for both priors.
    ax = axes[0]
    for prior, sub in stochastic.groupby("prior"):
        sub = sub.sort_values("sigma_nominal")
        ax.plot(
            sub["sigma_nominal"], sub["sigma_realized"], "o-", color=colors[prior],
            label=prior,
        )
    lim = [0, stochastic["sigma_nominal"].max() * 1.05]
    ax.plot(lim, lim, "--", color="0.6", lw=1, label="identity")
    ax.set_xlabel(r"nominal $\sigma$")
    ax.set_ylabel(r"realized injected noise $\hat\sigma$")
    ax.set_title("(a) Injected noise is matched")
    ax.legend(frameon=False, fontsize=9)

    # (b) Total latent scale seen by the decoder.
    ax = axes[1]
    for prior, sub in stochastic.groupby("prior"):
        sub = sub.sort_values("sigma_nominal")
        ax.plot(
            sub["sigma_nominal"], sub["std_total"], "o-", color=colors[prior],
            label=f"{prior} (total)",
        )
        ax.plot(
            sub["sigma_nominal"], sub["std_signal"], "s--", color=colors[prior],
            alpha=0.6, label=f"{prior} (signal)",
        )
    ax.set_xlabel(r"nominal $\sigma$")
    ax.set_ylabel("per-dim latent s.d.")
    ax.set_title("(b) Total vs mean-driven scale")
    ax.legend(frameon=False, fontsize=7)

    # (c) The headline re-plot: diversity against realized latent scale.
    ax = axes[2]
    for prior, sub in stochastic.groupby("prior"):
        sub = sub.sort_values("std_total")
        ax.plot(sub["std_total"], sub["vendi"], "o-", color=colors[prior], label=prior)
        ax.fill_between(
            sub["std_total"], sub["vendi_lo"], sub["vendi_hi"],
            color=colors[prior], alpha=0.2, lw=0,
        )
    ax.set_xlabel("per-dim latent s.d. (realized)")
    ax.set_ylabel("Vendi score")
    ax.set_title("(c) Diversity at matched latent scale")
    ax.legend(frameon=False, fontsize=9)

    # (d) Temporal roughness of the latent trajectory.
    ax = axes[3]
    for prior, sub in stochastic.groupby("prior"):
        sub = sub.sort_values("latent_step")
        ax.plot(
            sub["latent_step"], sub["vendi"], "o-", color=colors[prior], label=prior
        )
    ax.set_xlabel(r"mean one-step latent displacement $\|\Delta z\|$")
    ax.set_ylabel("Vendi score")
    ax.set_title("(d) Diversity at matched jitter")
    ax.legend(frameon=False, fontsize=9)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    main()
