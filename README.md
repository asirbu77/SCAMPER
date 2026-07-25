# SCAMPER

SCAMPER trains physically-grounded control policies for biomechanical rodent
models by combining motion-capture imitation, hierarchical distillation, and
task-transfer reinforcement learning. The repository implements a four-stage
pipeline — expert PPO → DAgger distillation into a VAE student → behavior-prior
training → task-specific transfer — plus an alternate one-shot mimic-mjx PPO
pipeline that skips distillation. Environments are built on
[MuJoCo](https://github.com/google-deepmind/mujoco) /
[mujoco_playground](https://github.com/google-deepmind/mujoco_playground), with
training accelerated via [Brax](https://github.com/google/brax) PPO on JAX.


## Installation

### Prerequisites

- Python 3.11 or 3.12
- A CUDA 12.x or 13.x toolchain (for GPU training; a CPU-only install also works
  but is impractical for full training runs)

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(Or install via your package manager: `brew install uv`, `pip install uv`,
etc. See the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).)

### 2. Clone and create a virtual environment

```bash
git clone https://github.com/asirbu77/SCAMPER.git
cd SCAMPER
uv venv
```

### 3. Install the package (editable, with CUDA extras)

Pick the CUDA extra that matches your driver:

```bash
# CUDA 13.x
uv pip install -e ".[cuda13]"

# CUDA 12.x
uv pip install -e ".[cuda12]"

# CPU-only
uv pip install -e .
```

The `-e` flag installs in editable mode so changes to the source tree take
effect without reinstalling.

### 4. Activate the venv

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 5. Verify

```bash
python -c "import jax; print('JAX', jax.__version__, 'devices', jax.devices())"
```

GPU installs should list `cuda` or `gpu` devices.


## Download the data

Reference motion-capture clips and pretrained checkpoints live in a
HuggingFace dataset. Pull them into `./data` with:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sirbu/SCAMPER', repo_type='dataset', local_dir='./data')"
```


## Repository layout

```
scamper/
├── train_expert.py              # Stage 1: expert MLP-policy PPO
├── train_distill_expert.py      # Stage 2 (main): DAgger into VAE student
├── train_behavior_prior.py      # Stage 3: behavior-prior on top of distill checkpoint
├── train_mimic_mjx.py           # Alternate stage 2: end-to-end VAE PPO
├── config/                      # Hydra configs (one per train script)
├── agent/
│   ├── expert/                  # Expert MLP networks + PPO
│   ├── imitation/               # VAE intention networks + PPO + losses
│   ├── distill_expert/          # DAgger distillation pieces
│   ├── behavior_prior/          # Behavior-prior networks + losses
│   └── task_transfer/           # arena_forage / maintain_velocity / walk_rear
├── environment/                 # Rodent envs + reference-clip loader
├── scripts/                     # Auxiliary scripts (rollout dataset, IS, ...)
└── utils/                       # Checkpointing, observation utils, gradients,
                                 # network masks, wandb logging, demo helpers
notebooks/
└── scamper_demo.ipynb           # End-to-end demo (rollouts + videos per stage)
```


## Training pipeline

The full SCAMPER pipeline is three sequential stages. Each stage saves an Orbax
checkpoint that the next stage consumes via `teacher_config.checkpoint_path`.

### Stage 1: Expert (MLP-policy PPO)

```bash
python -m scamper.train_expert
```

Trains a plain MLP policy to imitate reference clips via PPO. Saves under
`step_prefix="PPONetwork"`. Override the default save path with
`logging_config.model_path=...`.

### Stage 2: Distill expert (DAgger into a VAE student)

```bash
python -m scamper.train_distill_expert \
    teacher_config.checkpoint_path=/path/to/expert_checkpoint
```

Distills the frozen expert teacher into a VAE student (encoder + decoder +
motion prior) via online DAgger. Saves under `step_prefix="DistillNetwork"`.

### Stage 3: Behavior prior

```bash
python -m scamper.train_behavior_prior \
    teacher_config.checkpoint_path=/path/to/distill_expert_checkpoint
```

Trains a behavior prior on top of the frozen distill_expert encoder / decoder /
motion prior, using a KL loss. Saves under `step_prefix="DistillPriorNetwork"`.

All entry points are Hydra apps; pass arbitrary config overrides on the
command line (e.g., `train_setup.train_config.num_envs=4096`,
`logging_config.exp_name=...`). The Hydra config files are in `scamper/config/`.


## Task transfer

After training (or instead of it, for `scratch`), transfer the policy to a
downstream task by running the corresponding task-transfer entry point.

### Tasks

- `arena_forage` — collect treats scattered in an arena.
- `maintain_velocity` — locomote at a target forward velocity.
- `walk_rear` — walk to a target and rear up onto hind legs.

### Modes (per task)

| Task | `mimic_mjx` | `res_control` | `bp_res_control` | `prior_gating` | `prior_interp` | `scratch` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `arena_forage`     | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `maintain_velocity`| ✓ | ✓ |   |   |   | ✓ |
| `walk_rear`        | ✓ | ✓ |   |   |   | ✓ |

- **`mimic_mjx`** — freeze the decoder; train a fresh task encoder over it
  (input: a mimic-mjx checkpoint).
- **`res_control`** — freeze the motion prior + decoder; train a residual
  encoder on top (input: a distill_expert checkpoint).
- **`bp_res_control`** — freeze the behavior prior (RGM) + decoder; train a
  residual encoder that additively controls the RGM latent, i.e. the
  behavior-prior analogue of `res_control` (input: a behavior_prior checkpoint).
- **`prior_gating`** — freeze the motion prior, behavior prior, and decoder;
  train a residual encoder + a gate that switches between priors (input: a
  behavior_prior checkpoint).
- **`prior_interp`** — freeze the motion prior, behavior prior, and decoder;
  train a residual encoder whose latent is a *fixed* linear blend of both
  priors, `latent = alpha * (prior_mean + residual) + (1 - alpha) * rgm_z`.
  Unlike `prior_gating` the blend weight `alpha` is a constant hyperparameter
  (default 0.5), not learned, so the policy outputs only the residual (input: a
  behavior_prior checkpoint).
- **`scratch`** — train policy and decoder from random initialization; no
  upstream checkpoint needed beyond config-shape inference.

### CLI

```bash
# Generic form
python -m scamper.agent.task_transfer.<task>.train \
    mode=<mode> \
    checkpoint.path=/path/to/upstream_checkpoint \
    [checkpoint.step=<step>]

# Examples
python -m scamper.agent.task_transfer.maintain_velocity.train \
    mode=res_control \
    checkpoint.path=/path/to/distill_expert_checkpoint

python -m scamper.agent.task_transfer.arena_forage.train \
    mode=prior_gating \
    checkpoint.path=/path/to/behavior_prior_checkpoint

python -m scamper.agent.task_transfer.arena_forage.train \
    mode=bp_res_control \
    checkpoint.path=/path/to/behavior_prior_checkpoint

python -m scamper.agent.task_transfer.arena_forage.train \
    mode=prior_interp \
    prior_interp.alpha=0.5 \
    checkpoint.path=/path/to/behavior_prior_checkpoint

python -m scamper.agent.task_transfer.walk_rear.train mode=scratch
```

Per-task Hydra configs live alongside each task at
`scamper/agent/task_transfer/<task>/config.yaml`.


## Mimic-MJX (alternate pipeline)

Skip distillation and train the encoder + decoder end-to-end via PPO:

```bash
python -m scamper.train_mimic_mjx
```

The resulting checkpoint plugs into task transfer's `mimic_mjx` mode in place
of a distill_expert teacher.


## Demo notebook

[`notebooks/scamper_demo.ipynb`](notebooks/scamper_demo.ipynb) walks through
each stage with cached rollouts and rendered videos: imitation grid, behavior
prior, motion prior (low/high noise), and per-task transfer comparisons across
SCAMPER / MIMIC-MJX / scratch checkpoints. It's the fastest way to see what
each pipeline stage produces.


## Third-party data

`data/empirical_rat/df_phases.h5` is redistributed from
[`dannerlab/rat-sci-locomotion`](https://github.com/dannerlab/rat-sci-locomotion)
under the
[Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)
and is associated with Danner et al., *Experimental Neurology* 368, 114496
(2023), doi: [10.1016/j.expneurol.2023.114496](https://doi.org/10.1016/j.expneurol.2023.114496).
See [`data/empirical_rat/README.md`](data/empirical_rat/README.md) for full
attribution. The CC BY-SA 4.0 terms apply to that file (and any adapted
version of it) independently of this repository's Apache-2.0 license.


## License

Distributed under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE) for the full text.

Some files are derived from upstream projects and retain their original
Apache 2.0 copyright notices:

- [Brax](https://github.com/google/brax): `scamper/utils/gradients.py`,
  `scamper/agent/imitation/ppo.py`, `scamper/agent/imitation/losses.py`,
  `scamper/agent/expert/ppo.py`.
- [mujoco_playground](https://github.com/google-deepmind/mujoco_playground)
  (DeepMind): `scamper/utils/domain_randomization.py`.
