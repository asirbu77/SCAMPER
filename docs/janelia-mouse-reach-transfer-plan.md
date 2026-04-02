# Janelia Mouse Forelimb: SCAMPER Transfer to Multi-Target Reach

## Goal

Transfer the Janelia mouse forelimb intention network (trained in vnl-playground on branch `eric/janelia`) into SCAMPER for prior training and downstream multi-target reach task transfer. This document is the implementation guide for the `eric/janelia` branch of SCAMPER.

---

## Source Checkpoint

- **vnl-playground branch**: `eric/janelia`
- **Training script**: `/root/vast/eric/vnl-playground/train_mouse_janelia_intention.py`
- **Checkpoint**: `/root/vast/eric/vnl-playground/checkpoints/janelia-mouse-intention-20260402-141508/150036480`
- **wandb**: `vnl-mjx-rl` project, `janelia-intention-latent-sweep` group

### Verified Checkpoint Dimensions

| Dimension | Size |
|---|---|
| Normalizer (flat) | 24 |
| Proprioception (qpos + qvel) | 8 |
| Task obs (joint deltas + body pos deltas) | 16 |
| Action size (12 muscles x 2 for NormalTanh) | 24 params / 12 actions |
| Latent size | 4 |
| Encoder layers | [512, 512, 512] |
| Decoder layers | [512, 512, 512] (+ output layer 512->24) |

### Encoder Keys
`LayerNorm_0`, `LayerNorm_1`, `LayerNorm_2`, `fc_logvar`, `fc_mean`, `hidden_0`, `hidden_1`, `hidden_2`

### Decoder Keys
`LayerNorm_0`, `LayerNorm_1`, `LayerNorm_2`, `hidden_0` (12->512), `hidden_1` (512->512), `hidden_2` (512->512), `hidden_3` (512->24)

---

## What eric/mouse_arm Already Built

The `eric/mouse_arm` branch implemented the complete 3-stage pipeline. The `eric/janelia` branch should adapt this pipeline for the Janelia model. Here's what exists and what needs to change.

### Stage 1: Checkpoint Conversion (`scamper/scripts/convert_vnl_checkpoint.py`)

**Already works as-is.** The script auto-detects all sizes from checkpoint params. No Janelia-specific changes needed.

**Conversion command:**
```bash
python scamper/scripts/convert_vnl_checkpoint.py \
    /root/vast/eric/vnl-playground/checkpoints/janelia-mouse-intention-20260402-141508/150036480 \
    /root/vast/eric/SCAMPER/model_checkpoints/janelia-intention-converted \
    --output-step 0 --verify
```

**Expected output config:**
```json
{
    "network_config": {
        "arch_name": "intention",
        "obs_sizes": {
            "imitation_target": 16,
            "proprioception": 8
        },
        "action_size": 12,
        "intention_size": 4,
        "encoder_layer_sizes": [512, 512, 512],
        "decoder_layer_sizes": [512, 512, 512]
    }
}
```

**Key renaming:** `fc_mean` -> `fc2_mean`, `fc_logvar` -> `fc2_logvar`

**Normalizer split:** flat 24-dim -> `proprioception` (first 8) + `imitation_target` (last 16)

### Stage 2: Prior Training (`scamper/train_prior.py` + `scamper/config/mouse-prior-jax.yaml`)

**Needs Janelia-specific config updates.** The prior training config must match the Janelia model.

**Key differences from mouse_arm config:**

| Setting | mouse_arm (current) | Janelia (needed) |
|---|---|---|
| `tracked_bodies` | `[scapula, humerus, ulna, radius, wrist_body]` | `[scapula, humerus, ulna, wrist]` |
| `end_effector` | `wrist_body` | `wrist` |
| `clip_length` | 100 | 50 |
| `qvel_init` | `reference` | `zeros` |
| `joints_vel weight` | 0.0 | 0.5 |
| `wrist_pos weight` | 0.0 | 0.1 |
| `bodies_pos weight` | 0.0 | 0.1 |

**Action:** Create `scamper/config/mouse-prior-janelia.yaml` (or update the existing config). Set:

```yaml
teacher_config:
  checkpoint_path: /root/vast/eric/SCAMPER/model_checkpoints/janelia-intention-converted
  checkpoint_step: 0

env_config:
  env_name: "MouseImitation"
  mujoco_impl: "jax"
  solver: "cg"
  iterations: 6
  ls_iterations: 6
  noslip_iterations: 0
  sim_dt: 0.00125
  ctrl_dt: 0.0025
  mocap_hz: 200
  clip_length: 50
  clip_set: "all"
  reference_length: 1
  start_frame_range: [0, 1]
  qvel_init: "zeros"
  keep_clips_idx: null
  tracked_bodies: ["scapula", "humerus", "ulna", "wrist"]
  end_effector: "wrist"
  domain_randomization:
    use_domain_randomization: false
  reward_terms:
    joints:
      exp_scale: 0.2
      weight: 5.0
    joints_vel:
      exp_scale: 0.2
      weight: 0.5
    wrist_pos:
      exp_scale: 0.1
      weight: 0.1
    bodies_pos:
      exp_scale: 0.1
      weight: 0.1
    control_cost:
      weight: 0.01
    control_diff_cost:
      weight: 0.0
  termination_criteria:
    pose_error:
      max_l2_error: 3.0
    nan_termination: {}

walker_config:
  walker_name: "mouse"
  xml_variant: null
  torque_actuators: false
  rescale_factor: 1.0

network_config:
  arch_name: prior
  prior_layer_sizes: [512, 512, 512]

prior_config:
  kl_weight: 1
  grad_clip_norm: 10.0
  use_kl_schedule: false

train_setup:
  checkpoint_to_restore: null
  restore_from_run_state: null
  train_subset_ratio: 0.8
  eval_every: 100_000
  reset_every: 100_000
  train_config:
    num_envs: 4096
    num_timesteps: 5_000_000
    batch_size: 1024
    num_minibatches: 16
    num_updates_per_batch: 4
    learning_rate: 1e-4
    action_repeat: 1
    normalize_observations: true
    unroll_length: 20
    seed: 0
    deterministic_eval: true

logging_config:
  project_name: SCAMPER
  group_name: janelia-prior
  exp_name: null
  clip_id: -1
  model_path: null
```

**Prior training command:**
```bash
python scamper/train_prior.py --config-path=config --config-name=mouse-prior-janelia
```

**What the prior learns:** Given only proprioception (qpos + qvel, 8-dim), predict the latent distribution that the encoder would produce from the full task observation (16-dim). KL(encoder || prior) loss. Only the prior network is trainable; encoder + decoder are frozen.

**Output checkpoint structure:**
```
(DictRunningStatisticsState, {"params": {"encoder": ..., "decoder": ..., "prior": ...}})
```

### Stage 3: Task Transfer — Mouse Reach (`scamper/agent/task_transfer/mouse_reach/`)

**Needs Janelia-specific adaptations.** The MouseReach environment lives in vnl-playground and the task transfer code references it.

#### MouseReach Environment (vnl-playground)

**Location:** `vnl_playground/tasks/mouse/mouse_reach.py`

**Observations:**
```python
obs = OrderedDict(
    task_obs=to_target,          # shape (3,) — direction to target
    proprioception=concat([
        data.qpos,               # shape (4,) — joint angles
        data.qvel,               # shape (4,) — joint velocities
        wrist_pos,               # shape (3,) — wrist body position
    ]),                           # total: (11,)
)
return OrderedDict(state=obs)
```

**Observation sizes for task transfer:**

| Observation | Size | Notes |
|---|---|---|
| task_obs (to_target) | 3 | Direction vector to reach target |
| proprioception (full) | 11 | qpos(4) + qvel(4) + wrist_pos(3) |
| **Policy flat obs** | **14** | task_obs(3) + proprioception(11) |
| decoder_proprio_size | 8 | qpos(4) + qvel(4) — what decoder expects |
| full_proprio_size | 11 | What the reach env provides |

**Critical:** The decoder/prior were trained with 8-dim proprioception (qpos + qvel). The reach env provides 11-dim proprioception (qpos + qvel + wrist_pos). The wrappers handle this by slicing: `proprio[..., :8]` for frozen networks, full 11 for the policy.

**Body naming:** The Janelia XML uses `wrist` not `wrist_body`. The MouseReach env constructs bodies with `-mouse` suffix from `add_mouse()`, so the compiled name is `wrist_body-mouse` in the current code. **Check that the Janelia XML body name `wrist` (which becomes `wrist-mouse` after attachment) is handled correctly in MouseReach.** This may require:
- Updating `self._wrist_body_id` lookup from `wrist_body-mouse` to `wrist-mouse`
- Updating `self._wrist_marker_geom_id` if the geom name differs

#### Task Transfer Config

**Create/update** `scamper/agent/task_transfer/mouse_reach/config.yaml`:

```yaml
mode: "decoder_only"  # Start with decoder_only, try prior_decoder later

checkpoint:
  path: "<path-to-janelia-prior-checkpoint>"
  step: null  # latest

network:
  policy_layers: [512, 512, 512]
  value_layers: [512, 512, 512]

train:
  num_timesteps: 10_000_000_000
  episode_length: 100
  num_envs: 4096
  batch_size: 1024
  num_minibatches: 16
  num_updates_per_batch: 4
  learning_rate: 1e-4
  entropy_cost: 1e-4
  discounting: 0.99
  clipping_epsilon: 0.2
  gae_lambda: 0.95
  max_grad_norm: 50.0
  vf_loss_coefficient: 5.0
  reward_scaling: 1.0
  normalize_observations: true
  unroll_length: 20
  action_repeat: 1
  seed: 0
  eval_every: 100_000_000

env:
  mujoco_impl: "jax"
  target_mode: "random_volume"
  target_size: 0.001
  target_margin: 0.003
  ctrl_cost_weight: 0.01

logging:
  project: "SCAMPER"
  group_name: "janelia_reach_transfer"
  exp_name: "janelia_reach_transfer"
```

#### Task Transfer Training Modes

**Mode 1: `decoder_only`** (recommended start)
- Freeze decoder (from imitation)
- Train new PPO policy that outputs 4-dim latent intentions
- Policy sees: flat obs [task_obs(3) | proprioception(11)] = 14-dim
- Policy outputs: latent intention (4-dim)
- Wrapper: `DecoderHighLevelWrapper` concatenates [latent(4), decoder_proprio(8)] -> decoder -> 12 muscle activations

**Mode 2: `prior_decoder`** (after prior is trained)
- Freeze prior + decoder
- Train PPO policy that outputs 4-dim residual corrections to prior mean
- Final latent = residual + prior_mean (+ optional noise)
- Wrapper: `PriorDecoderHighLevelWrapper`

**Training command:**
```bash
# decoder_only mode
python -m scamper.agent.task_transfer.mouse_reach.train mode=decoder_only \
    checkpoint.path=<prior_checkpoint_path>

# prior_decoder mode
python -m scamper.agent.task_transfer.mouse_reach.train mode=prior_decoder \
    checkpoint.path=<prior_checkpoint_path>
```

---

## Implementation Steps

### Step 1: Convert Checkpoint

```bash
python scamper/scripts/convert_vnl_checkpoint.py \
    /root/vast/eric/vnl-playground/checkpoints/janelia-mouse-intention-20260402-141508/150036480 \
    /root/vast/eric/SCAMPER/model_checkpoints/janelia-intention-converted \
    --output-step 0 --verify
```

Verify output:
- `imitation_target` normalizer shape: (16,)
- `proprioception` normalizer shape: (8,)
- Encoder has `fc2_mean`, `fc2_logvar` (renamed from `fc_mean`, `fc_logvar`)
- Latent size: 4, action size: 12

### Step 2: Verify/Adapt MouseImitation Environment for Janelia

The prior training uses `MouseImitation` env from vnl-playground. Ensure:

1. **Reference data path** points to Janelia clips (46 trials, 50 frames each at 200 Hz)
   - Location: `vnl_playground/tasks/mouse/reference_data/`
2. **Tracked bodies** match: `[scapula, humerus, ulna, wrist]` (not `wrist_body`)
3. **XML** used is `mouse_forelimb_right.xml`
4. **Physics** match Janelia config:
   - sim_dt: 0.00125, ctrl_dt: 0.0025
   - joint_damping: 1e-6, joint_armature: 4e-10
   - force_scale: 1.0 (XML defaults)

### Step 3: Create Janelia Prior Config

Create `scamper/config/mouse-prior-janelia.yaml` with settings from Section "Stage 2" above. Key differences from `mouse-prior-jax.yaml`:
- `tracked_bodies`: `[scapula, humerus, ulna, wrist]`
- `end_effector`: `wrist`
- `clip_length`: 50
- `qvel_init`: `zeros`
- Non-zero reward weights for `joints_vel`, `wrist_pos`, `bodies_pos`

### Step 4: Train Prior Network

```bash
python scamper/train_prior.py --config-path=config --config-name=mouse-prior-janelia \
    teacher_config.checkpoint_path=/root/vast/eric/SCAMPER/model_checkpoints/janelia-intention-converted \
    teacher_config.checkpoint_step=0
```

Monitor on wandb: SCAMPER project, `janelia-prior` group. Watch for KL loss convergence.

### Step 5: Verify/Adapt MouseReach for Janelia Body Names

The `MouseReach` env in `vnl_playground/tasks/mouse/mouse_reach.py` hardcodes body lookups:
```python
self._wrist_body_id = self._mj_model.body("wrist_body-mouse").id
self._wrist_marker_geom_id = self._mj_model.geom("wrist_marker-mouse").id
```

**The Janelia XML has `wrist` not `wrist_body`.** After `add_mouse(suffix="-mouse")`, the body becomes `wrist-mouse`.

**Options:**
1. Make MouseReach body name configurable (cleanest)
2. Add a Janelia-specific subclass
3. Update the XML to use `wrist_body` naming

Check if `wrist_marker` geom exists in the Janelia XML. If not, use `wrist-mouse` body xpos directly instead of the geom.

### Step 6: Adapt Task Transfer Observation Handling

Verify the observation flow works with Janelia dimensions:

1. **MouseReach._get_obs()** returns:
   - `task_obs`: 3-dim (to_target vector)
   - `proprioception`: 11-dim (qpos:4 + qvel:4 + wrist_pos:3)

2. **concat_flat_dict_obs()** in `observation_utils.py` concatenates with proprioception LAST:
   - Result: `[task_obs(3) | proprioception(11)]` = 14-dim flat

3. **DecoderHighLevelWrapper.step()** extracts:
   - `full_proprio = obs[..., -11:]` (full_proprio_size=11)
   - `decoder_proprio = full_proprio[..., :8]` (decoder_proprio_size=8)
   - Passes `[latent(4), decoder_proprio(8)]` to frozen decoder

4. **PPO policy** sees 14-dim flat obs, outputs 4-dim latent

This should work without code changes as long as `full_proprio_size` and `decoder_proprio_size` are set correctly from the checkpoint config.

### Step 7: Run Task Transfer (decoder_only)

```bash
python -m scamper.agent.task_transfer.mouse_reach.train \
    mode=decoder_only \
    checkpoint.path=<janelia-prior-checkpoint-path> \
    logging.group_name=janelia_reach_decoder_only
```

### Step 8: Run Task Transfer (prior_decoder)

```bash
python -m scamper.agent.task_transfer.mouse_reach.train \
    mode=prior_decoder \
    checkpoint.path=<janelia-prior-checkpoint-path> \
    logging.group_name=janelia_reach_prior_decoder
```

---

## Critical Watch-Outs

### 1. Body Name Mismatch
The biggest risk. Janelia XML uses `wrist` while existing SCAMPER code may expect `wrist_body`. Grep for `wrist_body` across all SCAMPER files and update to `wrist` (or make configurable).

### 2. Decoder LayerNorm
The Janelia decoder has 3 LayerNorm layers (`LayerNorm_0/1/2`) — verify the `intention_network.Decoder` in SCAMPER handles this. The `eric/mouse_arm` decoder may or may not have used LayerNorm. Check `activate_final` parameter.

### 3. Reference Data Format
Prior training replays imitation clips through the frozen encoder+decoder. Ensure reference data at `vnl_playground/tasks/mouse/reference_data/` has the Janelia IK results (46 trials x 50 frames, 4 DOF).

### 4. Observation Ordering
vnl-playground sorts observation keys alphabetically for flat concatenation: `[proprioception(8) | task_obs(16)]`. The conversion script splits at `proprio_size=8`. This is different from the task transfer flat ordering which puts `[task_obs | proprioception]`. Both are handled correctly by their respective code paths — just don't mix them up.

### 5. ctrl_dt Consistency
The prior checkpoint stores `ctrl_dt=0.0025` in its config. The task transfer code reads this and sets the env's ctrl_dt to match. Verify this propagation works.

---

## Data Flow Summary

```
vnl-playground (eric/janelia branch)
  train_mouse_janelia_intention.py
  checkpoint: janelia-mouse-intention-20260402-141508/150036480
       |
       | convert_vnl_checkpoint.py (auto-detect sizes)
       v
SCAMPER imitation checkpoint (encoder + decoder, latent_size=4)
  model_checkpoints/janelia-intention-converted/
       |
       | train_prior.py + mouse-prior-janelia.yaml
       | (freeze encoder+decoder, train prior on KL loss)
       v
SCAMPER prior checkpoint (encoder + decoder + prior)
       |
       | task_transfer/mouse_reach/train.py
       |   mode=decoder_only: PPO policy -> latent(4) -> frozen decoder -> ctrl(12)
       |   mode=prior_decoder: PPO policy -> residual(4) + prior_mean -> decoder -> ctrl(12)
       v
Trained reach policy (frozen networks + PPO state)
```

---

## File Reference

| File | Role | Janelia Changes Needed |
|---|---|---|
| `scamper/scripts/convert_vnl_checkpoint.py` | Stage 1: Convert checkpoint | None (auto-detects) |
| `scamper/config/mouse-prior-jax.yaml` | Stage 2: Prior config template | Create Janelia variant |
| `scamper/train_prior.py` | Stage 2: Prior training entry | None |
| `scamper/agent/mlp_prior/prior_networks.py` | Prior network architecture | None |
| `scamper/agent/mlp_prior/prior_train.py` | Prior training loop | None |
| `scamper/agent/mlp_prior/losses.py` | KL loss computation | None |
| `scamper/agent/imitation/intention_network.py` | Encoder/Decoder architecture | Verify LayerNorm compat |
| `scamper/agent/observation_utils.py` | Dict normalizer, flatten utils | None |
| `vnl_playground/tasks/mouse/mouse_reach.py` | MouseReach env | Body name fix (wrist) |
| `scamper/agent/task_transfer/mouse_reach/train.py` | Stage 3: Task transfer entry | Update config path |
| `scamper/agent/task_transfer/mouse_reach/config.yaml` | Stage 3: Task transfer config | Update for Janelia |
| `scamper/agent/task_transfer/mouse_reach/wrappers.py` | Decoder/Prior wrappers | None (generic) |
| `scamper/agent/task_transfer/mouse_reach/checkpoint_utils.py` | Load/save checkpoints | None |
| `scamper/agent/task_transfer/mouse_reach/observation_utils.py` | Flatten/concat obs | None |
