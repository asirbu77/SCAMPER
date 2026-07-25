"""Helpers for the SCAMPER demo notebook.

Houses env setup, rollout drivers, rendering, and gait-analysis code that
``notebooks/scamper_demo.ipynb`` calls into. The notebook stays focused on
demo flow; logic lives here.
"""

import functools
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import imageio
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import orbax.checkpoint as ocp
from brax.training import distribution
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_networks
from jax import random
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import blended_transform_factory
from mujoco import mjx
from omegaconf import OmegaConf
from scipy.ndimage import binary_closing
from scipy.signal import medfilt

from scamper.agent.behavior_prior.behavior_prior_networks import (
    create_abstract_behavior_prior_policy as _af_create_abstract_distill_prior_policy,
)
from scamper.agent.distill_expert.teacher import (
    load_frozen_teacher,
    make_teacher_query_fn,
)
from scamper.agent.expert.mlp_network import MLPPolicy
from scamper.agent.task_transfer.arena_forage.checkpoint_utils import (
    _STEP_PREFIX as _AF_STEP_PREFIX,
    _create_abstract_prior_policy_dict as _af_create_abstract_prior_policy_dict,
    load_task_transfer_brax_params as af_load_task_transfer_brax_params,
    load_task_transfer_metadata as af_load_task_transfer_metadata,
    make_behavior_prior_inference_fn as af_make_behavior_prior_inference_fn,
    make_decoder_inference_fn as af_make_decoder_inference_fn,
    make_prior_inference_fn as af_make_motion_prior_inference_fn,
)
from scamper.agent.task_transfer.arena_forage.scratch_networks import (
    make_scratch_ppo_networks as _af_make_scratch_ppo_networks,
)
from scamper.agent.task_transfer.arena_forage.wrappers import (
    DecoderHighLevelWrapper as AFDecoderWrapper,
    FlatObsWrapper as AFFlatObsWrapper,
    PriorDecoderHighLevelWrapper as AFMotionPriorDecoderWrapper,
    PriorRGMDecoderHighLevelWrapper as AFMotionPriorBehaviorPriorDecoderWrapper,
    PriorRGMInterpDecoderHighLevelWrapper as AFMotionPriorBehaviorPriorInterpDecoderWrapper,
    PriorScheduleDecoderHighLevelWrapper as AFMotionPriorBehaviorPriorScheduleDecoderWrapper,
)
from scamper.agent.task_transfer.maintain_velocity.checkpoint_utils import (
    load_task_transfer_brax_params as mv_load_task_transfer_brax_params,
    load_task_transfer_frozen_networks as mv_load_task_transfer_frozen_networks,
    load_task_transfer_metadata as mv_load_task_transfer_metadata,
    make_decoder_inference_fn as mv_make_decoder_inference_fn,
    make_prior_inference_fn as mv_make_prior_inference_fn,
)
from scamper.agent.task_transfer.maintain_velocity.scratch_networks import (
    make_scratch_ppo_networks as mv_make_scratch_ppo_networks,
)
from scamper.agent.task_transfer.maintain_velocity.wrappers import (
    DecoderHighLevelWrapper as MVDecoderWrapper,
    FlatObsWrapper as MVFlatObsWrapper,
    PriorDecoderHighLevelWrapper as MVPriorDecoderWrapper,
)
from scamper.agent.task_transfer.walk_rear.checkpoint_utils import (
    _STEP_PREFIX as _WR_STEP_PREFIX,
    _create_abstract_prior_policy as _wr_create_abstract_prior_policy,
    load_task_transfer_brax_params as wr_load_task_transfer_brax_params,
    load_task_transfer_metadata as wr_load_task_transfer_metadata,
    make_decoder_inference_fn as wr_make_decoder_inference_fn,
    make_prior_inference_fn as wr_make_motion_prior_inference_fn,
)
from scamper.agent.task_transfer.walk_rear.scratch_networks import (
    make_scratch_ppo_networks as _wr_make_scratch_ppo_networks,
)
from scamper.agent.task_transfer.walk_rear.wrappers import (
    DecoderHighLevelWrapper as WRDecoderWrapper,
    FlatObsWrapper as WRFlatObsWrapper,
    PriorDecoderHighLevelWrapper as WRMotionPriorDecoderWrapper,
)
from scamper.environment.rodent import (
    arena_forage as af_env,
    consts as rodent_consts,
    maintain_velocity,
    walk_rear,
)
from scamper.environment.utils import _recolour_tree, dm_scale_spec
from scamper.scripts.collect_rollout_dataset import (
    create_environment as create_distill_motion_prior_env,
    fix_config_paths,
    load_distill_prior_checkpoint_for_rollouts as load_distill_motion_prior_checkpoint_for_rollouts,
)
from scamper.utils import rollout as scamper_rollout
from scamper.utils.observation_utils import (
    DictRunningStatisticsState,
    flatten_obs_dict,
)

mpl.rcParams["font.family"] = "DM Sans"


# Grid layout
GRID_ROWS = 3
GRID_COLS = 3
GRID_SPACING = 0.3  # meters between adjacent grid cells (xy)
GHOST_ALPHA = 0.3  # ghost transparency for the imitation grid
GHOST_COLOR = (0.5, 0.5, 0.5)  # uniform gray for imitation-grid ghosts

# Free-camera defaults for the grid scene
GRID_CAM_AZIMUTH = -135.0
GRID_CAM_ELEVATION = -21.0
GRID_CAM_DISTANCE = 1.0
GRID_CAM_LOOKAT = (0.04, 0.04, 0.0)

# Lighting for the grid scene
GRID_LIGHT_DIR = (0.0, 0.0, -1.0)
GRID_LIGHT_AMBIENT = (0.20, 0.20, 0.20)
GRID_LIGHT_DIFFUSE = (1.2, 1.2, 1.2)
GRID_LIGHT_SPECULAR = (0.3, 0.3, 0.3)

# Render dims
RENDER_HEIGHT = 720
RENDER_WIDTH = 1280
RENDER_FPS = 50

# Walk-rear single-rodent free camera (matches render_videos.ipynb)
WR_CAM_DISTANCE = 1.0
WR_CAM_AZIMUTH = 90.0
WR_CAM_ELEVATION = -15.0
WR_CAM_LOOKAT = (0.46, 0.0, 0.05)

# Maintain-velocity composite render
MV_RENDER_CAMERA = "close_profile-rodent"
MV_RENDER_W, MV_RENDER_H = 1920, 1440  # mujoco render resolution (4:3)
MV_TARGET_W, MV_TARGET_H = 1920, 1440  # composite output resolution
MV_COMPOSITE_DPI = 100
MV_VISIBLE = 120  # frames in the scrolling stance window

# Rodent-specific defaults
REFERENCE_DATA_PATH = "../data/rodent/rodent_reference_clips.h5"
CLIP_LENGTH = 250
RESCALE_FACTOR = 0.9
WALKER_XML = str(rodent_consts.RODENT_NO_TAIL_COLLISION_XML)
ARENA_XML = str(rodent_consts.ARENA_XML_PATH)
SINGLE_RODENT_CAMERA = "close_profile-rodent"


def xml_path_for_variant(xml_variant):
    """Resolve an ``xml_variant`` config string to its rodent XML path."""
    if xml_variant == "limited-collisions":
        return rodent_consts.RODENT_XML_PATH
    if xml_variant == "full-collisions":
        return rodent_consts.RODENT_FULL_COLLISION_XML
    if xml_variant == "no-tail-collisions":
        return rodent_consts.RODENT_NO_TAIL_COLLISION_XML
    raise ValueError(f"Unknown xml_variant: {xml_variant!r}")


# ---- shared helpers ----


def save_rollout_h5(path, qpos_dict, attrs=None, extra_groups=None):
    """Save {clip_idx: ndarray} (or any int-keyed dict) to h5 with file-level attrs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in (attrs or {}).items():
            try:
                f.attrs[k] = v
            except TypeError:
                f.attrs[k] = json.dumps(v)
        for k, arr in qpos_dict.items():
            f.create_dataset(f"clip_{int(k)}", data=np.asarray(arr), compression="gzip")
        for group_name, group_dict in (extra_groups or {}).items():
            grp = f.create_group(group_name)
            for k, arr in group_dict.items():
                grp.create_dataset(
                    f"clip_{int(k)}", data=np.asarray(arr), compression="gzip"
                )
    print(f"saved {len(qpos_dict)} entries to {path}")
    return path


def load_rollout_h5(path, extra_group_names=()):
    """Return ({clip_idx: ndarray}, attrs_dict, {group_name: {clip_idx: ndarray}})."""
    qpos_dict = {}
    attrs = {}
    extras = {name: {} for name in extra_group_names}
    with h5py.File(path, "r") as f:
        for k, v in f.attrs.items():
            attrs[k] = v
        for key in f.keys():
            if key.startswith("clip_"):
                qpos_dict[int(key[len("clip_") :])] = f[key][:]
        for name in extra_group_names:
            if name in f:
                for key in f[name].keys():
                    if key.startswith("clip_"):
                        extras[name][int(key[len("clip_") :])] = f[name][key][:]
    print(f"loaded {len(qpos_dict)} entries from {path}")
    return qpos_dict, attrs, extras


def grid_xy_offsets(rows=GRID_ROWS, cols=GRID_COLS, spacing=GRID_SPACING):
    """xy offsets centered at the origin, row-major."""
    return np.array(
        [
            (
                (c - (cols - 1) / 2.0) * spacing,
                (r - (rows - 1) / 2.0) * spacing,
            )
            for r in range(rows)
            for c in range(cols)
        ],
        dtype=np.float32,
    )


def _strip_lights(spec):
    def visit(body):
        for light in list(body.lights):
            spec.delete(light)
        for child in body.bodies:
            visit(child)

    visit(spec.worldbody)


def build_grid_scene(n_real, n_ghost, ghost_alpha=GHOST_ALPHA):
    """Build an MjModel containing n_real opaque rodents + n_ghost transparent rodents.

    Returns (model, data, nq_single, real_xy, ghost_xy, colors).
    """
    n_total = n_real + n_ghost
    real_xy = grid_xy_offsets()
    ghost_xy = grid_xy_offsets()
    colors = [tuple(cm.tab10(i % 10)) for i in range(max(n_real, n_ghost, 1))]

    spec = mujoco.MjSpec.from_file(ARENA_XML)
    spec.worldbody.add_light(
        name="grid_overhead_light",
        type=int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL),
        dir=list(GRID_LIGHT_DIR),
        ambient=list(GRID_LIGHT_AMBIENT),
        diffuse=list(GRID_LIGHT_DIFFUSE),
        specular=list(GRID_LIGHT_SPECULAR),
        castshadow=1,
    )
    for i in range(n_real):
        rod = mujoco.MjSpec.from_file(WALKER_XML)
        if RESCALE_FACTOR != 1.0:
            rod = dm_scale_spec(rod, RESCALE_FACTOR)
        _strip_lights(rod)
        for body in rod.worldbody.bodies:
            _recolour_tree(body, rgba=[*colors[i][:3], 1.0])
        frame = spec.worldbody.add_frame(
            pos=(float(real_xy[i, 0]), float(real_xy[i, 1]), 0.0)
        )
        body = frame.attach_body(rod.worldbody, "", suffix=f"-r{i}")
        body.add_freejoint()
    for i in range(n_ghost):
        rod = mujoco.MjSpec.from_file(WALKER_XML)
        if RESCALE_FACTOR != 1.0:
            rod = dm_scale_spec(rod, RESCALE_FACTOR)
        _strip_lights(rod)
        for body in rod.worldbody.bodies:
            _recolour_tree(body, rgba=[*GHOST_COLOR, ghost_alpha])
        frame = spec.worldbody.add_frame(
            pos=(float(ghost_xy[i, 0]), float(ghost_xy[i, 1]), 0.0)
        )
        body = frame.attach_body(rod.worldbody, "", suffix=f"-g{i}")
        body.add_freejoint()

    model = spec.compile()
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, RENDER_WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, RENDER_HEIGHT)
    data = mujoco.MjData(model)
    nq_single = model.nq // max(n_total, 1)
    return model, data, nq_single, real_xy, ghost_xy, colors


def recenter_to_grid(qpos_array, grid_xy):
    """Shift each trajectory so its initial xy lands on its grid cell.

    qpos_array: [N, T, nq]. grid_xy: [N, 2].
    Returns a copy with qpos[..., 0:2] adjusted.
    """
    out = qpos_array.copy()
    for i in range(qpos_array.shape[0]):
        start_xy = qpos_array[i, 0, :2]
        out[i, :, 0] += grid_xy[i, 0] - start_xy[0]
        out[i, :, 1] += grid_xy[i, 1] - start_xy[1]
    return out


def render_grid_frames(model, data, nq_single, qpos_real, qpos_ghost=None, cam=None):
    """Render frames of the grid scene given per-rodent qpos sequences.

    qpos_real: [N_real, T, nq]; qpos_ghost: [N_ghost, T, nq] or None.
    Returns list of HxWx3 uint8 frames.
    """
    n_real = qpos_real.shape[0]
    T = qpos_real.shape[1]
    n_ghost = 0 if qpos_ghost is None else qpos_ghost.shape[0]

    if cam is None:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = GRID_CAM_AZIMUTH
        cam.elevation = GRID_CAM_ELEVATION
        cam.distance = GRID_CAM_DISTANCE
        cam.lookat[:] = GRID_CAM_LOOKAT

    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
    frames = []
    try:
        for t in range(T):
            for i in range(n_real):
                data.qpos[i * nq_single : (i + 1) * nq_single] = qpos_real[i, t]
            for i in range(n_ghost):
                data.qpos[(n_real + i) * nq_single : (n_real + i + 1) * nq_single] = (
                    qpos_ghost[i, t]
                )
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    return frames


def pad_frames_for_codec(frames):
    pad = (-frames[0].shape[1]) % 16
    if pad:
        frames = [
            np.pad(f, ((0, 0), (0, pad), (0, 0)), constant_values=255) for f in frames
        ]
    return frames


def write_video(frames, path, fps=RENDER_FPS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = pad_frames_for_codec(frames)
    imageio.mimsave(str(path), frames, fps=fps, quality=8)
    print(f"saved video to {path} ({len(frames)} frames)")
    return path


def stack_grid_qpos(rollout_dict, clip_indices):
    """Stack a {clip_idx: [T, nq]} dict in the order given by clip_indices."""
    return np.stack([np.asarray(rollout_dict[c]) for c in clip_indices], axis=0)


def _infer_teacher_mlp_layer_sizes(restored_params, obs_size_total, action_param_size):
    """Chain 2-D kernel shapes through the restored teacher params.

    Walks 2-D leaves, finds the one whose in-dim equals ``obs_size_total``, then
    follows out-dim → next layer\'s in-dim until the action head. Returns the
    hidden layer out-dims (excluding the final action head).
    """
    kernels = []  # list of (in_dim, out_dim)
    for leaf in jax.tree_util.tree_leaves(restored_params):
        shape = tuple(getattr(leaf, "shape", ()))
        if len(shape) == 2:
            kernels.append((int(shape[0]), int(shape[1])))

    head = next((k for k in kernels if k[0] == obs_size_total), None)
    if head is None:
        raise RuntimeError(
            f"No 2-D kernel with input dim {obs_size_total} found in teacher params; "
            f"saw shapes={kernels}"
        )
    chain = [head]
    remaining = [k for k in kernels if k is not head]
    while True:
        last_out = chain[-1][1]
        nxt = next((k for k in remaining if k[0] == last_out), None)
        if nxt is None:
            break
        chain.append(nxt)
        remaining.remove(nxt)

    if chain[-1][1] != action_param_size:
        raise RuntimeError(
            f"Final teacher layer out-dim {chain[-1][1]} != "
            f"action_param_size {action_param_size}; chain={chain}"
        )
    return tuple(k[1] for k in chain[:-1])


def setup_imitation_env(
    distill_checkpoint,
    distill_step,
    teacher_checkpoint=None,
    teacher_step=None,
    mujoco_impl="warp",
    xml_variant_path=None,
):
    """Load a distill checkpoint, restore its teacher, and build the imitation env.

    By default the teacher is restored from the ``teacher`` slot saved inside the
    distill checkpoint itself; pass ``teacher_checkpoint`` to override with a
    standalone expert checkpoint.

    Returns (im_env, reference_clips, teacher_fn, teacher_normalizer, teacher_cfg, distill_cfg, action_dist).
    """
    distill_checkpoint = str(Path(distill_checkpoint).expanduser().resolve())
    print(f"Reading distill cfg: {distill_checkpoint}")
    options = ocp.CheckpointManagerOptions(create=False, step_prefix="DistillNetwork")
    with ocp.CheckpointManager(distill_checkpoint, options=options) as mgr:
        step = mgr.latest_step() if distill_step is None else distill_step
        cfg_dict = mgr.restore(
            step, args=ocp.args.Composite(config=ocp.args.JsonRestore())
        )["config"]
    distill_cfg = OmegaConf.create(cfg_dict)

    obs_sizes = dict(distill_cfg.network_config.obs_sizes)
    action_size = int(distill_cfg.network_config.action_size)
    action_param_size = action_size * 2
    obs_size_total = int(sum(obs_sizes.values()))

    if teacher_checkpoint is not None:
        print(
            f"Loading frozen teacher (override): {teacher_checkpoint} (step={teacher_step})"
        )
        teacher_fn, teacher_normalizer, teacher_cfg, _ = load_frozen_teacher(
            teacher_checkpoint, step=teacher_step
        )
    else:
        print(f"Loading teacher from distill checkpoint slot")
        with ocp.CheckpointManager(distill_checkpoint, options=options) as mgr:
            # Pass 1: restore raw arrays (no abstract tree) to discover layer sizes.
            raw = mgr.restore(
                step,
                args=ocp.args.Composite(teacher=ocp.args.StandardRestore()),
            )["teacher"]
            mlp_layer_sizes = _infer_teacher_mlp_layer_sizes(
                raw, obs_size_total, action_param_size
            )

            # Pass 2: restore properly using an abstract tree so we get a
            # DictRunningStatisticsState + Flax FrozenDict back.
            mlp_module = MLPPolicy(
                layer_sizes=list(mlp_layer_sizes) + [action_param_size]
            )
            dummy_obs = jnp.zeros((1, obs_size_total))
            init_params = mlp_module.init(jax.random.PRNGKey(0), dummy_obs)
            normalizer_state = DictRunningStatisticsState(
                imitation_target=running_statistics.init_state(
                    specs.Array(obs_sizes["imitation_target"], jnp.dtype("float32"))
                ),
                proprioception=running_statistics.init_state(
                    specs.Array(obs_sizes["proprioception"], jnp.dtype("float32"))
                ),
            )
            abstract_teacher = (normalizer_state, init_params)
            restored = mgr.restore(
                step,
                args=ocp.args.Composite(
                    teacher=ocp.args.StandardRestore(abstract_teacher)
                ),
            )["teacher"]
        teacher_normalizer, teacher_params = restored
        teacher_fn = make_teacher_query_fn(mlp_module, teacher_params)
        teacher_cfg = {
            "network_config": {
                "obs_sizes": obs_sizes,
                "action_size": action_size,
                "mlp_layer_sizes": list(mlp_layer_sizes),
            }
        }
        print(f"  inferred teacher mlp_layer_sizes={mlp_layer_sizes}")

    OmegaConf.set_struct(distill_cfg.env_config, False)
    OmegaConf.update(distill_cfg.env_config, "reference_data_path", REFERENCE_DATA_PATH)
    if mujoco_impl is not None:
        OmegaConf.update(distill_cfg.env_config, "mujoco_impl", mujoco_impl)
    if xml_variant_path is not None:
        OmegaConf.update(
            distill_cfg.env_config, "walker_xml_path", str(xml_variant_path)
        )
    OmegaConf.set_struct(distill_cfg.env_config, True)

    im_env = scamper_rollout.create_environment(distill_cfg)

    # Unwrap to get the reference_clips object.
    unwrapped = im_env
    while hasattr(unwrapped, "_env"):
        unwrapped = unwrapped._env
    reference_clips = unwrapped.reference_clips

    action_dist = distribution.NormalTanhDistribution(event_size=action_size)

    print(
        f"  env: {type(im_env).__name__}; clip_length={distill_cfg.env_config.clip_length}; "
        f"mujoco_impl={distill_cfg.env_config.mujoco_impl}"
    )
    return (
        im_env,
        reference_clips,
        teacher_fn,
        teacher_normalizer,
        teacher_cfg,
        distill_cfg,
        action_dist,
    )


def setup_distill_motion_prior_env(checkpoint, step):
    """Load a distill_motion_prior checkpoint and build its rollout env. Returns
    (env, reference_clips, cfg, policy_params)."""
    checkpoint = str(Path(checkpoint).expanduser().resolve())
    cfg, policy_params = load_distill_motion_prior_checkpoint_for_rollouts(
        checkpoint, step
    )
    OmegaConf.set_struct(cfg.env_config, False)
    OmegaConf.update(cfg.env_config, "mujoco_impl", "warp")
    OmegaConf.update(
        cfg.env_config,
        "walker_xml_path",
        str(rodent_consts.RODENT_NO_TAIL_COLLISION_XML),
    )
    OmegaConf.update(cfg.env_config, "reference_data_path", REFERENCE_DATA_PATH)
    OmegaConf.set_struct(cfg.env_config, True)
    cfg = fix_config_paths(cfg)
    env, reference_clips = create_distill_motion_prior_env(cfg)
    print(f"Loaded distill_motion_prior {Path(checkpoint).name} step={step}")
    return env, reference_clips, cfg, policy_params


def make_clip_initial_state(env, reference_clips, rng_key, clip_idx):
    """Reset env, then overwrite qpos with reference clip's frame-0 qpos and zero qvel."""
    state = env.reset(rng_key)
    qpos0 = reference_clips.qpos[clip_idx, 0]
    qpos0 = jnp.asarray(qpos0)
    data = state.data.replace(qpos=qpos0, qvel=jnp.zeros(env.mjx_model.nv))
    data = mjx.forward(env.mjx_model, data)
    return state.replace(data=data)


def run_proprio_rollout(env, policy_fn, starting_state, num_steps, rng_key):
    """Generic scan-based rollout loop where policy_fn(proprio, key) -> (action, _)."""
    jit_step = jax.jit(env.step)

    def step_fn(carry, _):
        state, key = carry
        key, action_key = random.split(key)
        if isinstance(state.obs, dict) or hasattr(state.obs, "get"):
            proprio = flatten_obs_dict(state.obs)["proprioception"]
        else:
            proprio = state.obs
        action, _ = policy_fn(proprio, action_key)
        next_state = jit_step(state, action)
        return (next_state, key), next_state

    (_, _), all_states = jax.lax.scan(
        step_fn, (starting_state, rng_key), None, length=num_steps
    )
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), all_states)
    return all_states


def rollout_distill_motion_prior_grid(
    env, reference_clips, policy_fn, clip_indices, num_steps, seed=42, chunk_size=None
):
    """vmap'd rollout from each clip's start-pose. Returns {clip_idx: [T, nq]}.

    All clips run in parallel under a single jitted ``jax.vmap`` + ``lax.scan``.
    If memory is tight, set ``chunk_size`` to slice the vmap into batches; pick a
    divisor of ``len(clip_indices)`` so every chunk has the same shape (otherwise
    jax recompiles the trailing chunk).
    """

    def step_fn(carry, _):
        state, key = carry
        key, action_key = random.split(key)
        if isinstance(state.obs, dict) or hasattr(state.obs, "get"):
            proprio = flatten_obs_dict(state.obs)["proprioception"]
        else:
            proprio = state.obs
        action, _ = policy_fn(proprio, action_key)
        next_state = env.step(state, action)
        return (next_state, key), next_state.data.qpos

    def rollout_one(clip_idx, rng):
        rng_init, rng_rollout = random.split(rng)
        start = make_clip_initial_state(env, reference_clips, rng_init, clip_idx)
        (_, _), qpos_seq = jax.lax.scan(
            step_fn, (start, rng_rollout), None, length=num_steps
        )
        return qpos_seq

    rollout_batch = jax.jit(jax.vmap(rollout_one))

    n = len(clip_indices)
    if chunk_size is None or chunk_size >= n:
        chunk_size = n

    clip_idxs_all = jnp.asarray([int(c) for c in clip_indices], dtype=jnp.int32)
    rngs_all = jax.vmap(lambda i: random.PRNGKey(seed + i))(clip_idxs_all)

    qpos_chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        qpos = rollout_batch(clip_idxs_all[start:end], rngs_all[start:end])
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), qpos)
        qpos_chunks.append(np.asarray(qpos))
        print(f"  vmap rollout [{start}:{end}/{n}]: qpos {qpos_chunks[-1].shape}")

    qpos_arr = np.concatenate(qpos_chunks, axis=0)  # [N, T, nq]
    return {int(c): qpos_arr[i] for i, c in enumerate(clip_indices)}


# ---- walk-rear ----


def _wr_load_frozen_networks(checkpoint_path, step, ckpt_cfg):
    """Restore frozen upstream networks from inside a walk_rear task-transfer checkpoint.

    Returns a dict with keys ``encoder_params``, ``prior_params``, ``decoder_params``,
    and ``imitation_normalizer``. Avoids following the (training-machine-local)
    upstream path baked into ``config["checkpoint"]["path"]``.
    """
    abstract_normalizer, abstract_combined = _wr_create_abstract_prior_policy(
        OmegaConf.create(ckpt_cfg)
    )
    abstract_frozen = {
        "encoder_params": abstract_combined["params"]["encoder"],
        "prior_params": abstract_combined["params"]["prior"],
        "decoder_params": abstract_combined["params"]["decoder"],
        "imitation_normalizer": abstract_normalizer,
    }
    mgr_options = ocp.CheckpointManagerOptions(
        create=False, step_prefix=_WR_STEP_PREFIX
    )
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as mgr:
        data = mgr.restore(
            step,
            args=ocp.args.Composite(
                frozen_networks=ocp.args.StandardRestore(abstract_frozen),
            ),
        )
    return data["frozen_networks"]


def _wr_build_env_cfg(config):
    env_cfg = walk_rear.default_config()
    env_cfg.walker_xml_path = xml_path_for_variant(
        config["env"].get("xml_variant", "full-collisions")
    )
    env_cfg.mujoco_impl = "warp"
    env_cfg.ctrl_dt = config["checkpoint_config"]["env_config"]["ctrl_dt"]
    if config.get("mode") == "scratch":
        env_cfg.torque_actuators = config.get("scratch", {}).get(
            "torque_actuators", False
        )
    else:
        env_cfg.torque_actuators = True
    for key in (
        "target_x",
        "target_y",
        "target_z",
        "xy_tolerance",
        "z_tolerance",
        "hold_duration",
        "distance_margin",
    ):
        if key in config["env"]:
            setattr(env_cfg, key, config["env"][key])
    if "reward_weights" in config["env"]:
        env_cfg.reward_terms = {
            name: {"weight": w} for name, w in config["env"]["reward_weights"].items()
        }
    return env_cfg


def setup_walkrear(checkpoint_path, step):
    """Build a wrapped walk_rear env + jit'd inference fn for a task-transfer checkpoint.

    Returns (env, base_env, jit_step, jit_inference_fn, inference_params, start_state, config, mode).
    """
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    info = wr_load_task_transfer_metadata(checkpoint_path, step=step)
    config = info["config"]
    resolved_step = info["step"]
    ckpt_cfg = config["checkpoint_config"]
    mode = config["mode"]
    # Old → new mode-name aliases for checkpoints persisted before the rename.
    # Old checkpoints store mode strings under their old names (`decoder_only`,
    # `distill`, `distill_prior`); new ones use `mimic_mjx` / `res_control` /
    # `prior_gating`. The alias map normalizes both into the new names so the
    # dispatch below only needs to handle current-name strings.
    _MODE_ALIASES = {
        "decoder_only": "mimic_mjx",
        "distill": "res_control",
        "distill_prior": "prior_gating",
    }
    mode = _MODE_ALIASES.get(mode, mode)
    print(f"Walk-rear mode={mode!r}, step={resolved_step}")

    env_cfg = _wr_build_env_cfg(config)
    base_env = walk_rear.WalkRear(config=env_cfg)

    if mode == "res_control":
        frozen = _wr_load_frozen_networks(checkpoint_path, resolved_step, ckpt_cfg)
        motion_prior_p = frozen["prior_params"]
        decoder_p = frozen["decoder_params"]
        normalizer_p = frozen["imitation_normalizer"]
        latent_size = ckpt_cfg["network_config"]["intention_size"]
        proprio_size = ckpt_cfg["network_config"]["obs_sizes"]["proprioception"]
        decoder_fn = wr_make_decoder_inference_fn(decoder_p, normalizer_p, ckpt_cfg)
        motion_prior_fn = wr_make_motion_prior_inference_fn(
            motion_prior_p, normalizer_p, ckpt_cfg
        )
        # Sub-block fallback handles old-name `distill` checkpoints.
        rc_cfg = config.get("res_control", config.get("distill", {}))
        env = WRMotionPriorDecoderWrapper(
            base_env,
            motion_prior_fn,
            decoder_fn,
            latent_size,
            proprio_size,
            deterministic_prior=rc_cfg["deterministic_prior"],
            noise_logvar=rc_cfg["noise_logvar"],
        )
    elif mode == "mimic_mjx":
        frozen = _wr_load_frozen_networks(checkpoint_path, resolved_step, ckpt_cfg)
        decoder_p = frozen["decoder_params"]
        normalizer_p = frozen["imitation_normalizer"]
        latent_size = ckpt_cfg["network_config"]["intention_size"]
        proprio_size = ckpt_cfg["network_config"]["obs_sizes"]["proprioception"]
        decoder_fn = wr_make_decoder_inference_fn(decoder_p, normalizer_p, ckpt_cfg)
        env = WRDecoderWrapper(base_env, decoder_fn, latent_size, proprio_size)
    elif mode == "scratch":
        env = WRFlatObsWrapper(base_env)
    else:
        raise ValueError(f"Unknown walk-rear mode: {mode!r}")

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    start_state = jit_reset(rng)

    if mode == "scratch":
        wr_proprio_size = ckpt_cfg["network_config"]["obs_sizes"]["proprioception"]
        wr_latent_size = ckpt_cfg["network_config"]["intention_size"]
        wr_decoder_layer_sizes = tuple(
            ckpt_cfg["network_config"]["decoder_layer_sizes"]
        )
        wr_task_obs_size = start_state.obs.shape[-1] - wr_proprio_size

        def _wr_scratch_factory(*_args, **_kwargs):
            return _wr_make_scratch_ppo_networks(
                task_obs_size=wr_task_obs_size,
                proprio_size=wr_proprio_size,
                action_size=env.action_size,
                latent_size=wr_latent_size,
                policy_hidden_layer_sizes=tuple(config["network"]["policy_layers"]),
                decoder_hidden_layer_sizes=wr_decoder_layer_sizes,
                value_hidden_layer_sizes=tuple(config["network"]["value_layers"]),
            )

        network_factory = _wr_scratch_factory
        ppo_network = network_factory()
    else:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=tuple(config["network"]["policy_layers"]),
            value_hidden_layer_sizes=tuple(config["network"]["value_layers"]),
        )
        ppo_network = network_factory(
            start_state.obs.shape[-1],
            env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
        )
    key_p, key_v = jax.random.split(jax.random.PRNGKey(0))
    dummy_normalizer = running_statistics.init_state(
        specs.Array(start_state.obs.shape[-1:], jnp.dtype("float32"))
    )
    dummy_policy = ppo_network.policy_network.init(key_p)
    dummy_value = ppo_network.value_network.init(key_v)
    abstract_brax_params = (dummy_normalizer, dummy_policy, dummy_value)
    brax_params = wr_load_task_transfer_brax_params(
        checkpoint_path=checkpoint_path,
        abstract_brax_params=abstract_brax_params,
        step=resolved_step,
    )
    normalizer_params_ppo, policy_params, _ = brax_params
    policy_network = ppo_network.policy_network
    parametric_action_distribution = ppo_network.parametric_action_distribution

    def deterministic_policy(params, observations, key_sample):
        del key_sample
        logits = policy_network.apply(params[0], params[1], observations)
        return parametric_action_distribution.mode(logits), {}

    jit_inference_fn = jax.jit(deterministic_policy)
    inference_params = (normalizer_params_ppo, policy_params)
    return (
        env,
        base_env,
        jit_step,
        jit_inference_fn,
        inference_params,
        start_state,
        config,
        mode,
    )


def task_transfer_rollout(
    start_state, jit_step, jit_inference_fn, inference_params, num_steps, seed
):
    """Single-episode rollout. Returns ndarray qpos[T+1, nq] (clipped at done)."""
    state = start_state
    states = [state]
    act_rng = jax.random.PRNGKey(seed + 1)
    done_step = None
    for t in range(num_steps):
        act_rng, key = jax.random.split(act_rng)
        action, _ = jit_inference_fn(inference_params, state.obs, key)
        state = jit_step(state, action)
        states.append(state)
        if bool(state.done) and done_step is None:
            done_step = t + 1
            print(f"Episode done at step {done_step}")
            break
    qpos = np.stack([np.asarray(s.data.qpos) for s in states], axis=0)
    return qpos, states


def render_single_rodent_qpos(
    base_env,
    qpos_traj,
    camera_name=SINGLE_RODENT_CAMERA,
    height=RENDER_HEIGHT,
    width=RENDER_WIDTH,
):
    """Render a single rodent through qpos_traj using its base mj_model and a named camera."""
    model = base_env.mj_model
    data = mujoco.MjData(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    cam_arg = cam_id if cam_id >= 0 else None
    if cam_arg is None:
        print(f"  (camera {camera_name!r} not found; using free camera)")
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames = []
    try:
        for t in range(qpos_traj.shape[0]):
            data.qpos[:] = qpos_traj[t]
            mujoco.mj_forward(model, data)
            if cam_arg is None:
                renderer.update_scene(data)
            else:
                renderer.update_scene(data, camera=cam_arg)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    return frames


def render_walkrear_qpos(
    base_env,
    qpos_traj,
    distance=WR_CAM_DISTANCE,
    azimuth=WR_CAM_AZIMUTH,
    elevation=WR_CAM_ELEVATION,
    lookat=WR_CAM_LOOKAT,
    height=RENDER_HEIGHT,
    width=RENDER_WIDTH,
):
    """Render walk_rear with a fixed-in-world free camera matching
    notebooks/render_videos.ipynb."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.trackbodyid = -1
    cam.lookat[:] = lookat
    cam.distance = float(distance)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    model = base_env.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames = []
    try:
        for t in range(qpos_traj.shape[0]):
            data.qpos[:] = qpos_traj[t]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    return frames


# ---- arena-forage ----

_AF_RGM_MODES = ("prior_gating", "prior_interp", "prior_schedule")


def _af_load_frozen_networks(checkpoint_path, step, ckpt_cfg, mode):
    """Restore frozen upstream networks from inside an arena_forage task-transfer checkpoint.

    Returns a dict with at least ``encoder_params``, ``prior_params``, ``decoder_params``,
    ``imitation_normalizer``; for modes in ``_AF_RGM_MODES`` also includes ``rgm_params``.
    """
    cfg_omega = OmegaConf.create(ckpt_cfg)
    if mode in _AF_RGM_MODES:
        rgm_layer_sizes = tuple(cfg_omega.network_config.rgm_layer_sizes)
        abstract_normalizer, abstract_combined = (
            _af_create_abstract_distill_prior_policy(ckpt_cfg, rgm_layer_sizes)
        )
        abstract_frozen = {
            "encoder_params": abstract_combined["params"]["encoder"],
            "prior_params": abstract_combined["params"]["prior"],
            "decoder_params": abstract_combined["params"]["decoder"],
            "rgm_params": abstract_combined["params"]["rgm"],
            "imitation_normalizer": abstract_normalizer,
        }
    else:
        abstract_normalizer, abstract_combined = _af_create_abstract_prior_policy_dict(
            cfg_omega
        )
        abstract_frozen = {
            "encoder_params": abstract_combined["params"]["encoder"],
            "prior_params": abstract_combined["params"]["prior"],
            "decoder_params": abstract_combined["params"]["decoder"],
            "imitation_normalizer": abstract_normalizer,
        }
    mgr_options = ocp.CheckpointManagerOptions(
        create=False, step_prefix=_AF_STEP_PREFIX
    )
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as mgr:
        data = mgr.restore(
            step,
            args=ocp.args.Composite(
                frozen_networks=ocp.args.StandardRestore(abstract_frozen),
            ),
        )
    return data["frozen_networks"]


def _af_build_env_cfg(config):
    env_cfg = af_env.default_config()
    env_cfg.walker_xml_path = xml_path_for_variant(
        config["env"].get("xml_variant", "no-tail-collisions")
    )
    env_cfg.mujoco_impl = "warp"
    env_cfg.ctrl_dt = config["checkpoint_config"]["env_config"]["ctrl_dt"]
    env_cfg.num_treats = config["env"]["num_treats"]
    env_cfg.treat_sphere_radius = config["env"]["treat_sphere_radius"]
    env_cfg.treat_height = config["env"]["treat_height"]
    env_cfg.min_spawn_radius = config["env"]["min_spawn_radius"]
    env_cfg.spawn_radius = config["env"]["spawn_radius"]
    env_cfg.scent_sigma = config["env"]["scent_sigma"]
    env_cfg.reward_terms = {
        "treat_collection": {
            "weight": config["env"]["reward_weights"]["treat_collection"]
        },
        "scent_proximity": {
            "weight": config["env"]["reward_weights"]["scent_proximity"]
        },
    }
    if config["mode"] == "scratch":
        env_cfg.torque_actuators = config.get("scratch", {}).get(
            "torque_actuators", False
        )
    else:
        env_cfg.torque_actuators = True
    return env_cfg


def setup_arena_forage(checkpoint_path, step):
    """Build a wrapped arena_forage env + jit'd inference fn.

    Returns (env, base_env, jit_step, jit_inference_fn, inference_params, start_state, config, mode).
    """
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    info = af_load_task_transfer_metadata(checkpoint_path, step=step)
    config = info["config"]
    resolved_step = info["step"]
    ckpt_cfg = config["checkpoint_config"]
    mode = config["mode"]
    # Old → new mode-name aliases for checkpoints persisted before the rename.
    # Old checkpoints store mode strings under their old names (`decoder_only`,
    # `distill`, `distill_prior`); new ones use `mimic_mjx` / `res_control` /
    # `prior_gating`. The alias map normalizes both into the new names so the
    # dispatch below only needs to handle current-name strings.
    _MODE_ALIASES = {
        "decoder_only": "mimic_mjx",
        "distill": "res_control",
        "distill_prior": "prior_gating",
    }
    mode = _MODE_ALIASES.get(mode, mode)
    print(f"Arena-forage mode={mode!r}, step={resolved_step}")

    # Load frozen networks from inside the task-transfer checkpoint itself
    # (avoid following the upstream path baked into config["checkpoint"]["path"]).
    if mode == "scratch":
        encoder_p = motion_prior_p = decoder_p = behavior_prior_p = normalizer_p = None
    else:
        frozen = _af_load_frozen_networks(
            checkpoint_path, resolved_step, ckpt_cfg, mode
        )
        encoder_p = frozen["encoder_params"]
        motion_prior_p = frozen["prior_params"]
        decoder_p = frozen["decoder_params"]
        normalizer_p = frozen["imitation_normalizer"]
        behavior_prior_p = frozen.get("rgm_params")

    latent_size = ckpt_cfg["network_config"]["intention_size"] if ckpt_cfg else None
    proprio_size = (
        ckpt_cfg["network_config"]["obs_sizes"]["proprioception"] if ckpt_cfg else None
    )
    if mode != "scratch":
        decoder_fn = af_make_decoder_inference_fn(decoder_p, normalizer_p, ckpt_cfg)
    else:
        decoder_fn = None

    env_cfg = _af_build_env_cfg(config)
    base_env = af_env.ArenaForage(config=env_cfg)
    base_env._mjx_model = mjx.put_model(base_env._mj_model, impl=env_cfg.mujoco_impl)

    if mode == "mimic_mjx":
        env = AFDecoderWrapper(base_env, decoder_fn, latent_size, proprio_size)
    elif mode == "scratch":
        env = AFFlatObsWrapper(base_env)
    elif mode == "prior_gating":
        motion_prior_fn = af_make_motion_prior_inference_fn(
            motion_prior_p, normalizer_p, ckpt_cfg
        )
        behavior_prior_fn = af_make_behavior_prior_inference_fn(
            behavior_prior_p, normalizer_p, ckpt_cfg
        )
        # Sub-block fallback handles old-name `distill_prior` checkpoints.
        mc = config.get("prior_gating", config.get("distill_prior", {}))
        env = AFMotionPriorBehaviorPriorDecoderWrapper(
            base_env,
            motion_prior_fn,
            behavior_prior_fn,
            decoder_fn,
            latent_size,
            proprio_size,
            gate_bias=mc["gate_bias"],
            gate_margin=mc.get("gate_margin", 0.0),
            rgm_noise=mc["rgm_noise"],
            rgm_logvar=mc.get("rgm_logvar", None),
        )
    elif mode == "prior_schedule":
        motion_prior_fn = af_make_motion_prior_inference_fn(
            motion_prior_p, normalizer_p, ckpt_cfg
        )
        behavior_prior_fn = af_make_behavior_prior_inference_fn(
            behavior_prior_p, normalizer_p, ckpt_cfg
        )
        mc = config.get("prior_schedule", {})
        # Rendering a trained checkpoint means the deployment regime: freeze the
        # schedule (progress_per_step=0) at the end of the ramp.
        env = AFMotionPriorBehaviorPriorScheduleDecoderWrapper(
            base_env,
            motion_prior_fn,
            behavior_prior_fn,
            decoder_fn,
            latent_size,
            proprio_size,
            progress_per_step=0.0,
            initial_progress=1.0,
            p_start=mc["p_start"],
            p_end=mc["p_end"],
            warmup_frac=mc["warmup_frac"],
            ramp_frac=mc["ramp_frac"],
            ramp_shape=mc.get("ramp_shape", "cosine"),
            bout_tau=mc["bout_tau"],
            rgm_noise=mc["rgm_noise"],
            rgm_logvar=mc.get("rgm_logvar", None),
        )
    elif mode == "prior_interp":
        motion_prior_fn = af_make_motion_prior_inference_fn(
            motion_prior_p, normalizer_p, ckpt_cfg
        )
        behavior_prior_fn = af_make_behavior_prior_inference_fn(
            behavior_prior_p, normalizer_p, ckpt_cfg
        )
        mc = config.get("prior_interp", {})
        env = AFMotionPriorBehaviorPriorInterpDecoderWrapper(
            base_env,
            motion_prior_fn,
            behavior_prior_fn,
            decoder_fn,
            latent_size,
            proprio_size,
            alpha=mc["alpha"],
            rgm_noise=mc["rgm_noise"],
            rgm_logvar=mc.get("rgm_logvar", None),
        )
    elif mode == "res_control":
        motion_prior_fn = af_make_motion_prior_inference_fn(
            motion_prior_p, normalizer_p, ckpt_cfg
        )
        # Sub-block fallback handles old-name `distill` checkpoints.
        mc = config.get("res_control", config.get("distill", {}))
        env = AFMotionPriorDecoderWrapper(
            base_env,
            motion_prior_fn,
            decoder_fn,
            latent_size,
            proprio_size,
            deterministic_prior=mc["deterministic_prior"],
            noise_logvar=mc["noise_logvar"],
        )
    else:
        raise ValueError(f"Unhandled arena-forage mode: {mode!r}")

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    start_state = jit_reset(rng)

    if mode == "scratch":

        def _af_scratch_factory(*_args, **_kwargs):
            return _af_make_scratch_ppo_networks(
                obs_size=start_state.obs.shape[-1],
                action_size=env.action_size,
                policy_hidden_layer_sizes=tuple(config["network"]["policy_layers"]),
                value_hidden_layer_sizes=tuple(config["network"]["value_layers"]),
            )

        network_factory = _af_scratch_factory
        ppo_network = network_factory()
    else:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=tuple(config["network"]["policy_layers"]),
            value_hidden_layer_sizes=tuple(config["network"]["value_layers"]),
        )
        ppo_network = network_factory(
            start_state.obs.shape[-1],
            env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
        )
    key_p, key_v = jax.random.split(jax.random.PRNGKey(0))
    dummy_normalizer = running_statistics.init_state(
        specs.Array(start_state.obs.shape[-1:], jnp.dtype("float32"))
    )
    dummy_policy = ppo_network.policy_network.init(key_p)
    dummy_value = ppo_network.value_network.init(key_v)
    abstract_brax_params = (dummy_normalizer, dummy_policy, dummy_value)
    brax_params = af_load_task_transfer_brax_params(
        checkpoint_path=checkpoint_path,
        abstract_brax_params=abstract_brax_params,
        step=resolved_step,
    )
    normalizer_params_ppo, policy_params, _ = brax_params
    policy_network = ppo_network.policy_network
    parametric_action_distribution = ppo_network.parametric_action_distribution

    def deterministic_policy(params, observations, key_sample):
        del key_sample
        logits = policy_network.apply(params[0], params[1], observations)
        return parametric_action_distribution.mode(logits), {}

    jit_inference_fn = jax.jit(deterministic_policy)
    inference_params = (normalizer_params_ppo, policy_params)
    return (
        env,
        base_env,
        jit_step,
        jit_inference_fn,
        inference_params,
        start_state,
        config,
        mode,
    )


def arena_forage_rollout(
    start_state, jit_step, jit_inference_fn, inference_params, num_steps, seed
):
    """Single-episode arena-forage rollout. Captures qpos and treat_collected per step.

    Returns dict with keys:
      qpos:             [T+1, nq]
      treat_positions:  [num_treats, 3]   (from initial state.info)
      treat_collected:  [T+1, num_treats]
      scent_sigma, spawn_radius           (env constants)
    """
    state = start_state
    states = [state]
    act_rng = jax.random.PRNGKey(seed + 1)
    done_step = None
    for t in range(num_steps):
        act_rng, key = jax.random.split(act_rng)
        action, _ = jit_inference_fn(inference_params, state.obs, key)
        state = jit_step(state, action)
        states.append(state)
        if bool(state.done) and done_step is None:
            done_step = t + 1
            print(f"Episode done at step {done_step}")
            break

    qpos = np.stack([np.asarray(s.data.qpos) for s in states], axis=0)
    treat_positions = np.asarray(states[0].info["treat_positions"])
    treat_collected = np.stack(
        [np.asarray(s.info["treat_collected"]) for s in states], axis=0
    )
    return {
        "qpos": qpos,
        "treat_positions": treat_positions,
        "treat_collected": treat_collected.astype(np.bool_),
    }, states


def save_arena_forage_rollout(path, rollout, attrs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in attrs.items():
            try:
                f.attrs[k] = v
            except TypeError:
                f.attrs[k] = json.dumps(v)
        f.create_dataset("qpos", data=np.asarray(rollout["qpos"]), compression="gzip")
        f.create_dataset("treat_positions", data=np.asarray(rollout["treat_positions"]))
        f.create_dataset(
            "treat_collected",
            data=np.asarray(rollout["treat_collected"]),
            compression="gzip",
        )
    print(f"saved arena-forage rollout to {path}")
    return path


def load_arena_forage_rollout(path):
    with h5py.File(path, "r") as f:
        rollout = {
            "qpos": f["qpos"][:],
            "treat_positions": f["treat_positions"][:],
            "treat_collected": f["treat_collected"][:],
        }
        attrs = {k: v for k, v in f.attrs.items()}
    print(f"loaded arena-forage rollout from {path}")
    return rollout, attrs


def render_arena_forage_3panel(
    base_env,
    states,
    mode,
    mode_config=None,
    out_path=None,
    fps=RENDER_FPS,
    sim_camera=SINGLE_RODENT_CAMERA,
    panel_height=480,
    panel_width=640,
    grid_resolution=60,
):
    """Render arena_forage rollout as a side-by-side mp4 mirroring
    notebooks/render_videos.ipynb (cell 44).

    Panels (left to right):
      1. Sim render via base_env.render(states, render_scent=True).
      2. Scent heatmap with growing rodent trajectory and treat collection state.
      3. Gate dynamics (gate_logit + binarized gate). Auto-omitted when no
         gate_logit metric is present (i.e. mimic-mjx / scratch modes).
    """
    import seaborn as sns
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def _round16(x):
        return ((int(x) + 15) // 16) * 16

    env_cfg = base_env._config

    num_treats = env_cfg.num_treats
    sigma = env_cfg.scent_sigma
    extent = env_cfg.spawn_radius * 1.5
    grid_1d = np.linspace(-extent, extent, grid_resolution)
    gx, gy = np.meshgrid(grid_1d, grid_1d)

    treat_pos = np.array(states[0].info["treat_positions"])
    scent_per_treat = np.zeros((num_treats, grid_resolution, grid_resolution))
    for i in range(num_treats):
        d2 = (gx - treat_pos[i, 0]) ** 2 + (gy - treat_pos[i, 1]) ** 2
        scent_per_treat[i] = np.exp(-d2 / (2 * sigma**2))
    vmax = max(float(scent_per_treat.sum(axis=0).max()), 1.0)

    treat_collected_per_frame = np.array(
        [np.array(s.info["treat_collected"]) for s in states]
    )
    rodent_xy = np.array([np.array(s.data.qpos[:2]) for s in states])

    traj_colors = plt.cm.YlOrRd(np.linspace(0.4, 1.0, 256))
    traj_cmap = LinearSegmentedColormap.from_list("YlOrRd_trunc", traj_colors)
    start_color = traj_cmap(0.0)

    print("Rendering sim frames...")
    sim_frames = base_env.render(
        states,
        camera=sim_camera,
        height=panel_height,
        width=panel_width,
        render_scent=True,
    )
    sim_frames = [np.asarray(f) for f in sim_frames]
    n_frames = len(sim_frames)
    print(f"  {n_frames} sim frames at {sim_frames[0].shape}")

    dpi = 100
    hm_fig, hm_ax = plt.subplots(
        figsize=(panel_width / dpi, panel_height / dpi), dpi=dpi
    )
    hm_canvas = FigureCanvasAgg(hm_fig)
    hm_im = hm_ax.imshow(
        scent_per_treat.sum(axis=0),
        extent=[-extent, extent, -extent, extent],
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        aspect="equal",
    )
    (hm_active,) = hm_ax.plot(
        treat_pos[:, 0],
        treat_pos[:, 1],
        "o",
        color="dodgerblue",
        markersize=9,
        markeredgewidth=2,
        linestyle="None",
    )
    (hm_collected,) = hm_ax.plot(
        [],
        [],
        "x",
        color="gray",
        markersize=9,
        markeredgewidth=2,
        linestyle="None",
    )
    hm_lc = LineCollection(
        [],
        cmap=traj_cmap,
        norm=plt.Normalize(0, n_frames),
        linewidths=1.5,
        alpha=0.9,
    )
    hm_ax.add_collection(hm_lc)
    hm_ax.plot(
        [rodent_xy[0, 0]], [rodent_xy[0, 1]], "o", color=start_color, markersize=8
    )
    (hm_end,) = hm_ax.plot([], [], "s", color="red", markersize=8)
    hm_ax.set_xlim(-extent, extent)
    hm_ax.set_ylim(-extent, extent)
    hm_ax.axis("off")
    divider = make_axes_locatable(hm_ax)
    cax = divider.append_axes("right", size="7%", pad=0.15)
    cbar = hm_fig.colorbar(hm_im, cax=cax)
    cbar.set_label("Scent Intensity", fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    hm_fig.tight_layout(pad=0.3)

    gate_logits, gate_values = [], []
    for s in states:
        if hasattr(s, "metrics") and "gate_logit" in s.metrics:
            gate_logits.append(float(np.asarray(s.metrics["gate_logit"])))
            gate_values.append(float(np.asarray(s.metrics["gate"])))
    have_gate = bool(gate_logits)

    gate_fig = gate_canvas = None
    gate_logit_line = gate_step_line = gate_cursor = None
    if have_gate:
        gate_logits = np.array(gate_logits)
        gate_values = np.array(gate_values)
        T_gate = len(gate_logits)
        if mode_config is None:
            mode_config = {}
        gate_bias = float(mode_config.get("gate_bias", 0.0))
        gate_margin = float(mode_config.get("gate_margin", 0.0))

        gate_h = _round16(panel_height)
        gate_w = _round16(panel_height * 1.7)
        gate_fig, gate_ax = plt.subplots(figsize=(gate_w / dpi, gate_h / dpi), dpi=dpi)
        gate_canvas = FigureCanvasAgg(gate_fig)

        scamper_color = tuple(c / 255 for c in (0, 170, 188))
        gate_color = (148 / 255, 103 / 255, 189 / 255)

        (gate_logit_line,) = gate_ax.plot(
            [], [], color=scamper_color, linewidth=2.5, zorder=3
        )
        gate_cursor = gate_ax.axvline(
            0,
            color=tuple(c / 255 for c in (189, 19, 0)),
            linewidth=2,
            alpha=0.9,
            zorder=4,
        )
        gate_ax.axhline(
            gate_bias,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Gate Bias = {gate_bias:.2f}",
            zorder=2,
        )
        if gate_margin > 0:
            gate_ax.axhspan(
                gate_bias - gate_margin,
                gate_bias + gate_margin,
                color="black",
                alpha=0.08,
                zorder=0,
                label=f"Hysteresis: \u00b1{gate_margin:.2f}",
            )
        gx_max = max(T_gate - 1, 1)
        gate_ax.set_xlim(0, gx_max)
        gate_ax.set_ylim(-1, 1)
        gate_ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        gate_ax.set_xlabel("Frame", fontsize=20, labelpad=10)
        gate_ax.set_ylabel(r"Gate Logit ($\beta_t$)", fontsize=20, labelpad=12)
        gate_ax.tick_params(labelsize=16, width=2, length=5)
        gate_ax.tick_params(axis="y", pad=8)
        sns.despine(ax=gate_ax)
        gate_ax.spines["bottom"].set_bounds(0, gx_max)
        gate_ax.spines["left"].set_bounds(-1, 1)
        gate_ax.spines["left"].set_position(("outward", 16))
        for s_name in ("left", "bottom"):
            gate_ax.spines[s_name].set_linewidth(2)
        gate_ax.spines["left"].set_color(scamper_color)
        gate_ax.yaxis.label.set_color(scamper_color)
        gate_ax.tick_params(axis="y", colors=scamper_color)

        gate_ax2 = gate_ax.twinx()
        (gate_step_line,) = gate_ax2.plot(
            [],
            [],
            color=gate_color,
            linewidth=2.5,
            drawstyle="steps-post",
            zorder=3,
        )
        gate_ax2.set_xlim(0, gx_max)
        gate_ax2.set_ylim(-0.05, 1.05)
        gate_ax2.set_yticks([0, 1])
        gate_ax2.set_ylabel(
            r"Binarized Gate ($b_t$)",
            fontsize=20,
            labelpad=12,
            color=gate_color,
        )
        gate_ax2.tick_params(
            axis="y",
            labelsize=16,
            width=2,
            length=5,
            pad=8,
            colors=gate_color,
        )
        gate_ax2.spines["top"].set_visible(False)
        gate_ax2.spines["left"].set_visible(False)
        gate_ax2.spines["right"].set_visible(True)
        gate_ax2.spines["right"].set_color(gate_color)
        gate_ax2.spines["right"].set_linewidth(2)
        gate_ax2.spines["right"].set_bounds(0, 1)
        gate_ax2.spines["right"].set_position(("outward", 16))
        gate_ax2.spines["bottom"].set_visible(False)
        gate_fig.subplots_adjust(top=0.85, left=0.22, right=0.82, bottom=0.20)
        gate_ax.legend(
            fontsize=14,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.45, 1.02),
            borderaxespad=0.0,
            ncol=2,
            columnspacing=1.2,
            handlelength=2.0,
        )
    else:
        print(f"  no gate metrics for mode={mode!r} - third panel will be omitted.")

    print("Compositing panels...")
    combined = []
    for t in range(n_frames):
        collected = treat_collected_per_frame[t]
        active = ~collected
        if active.any():
            hm_im.set_data(scent_per_treat[active].sum(axis=0))
        else:
            hm_im.set_data(np.zeros_like(gx))
        hm_active.set_data(treat_pos[active, 0], treat_pos[active, 1])
        hm_collected.set_data(treat_pos[collected, 0], treat_pos[collected, 1])
        pts = rodent_xy[: t + 1].reshape(-1, 1, 2)
        if len(pts) >= 2:
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            hm_lc.set_segments(segs)
            hm_lc.set_array(np.arange(len(pts)))
        hm_end.set_data([rodent_xy[t, 0]], [rodent_xy[t, 1]])
        hm_canvas.draw()
        hm_frame = np.asarray(hm_canvas.buffer_rgba())[:, :, :3]
        panels = [sim_frames[t], hm_frame]

        if have_gate:
            gt = min(t, len(gate_logits) - 1)
            gate_logit_line.set_data(np.arange(gt + 1), gate_logits[: gt + 1])
            gate_step_line.set_data(np.arange(gt + 1), gate_values[: gt + 1])
            gate_cursor.set_xdata([gt, gt])
            gate_canvas.draw()
            panels.append(np.asarray(gate_canvas.buffer_rgba())[:, :, :3])

        h = min(p.shape[0] for p in panels)
        panels = [p[:h] for p in panels]
        combined.append(np.concatenate(panels, axis=1))

    plt.close(hm_fig)
    if gate_fig is not None:
        plt.close(gate_fig)

    final_w = combined[0].shape[1]
    pad = (-final_w) % 16
    if pad:
        combined = [
            np.pad(f, ((0, 0), (0, pad), (0, 0)), constant_values=255) for f in combined
        ]
    print(f"  built {len(combined)} frames at {combined[0].shape}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(out_path), fps=fps) as writer:
        for f in combined:
            writer.append_data(f)
    print(f"Saved {out_path}")
    return out_path


# ---- maintain-velocity ----


def setup_maintain_velocity(checkpoint_path, step):
    """Returns (env, base_env, jit_reset, jit_step, jit_inference_fn,
    inference_params, config, ckpt_cfg, mode)."""
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    info = mv_load_task_transfer_metadata(checkpoint_path, step=step)
    config = info["config"]
    resolved_step = info["step"]
    ckpt_cfg = config["checkpoint_config"]
    mode = config["mode"]
    # Old → new mode-name aliases for checkpoints persisted before the rename.
    # Old checkpoints store mode strings under their old names (`decoder_only`,
    # `distill`, `distill_prior`); new ones use `mimic_mjx` / `res_control` /
    # `prior_gating`. The alias map normalizes both into the new names so the
    # dispatch below only needs to handle current-name strings.
    _MODE_ALIASES = {
        "decoder_only": "mimic_mjx",
        "distill": "res_control",
        "distill_prior": "prior_gating",
    }
    mode = _MODE_ALIASES.get(mode, mode)

    encoder_p, prior_p, decoder_p, normalizer_p = mv_load_task_transfer_frozen_networks(
        checkpoint_path, ckpt_cfg["network_config"], step=resolved_step
    )
    del encoder_p

    latent_size = ckpt_cfg["network_config"]["intention_size"]
    proprio_size = ckpt_cfg["network_config"]["obs_sizes"]["proprioception"]
    action_size = ckpt_cfg["network_config"]["action_size"]

    env_cfg = maintain_velocity.default_config()
    env_cfg.walker_xml_path = xml_path_for_variant(
        config["env"].get("xml_variant", "full-collisions")
    )
    env_cfg.mujoco_impl = config["env"].get("mujoco_impl", "warp")
    env_cfg.ctrl_dt = ckpt_cfg["env_config"]["ctrl_dt"]
    env_cfg.init_z = config["env"]["init_z"]
    env_cfg.target_speed = config["env"]["target_speed"]
    if mode == "scratch":
        env_cfg.torque_actuators = False

    decoder_fn = mv_make_decoder_inference_fn(decoder_p, normalizer_p, ckpt_cfg)
    base_env = maintain_velocity.MaintainVelocity(config=env_cfg)

    if mode == "mimic_mjx":
        env = MVDecoderWrapper(base_env, decoder_fn, latent_size, proprio_size)
    elif mode == "scratch":
        env = MVFlatObsWrapper(base_env)
    elif mode == "res_control":
        prior_fn = mv_make_prior_inference_fn(prior_p, normalizer_p, ckpt_cfg)
        # Sub-block fallback handles old-name `distill` checkpoints.
        mode_cfg = config.get("res_control", config.get("distill", {}))
        env = MVPriorDecoderWrapper(
            base_env,
            prior_fn,
            decoder_fn,
            latent_size,
            proprio_size,
            deterministic_prior=mode_cfg["deterministic_prior"],
            noise_logvar=mode_cfg["noise_logvar"],
        )
    else:
        raise ValueError(f"Unknown maintain_velocity mode: {mode!r}")

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    start_state = jit_reset(rng)

    if mode == "scratch":
        task_obs_size = start_state.obs.shape[-1] - proprio_size
        decoder_layer_sizes = tuple(ckpt_cfg["network_config"]["decoder_layer_sizes"])

        def network_factory(*_args, **_kwargs):
            return mv_make_scratch_ppo_networks(
                task_obs_size=task_obs_size,
                proprio_size=proprio_size,
                action_size=action_size,
                latent_size=latent_size,
                policy_hidden_layer_sizes=tuple(config["network"]["policy_layers"]),
                decoder_hidden_layer_sizes=decoder_layer_sizes,
                value_hidden_layer_sizes=tuple(config["network"]["value_layers"]),
            )

        ppo_network = network_factory()
    else:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=tuple(config["network"]["policy_layers"]),
            value_hidden_layer_sizes=tuple(config["network"]["value_layers"]),
        )
        ppo_network = network_factory(
            start_state.obs.shape[-1],
            env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
        )

    key_p, key_v = jax.random.split(jax.random.PRNGKey(0))
    dummy_normalizer = running_statistics.init_state(
        specs.Array(start_state.obs.shape[-1:], jnp.dtype("float32"))
    )
    dummy_policy = ppo_network.policy_network.init(key_p)
    dummy_value = ppo_network.value_network.init(key_v)
    abstract_brax_params = (dummy_normalizer, dummy_policy, dummy_value)

    brax_params = mv_load_task_transfer_brax_params(
        checkpoint_path=checkpoint_path,
        abstract_brax_params=abstract_brax_params,
        step=resolved_step,
    )
    normalizer_params_ppo, policy_params, _ = brax_params
    policy_network = ppo_network.policy_network
    parametric_action_distribution = ppo_network.parametric_action_distribution

    def deterministic_policy(params, observations, key_sample):
        del key_sample
        logits = policy_network.apply(params[0], params[1], observations)
        return parametric_action_distribution.mode(logits), {}

    jit_inference_fn = jax.jit(deterministic_policy)
    inference_params = (normalizer_params_ppo, policy_params)
    return (
        env,
        base_env,
        jit_reset,
        jit_step,
        jit_inference_fn,
        inference_params,
        config,
        ckpt_cfg,
        mode,
    )


def maintain_velocity_rollout(
    env, jit_reset, jit_step, jit_inference_fn, inference_params, num_steps, seed
):
    """Roll out one episode and return a dict of stacked arrays + raw state list.

    The arrays mirror what gait_analysis_combined caches per-frame, so the
    downstream gait analysis + composite render code can consume them directly.
    """
    rng = jax.random.PRNGKey(seed)
    reset_rng, act_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)
    rollout = [state]
    for _ in range(num_steps):
        _, act_rng = jax.random.split(act_rng)
        action, _ = jit_inference_fn(inference_params, state.obs, act_rng)
        state = jit_step(state, action)
        rollout.append(state)
    arrays = {
        "qpos": np.array([np.array(s.data.qpos) for s in rollout]),
        "qvel": np.array([np.array(s.data.qvel) for s in rollout]),
        "xpos": np.array([np.array(s.data.xpos) for s in rollout]),
        "sensordata": np.array([np.array(s.data.sensordata) for s in rollout]),
        "obs": np.array([np.array(s.obs) for s in rollout]),
        "reward": np.array([np.array(s.reward) for s in rollout]),
        "done": np.array([np.array(s.done) for s in rollout]),
    }
    return arrays, rollout


def save_mv_rollout_h5(path, arrays, attrs=None):
    """Save dict of named ndarrays at file root with file-level attrs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in (attrs or {}).items():
            try:
                f.attrs[k] = v
            except TypeError:
                f.attrs[k] = json.dumps(v)
        for name, arr in arrays.items():
            f.create_dataset(name, data=np.asarray(arr), compression="gzip")
    print(f"saved mv rollout to {path}")
    return path


def load_mv_rollout_h5(path):
    """Load named ndarrays + attrs from a file written by save_mv_rollout_h5."""
    arrays, attrs = {}, {}
    with h5py.File(path, "r") as f:
        for k, v in f.attrs.items():
            attrs[k] = v
        for name in f.keys():
            arrays[name] = f[name][:]
    return arrays, attrs


# ---- gait analysis ----

MV_LIMB_TO_BODY = {0: 17, 1: 66, 2: 61, 3: 13}


def mv_extract_touch_sensors(sensordata, mj_model):
    """Pull palm/sole touch traces by sensor name (idx-stable across XML edits)."""
    touch = {}
    for i in range(mj_model.nsensor):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        if name and ("palm" in name.lower() or "sole" in name.lower()):
            touch[name] = (mj_model.sensor_adr[i], mj_model.sensor_dim[i])
    palm_l_idx = touch.get("palm_L-rodent", (9, 1))[0]
    palm_r_idx = touch.get("palm_R-rodent", (10, 1))[0]
    sole_l_idx = touch.get("sole_L-rodent", (11, 1))[0]
    sole_r_idx = touch.get("sole_R-rodent", (12, 1))[0]
    palm_l = np.asarray(sensordata)[:, palm_l_idx]
    palm_r = np.asarray(sensordata)[:, palm_r_idx]
    sole_l = np.asarray(sensordata)[:, sole_l_idx]
    sole_r = np.asarray(sensordata)[:, sole_r_idx]
    return palm_l, palm_r, sole_l, sole_r


def mv_compute_contact_barcode(palm_l, palm_r, sole_l, sole_r, cfg):
    palm_l_s = medfilt(palm_l, kernel_size=1)
    palm_r_s = medfilt(palm_r, kernel_size=1)
    sole_l_s = medfilt(sole_l, kernel_size=1)
    sole_r_s = medfilt(sole_r, kernel_size=1)
    if cfg.get("spike_removal_sensor"):
        sensor_map = {
            "palm_L": palm_l_s,
            "palm_R": palm_r_s,
            "sole_L": sole_l_s,
            "sole_R": sole_r_s,
        }
        target = sensor_map[cfg["spike_removal_sensor"]]
        above = (target > cfg["spike_threshold"]).astype(int)
        diffs = np.diff(np.concatenate(([0], above, [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) <= cfg["spike_max_len"]:
                target[s:e] = 0.0
    stance_palm_l = (palm_l_s > cfg["touch_threshold_palm_l"]).astype(int)
    stance_palm_r = (palm_r_s > cfg["touch_threshold_palm_r"]).astype(int)
    stance_sole_l = (sole_l_s > cfg["touch_threshold_sole_l"]).astype(int)
    stance_sole_r = (sole_r_s > cfg["touch_threshold_sole_r"]).astype(int)
    structure = np.ones(cfg["min_stance_gap"])
    stance_palm_l = binary_closing(stance_palm_l, structure=structure).astype(int)
    stance_palm_r = binary_closing(stance_palm_r, structure=structure).astype(int)
    stance_sole_l = binary_closing(stance_sole_l, structure=structure).astype(int)
    stance_sole_r = binary_closing(stance_sole_r, structure=structure).astype(int)
    # Stack: [RH, RF, LF, LH]
    return np.stack(
        [stance_sole_r, stance_palm_r, stance_palm_l, stance_sole_l], axis=0
    )


# ---- maintain-velocity composite render ----


def _mv_rollout_states_from_arrays(env, arrays):
    """Reconstruct a list of SimpleNamespace states the wrapper's .render expects."""
    jit_reset = jax.jit(env.reset)
    ref_state = jit_reset(jax.random.PRNGKey(99))
    default_mocap_pos = np.array(ref_state.data.mocap_pos)
    default_mocap_quat = np.array(ref_state.data.mocap_quat)
    default_xfrc = np.zeros_like(np.array(ref_state.data.xfrc_applied))
    states = []
    n = arrays["qpos"].shape[0]
    for i in range(n):
        d = SimpleNamespace(
            qpos=arrays["qpos"][i],
            qvel=arrays["qvel"][i],
            xpos=arrays["xpos"][i],
            sensordata=arrays["sensordata"][i],
            mocap_pos=default_mocap_pos,
            mocap_quat=default_mocap_quat,
            xfrc_applied=default_xfrc,
        )
        states.append(
            SimpleNamespace(
                data=d,
                obs=arrays["obs"][i],
                reward=arrays["reward"][i],
                done=arrays["done"][i],
            )
        )
    return states


def render_mv_composite(
    env,
    arrays,
    contact_barcode,
    out_path,
    render_w=MV_RENDER_W,
    render_h=MV_RENDER_H,
    target_w=MV_TARGET_W,
    target_h=MV_TARGET_H,
    dpi=MV_COMPOSITE_DPI,
    visible=MV_VISIBLE,
    fps=RENDER_FPS,
):
    """Render the wrapped env at high res, then composite the swing/stance overlay."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    states = _mv_rollout_states_from_arrays(env, arrays)
    print(f"  rendering {len(states)} mujoco frames at {render_w}x{render_h}...")
    frames = env.render(
        states, camera=MV_RENDER_CAMERA, height=render_h, width=render_w
    )

    T = min(len(frames), contact_barcode.shape[1])
    frames = frames[:T]
    contact_barcode = contact_barcode[:, :T]

    # Reorder source [RH, RF, LF, LH] -> display [RF, LF, RH, LH] (top to bottom).
    row_perm = [1, 2, 0, 3]
    contact_reord = contact_barcode[row_perm]
    limb_labels = ["RF", "LF", "RH", "LH"]
    limb_colors = [
        tuple(c / 255 for c in (0, 170, 188)),
        tuple(c / 255 for c in (148, 103, 189)),
        tuple(c / 255 for c in (189, 19, 0)),
        tuple(c / 255 for c in (150, 150, 150)),
    ]
    row_px, gap_px = 44, 12
    strip_h = 4 * row_px + 3 * gap_px
    strip_full = np.zeros((strip_h, T, 4))
    for i, color in enumerate(limb_colors):
        y0 = i * (row_px + gap_px)
        y1 = y0 + row_px
        stance = contact_reord[i] > 0
        strip_full[y0:y1, stance, :3] = color
        strip_full[y0:y1, stance, 3] = 1.0

    fig = plt.figure(figsize=(target_w / dpi, target_h / dpi), dpi=dpi)
    ax_video = fig.add_axes([0, 0, 1, 1])
    ax_video.set_axis_off()
    video_im = ax_video.imshow(frames[0], aspect="auto")

    bar_left, bar_bot, bar_w, bar_h = 0.06, 0.08, 0.88, 0.22
    ax_bar = fig.add_axes([bar_left, bar_bot, bar_w, bar_h])
    ax_bar.set_axis_off()
    ax_bar.patch.set_alpha(0)
    ax_bar.set_xlim(0, visible)
    ax_bar.set_ylim(strip_h + 5, -5)

    init_rgba = np.zeros((strip_h, visible, 4))
    bar_im = ax_bar.imshow(
        init_rgba,
        aspect="auto",
        interpolation="nearest",
        extent=[0, visible, strip_h, 0],
    )
    ax_bar.axvline(visible, color="#FF3B30", lw=8, alpha=0.95, zorder=5)

    trans = blended_transform_factory(ax_bar.transAxes, ax_bar.transData)
    for i, lbl in enumerate(limb_labels):
        y_center = i * (row_px + gap_px) + row_px / 2
        txt = ax_bar.text(
            -0.015,
            y_center,
            lbl,
            transform=trans,
            ha="right",
            va="center",
            fontsize=40,
            fontweight="bold",
            color="white",
            fontfamily="DM Sans",
            clip_on=False,
        )
        txt.set_path_effects(
            [path_effects.withStroke(linewidth=4.5, foreground="black")]
        )

    def update(t):
        video_im.set_data(frames[t])
        vis = np.zeros((strip_h, visible, 4))
        start = max(0, t - visible + 1)
        n_show = t - start + 1
        vis[:, visible - n_show :, :] = strip_full[:, start : t + 1, :]
        bar_im.set_data(vis)
        return [video_im, bar_im]

    anim = FuncAnimation(fig, update, frames=T, interval=1000.0 / fps, blit=True)
    print(f"  saving composite video to {out_path}...")
    anim.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)
    plt.close(fig)
    return out_path
