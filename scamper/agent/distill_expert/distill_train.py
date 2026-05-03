"""DAgger-based distillation training loop for the expert teacher.

Student (encoder + decoder + motion prior) rolls out in the environment,
the frozen expert teacher annotates the collected states, and the student
is trained via supervised learning through a variational information bottleneck.
"""

import functools
import time
from typing import Callable, Optional, Tuple

from absl import logging
from brax import base, envs
from brax.training import acting, pmap, types
from brax.training.acme import running_statistics, specs
from brax.training.types import PRNGKey
from mujoco_playground import wrapper as mp_wrapper

import flax
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from scamper.agent import checkpointing, gradients
from scamper.agent.distill_expert import losses as distill_losses
from scamper.agent.mlp_prior import prior_rollout_eval
from scamper.agent.distill_expert.distill_networks import (
    DistillDecoder,
    DistillEncoder,
    DistillMotionPrior,
    DistillNetworkParams,
    create_compatible_checkpoint_params,
    init_distill_params,
    make_distill_networks,
    make_student_inference_fn,
)
from scamper.agent.distill_expert.teacher import (
    load_frozen_teacher,
    make_teacher_query_fn,
)
from scamper.agent.expert.ppo_networks import (
    make_inference_fn as make_mlp_inference_fn,
    make_mlp_ppo_networks,
)
from scamper.agent.observation_utils import (
    DictRunningStatisticsState,
    get_obs_sizes,
    init_dict_normalizer,
    update_dict_normalizer,
)

Metrics = types.Metrics
STEPS_IN_THOUSANDS = 1e3
_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class DistillTrainingState:
    """Training state for DAgger distillation."""

    optimizer_state: optax.OptState
    params: DistillNetworkParams
    normalizer_params: DictRunningStatisticsState
    env_steps: jnp.ndarray


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def train(
    environment: envs.Env,
    num_timesteps: int,
    episode_length: int,
    ckpt_mgr: ocp.CheckpointManager,
    config_dict: dict,
    teacher_checkpoint_path: str,
    teacher_checkpoint_step: Optional[int] = None,
    checkpoint_to_restore: Optional[str] = None,
    action_repeat: int = 1,
    num_envs: int = 4096,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    kl_weight: float = 1.0,
    ar1_weight: float = 0.1,
    grad_clip_norm: float = 10.0,
    seed: int = 0,
    use_pmap_on_reset: bool = True,
    unroll_length: int = 20,
    batch_size: int = 1024,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 4,
    num_evals: int = 100,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = True,
    deterministic_eval: bool = True,
    latent_size: int = 16,
    encoder_hidden_layer_sizes: Tuple[int, ...] = (2048, 1024, 1024),
    decoder_hidden_layer_sizes: Tuple[int, ...] = (1024, 1024, 512),
    motion_prior_hidden_layer_sizes: Tuple[int, ...] = (1536, 1024, 512),
    clamp_logvar: bool = True,
    logvar_min: float = -5.0,
    logvar_max: float = 0.0,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None,
    eval_env_test_set: Optional[envs.Env] = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    use_kl_schedule: bool = False,
    kl_schedule_params: Optional[dict] = None,
    use_ar1_schedule: bool = False,
    ar1_schedule_params: Optional[dict] = None,
    checkpoint_callback: Optional[Callable[[int], None]] = None,
    wrap_for_training: Callable[..., mp_wrapper.Wrapper] = functools.partial(
        mp_wrapper.wrap_for_brax_training, full_reset=False
    ),
    hard_negative_mining: bool = True,
    motion_prior_rollout_config: Optional[dict] = None,
):
    """DAgger distillation training.

    Args:
        environment: The imitation environment to train on.
        num_timesteps: Total number of environment steps.
        episode_length: Length of an episode.
        ckpt_mgr: Orbax checkpoint manager.
        config_dict: Configuration dictionary for checkpointing.
        teacher_checkpoint_path: Path to the pretrained expert checkpoint.
        teacher_checkpoint_step: Optional step to load from teacher checkpoint.
        checkpoint_to_restore: Optional path to restore distill training from.
        ... (remaining args match prior_train.py pattern)

    Returns:
        Tuple of (make_policy_fn, final_params, final_metrics).
    """
    assert batch_size * num_minibatches % num_envs == 0
    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, "
        "devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count

    env_step_per_training_step = (
        batch_size * unroll_length * num_minibatches * action_repeat
    )
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
            num_evals_after_init
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        )
    ).astype(int)

    # --- Load frozen teacher ---
    logging.info(f"Loading teacher from: {teacher_checkpoint_path}")
    teacher_fn, teacher_normalizer_params, teacher_cfg, teacher_raw_params = (
        load_frozen_teacher(teacher_checkpoint_path, step=teacher_checkpoint_step)
    )
    teacher_obs_sizes = teacher_cfg["network_config"]["obs_sizes"]
    action_size = teacher_cfg["network_config"]["action_size"]
    logging.info("Teacher loaded successfully")

    # --- Initialize keys ---
    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    key_networks, policy_params_fn_key = jax.random.split(global_key, 2)
    del global_key

    assert num_envs % device_count == 0

    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // local_device_count
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)

    # --- Create environment ---
    env = wrap_for_training(
        environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])

    def reset_fn_donated(env_state_donated, key_envs):
        return env.reset(key_envs)

    if local_devices_to_use > 1 or use_pmap_on_reset:
        reset_fn_ = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
        env_state = reset_fn_(key_envs)
        reset_fn = jax.pmap(
            reset_fn_donated,
            axis_name=_PMAP_AXIS_NAME,
            donate_argnums=(0,),
        )
    else:
        reset_fn_ = jax.jit(jax.vmap(env.reset))
        env_state = reset_fn_(key_envs)
        reset_fn = jax.jit(reset_fn_donated, donate_argnums=(0,), keep_unused=True)(
            key_envs
        )

    obs_sizes = get_obs_sizes(env_state.obs)

    # Update config with network info
    config_dict["network_config"].update(
        {
            "obs_sizes": obs_sizes,
            "action_size": action_size,
            "normalize_observations": normalize_observations,
            "intention_size": latent_size,
            "encoder_layer_sizes": list(encoder_hidden_layer_sizes),
            "decoder_layer_sizes": list(decoder_hidden_layer_sizes),
            # Persisted dict key kept as 'prior_layer_sizes' for compat with
            # existing checkpoints + downstream task_transfer config reads.
            "prior_layer_sizes": list(motion_prior_hidden_layer_sizes),
        }
    )

    # --- Create student networks ---
    encoder, decoder, motion_prior, action_dist = make_distill_networks(
        obs_sizes=obs_sizes,
        action_size=action_size,
        latent_size=latent_size,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        motion_prior_hidden_layer_sizes=motion_prior_hidden_layer_sizes,
        clamp_logvar=clamp_logvar,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )

    init_params = init_distill_params(
        encoder, decoder, motion_prior, obs_sizes, latent_size, key_networks
    )

    # --- Create student policy for data collection ---
    make_policy = make_student_inference_fn(
        encoder, decoder, action_dist, obs_sizes, latent_size
    )

    # --- Setup optimizer ---
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=0.0, eps=1e-5),
    )

    # --- Setup schedules ---
    kl_schedule_fn = None
    if use_kl_schedule and kl_schedule_params is not None:
        kl_schedule_fn = distill_losses.create_ramp_schedule(
            start_value=kl_schedule_params.get("kl_start_weight", 1e-5),
            end_value=kl_schedule_params.get("kl_end_weight", kl_weight),
            total_steps=num_evals,
            start_frac=kl_schedule_params.get("kl_start_ramp", 0.0),
            end_frac=kl_schedule_params.get("kl_end_ramp", 0.25),
            schedule="linear",
        )

    ar1_schedule_fn = None
    if use_ar1_schedule and ar1_schedule_params is not None:
        ar1_schedule_fn = distill_losses.create_ramp_schedule(
            start_value=ar1_schedule_params.get("ar1_start_weight", 0.01),
            end_value=ar1_schedule_params.get("ar1_end_weight", ar1_weight),
            total_steps=num_evals,
            start_frac=ar1_schedule_params.get("ar1_start_ramp", 0.0),
            end_frac=ar1_schedule_params.get("ar1_end_ramp", 0.25),
            schedule="linear",
        )

    # --- Create loss function ---
    loss_fn = functools.partial(
        distill_losses.compute_distill_loss,
        teacher_fn=teacher_fn,
        teacher_normalizer_params=teacher_normalizer_params,
        encoder=encoder,
        decoder=decoder,
        motion_prior=motion_prior,
        kl_weight=kl_weight,
        ar1_weight=ar1_weight,
        kl_schedule_fn=kl_schedule_fn,
        ar1_schedule_fn=ar1_schedule_fn,
    )

    # --- Initialize training state ---
    training_state = DistillTrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        normalizer_params=teacher_normalizer_params,
        env_steps=jnp.int32(0),
    )

    if checkpoint_to_restore is not None:
        logging.info(f"Restoring checkpoint from {checkpoint_to_restore}")
        training_state = checkpointing.load_training_state(
            checkpoint_to_restore, training_state, step_prefix="DistillNetwork"
        )
        logging.info(f"Restored checkpoint from {checkpoint_to_restore}")

    # --- Create gradient update function ---
    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )

    # --- Define training steps ---
    def minibatch_step(carry, data, normalizer_params):
        optimizer_state, params, key, it = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params,
            normalizer_params,
            data,
            key_loss,
            it,
            optimizer_state=optimizer_state,
            params=params,
        )
        return (optimizer_state, params, key, it), metrics

    def sgd_step(carry, unused_t, data, normalizer_params):
        optimizer_state, params, key, it = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _, _), metrics = jax.lax.scan(
            functools.partial(
                minibatch_step,
                normalizer_params=normalizer_params,
            ),
            (optimizer_state, params, key_grad, it),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, key, it), metrics

    def training_step(carry, unused_t):
        training_state, state, key, it = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        # Student policy for data collection (DAgger: student generates states)
        policy = make_policy(
            (training_state.normalizer_params, training_state.params),
            deterministic=False,
        )

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env,
                current_state,
                policy,
                current_key,
                unroll_length,
                extra_fields=("truncation",),
            )
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            f,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )

        # Reshape: [scans, envs, T, ...] -> [B, T, ...]
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        assert data.discount.shape[1:] == (unroll_length,)

        # Update normalizer
        normalizer_params = update_dict_normalizer(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        # SGD updates (teacher annotation happens inside loss_fn)
        (optimizer_state, params, _, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step,
                data=data,
                normalizer_params=normalizer_params,
            ),
            (training_state.optimizer_state, training_state.params, key_sgd, it),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = DistillTrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=jnp.int32(
                training_state.env_steps
                + env_step_per_training_step / STEPS_IN_THOUSANDS
            ),
        )
        return (new_training_state, state, new_key, it), metrics

    def training_epoch(
        training_state: DistillTrainingState, state: envs.State, key: PRNGKey, it: int
    ):
        (training_state, state, _, _), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, key, it),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(
        training_epoch,
        axis_name=_PMAP_AXIS_NAME,
        donate_argnums=(0, 1),
    )

    training_walltime = 0

    def training_epoch_with_timing(training_state, env_state, key, it):
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        step = jnp.ones_like(training_state.env_steps) * it
        result = training_epoch(training_state, env_state, key, step)
        training_state, env_state, metrics = _strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            num_training_steps_per_epoch
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return training_state, env_state, metrics

    # Replicate training state across devices
    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    # --- Setup evaluation ---
    if eval_env is None:
        eval_env = environment
    eval_env = wrap_for_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=None,
    )

    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # Test set evaluator
    test_evaluator = None
    if eval_env_test_set is not None:
        test_eval_env = wrap_for_training(
            eval_env_test_set,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=None,
        )
        test_evaluator = acting.Evaluator(
            test_eval_env,
            functools.partial(make_policy, deterministic=deterministic_eval),
            num_eval_envs=num_eval_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            key=eval_key,
        )

    # --- Teacher evaluator (for baseline reward comparison) ---
    teacher_mlp_networks = make_mlp_ppo_networks(
        obs_sizes=obs_sizes,
        action_size=action_size,
        mlp_hidden_layer_sizes=tuple(teacher_cfg["network_config"]["mlp_layer_sizes"]),
    )
    teacher_make_policy = make_mlp_inference_fn(teacher_mlp_networks)
    teacher_policy_params = (teacher_normalizer_params, teacher_raw_params)
    teacher_evaluator = acting.Evaluator(
        eval_env,
        functools.partial(teacher_make_policy, deterministic=True),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )
    teacher_test_evaluator = None
    if eval_env_test_set is not None:
        teacher_test_evaluator = acting.Evaluator(
            test_eval_env,
            functools.partial(teacher_make_policy, deterministic=True),
            num_eval_envs=num_eval_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            key=eval_key,
        )

    # --- Helper: create checkpoint params ---
    def get_policy_params(training_state):
        ts = _unpmap(training_state)
        return create_compatible_checkpoint_params(
            network_params=ts.params,
            normalizer_params=ts.normalizer_params,
            obs_sizes=obs_sizes,
            latent_size=latent_size,
            encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        )

    # --- Initialize multi-mode motion-prior rollout evaluator ---
    multi_mode_evaluator = None
    if motion_prior_rollout_config is not None and motion_prior_rollout_config.get(
        "enabled", False
    ):
        logging.info("Initializing multi-mode motion-prior rollout evaluator")
        render_config = config_dict.get("render_config", {})
        proprio_size = obs_sizes["proprioception"]

        normalize = running_statistics.normalize
        # The MultiModePriorRolloutEvaluator API (in scamper.agent.mlp_prior) takes
        # `prior_hidden_layer_sizes` as the kwarg name; that's an external API and
        # kept as-is. We pass our local motion_prior_hidden_layer_sizes through.
        multi_mode_evaluator = prior_rollout_eval.MultiModePriorRolloutEvaluator(
            env=environment,
            intention_latent_size=latent_size,
            action_size=action_size,
            proprioceptive_obs_size=proprio_size,
            decoder_hidden_layer_sizes=list(decoder_hidden_layer_sizes),
            prior_hidden_layer_sizes=list(motion_prior_hidden_layer_sizes),
            preprocess_observations_fn=normalize,
            max_steps=motion_prior_rollout_config.get("max_steps", 200),
            eval_interval=motion_prior_rollout_config.get("eval_interval", 1),
            render_fps=render_config.get("render_fps", 50),
            render_camera_name=render_config.get("render_camera_name", "close_profile"),
            model_path=str(ckpt_mgr.directory),
        )
        logging.info("Multi-mode motion-prior rollout evaluator initialized")

    # --- Get starting iteration ---
    start_it = 0
    if ckpt_mgr is not None and ckpt_mgr.latest_step() is not None:
        num_evals_after_init -= ckpt_mgr.latest_step()
        start_it = ckpt_mgr.latest_step()

    logging.info(
        f"Starting at iteration: {start_it} with {num_evals_after_init} evals left"
    )

    # --- Evaluate teacher once (constant baseline) ---
    teacher_baseline_metrics = {}
    if process_id == 0:
        logging.info("Evaluating teacher baseline reward (train set)")
        teacher_eval_metrics = teacher_evaluator.run_evaluation(
            teacher_policy_params,
            training_metrics={},
        )
        teacher_baseline_metrics = {
            k.replace("eval/", "eval/teacher/", 1): v
            for k, v in teacher_eval_metrics.items()
            if k.startswith("eval/") and not k.endswith("_std")
        }
        logging.info(
            f"Teacher baseline reward (train): "
            f"{teacher_eval_metrics.get('eval/episode_reward', 'N/A')}"
        )

        if teacher_test_evaluator is not None:
            logging.info("Evaluating teacher baseline reward (test set)")
            teacher_test_metrics = teacher_test_evaluator.run_evaluation(
                teacher_policy_params,
                training_metrics={},
            )
            teacher_baseline_metrics.update(
                {
                    k.replace("eval/", "eval/teacher/test_set/", 1): v
                    for k, v in teacher_test_metrics.items()
                    if k.startswith("eval/") and not k.endswith("_std")
                }
            )
            logging.info(
                f"Teacher baseline reward (test): "
                f"{teacher_test_metrics.get('eval/episode_reward', 'N/A')}"
            )

    # --- Initial eval ---
    metrics = {}
    if process_id == 0 and num_evals > 1 and start_it == 0:
        logging.info("Running initial evaluation")
        policy_param = get_policy_params(training_state)
        ts = _unpmap(training_state)

        # Teacher baseline (constant, logged every step for easy comparison)
        metrics.update(teacher_baseline_metrics)

        # Eval on train set
        eval_metrics = evaluator.run_evaluation(
            (ts.normalizer_params, ts.params),
            training_metrics={},
        )
        metrics.update(eval_metrics)

        # Eval on test set
        if test_evaluator is not None:
            test_metrics = test_evaluator.run_evaluation(
                (ts.normalizer_params, ts.params),
                training_metrics={},
            )
            metrics.update(
                {
                    k.replace("eval/", "eval/test_set/", 1): v
                    for k, v in test_metrics.items()
                    if k.startswith("eval/")
                }
            )

        # Generate shared reset key for all rollouts
        policy_params_fn_key, reset_key = jax.random.split(policy_params_fn_key)

        # Run multi-mode motion-prior rollout evaluation
        if multi_mode_evaluator is not None:
            motion_prior_metrics = multi_mode_evaluator.run_evaluation(
                policy_params=policy_param,
                eval_step=0,
                reset_key=reset_key,
            )
            if motion_prior_metrics is not None:
                metrics.update(motion_prior_metrics)
                logging.info(
                    f"Initial motion-prior rollout metrics: {motion_prior_metrics}"
                )

        # Rollout logging callback
        _, policy_params_fn_key = jax.random.split(policy_params_fn_key)
        policy_params_fn(
            current_step=0,
            params=policy_param,
            normalizer_params=ts.normalizer_params,
            network_params=ts.params,
            policy_params_fn_key=policy_params_fn_key,
            reset_key=reset_key,
            render_video=True,
        )

        logging.info(metrics)
        progress_fn(start_it, metrics)

        if ckpt_mgr is not None:
            ckpt_mgr.save(
                step=0,
                args=ocp.args.Composite(
                    policy=ocp.args.StandardSave(policy_param),
                    train_state=ocp.args.StandardSave(ts),
                    teacher=ocp.args.StandardSave(teacher_policy_params),
                    config=ocp.args.JsonSave(config_dict),
                ),
            )
            if checkpoint_callback is not None:
                try:
                    checkpoint_callback(0)
                except Exception as e:
                    logging.warning(f"Initial checkpoint callback failed: {e}")

    # --- Training loop ---
    training_metrics = {}
    start_it += 1
    current_step = 0

    for it in range(start_it, num_evals_after_init + start_it):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        for _ in range(max(num_resets_per_eval, 1)):
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            training_state, env_state, training_metrics = training_epoch_with_timing(
                training_state, env_state, epoch_keys, it
            )
            current_step = int(_unpmap(training_state.env_steps))

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
            )(key_envs, key_envs.shape[1])
            if num_resets_per_eval > 0:
                env_state = reset_fn((training_state, env_state), key_envs)

        if process_id == 0:
            policy_param = get_policy_params(training_state)
            ts = _unpmap(training_state)

            metrics = training_metrics.copy()

            # Teacher baseline (constant, logged every step for easy comparison)
            metrics.update(teacher_baseline_metrics)

            # Eval on train set (tracking reward)
            eval_metrics = evaluator.run_evaluation(
                (ts.normalizer_params, ts.params),
                training_metrics={},
            )
            metrics.update(eval_metrics)

            # Eval on test set
            if test_evaluator is not None:
                test_metrics = test_evaluator.run_evaluation(
                    (ts.normalizer_params, ts.params),
                    training_metrics={},
                )
                metrics.update(
                    {
                        k.replace("eval/", "eval/test_set/", 1): v
                        for k, v in test_metrics.items()
                        if k.startswith("eval/")
                    }
                )

            # Generate shared reset key for all rollouts
            policy_params_fn_key, reset_key = jax.random.split(policy_params_fn_key)

            # Run multi-mode motion-prior rollout evaluation
            if multi_mode_evaluator is not None:
                motion_prior_metrics = multi_mode_evaluator.run_evaluation(
                    policy_params=policy_param,
                    eval_step=it,
                    reset_key=reset_key,
                )
                if motion_prior_metrics is not None:
                    metrics.update(motion_prior_metrics)
                    logging.info(
                        f"Motion-prior rollout metrics: {motion_prior_metrics}"
                    )

            # Rollout logging callback
            _, policy_params_fn_key = jax.random.split(policy_params_fn_key)
            render_config = config_dict.get("render_config", {})
            render_interval = render_config.get("render_interval", 1)
            policy_params_fn(
                current_step=it,
                params=policy_param,
                normalizer_params=ts.normalizer_params,
                network_params=ts.params,
                policy_params_fn_key=policy_params_fn_key,
                reset_key=reset_key,
                render_video=(it % render_interval == 0),
            )

            logging.info(metrics)
            progress_fn(current_step, metrics)

            if ckpt_mgr is not None:
                ckpt_mgr.save(
                    step=it,
                    args=ocp.args.Composite(
                        policy=ocp.args.StandardSave(policy_param),
                        train_state=ocp.args.StandardSave(ts),
                        teacher=ocp.args.StandardSave(teacher_policy_params),
                        config=ocp.args.JsonSave(config_dict),
                    ),
                )
                if checkpoint_callback is not None:
                    try:
                        checkpoint_callback(it)
                    except Exception as e:
                        logging.warning(f"Checkpoint callback failed: {e}")

    total_steps = current_step
    pmap.assert_is_replicated(training_state)
    params = get_policy_params(training_state)
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
