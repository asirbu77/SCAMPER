"""Maintain multi-velocity task for virtual rodent.

The rodent is initialized in a neutral standing pose facing forward (+x axis)
and must maintain a target forward velocity that varies per episode. The target
velocity is sampled uniformly from [min_velocity, max_velocity] at each episode
reset and is included in the task observation so the policy can condition on it.

Termination occurs if:
- Torso becomes too tilted (fallen)
- Torso goes below ground level
- NaN detected in simulation data
"""

import collections
from typing import Any, Callable, Dict, Mapping, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from jax import flatten_util
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env

from scamper.environment.rodent import base as rodent_base
from scamper.environment.rodent import consts


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the MaintainMultiVelocity environment.

    Returns:
        config_dict.ConfigDict: The default configuration dictionary.
    """
    return config_dict.create(
        walker_xml_path=consts.RODENT_BOX_FEET_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        ctrl_dt=0.01,
        sim_dt=0.002,
        solver="cg",
        mujoco_impl="jax",
        naconmax=16 * 8192,
        njmax=512,
        iterations=10,
        ls_iterations=5,
        noslip_iterations=0,
        torque_actuators=True,
        rescale_factor=0.9,
        init_z=0.0,
        min_velocity=0.1,
        max_velocity=2.0,
        episode_length=2000,
        action_repeat=1,
        reward_terms={
            "forward_velocity": {"weight": 1.0},
            "lateral_velocity": {"weight": 0.0},
            "angular_velocity_z": {"weight": 0.0},
            "control_cost": {"weight": 0.0},
            "energy_cost": {"weight": 0.0, "max_value": 50.0},
        },
        termination_criteria={
            "fallen": {"min_torso_z": 0.03, "max_torso_angle": 60},
            "out_of_bounds_y": {"max_y": 0.15},
            "nan_termination": {},
        },
    )


_REWARD_FCN_REGISTRY: dict[str, Callable] = {}
_TERMINATION_FCN_REGISTRY: dict[str, Callable] = {}


class MaintainMultiVelocity(rodent_base.RodentEnv):
    """Maintain multi-velocity environment.

    The rodent must maintain a target forward velocity in the +x direction.
    The target velocity is randomized per episode from a uniform distribution
    over [min_velocity, max_velocity] and is included in the task observation.
    """

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """Initialize the MaintainMultiVelocity environment.

        Args:
            rng: Random number generator key.
            config: Configuration dictionary.
            config_overrides: Optional configuration overrides.
        """
        super().__init__(config, config_overrides)
        self._rng = rng

        # Initialize rodent at origin facing forward (+x direction)
        # quat (1, 0, 0, 0) = identity = facing +x by default
        init_x, init_y, init_z = 0.0, 0.0, self._config.init_z
        init_quat = (1, 0, 0, 0)

        self.add_rodent(
            torque_actuators=self._config.torque_actuators,
            rescale_factor=self._config.rescale_factor,
            pos=[init_x, init_y, init_z],
            quat=init_quat,
        )
        self._spec.worldbody.add_light(pos=[0, 0, 10], dir=[0, 0, -1])
        self.compile()

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment state.

        Samples a random target velocity for this episode from
        U(min_velocity, max_velocity) and stores it in state.info.

        Args:
            rng: Random number generator state.

        Returns:
            mjx_env.State: The initial environment state after reset.
        """
        rng, speed_rng = jax.random.split(rng)

        target_speed = jax.random.uniform(
            speed_rng,
            minval=self._config.min_velocity,
            maxval=self._config.max_velocity,
        )

        info = {
            "prev_action": self.null_action(),
            "action": self.null_action(),
            "target_speed": target_speed,
        }

        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )

        metrics = {}
        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step the environment forward by one timestep.

        Args:
            state: Current environment state.
            action: Action to apply.

        Returns:
            mjx_env.State: The new environment state after stepping.
        """
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        obs = self._get_obs(data, info)

        info["prev_action"] = info["action"]
        info["action"] = action

        done = self._is_done(data, info, state.metrics)
        reward = self._get_reward(data, info, state.metrics)
        reward = jp.nan_to_num(reward)

        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )
        return state

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> collections.OrderedDict:
        """Get the current observation from the simulation data.

        Args:
            data: The simulation data.
            info: State info dictionary (must contain 'target_speed').

        Returns:
            OrderedDict with task_obs and proprioception keys.
            task_obs includes the target speed as its last element.
        """
        kinematic_sensors = self._get_kinematic_sensors(data)
        touch_sensors = self._get_touch_sensors(data)
        origin = self._get_origin(data)

        task_obs = jp.concatenate(
            [
                info["prev_action"],
                kinematic_sensors,
                touch_sensors,
                origin,
                jp.atleast_1d(info["target_speed"]),
            ]
        )

        return collections.OrderedDict(
            task_obs=task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
        )

    def _is_done(self, data: mjx.Data, info: Mapping[str, Any], metrics) -> bool:
        """Check if episode should terminate.

        Args:
            data: Simulation data.
            info: State info dictionary.
            metrics: Metrics dictionary for logging.

        Returns:
            Boolean indicating if episode should terminate.
        """
        any_terminated = False
        for name, kwargs in self._config.termination_criteria.items():
            termination_fcn = _TERMINATION_FCN_REGISTRY[name]
            terminated = termination_fcn(self, data, info, **kwargs)
            any_terminated = jp.logical_or(any_terminated, terminated)
            metrics["terminations/" + name] = jp.astype(terminated, float)
        metrics["terminations/any"] = jp.astype(any_terminated, float)
        return any_terminated

    def _get_reward(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: Dict
    ) -> float:
        """Compute total reward.

        Args:
            data: Simulation data.
            info: State info dictionary.
            metrics: Metrics dictionary for logging.

        Returns:
            Total reward value.
        """
        net_reward = 0.0
        for name, kwargs in self._config.reward_terms.items():
            net_reward += _REWARD_FCN_REGISTRY[name](
                self, data, info, metrics, **kwargs
            )
        return net_reward

    def _named_reward(name: str):
        """Decorator to register reward functions."""

        def decorator(reward_fcn: Callable):
            _REWARD_FCN_REGISTRY[name] = reward_fcn
            return reward_fcn

        return decorator

    @_named_reward("forward_velocity")
    def _forward_velocity_reward(self, data, info, metrics, weight) -> float:
        """Reward for maintaining the per-episode target forward velocity.

        Args:
            data: Simulation data.
            info: State info (contains 'target_speed').
            metrics: Metrics dict for logging.
            weight: Reward weight multiplier.

        Returns:
            Weighted forward velocity reward.
        """
        body = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        forward_vel = body.subtree_linvel[0]

        target_speed = info["target_speed"]

        # JIT-compatible linear tolerance: equivalent to reward_fns.tolerance(
        #   forward_vel, bounds=(target_speed, target_speed),
        #   margin=target_speed, sigmoid="linear", value_at_margin=0.0)
        # but avoids Python-level conditionals on traced values.
        d = jp.abs(forward_vel - target_speed) / jp.maximum(target_speed, 1e-8)
        reward_value = jp.clip(1.0 - d, 0.0, 1.0)

        weighted_reward = reward_value * weight
        metrics["rewards/forward_velocity"] = weighted_reward

        return weighted_reward

    @_named_reward("lateral_velocity")
    def _lateral_velocity_cost(self, data, info, metrics, weight) -> float:
        """Cost for lateral (y-direction) velocity to encourage straight-line motion.

        Args:
            data: Simulation data.
            info: State info (unused).
            metrics: Metrics dict for logging.
            weight: Cost weight multiplier (positive value).

        Returns:
            Negative weighted cost (penalty for lateral movement).
        """
        del info
        body = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        lateral_vel = body.subtree_linvel[1]  # y-direction velocity
        cost = -weight * jp.square(lateral_vel)  # negative to penalize
        metrics["rewards/lateral_velocity"] = cost
        return cost

    @_named_reward("control_diff_cost")
    def _control_diff_cost(self, data, info, metrics, weight) -> float:
        """Cost for action smoothness (penalizes large frame-to-frame action changes).

        Args:
            data: Simulation data.
            info: State info (contains 'action' and 'prev_action').
            metrics: Metrics dict for logging.
            weight: Cost weight multiplier (positive value).

        Returns:
            Negative weighted cost (penalty for jerky actions).
        """
        ctrl_diff_sqr = jp.sum(jp.square(info["action"] - info["prev_action"]))
        cost = -weight * ctrl_diff_sqr
        metrics["rewards/control_diff_cost"] = cost
        return cost

    @_named_reward("control_cost")
    def _control_cost(self, data, info, metrics, weight) -> float:
        """Cost for action magnitude (penalizes large actuator commands).

        Args:
            data: Simulation data.
            info: State info (contains 'action').
            metrics: Metrics dict for logging.
            weight: Cost weight multiplier (positive value).

        Returns:
            Negative weighted cost (penalty for large actions).
        """
        ctrl_sqr = jp.sum(jp.square(info["action"]))
        cost = -weight * ctrl_sqr
        metrics["rewards/control_cost"] = cost
        return cost

    @_named_reward("energy_cost")
    def _energy_cost(self, data, info, metrics, weight, max_value) -> float:
        """Cost for energy consumption (penalizes mechanical power).

        Computes sum(|joint_velocity| * |joint_torque|), which approximates
        mechanical power. Clamped to max_value to prevent dominating the reward.

        Args:
            data: Simulation data.
            info: State info (unused).
            metrics: Metrics dict for logging.
            weight: Cost weight multiplier (positive value).
            max_value: Maximum energy value (clamp).

        Returns:
            Negative weighted cost (penalty for energy use).
        """
        del info
        energy_use = jp.sum(jp.abs(data.qvel) * jp.abs(data.qfrc_actuator))
        cost = -weight * jp.minimum(energy_use, max_value)
        metrics["rewards/energy_cost"] = cost
        return cost

    @_named_reward("angular_velocity_z")
    def _angular_velocity_z_cost(self, data, info, metrics, weight) -> float:
        """Cost for yaw rate (z-axis angular velocity) to prevent turning.

        Args:
            data: Simulation data.
            info: State info (unused).
            metrics: Metrics dict for logging.
            weight: Cost weight multiplier (positive value).

        Returns:
            Negative weighted cost (penalty for turning).
        """
        del info
        # Use gyro sensor for angular velocity
        gyro = data.bind(self.mjx_model, self._spec.sensor("gyro-rodent")).sensordata
        yaw_rate = gyro[2]  # z-axis angular velocity
        cost = -weight * jp.square(yaw_rate)  # negative to penalize
        metrics["rewards/angular_velocity_z"] = cost
        return cost

    def _named_termination_criterion(name: str):
        """Decorator to register termination functions."""

        def decorator(termination_fcn: Callable):
            _TERMINATION_FCN_REGISTRY[name] = termination_fcn
            return termination_fcn

        return decorator

    @_named_termination_criterion("fallen")
    def _fallen_termination(
        self,
        data: mjx.Data,
        info,
        min_torso_z: float = 0.03,
        max_torso_angle: float = 60,
    ) -> bool:
        """Check if rodent has fallen.

        Args:
            data: Simulation data.
            info: State info (unused).
            min_torso_z: Minimum z height threshold.
            max_torso_angle: Maximum angle from vertical in degrees.

        Returns:
            Boolean indicating if fallen.
        """
        del info

        torso_body = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        torso_z = torso_body.xpos[2]

        below_ground = torso_z < min_torso_z

        # xmat is 3x3 rotation matrix, [-1, -1] is element (2,2) = cos(angle from vertical)
        upright_z = torso_body.xmat[-1, -1]
        max_cos_angle = np.cos(np.deg2rad(max_torso_angle))
        too_tilted = upright_z < max_cos_angle

        return jp.logical_or(below_ground, too_tilted)

    @_named_termination_criterion("out_of_bounds_y")
    def _out_of_bounds_y(self, data, info, max_y: float = 0.15) -> bool:
        """Check if rodent has drifted too far laterally.

        Args:
            data: Simulation data.
            info: State info (unused).
            max_y: Maximum absolute y-position before termination.

        Returns:
            Boolean indicating if out of y-bounds.
        """
        del info
        torso_body = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        return jp.abs(torso_body.xpos[1]) > max_y

    @_named_termination_criterion("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        """Check for NaN values in simulation data.

        Args:
            data: Simulation data.
            info: State info (unused).

        Returns:
            Boolean indicating if NaN detected.
        """
        del info
        flattened_vals, _ = flatten_util.ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        return num_nans > 0

    def null_action(self) -> jp.ndarray:
        """Return zero action."""
        return jp.zeros(self.action_size)

    @property
    def proprioceptive_obs_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs_size["proprioception"])[0])

    @property
    def non_proprioceptive_obs_size(self) -> int:
        return self.observation_size - self.proprioceptive_obs_size

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        obs = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs)[0])

    @property
    def non_flattened_observation_size(self) -> mjx_env.ObservationSize:
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        return jax.tree_util.tree_map(lambda x: jp.prod(jp.array(x.shape)), obs)
