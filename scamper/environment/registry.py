"""Registry for all SCAMPER environments.

Usage:
    from scamper.environment import registry

    # List all environments
    print(registry.ALL_ENVS)

    # Load environment by name
    env = registry.load("RodentImitation", config=cfg, clips=clips)

    # Load reference clips
    clips = registry.load_reference_clips("RodentImitation", data_path, ...)
"""

from typing import Any, Optional

from ml_collections import config_dict

from scamper import environment as tasks

# Combine all task categories
ALL_ENVS = tasks.ALL_ENVS


def get_default_config(env_name: str) -> config_dict.ConfigDict:
    """Get the default configuration for an environment."""
    if env_name in tasks.ALL_ENVS:
        return tasks.get_default_config(env_name)
    raise ValueError(f"Env '{env_name}' not found. Available: {ALL_ENVS}")


def load(
    env_name: str,
    config: Optional[config_dict.ConfigDict] = None,
    clips: Any = None,
    flatten_obs: bool = True,
):
    """Load an environment instance.

    Args:
        env_name: Environment name (e.g., "RodentImitation").
        config: Configuration dict. Uses default if not provided.
        clips: ReferenceClips for imitation tasks.
        flatten_obs: Whether to apply observation wrapper.

    Returns:
        Instantiated environment.
    """
    if env_name in tasks.ALL_ENVS:
        return tasks.load(env_name, config, clips, flatten_obs)
    raise ValueError(f"Env '{env_name}' not found. Available: {ALL_ENVS}")


def load_reference_clips(
    env_name: str,
    data_path: str,
    n_frames_per_clip: int,
    keep_clips_idx=None,
):
    """Load reference clips for an environment."""
    if env_name in tasks.ALL_ENVS:
        return tasks.load_reference_clips(
            env_name, data_path, n_frames_per_clip, keep_clips_idx
        )
    raise ValueError(f"Env '{env_name}' has no reference clips.")
