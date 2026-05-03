"""OmegaConf helpers: walker path resolution, config merging."""

import json
import logging
import subprocess
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

from ml_collections import config_dict
from omegaconf import DictConfig, OmegaConf


def _get_package_commit(package_name: str) -> str:
    """Get the git commit hash for an installed package.

    Works for both editable installs (file://) and VCS installs (git+https://).

    Args:
        package_name: Name of the installed package.

    Returns:
        Git commit hash, or "unknown" if not available.
    """
    try:
        dist = distribution(package_name)
        direct_url = json.loads(dist.read_text("direct_url.json"))
        url = direct_url.get("url", "")

        # Editable install: file:// URL pointing to local repo
        if url.startswith("file://"):
            path = url[7:]
            result = subprocess.run(
                ["git", "-C", path, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()

        # VCS install: git+https:// with vcs_info
        if "vcs_info" in direct_url:
            return direct_url["vcs_info"].get("commit_id", "unknown")

    except Exception:
        pass
    return "unknown"


from scamper.environment.rodent import consts as rodent_consts

# Project root directory (scamper/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _resolve_data_path(relative_path: str) -> str:
    """Resolve a relative data path to an absolute path from the project root.

    Args:
        relative_path: Path relative to the project root (e.g., "data/rodent/file.h5").

    Returns:
        Absolute path as a string.
    """
    return str(_PROJECT_ROOT / relative_path)


def prepare_config(
    cfg: DictConfig,
) -> tuple[DictConfig, dict[str, Any], config_dict.ConfigDict]:
    """Prepare configuration by resolving walker-specific paths and creating config variants.

    Takes a Hydra/OmegaConf configuration and resolves walker-specific XML and
    reference data paths based on the walker_name. Updates the env_config section
    with these paths and walker settings, then returns multiple config formats.

    Args:
        cfg: The root OmegaConf configuration containing walker_config and env_config.

    Returns:
        A tuple containing:
            - cfg: The updated OmegaConf DictConfig with resolved paths.
            - cfg_dict: The full config as a plain Python dictionary.
            - env_cfg_ml: The env_config as an ml_collections ConfigDict.

    Raises:
        ValueError: If walker_name is not recognized.
        NotImplementedError: If the specified walker is not yet fully implemented.
    """
    walker_name = cfg.walker_config.walker_name
    xml_variant = cfg.walker_config.get("xml_variant", "limited-collisions")

    if walker_name == "rodent":
        logging.info("Using rodent walker")
        if xml_variant == "limited-collisions":
            walker_xml_path = str(rodent_consts.RODENT_XML_PATH)
        elif xml_variant == "full-collisions":
            walker_xml_path = str(rodent_consts.RODENT_FULL_COLLISION_XML)
        elif xml_variant == "no-tail-collisions":
            walker_xml_path = str(rodent_consts.RODENT_NO_TAIL_COLLISION_XML)
        else:
            raise ValueError(f"Unknown xml_variant: {xml_variant}")
        arena_xml_path = str(rodent_consts.ARENA_XML_PATH)
        reference_data_path = _resolve_data_path(
            "data/rodent/rodent_reference_clips.h5"
        )
    else:
        raise ValueError(f"Unknown walker name: {walker_name}")

    # Update env_config with resolved paths and walker settings
    OmegaConf.set_struct(cfg.env_config, False)
    OmegaConf.update(cfg.env_config, "walker_xml_path", walker_xml_path, merge=False)
    OmegaConf.update(cfg.env_config, "arena_xml_path", arena_xml_path, merge=False)
    OmegaConf.update(
        cfg.env_config, "reference_data_path", reference_data_path, merge=False
    )
    OmegaConf.update(
        cfg.env_config, "walker_name", cfg.walker_config.walker_name, merge=False
    )
    OmegaConf.update(
        cfg.env_config,
        "torque_actuators",
        cfg.walker_config.torque_actuators,
        merge=False,
    )
    OmegaConf.update(
        cfg.env_config, "rescale_factor", cfg.walker_config.rescale_factor, merge=False
    )
    OmegaConf.update(
        cfg.env_config,
        "scamper_commit",
        _get_package_commit("scamper"),
        merge=False,
    )
    OmegaConf.set_struct(cfg.env_config, True)

    # Create ml_collections ConfigDict for env_config
    env_cfg_dict = OmegaConf.to_container(cfg.env_config, resolve=True)
    env_cfg_ml = config_dict.ConfigDict(env_cfg_dict)

    # Convert full config to dict and log
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logging.info(f"Configs: {cfg_dict}")

    return cfg, cfg_dict, env_cfg_ml


def update_config(cfg: DictConfig, overrides: dict[str, Any]) -> DictConfig:
    """Update an OmegaConf configuration with override values.

    Temporarily disables struct mode to allow adding/modifying keys,
    applies all overrides, then re-enables struct mode.

    Args:
        cfg: The OmegaConf DictConfig to update.
        overrides: Dictionary of key-value pairs to apply. Keys can use
            dot notation for nested updates (e.g., "env_config.num_envs").

    Returns:
        The updated DictConfig (modified in-place).
    """
    OmegaConf.set_struct(cfg, False)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    OmegaConf.set_struct(cfg, True)
    return cfg
