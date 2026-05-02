"""This module exposes all high level APIs for scamper."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scamper")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
