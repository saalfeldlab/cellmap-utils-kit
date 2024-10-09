"""
cellmap-utils-kit: Collection of various utilities for cellmap data.

This package provides various CLIs for data preparation, including
- zarr format handling
- multiscale data generation
- h5 conversion

as well as functions and methods to interact with cellmap style data.
"""

from importlib.metadata import PackageNotFoundError, version

from cellmap_utils_kit import attribute_handler, h5_xarray_reader

try:
    __version__ = version("cellmap-utils-kit")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Larissa Heinrich"
__email__ = "heinrichl@janelia.hhmi.org"
__all__ = ["attribute_handler", "h5_xarray_reader"]  # Explicit re-export
