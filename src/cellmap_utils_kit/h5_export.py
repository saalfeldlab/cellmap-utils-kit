"""
Module for exporting Zarr datasets to HDF5 format.

This module provides functionality to copy Zarr datasets into HDF5 files while
maintaining the original structure and converting attributes to JSON strings. The
export process is asynchronous, allowing for concurrent processing of multiple crops
to enhance performance.

Key Functions:
-------------
- `h5_export_main(data_yaml: str, destination: str, concurrence: int = 4) -> None`:
  Main entry point for exporting Zarr data to HDF5 format. It reads a YAML configuration
  file to determine which datasets to process and manages the execution of the export
  tasks, supporting concurrent processing.

Dependencies:
------------
- asyncio: For asynchronous programming.
- h5py: For reading and writing HDF5 files.
- json: For handling JSON data.
- logging: For logging messages and errors.
- numpy: For numerical operations and array handling.
- pathlib: For manipulating filesystem paths.
- yaml: For reading configuration files in YAML format.
- zarr: For accessing and manipulating Zarr datasets.
- cellmap_utils_kit.misc_utils: For utility functions like `extract_crop_name`.
- cellmap_utils_kit.parallel_utils: For managing background tasks.

Usage:
-----
1. Prepare a YAML configuration file specifying the paths to Zarr datasets and crops.
2. Call `h5_export_main(data_yaml, destination, concurrence)` with the path to the YAML
   file, the desired output directory for HDF5 files, and the level of concurrency for
   processing. A CLI is provided in `cellmap_utils_kit.cli`.

Example:
-------
```python
h5_export_main("path/to/config.yaml", "output/directory", concurrence=4)
```

"""

import asyncio
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import yaml
import zarr

from cellmap_utils_kit.misc_utils import extract_crop_name
from cellmap_utils_kit.parallel_utils import background

logger = logging.getLogger(__name__)


def _copy_data(zfh: zarr.Group, h5fh: h5py.Group | h5py.File, dataset: str) -> None:
    data = np.array(zfh[dataset])
    chunks = tuple(min(8, sh) for sh in data.shape)
    h5fh.create_dataset(dataset, data=data, chunks=chunks)
    for k, attr in zfh[dataset].attrs.asdict().items():
        h5fh.attrs.create(k, json.dumps(attr))


def _copy_group(zfh: zarr.Group, h5fh: h5py.Group | h5py.File, group: str) -> None:
    logger.debug(f"Copy group {group}")
    h5fhg = h5fh.create_group(group)
    for k, attr in zfh[group].attrs.asdict().items():
        h5fhg.attrs.create(k, json.dumps(attr))
    for el in zfh[group].keys():
        logger.debug(f"Processing {group}'s {el}")
        if isinstance(zfh[group][el], zarr.Group):
            _copy_group(zfh[group], h5fhg, el)
        else:
            _copy_data(zfh[group], h5fhg, el)


@background
def _export_crop(src_crop_path: str, destination: str, dataname: str) -> None:
    logger.info(src_crop_path)
    cropname = extract_crop_name(src_crop_path)
    h5f = h5py.File(Path(destination) / dataname / f"{cropname}.h5", "w")
    zf = zarr.open(src_crop_path, "r")
    for group in zf.keys():
        _copy_group(zf, h5f, group)
    h5f.close()


def h5_export_main(
    data_yaml: str, destination: str, max_concurrency: int | None = None
) -> None:
    """Copy zarr data to h5 and rechunk to (8,8,8).

    This function resaves zarrs specified in a data configuration yaml to a new
    directory as hdf5 files. The structure within the hdf5 files follows that of the
    zarr files. Attributes are encoded as json strings.

    Args:
        data_yaml (str): Path to data configuration yaml with zarrs.
        destination (str): Path to destination directory
        max_concurrency (int, optional): Maximum number of concurrent processes. If
            None, no limit is set. Defaults to None.

    """
    loop = asyncio.get_event_loop()
    task_list = []
    with open(data_yaml) as f:
        datasets = yaml.safe_load(f)["datasets"]
        for dataname, datainfo in datasets.items():
            (Path(destination) / dataname).mkdir(exist_ok=True)
            for crop in datainfo["crops"]:
                crop_path = str(Path(datainfo["crop_group"]) / crop)
                task_list.append(_export_crop(crop_path, destination, dataname))
                if len(task_list) == max_concurrency:
                    looper = asyncio.gather(*task_list)
                    loop.run_until_complete(looper)
                    task_list = []
    looper = asyncio.gather(*task_list)
    loop.run_until_complete(looper)
