"""
Module for handling the copying of crop datasets from CellMap data.

This module provides functionality to copy crop datasets described in a YAML
configuration file to a specified destination directory. It utilizes asynchronous
processing to copy multiple datasets in parallel.

Key Functions:
--------------
- `copy_crops_main(data_yaml: str, destination: str) -> None`:
    Reads crop datasets from the provided YAML configuration and initiates the copying
    process to the specified destination directory, creating Zarr files for each crop
    and organizing them into appropriate directories.

Dependencies:
-------------
- asyncio: For asynchronous processing.
- logging: For logging progress and information during execution.
- pathlib: For path manipulation and handling.
- yaml: For reading the dataset configuration from a YAML file.
- zarr: For managing Zarr file storage and dataset organization.

Usage:
------
1. Prepare a YAML configuration file specifying the datasets to be copied.
2. Call the `copy_crops_main` function with the path to the YAML file and the desired
   destination directory. A CLI is provided in `cellmap_utils_kit.cli`.

Example:
-------
```python
copy_crops_main("path/to/config.yaml", "path/to/destination")
```

"""

import asyncio
import logging
from pathlib import Path

import yaml
import zarr

from cellmap_utils_kit.attribute_handler import extract_single_scale_attrs
from cellmap_utils_kit.parallel_utils import background

logger = logging.getLogger(__name__)


@background
def _copy_cellmap_dataset(dataname: str, datainfo: dict, destination: str) -> None:
    destination_path = Path(destination)
    for crop in datainfo["crops"]:
        (destination_path / dataname).mkdir(exist_ok=True)
        logger.info(f"Copying crop {Path(datainfo['crop_group'])/crop}")
        cropstore_dst = zarr.DirectoryStore(
            destination_path / dataname / (crop + ".zarr"),
            dimension_separator="/",
            normalize_keys=True,
        )
        cropgroup_dst = zarr.open_group(cropstore_dst, "w")
        cropgroup_src = zarr.open(Path(datainfo["crop_group"]) / crop, "r")
        labelgroup_dst = cropgroup_dst.create_group("labels")
        labelgroup_dst.attrs.put(cropgroup_src.attrs.asdict())
        for ds in cropgroup_src.keys():
            ds_attrs = extract_single_scale_attrs(
                cropgroup_src[ds].attrs.asdict(), "s0", "s0"
            )
            dsgroup_dst = labelgroup_dst.create_group(ds, overwrite=True)
            dsgroup_dst.attrs.put(ds_attrs)
            arr = cropgroup_src[ds]["s0"]
            chunksize = (1, *arr.shape[1:])
            dsgroupms_dst = dsgroup_dst.create_dataset("s0", data=arr, chunks=chunksize)
            dsgroupms_dst.attrs.put(cropgroup_src[ds]["s0"].attrs.asdict())


def copy_crops_main(
    data_yaml: str, destination: str, max_concurrency: int | None = None
) -> None:
    """Copy crops described in data configuration yaml located at `data_yaml` to
    directory `destination`.

    This function will create one directory per dataset in the data_yaml. In the
    dataset directory there will be an individual zarr file for each crop. The labels
    will be put in a zarr group called "labels". Zarrs will be rechunked to have 2D
    chunks. Only the "s0" scale level will be copied over.

    Args:
        data_yaml (str): Path to data configuration yaml.
        destination (str): Path of parent directory for copy.
        max_concurrency (int, optional): Maximum number of concurrent processes. If
            None, no limit is set. Defaults to None.

    """
    loop = asyncio.get_event_loop()
    task_list = []
    with open(data_yaml) as f:
        datasets = yaml.safe_load(f)["datasets"]
        for dataname, datainfo in datasets.items():
            task_list.append(_copy_cellmap_dataset(dataname, datainfo, destination))
            if len(task_list) == max_concurrency:
                looper = asyncio.gather(*task_list)
                loop.run_until_complete(looper)
                task_list = []
    looper = asyncio.gather(*task_list)
    loop.run_until_complete(looper)
