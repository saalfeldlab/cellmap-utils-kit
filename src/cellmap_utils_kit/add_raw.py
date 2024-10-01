"""
Module for processing and adding raw data to specified crops within a dataset.

This module provides functionality to add raw data alongside labels for crops
specified in a configuration YAML file. It uses asynchronous processing to handle
multiple crops concurrently, leveraging background tasks for efficiency.

Key Functions:
-------------
- `add_raw_main(data_yaml: str, concurrence: int = 1) -> None`:
  Main entry point for adding raw data. It reads a YAML configuration file and
  manages the execution of data processing for each dataset, supporting concurrent
  processing to enhance performance.

Dependencies:
------------
- asyncio: For asynchronous programming.
- logging: For logging messages and errors.
- pathlib: For handling filesystem paths.
- fibsem_tools: Custom library for reading xarray data.
- numpy: For numerical operations and array handling.
- yaml: For reading configuration files in YAML format.
- zarr: For storing large arrays in a hierarchical format.

Usage:
-----
1. Prepare a YAML configuration file containing paths to datasets and crops.
2. Call `add_raw_main(data_yaml, concurrence)` with the path to the YAML file
   and the desired level of concurrency for processing. A CLI is provided in
   `cellmap_utils_kit.cli`.

Example:
-------
```python
add_raw_main("path/to/config.yaml", concurrence=2)
```

"""

import asyncio
import logging
from pathlib import Path

import fibsem_tools as fst
import numpy as np
import yaml
import zarr

from cellmap_utils_kit.attribute_handler import (
    add_scalelevel_to_attributes,
    initialize_multiscale_attributes,
)
from cellmap_utils_kit.parallel_utils import background

logger = logging.getLogger(__name__)


@background
def _add_raw(dataname: str, datainfo: dict) -> None:
    for crop in datainfo["crops"]:
        cropgroup = Path(datainfo["crop_group"]) / crop
        cropgroup_zarr = zarr.open_group(cropgroup, "a")
        rawgroup_dst = cropgroup_zarr.create_group("raw", overwrite=True)
        raw_src_xarr = fst.read_xarray(Path(datainfo["raw"]) / "s0")
        raw_res = {dim: float(c[1] - c[0]) for dim, c in raw_src_xarr.coords.items()}
        if "labels" in cropgroup_zarr:
            labels = cropgroup_zarr["labels"].attrs["cellmap"]["annotation"][
                "class_names"
            ]
            ref_lbl = labels[0]
            ref_crop = Path(datainfo["crop_group"]) / crop / "labels" / ref_lbl
        else:
            labels = cropgroup_zarr.attrs["cellmap"]["annotation"]["class_names"]
            ref_lbl = labels[0]
            ref_crop = Path(datainfo["crop_group"]) / crop / ref_lbl

        lbl_src_xarr = fst.read_xarray(ref_crop / "s0")
        lbl_res = {dim: c[1] - c[0] for dim, c in lbl_src_xarr.coords.items()}
        start = {dim: c[0] for dim, c in lbl_src_xarr.coords.items()}
        end = {dim: c[-1] for dim, c in lbl_src_xarr.coords.items()}
        raw_sel_coords = {}
        for dim in start.keys():
            start_raw = start[dim] - lbl_res[dim] / 2 + raw_res[dim] / 2
            end_raw = end[dim] + lbl_res[dim] / 2 - raw_res[dim] / 2
            raw_sel_coords[dim] = np.arange(
                start_raw, end_raw + raw_res[dim], raw_res[dim]
            ).tolist()
        try:
            raw_crop = raw_src_xarr.sel(raw_sel_coords)
        except KeyError as e:
            msg = f"Could not extract raw in {dataname}: {crop}"
            logger.info(msg)
            logger.error(e)
            continue
        raw_attrs = add_scalelevel_to_attributes(
            initialize_multiscale_attributes(),
            "s0",
            [raw_res[k] for k in "zyx"],
            [raw_sel_coords[k][0] for k in "zyx"],
        )

        rawgroup_dst.attrs.put(raw_attrs)
        chunksize = (1, *raw_crop.shape[1:])
        rawgroup_dst.create_dataset(
            "s0", data=raw_crop.data.compute(), chunks=chunksize, overwrite=True
        )
        logger.info(f"Successfully added raw in {dataname}: {crop}")


def add_raw_main(data_yaml: str, concurrence: int = 1) -> None:
    """Adds raw data alongside labels to crops specified in `data_yaml`.

    Args:
        data_yaml (str): Path to data configuration yaml
        concurrence (int, optional): Number of concurrent processes. Defaults to 1.

    """
    loop = asyncio.get_event_loop()
    with open(data_yaml) as f:
        datasets = yaml.safe_load(f)["datasets"]
        task_list = []
        for dataname, datainfo in datasets.items():
            task_list.append(_add_raw(dataname, datainfo))
            if len(task_list) == concurrence:
                looper = asyncio.gather(*task_list)
                loop.run_until_complete(looper)
                task_list = []
        looper = asyncio.gather(*task_list)
        loop.run_until_complete(looper)
