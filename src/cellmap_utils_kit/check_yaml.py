"""
Module for verifying the integrity and existence of datasets in CellMap data.

This module provides functions to check the availability and consistency of dataset
paths, crops, and scale levels as described in a YAML configuration file. It ensures
that the necessary raw data and labeled crops exist and are accessible, logging any
errors that occur.

Key Functions:
--------------

- `check_data_yaml_main(data_yaml: str, label_scalelevels: tuple[str, ...] = (),
    raw_scalelevels: tuple[str, ...] = ()) -> None`: Iterates through all datasets
    specified in a YAML file, checks for their existence, and validates them.

Dependencies:
-------------
- asyncio: For managing asynchronous checks of multiple datasets.
- logging: For logging error messages and progress information.
- pathlib: For path manipulation and handling.
- yaml: For loading dataset configurations from a YAML file.
- zarr: For accessing Zarr datasets.
- fibsem_tools: For reading data stored in Zarr files.
- cellmap_utils_kit.parallel_utils: For running background tasks.

Usage:
------
1. Prepare a YAML configuration file that specifies the datasets, crops, and paths to
   be verified.
2. Call the `check_data_yaml_main` function with the path to the YAML file. A CLI is
   provided in cellmap_utils_kit.cli

Example:
-------
```python
check_data_yaml_main("path/to/data_config.yaml")
```

"""

import asyncio
import logging
from pathlib import Path

import fibsem_tools as fst
import yaml
import zarr

from cellmap_utils_kit.parallel_utils import background

logger = logging.getLogger(__name__)


def _check_crop(datainfo: dict, crop: str, scalelevels: tuple[str, ...] = ()) -> None:
    try:
        # check that crop exists in the crop group
        if not (Path(datainfo["crop_group"]) / crop).exists():
            msg = f"{Path(datainfo['crop_group'])/crop} does not exist"
            raise ValueError(msg)
        # check that scale levels exists for all labels
        crop_zarr = zarr.open(Path(datainfo["crop_group"]) / crop)
        if "labels" in crop_zarr:
            crop_zarr = crop_zarr["labels"]
            croplbl_path = Path(datainfo["crop_group"]) / crop / "labels"
        else:
            croplbl_path = Path(datainfo["crop_group"]) / crop
        labels = zarr.open(croplbl_path).attrs["cellmap"]["annotation"]["class_names"]
        for lbl in labels:
            for sclvl in scalelevels:
                fst.read_xarray(croplbl_path / lbl / sclvl)
    except Exception as e:
        logger.error(f"{e}, crop: {crop}")


def _check_crop_for_raw(
    datainfo: dict, crop: str, scalelevels: tuple[str, ...] = ()
) -> None:
    try:
        if not (Path(datainfo["crop_group"]) / crop / "raw").exists():
            msg = f"{Path(datainfo['crop_group'])/crop/'raw'} does not exist"
            raise ValueError(msg)
        for sclvl in scalelevels:
            fst.read_xarray(Path(datainfo["crop_group"]) / crop / "raw" / sclvl)
    except Exception as e:
        logger.error(f"{e}, crop: {crop}")


@background
def _check_dataset(
    dataname: str,
    datainfo: dict,
    label_scalelevels: tuple[str, ...] = (),
    raw_scalelevels: tuple[str, ...] = (),
) -> None:
    try:
        if "raw" in datainfo:
            # check that raw path exists
            if not Path(datainfo["raw"]).exists():
                msg = f"{Path(datainfo['raw'])} does not exist"
                raise ValueError(msg)
            # check that scale levels are openable for raw
            for sclvl in raw_scalelevels:
                fst.read_xarray(Path(datainfo["raw"]) / sclvl)
        # check that crop group exists
        if not Path(datainfo["crop_group"]).exists():
            msg = f"{Path(datainfo['crop_group'])} does not exist"
            raise ValueError(msg)
        # check each of the crops
        for crop in datainfo["crops"]:
            _check_crop(datainfo, crop, scalelevels=label_scalelevels)
            if "raw" not in datainfo:
                _check_crop_for_raw(datainfo, crop, scalelevels=raw_scalelevels)
    except Exception as e:
        logger.error(f"{e}, {dataname}")


def check_data_yaml_main(
    data_yaml: str,
    label_scalelevels: tuple[str, ...] = (),
    raw_scalelevels: tuple[str, ...] = (),
) -> None:
    """Check that data specified in a data configuration yaml exists and is readable.

    Iterates through crops and raw data in a data configuration file and checks that
    it all exists. Will log any errors that are encountered.

    Args:
        data_yaml (str): Path to data configuration yaml
        label_scalelevels (tuple[str, ...], optional): Scale levels that should be
            checked for labels. Defaults to ().
        raw_scalelevels (tuple[str, ...], optional): Scale levels that should be
            checked for raw data. Defaults to ().

    """
    loop = asyncio.get_event_loop()
    with open(data_yaml) as f:
        datasets = yaml.safe_load(f)["datasets"]
        looper = asyncio.gather(
            *[
                _check_dataset(
                    dataname,
                    datainfo,
                    label_scalelevels=label_scalelevels,
                    raw_scalelevels=raw_scalelevels,
                )
                for dataname, datainfo in datasets.items()
            ]
        )
    loop.run_until_complete(looper)
    logger.info("all done!")
