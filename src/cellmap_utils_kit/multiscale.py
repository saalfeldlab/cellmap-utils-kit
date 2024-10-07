"""
Module for generating multiscale pyramids for labeled and raw data from CellMap crops.

This module provides functionality to generate a multiscale representation (pyramid)
for labels and raw datasets described in a configuration YAML file. It uses
downsampling with smooth averaging for label datasets, allowing for "soft" labels where
there is a transition between "present" and "absent" values. The module leverages
asynchronous processing to manage multiple crops in parallel.

Key Functions:
--------------
- `check_encoding(encoding: dict[Literal["present", "absent", "unknown"],
   int]) -> None`:
    Validates the encoding attribute to ensure smooth downsampling is valid by checking
    that "unknown" values can be safely maintained after downsampling.

- `smooth_multiscale_labels_main(data_yaml: str, num_scales: int = 4) -> None`:
    Creates a multiscale pyramid for labeled datasets specified in the provided
    configuration YAML. Supports smooth downsampling with maintenance of "unknown"
    values.

- `smooth_multiscale_raw_main(data_yaml: str, num_scales: int = 4, concurrence: int = 4)
   -> None`:
    Creates a multiscale pyramid for raw datasets from the provided YAML configuration.
    The number of concurrent processes for creating multiscales can be adjusted.

Dependencies:
-------------
- asyncio: For managing asynchronous tasks.
- itertools: To help iterate over scale levels for creating pyramids.
- pathlib: For working with filesystem paths.
- skimage: For performing downscaling on image data.
- yaml: To read the dataset configuration from a YAML file.
- fibsem_tools: For handling dataset access.
- numcodecs: To apply compression while saving Zarr data.

Usage:
------
1. Prepare a YAML configuration file specifying the datasets with crop groups.
2. Call the `smooth_multiscale_labels_main` or `smooth_multiscale_raw_main` function
   with the path to the YAML file to generate a multiscale representation for labels or
   raw data respectively. CLIs are provided in `cellmap_utils_kit.cli`.

Example:
-------
```python
smooth_multiscale_labels_main("path/to/config.yaml", num_scales=4)
smooth_multiscale_raw_main("path/to/config.yaml", num_scales=4, concurrence=2)
```

"""

import asyncio
import itertools
import logging
from pathlib import Path
from typing import Literal

import fibsem_tools as fst
import numcodecs
import numpy as np
import skimage
import yaml

from cellmap_utils_kit.attribute_handler import (
    add_scalelevel_to_attributes,
    get_scale_and_translation,
)
from cellmap_utils_kit.parallel_utils import background

logger = logging.getLogger(__name__)


def check_encoding(
    encoding: dict[Literal["present", "absent", "unknown"], int],
) -> None:
    """Check encoding attribute to make sure smooth downsampling is valid. This
    function takes the encoding attribute from the cellmap segmentation attribute
    schema and checks that downsampling can be done safely by a factor of 2 (i.e.,
    "unknown" values remain "unknown").Assumes 3D data.

    Args:
        encoding (dict[Literal["present", "absent", "unknown"], int]): Dictionary
            mapping descriptors "present", "absent" and "unknown" to their integer
            representation.

    Raises:
        ValueError: If unknown values cannot safely be maintained after downsampling.

    """
    if encoding["unknown"] <= 8 * max(encoding["present"], encoding["absent"]):
        msg = "smoothing relies on large value for UNKNOWN"
        raise ValueError(msg)


@background
def _smooth_multiscale_labels(crop_path: str | Path, num_scales: int = 4) -> None:
    crop = fst.access(crop_path, "a")
    if "labels" in crop:
        crop = crop["labels"]
    labels = crop.attrs["cellmap"]["annotation"]["class_names"]
    for label in labels:
        logger.info(f"Processing {crop_path} for {label}")
        scales = [f"s{k}" for k in range(num_scales)]
        for l1, l2 in itertools.pairwise(scales):
            src = crop[f"{label}/{l1}"]
            if (
                src.attrs["cellmap"]["annotation"]["annotation_type"]["type"]
                != "semantic_segmentation"
            ):
                msg = (
                    f"smooth multiscaling not implemented for annotations of type "
                    f"{src.attrs['cellmap']['annotation']['annotation_type']['type']}"
                )
                raise NotImplementedError(msg)
            encoding = src.attrs["cellmap"]["annotation"]["annotation_type"]["encoding"]
            check_encoding(encoding)
            down = skimage.transform.downscale_local_mean(src[:], 2).astype("float32")
            downslice = tuple(slice(sh) for sh in (np.array(down.shape) // 2) * 2)
            down = down[downslice]
            down[down > max(encoding["present"], encoding["absent"])] = encoding[
                "unknown"
            ]
            attrs_as_dict = crop[f"{label}/{l1}"].attrs.asdict()
            chunksize = (1, *down.shape[1:])
            crop[label].create_dataset(
                l2,
                data=down,
                overwrite=True,
                dimension_separator="/",
                compressor=numcodecs.Zstd(level=3),
                chunks=chunksize,
            )

            attrs_as_dict["cellmap"]["annotation"]["complement_counts"]["absent"] = (
                round(
                    np.sum(encoding["present"] - down[down != encoding["unknown"]]), 2
                )
            )
            attrs_as_dict["cellmap"]["annotation"]["complement_counts"]["unknown"] = (
                np.sum(down == encoding["unknown"])
            )
            crop[label][l2].attrs.put(attrs_as_dict)
            l1_scale, l1_translation = get_scale_and_translation(
                crop[label].attrs.asdict(), l1
            )
            l2_scale = [sc * 2 for sc in l1_scale]
            l2_translation = [
                (sc * 0.5) + tr for sc, tr in zip(l1_scale, l1_translation)
            ]
            new_attrs = add_scalelevel_to_attributes(
                crop[label].attrs.asdict(), l2, l2_scale, l2_translation
            )
            crop[label].attrs.put(new_attrs)


@background
def _smooth_multiscale_raw(crop_path: str | Path, num_scales: int = 4) -> None:
    crop = fst.access(crop_path, "a")
    logger.info(f"Processing {crop_path} for raw")
    scales = [f"s{k}" for k in range(num_scales)]
    for l1, l2 in itertools.pairwise(scales):
        src = crop[f"raw/{l1}"]
        down = skimage.transform.downscale_local_mean(src[:], 2).astype("float32")
        downslice = tuple(slice(sh) for sh in (np.array(down.shape) // 2) * 2)
        down = down[downslice]
        chunksize = (1, *down.shape[1:])
        crop["raw"].create_dataset(
            l2,
            data=down,
            overwrite=True,
            dimension_separator="/",
            compressor=numcodecs.Zstd(level=3),
            chunks=chunksize,
        )
        l1_scale, l1_translation = get_scale_and_translation(
            crop["raw"].attrs.asdict(), l1
        )
        l2_scale = [sc * 2 for sc in l1_scale]
        l2_translation = [(sc * 0.5) + tr for sc, tr in zip(l1_scale, l1_translation)]
        new_attrs = add_scalelevel_to_attributes(
            crop["raw"].attrs.asdict(), l2, l2_scale, l2_translation
        )
        crop["raw"].attrs.put(new_attrs)


def smooth_multiscale_labels_main(
    data_yaml: str, num_scales: int = 4, max_concurrency: int | None = None
) -> None:
    """Generate multiscale pyramid for labels.

    This function will generate a multiscale pyramid for labels but allows for soft
    labels. That means the downsampled versions will not just contain integer values,
    but a smooth transition from "present" to "absent". The "unknown" values however are
    maintained as integers. Existence of s0 is assumed.

    Args:
        data_yaml (str): Data configuration yaml
        num_scales (int, optional): Desired number of scale levels. Defaults to 4.
        max_concurrency (int, optional): Maximum number of concurrent processes. If
            None, no limit is set. Defaults to None.

    """
    loop = asyncio.get_event_loop()
    all_funcs = []
    with open(data_yaml) as f:
        datasets = yaml.safe_load(f)["datasets"]
        for datainfo in datasets.values():
            all_funcs.append(
                asyncio.gather(
                    *[
                        _smooth_multiscale_labels(
                            Path(datainfo["crop_group"]) / crop, num_scales=num_scales
                        )
                        for crop in datainfo["crops"]
                    ]
                )
            )
            if len(all_funcs) == max_concurrency:
                looper = asyncio.gather(*all_funcs)
                loop.run_until_complete(looper)
                all_funcs = []
    looper = asyncio.gather(*all_funcs)
    loop.run_until_complete(looper)


def smooth_multiscale_raw_main(
    data_yaml: str, num_scales: int = 4, max_concurrency: int | None = None
) -> None:
    """Generate multiscale pyramid for raw data.

    This function will generate a multiscale pyramid for raw data. Existence of s0 is
    assumed.

    Args:
        data_yaml (str): Data configuration yaml
        num_scales (int, optional): Desired number of scale levels. Defaults to 4.
        max_concurrency (int, optional): Maximum number of concurrent processes. If
            None, no limit is set. Defaults to None.

    """
    loop = asyncio.get_event_loop()
    task_list = []
    with open(data_yaml) as f:
        datasets = yaml.safe_load(f)["datasets"]
        for datainfo in datasets.values():
            if "raw" in datainfo:
                msg = (
                    "The data configuration yaml contains a dataset-level path for "
                    "raw data. Consider removing this. This script will attempt to "
                    "multiscale `raw` group in each crop."
                )
                logger.warning(msg)
            for crop in datainfo["crops"]:
                task_list.append(
                    _smooth_multiscale_raw(
                        Path(datainfo["crop_group"]) / crop, num_scales=num_scales
                    )
                )
                if len(task_list) == max_concurrency:
                    looper = asyncio.gather(*task_list)
                    loop.run_until_complete(looper)
                    task_list = []
        looper = asyncio.gather(*task_list)
        loop.run_until_complete(looper)
