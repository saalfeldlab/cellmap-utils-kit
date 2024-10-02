"""
Module for correcting label attributes in CellMap dataset crops.

This module provides functionality to correct the `complement_counts` attributes
for labels in dataset crops listed in a YAML configuration file. It ensures the
attributes are properly updated, specifically the counts for "absent" and "unknown"
elements within the labeled data. The module is designed to work asynchronously
to efficiently process multiple crops concurrently.

Key Functions:
--------------
- `correct_label_attrs_main(data_yaml: str) -> None`: Read from a data configuration
  YAML file and correct label attributes for each crop listed.

Dependencies:
-------------
- asyncio: For managing asynchronous processing of crops.
- logging: For logging correction operations.
- pathlib.Path: For handling file paths.
- fibsem_tools (fst): A library for reading and accessing FIB-SEM data.
- numpy (np): For performing numerical operations like element counts.
- yaml: For reading the dataset configuration from a YAML file.
- cellmap_utils_kit.parallel_utils: Provides the `background` decorator to run functions
  asynchronously.

Usage:
------
1. Prepare a YAML configuration file specifying the dataset crops and paths for which
   label attributes need correction.
2. Call the `correct_label_attrs_main` function with the path to the YAML file to
   correct the label attributes for each crop.

Example:
-------
```python
correct_label_attrs_main("path/to/data_config.yaml")
```

"""

import asyncio
import logging
from pathlib import Path

import fibsem_tools as fst
import numpy as np
import yaml

from cellmap_utils_kit.parallel_utils import background

logger = logging.getLogger(__name__)


@background
def _correct_label_attrs(crop_path: str | Path) -> None:
    crop = fst.access(crop_path, "a")
    if "labels" in crop:
        crop = crop["labels"]
    labels = crop.attrs["cellmap"]["annotation"]["class_names"]
    for label in labels:
        logger.info(f"Correcting attributes in {crop_path} for {label}")
        for lvl in crop[label].keys():
            src = crop[label][lvl]
            srcarr = np.array(src)
            attrs_as_dict = src.attrs.asdict()
            encoding = src.attrs["cellmap"]["annotation"]["annotation_type"]["encoding"]
            num_unknown = np.sum(srcarr == encoding["unknown"])
            thr = abs(encoding["present"] - encoding["absent"]) / 2.0
            if encoding["present"] > encoding["absent"]:
                num_absent = np.sum(srcarr <= thr)
            else:
                num_present = np.sum(srcarr <= thr)
                num_absent = np.product(srcarr.shape - num_unknown - num_present)
            attrs_as_dict["cellmap"]["annotation"]["complement_counts"]["absent"] = int(
                num_absent
            )
            attrs_as_dict["cellmap"]["annotation"]["complement_counts"]["unknown"] = (
                int(num_unknown)
            )
            src.attrs.put(attrs_as_dict)


def correct_label_attrs_main(data_yaml: str) -> None:
    """Coorect the complement_counts attributes for labels in crops listed in data
    configuration yaml.

    Can handle smoothly downsample labels. The values are binarized (threshold halfway
    between present and absent) to count "absent".

    Args:
        data_yaml (str): Data configuration yaml.

    """
    loop = asyncio.get_event_loop()
    task_list = []
    with open(data_yaml) as f:
        datasets = yaml.safe_load(f)["datasets"]
        for datainfo in datasets.values():
            for crop in datainfo["crops"]:
                task_list.append(
                    _correct_label_attrs(Path(datainfo["crop_group"]) / crop)
                )
    looper = asyncio.gather(*task_list)
    loop.run_until_complete(looper)
