"""
Module for filtering crops based on various conditions: existence of a specific
scale/resolution, a minimum size of the crop and a minimum fraction of pixels/voxels
being annotated.

Key Functions:
-------------
- `filter_yaml_main(data_yaml: str,
                    data_yaml_filtered: str,
                    scale: None | Sequence[float] = None,
                    min_size: None | Sequence[int] = None,
                    min_frac_annotated: None | float = None,
                    labels: Sequence[str] = ()) -> None
    Filter a data configuration YAML file by removing crops that don't fulfill specified
    conditions and save as a new data configuration YAML.

Dependencies:
------------
- copy: Provides utilities for deep copying of data structures.
- pathlib.Path: For handling file paths.
- typing.Sequence: To annotate types for sequences.
- fibsem_tools (fst): A library for interacting with FIB-SEM data formats.
- numpy (np): Used for numerical calculations.
- yaml: To read and write configuration files.
- cellmap_utils_kit.attribute_handler: Used for attribute manipulation of cellmap
  data.

Usage:
------
1. Prepare a YAML configuration file that specifies the datasets and crops you would
   wanna use in principle.
2. Define the conditions that need to be met by a crop to be suitable for your training.
3. Call the `filteR_yaml_main` function to generate a reduced data YAML configuration.
   A CLI is provided in `cellmap_utils_kit.cli`

Example:
-------
```python
filter_yaml_main(
    data_yaml = "/path/to/data_config.yaml"
    filtered_yaml = "/path/to/new_filtered_data_config.yaml."
    scale = [8, 8, 8],
    min_size = [1, 64, 64],
    min_frac_annotated = 1.,
    labels = ["ves", "mito", "er"]
)
```

"""

import copy
from pathlib import Path
from typing import Sequence

import fibsem_tools as fst
import numpy as np
import yaml

from cellmap_utils_kit.attribute_handler import access_attributes, get_scalelevel


def check_res(crop_path: str | Path, check_scale: Sequence[float]) -> bool:
    """Check that crop is annotated at a specific scale.

    Check whether scale `check_scale` is contained in the multiscale pyramids of the
    labels in the crop located at `crop_path`. Assumes that the scale levels are the
    same for all labels (aka just checks for one).

    Args:
        crop_path (str | Path): path to the crop that should be checked
        check_scale (Sequence[float]): scale to check for

    Returns:
        bool: True if there is a scale level with scale `check_scale`. False if there
            isn't

    """
    crop = fst.read(crop_path)
    ref_lbl = access_attributes(crop["labels"].attrs["cellmap"])["annotation"][
        "class_names"
    ][0]
    try:
        get_scalelevel(crop["labels"][ref_lbl], check_scale)
    except ValueError:
        return False
    return True


def check_min_size(
    crop_path: str,
    min_size: Sequence[int],
    at_scale: None | Sequence[float] = None,
) -> bool:
    """Check crop for minimum size.

    Checks that the labels in the crop_path have at least size `min_size` at the scale
    level with scale `at_scale`. If `at_scale` is None (default) it checks for "s0".

    Args:
        crop_path (str): path to the crop that should be checked
        min_size (Sequence[int]): minimum size of labels in that crop
        at_scale (None | Sequence[float], optional): scale for which size should be
            checked. If None checks at s0. Defaults to None.

    Returns:
        bool: True if labels in the crop at the specified scale are at least of size
            `min_size`. Otherwise False

    """
    crop = fst.read(crop_path)
    ref_lbl = access_attributes(crop["labels"].attrs["cellmap"])["annotation"][
        "class_names"
    ][0]
    if at_scale is None:
        ref_scale = "s0"
    else:
        ref_scale = get_scalelevel(crop["labels"][ref_lbl], at_scale)
    ref_arr = fst.read(Path(crop_path) / "labels" / ref_lbl / ref_scale)
    if all(sh >= min_sh for sh, min_sh in zip(ref_arr.shape, min_size)):
        return True
    else:
        return False


def check_annotated_label(
    crop_path: str,
    label: str,
    min_frac_annotated: float,
    at_scale: None | Sequence[float] = None,
) -> bool:
    """Check that a minimum number of voxels in crop are annotated for the given label.

    For the crop located at `crop_path` find the label array for class `label` and check
    attributes for the fraction of annotated elements (not "unknown"). If at least
    `min_frac_annotated` are "present" or "absent" returns True. Otherwise
    returns False. The parameter `at_scale` controls which scale level to look at. If
    it is set to None "s0" will be used.

    Args:
        crop_path (str): path to the crop that should be checked
        label (str): name of the label that should be checked
        min_frac_annotated (float): minimum fraction of voxels that need to be annotated
            for crop to pass
        at_scale (None | Sequence[float], optional): scale for which annotation should
            be checked. If None checks at s0. Defaults to None.

    Raises:
        ValueError: If `min_frac_annotated` is larger than 1
    Returns:
        bool: True if at least `min_frac_annotated` of the voxels in the specified
        array of the crop are annotated. Otherwise False.

    """
    if min_frac_annotated > 1:
        msg = (
            "`min_frac_annotated` should be given as a fraction but got a value"
            f"larger than 1: {min_frac_annotated}. Did you use percent?"
        )
        raise ValueError(msg)
    crop = fst.read(crop_path)
    if (
        label
        not in access_attributes(crop["labels"].attrs["cellmap"])["annotation"][
            "class_names"
        ]
    ):
        return False
    if at_scale is None:
        ref_scale = "s0"
    else:
        ref_scale = get_scalelevel(crop["labels"][label], at_scale)
    ref_attrs = access_attributes(crop["labels"][label][ref_scale].attrs["cellmap"])[
        "annotation"
    ]
    num_elements = np.prod(crop["labels"][label][ref_scale].shape).item()
    if "unknown" in ref_attrs["complement_counts"]:
        num_annotated = num_elements - ref_attrs["complement_counts"]["unknown"]
    else:
        num_annotated = num_elements
    frac_annotated = num_annotated / num_elements
    if frac_annotated >= min_frac_annotated:
        return True
    else:
        return False


def check_annotated(
    crop_path: str,
    labels: Sequence[str],
    min_frac_annotated: float,
    at_scale: None | Sequence[float] = None,
) -> bool:
    """Check that a minimum number of voxels in crop are annotated for all given labels.

    For the crop located at `crop_path` find each of the label arrays in `labels` and
    check their attributes for the fraction of annotated elements (not "unknown"). If at
    least `min_frac_annotated` are "present" or "absent" for every label return True.
    Otherwise return False. The parameter `at_scale` controls which scale level to look
    at. If it is None, "s0" will be used.

    Args:
        crop_path (str): path to the crop that should be checked
        labels (Sequence[str]): names of all the labels that should be checked
        min_frac_annotated (float): minimum fraction of voxels that need to be annotated
            in each label array for crop to pass
        at_scale (None | Sequence[float], optional): scale for which annotation should
            be checked. If None checks at s0. Defaults to None.

    Returns:
        bool: True if at least `min_frac_annotated` of the voxels in each of the
            specified label arrays of the crop are annotated. Otherwise False.

    """
    for label in labels:
        if not check_annotated_label(
            crop_path, label, min_frac_annotated, at_scale=at_scale
        ):
            return False
    return True


def filter_crop(
    crop_path: str,
    scale: None | Sequence[float] = None,
    min_size: None | Sequence[int] = None,
    min_frac_annotated: None | float = None,
    labels: Sequence[str] = (),
) -> bool:
    """Check crop for various conditions (scale, min size, min annotated fraction).

    Check crop located at `crop_path` for whether it is annotated at scale `scale`, is
    at least of size `min_size` at that scale and has at least `min_frac_annotated` of
    its voxels annotated at that scale for each of the labels in `labels`. Each check
    can be turned off by setting the corresponding condition to None. If `scale` is set
    to None remaining checks will be done at scale level "s0".

    Args:
        crop_path (str): path to the crop that should be checked
        scale (None | Sequence[float], optional): scale to check for and that should be
            used for size and annotation checks. If None, the crop will not be checked
            for the existence of a specific scale and other checks will be done at full
            scale ("s0"). Defaults to None.
        min_size (None | Sequence[int], optional): minimum size of labels in that crop.
            If None crop will not be checked to have a minimum size. Defaults to None.
        min_frac_annotated (None | float, optional): minimum fraction of voxels that
            need to be annotated for each of the specified labels for crop to pass. If
            None crop will not be checked to have a minimum annotated fraction. Defaults
            to None.
        labels (Sequence[str], optional): Labels for which to check annotated fraction.
            Defaults to ().

    Returns:
        bool: True if all checks passed. If any of them fail False.

    """
    keep_crop = True
    if keep_crop and scale is not None:
        keep_crop = check_res(crop_path, scale)
    if keep_crop and min_size is not None:
        keep_crop = check_min_size(crop_path, min_size, at_scale=scale)
    if keep_crop and min_frac_annotated is not None:
        keep_crop = check_annotated(
            crop_path, labels, min_frac_annotated, at_scale=scale
        )
    return keep_crop


def filter_yaml_main(
    data_yaml: str,
    data_yaml_filtered: str,
    scale: None | Sequence[float] = None,
    min_size: None | Sequence[int] = None,
    min_frac_annotated: None | float = None,
    labels: Sequence[str] = (),
) -> None:
    """Make new data config yaml with crops removed that don't fulfill conditions.

    Checks all the crops in the data configuration yaml and checks them for the
    conditions (as described in `filter_crop`). Crops that don't pass the checks are
    not included in the new data configuration yaml.

    Args:
        data_yaml (str): Path to data configuration yaml
        data_yaml_filtered (str): Path in which to save filtered data configuration yaml
        scale (None | Sequence[float], optional): scale to check for and that should be
            used for size and annotation checks. If None, crops will not be checked for
            the existence of a specific scale and other checks will be done at full
            scale ("s0"). Defaults to None.
        min_size (None | Sequence[int], optional): minimum size for crops. If None crops
            will not be checked to have a minimum size. Defaults to None.
        min_frac_annotated (None | float, optional): minimum fraction of voxels that
            need to be annotated for each of the specified labels for a crop to pass. If
            None crops will not be checked to have a minimum annotated fraction.
            Defaults to None.
        labels (Sequence[str], optional): Labels for which to check annotated fraction.
            Defaults to ().

    """
    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)
        data_config_filtered = copy.deepcopy(data_config)
        datasets = data_config["datasets"]
        for dataname, datainfo in datasets.items():
            crops_filtered = []
            for crop in datainfo["crops"]:
                if filter_crop(
                    Path(datainfo["crop_group"]) / crop,
                    scale=scale,
                    min_size=min_size,
                    min_frac_annotated=min_frac_annotated,
                    labels=labels,
                ):
                    crops_filtered.append(crop)
            if len(crops_filtered) > 0:
                data_config_filtered["datasets"][dataname]["crops"] = crops_filtered
            else:
                del data_config_filtered["datasets"][dataname]
    with open(data_yaml_filtered, "w") as f:
        yaml.safe_dump(data_config_filtered, f)
