"""
cellmap-utils.kit.attribute_handler: Utility functions to deal with cellmap
    style attributes.
"""

import json
from typing import Sequence

import h5py
import zarr


def access_attributes(attr: str | dict) -> dict:
    """Decode a nested attribute if it is encoded as a json-string. Otherwise just
    return it.

    Args:
        attr (str | dict): Attribute that potentially needs to be decoded

    Returns:
        dict: Nested attribute as dictionary.

    """
    if isinstance(attr, str):
        cellmap_attr_decoded: dict = json.loads(attr)
        return cellmap_attr_decoded
    return attr


def get_res_dict_from_attrs(
    attrs: h5py.AttributesManager | zarr.attrs.Attributes | dict,
) -> dict[str, Sequence[float | int]]:
    """From attributes that contain multiscales pyramid description get a dictionary
    that goes from the name of the scale array (usually "s0"/"s1") to the scale
    parameter of the corresponding coordinate transformation.

    Args:
        attrs (h5py.AttributesManager | zarr.attrs.Attributes | dict): Attributes
        containing multiscale description

    Returns:
        dict[str, Sequence[float | int]]: Mapping <name of scale array> -> <scale>

    """
    result = {}
    ms_attrs = access_attributes(attrs["multiscales"])
    for ds in ms_attrs[0]["datasets"]:
        for ct in ds["coordinateTransformations"]:
            if ct["type"] == "scale":
                result[ds["path"]] = ct["scale"]
                break
    return result


def get_scalelevel(
    group: h5py.Group | zarr.Group, request_scale: Sequence[float]
) -> str:
    """Find the name of the array in a multiscale pyramid that has a specific scale.

    Args:
        group (h5py.Group | zarr.Group): multiscale group
        request_scale (Sequence[float]): scale of the array you're looking for

    Raises:
        ValueError: If that scale is not in the scale pyramid.

    Returns:
        str: name of the array with the scale `request_scale`

    """
    scales = get_res_dict_from_attrs(group.attrs)
    ref_scale = None
    for sclvl, scale in scales.items():
        if tuple(scale) == tuple(request_scale):
            ref_scale = sclvl
            break
    if ref_scale is None:
        msg = f"Did not find scale level with scale {request_scale} in {group.name}"
        raise ValueError(msg)
    return ref_scale


def add_scalelevel_to_attributes(
    attrs_as_dict: dict, scalelvl: str, scale: list[float], translation: list[float]
) -> dict:
    """Add an additional scale level to attributes describing a multiscale
    pyramid.

    Args:
        attrs_as_dict (dict): Attributes dictionary with multiscales.
        scalelvl (str): The name/path of the scale level that should be added.
        scale (list[float]): The scale factor for the added scale level
        translation (list[float]): The translation for the added scale level.

    Returns:
        dict: Updated attributes dictionary with new scale level added.

    """
    dataset_list = attrs_as_dict["multiscales"][0]["datasets"]
    dataset_list.append(
        {
            "coordinateTransformations": [
                {"type": "scale", "scale": scale},
                {"type": "translation", "translation": translation},
            ],
            "path": scalelvl,
        }
    )
    attrs_as_dict["multiscales"][0]["datasets"] = dataset_list
    return attrs_as_dict


def initialize_multiscale_attributes() -> dict:
    """Return a dictionary representing an empty mutliscale pyramid.

    Returns:
        dict: Dictionary representing an empty multiscale pyramid that can be
            used to initialize attributes.

    """
    axes = [{"name": axname, "type": "space", "unit": "nanometer"} for axname in "zyx"]
    ct = [{"scale": [1.0, 1.0, 1.0], "type": "scale"}]
    attrs = {
        "multiscales": [{"axes": axes, "coordinateTransformations": ct, "datasets": []}]
    }
    return attrs


def extract_single_scale_attrs(attrs_as_dict: dict, scalelvl: str, rename: str) -> dict:
    """Turn attributes describing a multiscale pyramid into a multiscale
    pyramid that just contains a single scale level.

    Args:
        attrs_as_dict (dict): Attributes describing multiscale pyramid.
        scalelvl (str): Name/path of scale level that should be extracted
        rename (str): Name/path for the single scale level in the reduced
            attributes.

    Raises:
        ValueError: If scale level `scalelvl` is not referenced in `attrs_as_dict`.

    Returns:
        dict: Dictionary describing a multiscale pyramid, but now only for a
        single scale level.

    """
    single_attr = next(
        (
            ms
            for ms in attrs_as_dict["multiscales"][0]["datasets"]
            if ms["path"] == scalelvl
        ),
        None,
    )
    if single_attr is not None:
        attrs_as_dict["multiscales"][0]["datasets"] = [single_attr]
    else:
        msg = f"Scale level {scalelvl} not found in attributes."
        raise ValueError(msg)

    if scalelvl != rename:
        attrs_as_dict["multiscales"][0]["datasets"][0]["path"] = rename
    return attrs_as_dict


def get_scale_and_translation(
    attrs_as_dict: dict, scalelvl: str
) -> tuple[list[float], list[float]]:
    """Infer scale and translation of scale level `scalelvl` from dictionary describing
    multiscale attributes `attrs_as_dict`.

    Args:
        attrs_as_dict (dict): Dictionary describing multiscale attributes.
        scalelvl (str): Name of scale level for which to get scale and translation.

    Raises:
        ValueError: If the specified scale level does not exist in the multiscale
            attributes.
        ValueError: If an unknown coordinate transformation type is encountered.
        ValueError: If the translation or scale value is not found for the specified
            scale level.

    Returns:
        tuple[list[float], list[float]]: Two lists representing the scale values and
            translation values, respectively.

    """
    for ms in attrs_as_dict["multiscales"][0]["datasets"]:
        if ms["path"] == scalelvl:
            scale = None
            translation = None
            for ct in ms["coordinateTransformations"]:
                if ct["type"] == "scale":
                    scale = ct["scale"]
                elif ct["type"] == "translation":
                    translation = ct["translation"]
                else:
                    msg = (
                        f"Unknown coordinate transformation type in attributes for "
                        f"{scalelvl}: {ct['type']}"
                    )
                    raise ValueError(msg)
            if translation is None or scale is None:
                msg = (
                    f"Did not find translation and scale value in attributes for "
                    f"{scalelvl}: {translation=}, {scale=}"
                )
                raise ValueError(msg)
            return scale, translation
    msg = f"Did not find attributes for {scalelvl} in {attrs_as_dict}"
    raise ValueError(msg)
